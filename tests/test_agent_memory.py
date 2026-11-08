"""tests/test_agent_memory.py — 双层 memory（TwoLayerMemory）无 GPU 单测。

运行: python -m pytest tests/test_agent_memory.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.agent_memory import (  # noqa: E402
    TwoLayerMemory, fallback_classify, legacy_to_v2, _extract_json,
)

REQ_DE = '语言为德语，说话人数量为1，医疗任务为喉炎检测。我希望匿名化后的MOS相对原始语音有提升。'
REQ_EN = '语言为英语，说话人数量为2，医疗任务为儿童语言障碍检测。我希望匿名化后的下游分类BACC不显著衰退。'
REQ_DEPRESS = '语言为中文，说话人数量为1，医疗任务为抑郁检测。我希望匿名化后的下游分类BACC不显著衰退。'

HIST = [
    {'vc_method': 'Resyn', 'metrics': {'mos': 3.5, 'cer': 0.03}, 'reason': 'tts 音质好'},
    {'vc_method': 'formant_spec', 'metrics': {'mos': 2.1, 'cer': 0.16}, 'reason': 'formant_spec 保病理特征'},
]


def _record(mem, input_dir, req, final='formant_spec', history=None, **kw):
    return mem.record_run(
        input_dir=input_dir, language='德语', request=req, output_dir='/tmp/o',
        history=history if history is not None else HIST,
        final_vc_method=final, task_meta=fallback_classify(req), llm_call=None, **kw)


# ── fallback_classify ────────────────────────────────────────────────────────

def test_fallback_classify_full_template():
    m = fallback_classify(REQ_DE)
    assert m['language'] == '德语'
    assert m['n_speakers'] == 1
    assert m['downstream_task'] == '喉炎检测'
    assert m['task_type'] == 'voice_pathology'
    assert m['goal'].startswith('匿名化后')
    assert m['warnings'] == []


@pytest.mark.parametrize('req,expected', [
    (REQ_DEPRESS, 'depression'),
    ('语言为中文，说话人数量为1，医疗任务为阿尔茨海默病三分类。我希望EER高于40%。', 'dementia'),
    ('语言为西班牙语，说话人数量为1，医疗任务为帕金森检测。我希望F0轮廓相关性PCC_F0在0.80以上。', 'parkinson'),
    (REQ_EN, 'developmental_language_disorder'),
])
def test_fallback_classify_task_types(req, expected):
    assert fallback_classify(req)['task_type'] == expected


def test_fallback_classify_missing_fields_no_guess():
    m = fallback_classify('随便一句话，没有任务信息')
    assert m['task_type'] is None
    assert m['language'] is None
    assert m['n_speakers'] is None
    assert m['downstream_task'] is None
    # 缺失要进 warnings，而不是被编造
    assert any('language' in w for w in m['warnings'])
    assert any('task_type' in w for w in m['warnings'])


def test_fallback_classify_unknown_domain_maps_other():
    m = fallback_classify('语言为俄语，说话人数量为2，医疗任务为年龄性别预测。我希望BACC不衰退。')
    assert m['task_type'] == 'other'
    assert any('词表' in w for w in m['warnings'])


# ── record / upsert / 持久化 ─────────────────────────────────────────────────

def test_record_run_writes_layer2_file_and_experience(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    run = _record(mem, '/x/SVD-L/wav', REQ_DE)
    assert os.path.isfile(os.path.join(str(tmp_path), 'runs', run['run_id'] + '.json'))
    assert len(mem.experiences) == 1
    exp = mem.experiences[0]
    assert exp['task_type'] == 'voice_pathology'
    assert exp['run_ids'] == [run['run_id']]
    assert exp['lesson']  # 规则模式也有 lesson
    assert exp['recommended_method'] == 'formant_spec'


def test_upsert_same_task_merges_runs(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    r1 = _record(mem, '/x/SVD-L/wav', REQ_DE)
    r2 = _record(mem, '/x/SVD-L/wav', REQ_DE, baseline_bacc=0.486)
    assert r1['run_id'] != r2['run_id']        # 同秒也不冲突
    assert len(mem.experiences) == 1
    exp = mem.experiences[0]
    assert sorted(exp['run_ids']) == sorted([r1['run_id'], r2['run_id']])
    assert exp['n_runs'] == 2


def test_different_task_same_dataset_separate_experiences(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    _record(mem, '/x/SVD-L/wav', '语言为德语，说话人数量为1，医疗任务为声带麻痹检测。我希望BACC不衰退。')
    assert len(mem.experiences) == 2


def test_reload_from_disk(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    _record(mem, '/x/ENNI/wav', REQ_EN)
    mem2 = TwoLayerMemory(root=str(tmp_path))
    assert len(mem2.runs) == 2
    assert len(mem2.experiences) == 2
    assert mem2.experiences[0]['task_type'] in ('voice_pathology', 'developmental_language_disorder')


def test_llm_failure_still_writes_run(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    run = mem.record_run(
        input_dir='/x/ENNI/wav', language='英文', request=REQ_EN, output_dir='/tmp/o',
        history=HIST, final_vc_method='NeuralVC', task_meta=fallback_classify(REQ_EN),
        llm_call=lambda s, u: '思考中... {不完整的')
    assert os.path.isfile(os.path.join(str(tmp_path), 'runs', run['run_id'] + '.json'))
    assert mem.experiences == []               # LLM 失败 → Layer-1 不写


def test_llm_success_writes_experience(tmp_path):
    def good_llm(sys_prompt, user_prompt):
        return json.dumps({
            'task_type': 'voice_pathology',
            'lesson': 'TTS 重合成会抹掉病理特征，BACC 明显下降；最终 formant_spec 在保 bacc 与可懂度间最平衡。',
            'recommended_method': 'formant_spec',
            'avoid_methods': ['Resyn'],
        }, ensure_ascii=False)
    mem = TwoLayerMemory(root=str(tmp_path))
    mem.record_run(input_dir='/x/SVD-L/wav', language='德语', request=REQ_DE,
                   output_dir='/tmp/o', history=HIST, final_vc_method='formant_spec',
                   task_meta=fallback_classify(REQ_DE), llm_call=good_llm)
    assert len(mem.experiences) == 1
    exp = mem.experiences[0]
    assert exp['lesson'].startswith('TTS 重合成')
    assert exp['avoid_methods'] == ['Resyn']
    assert exp['task_type_source'] == 'llm'


def test_llm_can_register_new_task_type(tmp_path):
    """agent 遇到词表外新领域：合法 snake_case 自动入表并用于经验。"""
    def new_type_llm(sys_prompt, user_prompt):
        return json.dumps({
            'task_type': 'dysarthria',
            'task_type_desc': '构音障碍检测',
            'lesson': '构音障碍依赖精细发音特征，TTS 重合成不可逆破坏构音细节，应避免。',
            'recommended_method': 'formant_spec',
            'avoid_methods': ['Resyn'],
        }, ensure_ascii=False)
    mem = TwoLayerMemory(root=str(tmp_path))
    meta = {'task_type': None, 'language': '中文', 'n_speakers': 1,
            'downstream_task': '构音障碍检测', 'goal': None,
            'source': 'llm', 'warnings': []}
    mem.record_run(input_dir='/x/pumch/wav', language='中文', request='r',
                   output_dir='/tmp/o', history=HIST, final_vc_method='formant_spec',
                   task_meta=meta, llm_call=new_type_llm)
    exp = mem.experiences[0]
    assert exp['task_type'] == 'dysarthria'
    assert 'dysarthria' in mem.task_types
    assert mem.task_types['dysarthria'] == '构音障碍检测'
    # 持久化：重新加载仍在
    mem2 = TwoLayerMemory(root=str(tmp_path))
    assert 'dysarthria' in mem2.task_types


def test_invalid_new_task_type_falls_to_other(tmp_path):
    def bad_type_llm(sys_prompt, user_prompt):
        return json.dumps({
            'task_type': 'Bad-Type!',
            'lesson': '构音障碍依赖精细发音特征，重合成不可逆破坏构音细节，应避免重合成方法。',
            'recommended_method': None, 'avoid_methods': None,
        }, ensure_ascii=False)
    mem = TwoLayerMemory(root=str(tmp_path))
    meta = {'task_type': None, 'language': '中文', 'n_speakers': 1,
            'downstream_task': '构音障碍检测', 'goal': None,
            'source': 'llm', 'warnings': []}
    mem.record_run(input_dir='/x/pumch/wav', language='中文', request='r',
                   output_dir='/tmp/o', history=HIST, final_vc_method='formant_spec',
                   task_meta=meta, llm_call=bad_type_llm)
    assert mem.experiences[0]['task_type'] == 'other'
    assert 'Bad-Type!' not in mem.task_types


def test_lesson_stored_in_full(tmp_path):
    """lesson 不截断：>200 字全文入库。"""
    long_lesson = ('构音障碍检测依赖精细发音特征。' * 20) + '结尾。'
    def long_llm(sys_prompt, user_prompt):
        return json.dumps({
            'task_type': 'articulation_disorder',
            'lesson': long_lesson,
            'recommended_method': 'formant_spec',
            'avoid_methods': ['Resyn'],
        }, ensure_ascii=False)
    mem = TwoLayerMemory(root=str(tmp_path))
    meta = fallback_classify('语言为中文，说话人数量为1，医疗任务为腭裂构音障碍检测。我希望BACC不衰退。')
    mem.record_run(input_dir='/x/pumch/wav', language='中文', request='r',
                   output_dir='/tmp/o', history=HIST, final_vc_method='formant_spec',
                   task_meta=meta, llm_call=long_llm)
    assert mem.experiences[0]['lesson'] == long_lesson


# ── recall ───────────────────────────────────────────────────────────────────

def test_recall_empty_memory(tmp_path):
    assert TwoLayerMemory(root=str(tmp_path)).recall_for_prompt() == ''


def test_recall_two_layers_and_filtering(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    _record(mem, '/x/ENNI/wav', REQ_EN)
    blk = mem.recall_for_prompt(task_meta=fallback_classify(REQ_DE))
    assert 'Layer-1' in blk and 'Layer-2' in blk
    assert 'voice_pathology' in blk
    # Layer-1 全量注入（两条任务经验都在），Layer-2 只注同类
    assert 'developmental_language_disorder' in blk
    assert blk.count('- [') == 3   # L1 两条 + L2 一条


def test_recall_layer2_cap_six(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    for i in range(8):
        _record(mem, f'/x/SVD-L{i}/wav', REQ_DE)
    blk = mem.recall_for_prompt(task_meta=fallback_classify(REQ_DE))
    l2 = blk.split('【Layer-2')[1]
    assert l2.count('- [') == 6


def test_recall_meta_none_falls_back_to_recent(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    blk = mem.recall_for_prompt(task_meta=None)
    assert 'Layer-2' in blk and 'SVD-L' in blk


def test_recall_no_same_type_notes_other_domains(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    blk = mem.recall_for_prompt(task_meta=fallback_classify(REQ_DEPRESS))
    assert '暂无同领域记录' in blk


def test_recall_exclude_dataset(tmp_path):
    mem = TwoLayerMemory(root=str(tmp_path))
    _record(mem, '/x/SVD-L/wav', REQ_DE)
    blk = mem.recall_for_prompt(task_meta=fallback_classify(REQ_DE), exclude_dataset='SVD-L')
    assert 'SVD-L' not in blk


# ── legacy 兼容 ──────────────────────────────────────────────────────────────

def test_legacy_single_file_readonly_and_migrate(tmp_path):
    legacy = {
        'runs': [{
            'run_id': 'DAIC-WOZ_backfill_20260828-141407', 'dataset': 'DAIC-WOZ',
            'language': '英文', 'request': '(backfill from evaluate CSVs)',
            'output_dir': '/o', 'timestamp': '2026-08-28T14:14:07', 'attempts': 7,
            'final_vc_method': None, 'success': True,
            'vc_results': {'pitch': {'mos': 1.24}}, 'reflections': [],
        }]
    }
    legacy_path = tmp_path / 'agent_memory.json'
    legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf-8')

    mem = TwoLayerMemory(root=str(legacy_path))
    # 只读并入，参与召回
    blk = mem.recall_for_prompt(task_meta=fallback_classify(REQ_DEPRESS))
    assert 'DAIC-WOZ' in blk
    assert '暂无同领域记录' in blk
    # 原文件不被改写
    assert json.loads(legacy_path.read_text(encoding='utf-8')) == legacy

    # 迁移到新布局
    new_root = tmp_path / 'memory'
    mem2 = TwoLayerMemory(root=str(legacy_path))
    mem2.migrate_legacy_runs()
    assert (new_root.parent / (legacy_path.stem + '_v2') / 'runs').is_dir() or \
           [p for p in tmp_path.rglob('DAIC-WOZ_backfill*.json')]


def test_legacy_to_v2_schema():
    old = {'run_id': 'x_1', 'dataset': 'x', 'language': '英文',
           'request': '语言为英语，说话人数量为2，医疗任务为抑郁检测。我希望BACC不衰退。',
           'vc_results': {'NeuralVC': {'mos': 2.0}}, 'reflections': ['r']}
    v2 = legacy_to_v2(old)
    assert v2['schema_version'] == 2
    assert v2['task_type'] == 'depression'
    assert v2['task_meta']['downstream_task'] == '抑郁检测'
    assert v2['vc_results'] == old['vc_results']
    assert legacy_to_v2({'no_run_id': 1}) is None


# ── _extract_json（思考模型残留的容错）───────────────────────────────────────

def test_extract_json_plain():
    assert _extract_json('{"a": 1}')['a'] == 1


def test_extract_json_with_think_prefix():
    raw = '思考: 用户要经验总结。\n{"task_type": "depression", "lesson": "ok"}'
    assert _extract_json(raw)['task_type'] == 'depression'


def test_extract_json_takes_last_balanced_block():
    # 思考文本里含平衡小块 + 真正 JSON 在最后 → 取最后一个平衡块
    raw = 'think: 方案 {a:1} 不行。结论 {"task_type": "other", "lesson": "真正输出"}'
    assert _extract_json(raw)['lesson'] == '真正输出'


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json('完全没有大括号的输出')


# ── task_types.json 词表扩展 ────────────────────────────────────────────────

def test_task_types_file_extends_vocab(tmp_path):
    (tmp_path / 'task_types.json').write_text(
        json.dumps({'vocab': {'dysarthria': '构音障碍检测'}}, ensure_ascii=False), encoding='utf-8')
    mem = TwoLayerMemory(root=str(tmp_path))
    assert 'dysarthria' in mem.task_types


# ── LOO 消融：recall_for_prompt(exclude_dataset=) 排除验证 ──────────────────

def _build_loo_mem(tmp_path):
    """3 数据集双层 memory：ADReSS(dementia) / Ivanova(dementia) / Crisis(crisis_risk)"""
    root = tmp_path / 'mem'
    (root / 'runs').mkdir(parents=True)
    exps = [
        {'exp_id': f'{d}__{t}', 'dataset': d, 'task_type': t, 'language': 'x',
         'lesson': f'来自{d}的教训', 'recommended_method': 'pitch',
         'avoid_methods': [], 'n_runs': 1, 'updated_at': f'2026-10-0{i}'}
        for i, (d, t) in enumerate(
            [('ADReSS', 'dementia'), ('Ivanova', 'dementia'), ('Crisis', 'crisis_risk')], 1)
    ]
    (root / 'experiences.json').write_text(
        json.dumps({'version': 2, 'experiences': exps}, ensure_ascii=False), encoding='utf-8')
    for d, t in [('ADReSS', 'dementia'), ('Ivanova', 'dementia')]:
        (root / 'runs' / f'{d}_20261001-000000.json').write_text(
            json.dumps({'run_id': f'{d}_x', 'dataset': d, 'task_type': t,
                        'final_vc_method': 'pitch', 'vc_results': {}}, ensure_ascii=False),
            encoding='utf-8')
    return TwoLayerMemory(root=str(root))


def test_recall_loo_excludes_dataset(tmp_path):
    mem = _build_loo_mem(tmp_path)
    meta = {'task_type': 'dementia', 'dataset': 'ADReSS'}
    full = mem.recall_for_prompt(task_meta=meta)
    assert 'ADReSS' in full and 'Ivanova' in full
    loo = mem.recall_for_prompt(task_meta=meta, exclude_dataset='ADReSS')
    assert 'ADReSS' not in loo, 'LOO 失败：被排除数据集的经验仍泄漏'
    assert 'Ivanova' in loo, 'LOO 失败：同领域其它数据集经验丢失'
