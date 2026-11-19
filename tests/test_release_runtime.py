"""Runtime contracts without model downloads, GPU work, or API calls."""
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('module_name', ['casa', 'casa_downstream'])
@pytest.mark.parametrize('decision', [
    {'done': True, 'best_vc_method': 'NeuralVC'},
    {'done': False, 'new_vc_method': 'NeuralVC'},
])
def test_run_uses_selected_metrics_and_records_memory(module_name, decision, tmp_path):
    module = importlib.import_module(module_name)
    agent = module.CASA(str(tmp_path/'input'), str(tmp_path/'output'), str(tmp_path/'middle'))
    if hasattr(agent, 'memory'):
        agent.memory = Mock()
        agent._classify_request = Mock(return_value={'task_type': 'other'})
    agent._generate_initial_plan = Mock(return_value={
        'language': '中文', 'steps': [{'tool': 'run_voice_conversion', 'params': {'method': 'NeuralVC'}}]})
    agent._execute_plan = Mock(return_value={'success': True, 'eval_metrics': {'eer': 49}})
    agent._reflect = Mock(side_effect=[{'done': False, 'new_vc_method': 'pitch'}, decision])
    agent._execute_step = Mock(side_effect=[(True, {}, 0), (True, {'mean_results': {'eer': 20}}, 0)])
    result = agent.run('中文，单说话人，不需要内容匿名', max_reflections=2)
    assert result['final_vc_method'] == 'NeuralVC'
    assert result['final_metrics'] == {'eer': 49}
    assert agent.ctx.vc_method_used == 'NeuralVC'
    if hasattr(agent, 'memory'):
        args = agent.memory.record_run.call_args.kwargs
        assert args['final_vc_method'] == 'NeuralVC'
        assert args['final_metrics'] == {'eer': 49}
        saved = json.loads((tmp_path/'output'/'agent_result.json').read_text())
        assert saved['final_vc_method'] == 'NeuralVC'


def test_bundled_memory_loads_from_other_working_directory(tmp_path, monkeypatch):
    from casa_downstream import CASA
    monkeypatch.chdir(tmp_path)
    agent = CASA(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'middle'))
    assert len(agent.memory.experiences) == 14
    assert len(agent.memory.runs) == 16
    assert agent.memory.recall_for_prompt({'task_type': 'dementia'})


def test_demo_uses_runtime_memory_and_selected_audio(tmp_path, monkeypatch):
    import demo.demo as demo
    original = demo.DemoCASA.run
    assert original is demo.CASA.run
    memory_root = tmp_path/'memory'
    events = []
    agent = demo.DemoCASA(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'middle'),
                                  memory_path=str(memory_root), on_update=lambda *args: events.append(args))
    agent._call_llm_with_retry = Mock(return_value={'language': '中文', 'steps': [
        {'tool': 'run_voice_conversion', 'params': {'method': 'NeuralVC'}}]})
    agent._generate_initial_plan('中文，不需要内容匿名')
    assert any(kind == 'memory' for kind, _ in events)
    for method in ['NeuralVC', 'pitch']:
        path=tmp_path/'out'/method/'sample.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(method.encode())
    preview=demo._find_output_audio(tmp_path/'out'/'pitch')
    assert Path(preview).read_bytes() == b'pitch'
    assert demo._find_output_audio(tmp_path/'out'/'Resyn') is None


def test_method_aliases_work_in_cli_and_plans(tmp_path):
    import subprocess
    import sys
    from casa_downstream import _validate_plan
    plan={'language': '中文', 'steps': [{'tool':'run_voice_conversion','params':{'method':'seedvc'}}]}
    assert _validate_plan(plan) is None
    assert plan['steps'][0]['params']['method'] == 'NeuralVC'
    cli=subprocess.run([sys.executable, 'agent.py', '--vc_method', 'seedvc', '--help'], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr


def test_mask_plan_never_reuses_unmasked_cache(tmp_path):
    import casa_downstream as module
    agent=module.CASA(str(tmp_path/'input'),str(tmp_path/'out'),str(tmp_path/'mid'),use_memory=False)
    (tmp_path/'out'/'evaluate_NeuralVC.csv').write_text('name,eer\n-mean-,45\n')
    agent._execute_step=Mock(return_value=(True, {'mean_results': {'eer': 49}}, 0))
    plan={'steps': [
        {'tool':'run_mask','params':{}},
        {'tool':'run_voice_conversion','params':{'method':'NeuralVC'}},
        {'tool':'run_evaluation','params':{}},
    ]}
    result=agent._execute_plan(plan)
    assert agent._execute_step.call_count==3
    assert result['eval_metrics']=={'eer':49}
    assert plan['steps'][1]['params']['with_mask'] is True
    assert plan['steps'][2]['params']['with_mask'] is True


def test_retuning_same_method_selects_the_matching_audio_metrics_and_memory(tmp_path, monkeypatch):
    import casa_downstream as runtime
    import voice_convert
    from utils.vc_parameters import VC_DEFAULTS

    audio = tmp_path / 'input'
    audio.mkdir()
    (audio / 'sample.wav').write_bytes(b'source')
    output = tmp_path / 'output'
    agent = runtime.CASA(str(audio), str(output), str(tmp_path / 'middle'),
                         memory_path=str(tmp_path / 'memory'))
    agent._classify_request = Mock(return_value={'task_type': 'other'})
    # A previous default run must not supply the audio or metrics for either new setting.
    (output / 'pitch').mkdir()
    (output / 'pitch' / 'sample.wav').write_bytes(b'default')
    (output / 'evaluate_pitch.csv').write_text('name,eer\n-mean-,99\n')

    plan = {'language': 'en', 'steps': [
        {'tool': 'run_voice_conversion', 'params': {'method': 'pitch', 'vc_params': {'n_steps': -2}}},
        {'tool': 'run_evaluation', 'params': {'language': 'en'}},
    ]}
    decisions = [
        plan,
        {'done': False, 'new_vc_method': 'pitch', 'vc_params': {'n_steps': -5}},
        {'done': True, 'best_vc_method': 'pitch', 'best_vc_params': {'n_steps': -2}},
        {'lesson': 'The first pitch setting preserved more utility.', 'recommended_method': 'pitch'},
    ]
    agent._call_llm = Mock(side_effect=[json.dumps(d) for d in decisions])
    calls = []

    def convert(input_dir, output_dir, method, n_steps, **kwargs):
        calls.append((method, n_steps))
        path = Path(output_dir)
        path.mkdir()
        (path / 'sample.wav').write_text(str(n_steps))

    monkeypatch.setattr(voice_convert, 'batch_convert', convert)
    monkeypatch.setattr(runtime, 'run_in_isolation', lambda func, *args, conda_env=None: func(*args))
    monkeypatch.setattr('utils.metrics.spk_audio_extractor.extract_dir', Mock())
    monkeypatch.setattr(runtime.TOOL_BY_NAME['run_evaluation'], 'handler',
                        lambda ctx, **kwargs: {'success': True, 'mean_results': {'eer': 50 + ctx.vc_params['n_steps']}})

    result = agent.run('Compare two pitch settings for English speech.', max_reflections=3)
    assert calls == [('pitch', -2), ('pitch', -5)]
    assert 'best_vc_params' in agent._call_llm.call_args_list[1].args[0]
    assert VC_DEFAULTS['pitch'] == {'n_steps': -3.0}
    assert result['final_vc_params'] == {'n_steps': -2}
    assert result['final_metrics'] == {'eer': 48}
    assert (Path(result['final_audio_dir']) / 'sample.wav').read_text() == '-2.0'
    run = agent.memory.runs[-1]
    assert run['vc_results']['pitch'] == result['final_metrics']
    assert run['vc_attempts'] == [
        {'vc_method': 'pitch', 'vc_params': {'n_steps': -2}, 'metrics': {'eer': 48}},
        {'vc_method': 'pitch', 'vc_params': {'n_steps': -5}, 'metrics': {'eer': 45}},
    ]


@pytest.mark.parametrize('method,params', [
    ('pitch', {'alpha': 0.7}),
    ('formant_lpc', {'alpha': 'low'}),
    ('formant_spec', {'scale': 0}),
    ('NeuralVC', {'diffusion_steps': 1.5}),
    ('Resyn', {'fish_speech_root': '/tmp'}),
])
def test_plan_rejects_invalid_conversion_settings(method, params):
    from casa_downstream import _validate_plan
    step = {'tool': 'run_voice_conversion', 'params': {'method': method, 'vc_params': params}}
    assert _validate_plan({'steps': [step]})


def test_zh_cli_routes_to_chinese_asr_masking_evaluation_and_tts(tmp_path, monkeypatch):
    import sys
    import agent
    import mask_llm
    import mask_ner
    from configs.config_loader import get_asr_config, get_llm_config, uses_character_timestamps
    from utils.metrics.cer_calc import normalize_text
    from utils.voice_convert.fishaudio_tts import _tts_lang_prefix

    invoke = Mock()
    monkeypatch.setattr(agent, 'run_in_isolation', invoke)
    monkeypatch.setattr(sys, 'argv', ['agent.py', '-i', str(tmp_path), '-o', str(tmp_path/'out'),
                                    '--trans', '--language', 'zh'])
    agent.main()
    assert invoke.call_args.args[0] is agent.trans
    assert invoke.call_args.args[2] == 'paraformer'
    assert invoke.call_args.args[5] == 'zh'
    assert invoke.call_args.kwargs['conda_env'] is None
    assert get_asr_config('ja')['model'] == 'whisperx'
    assert get_llm_config('zh') == get_llm_config('中文')
    assert uses_character_timestamps('zh')
    for module in (mask_llm, mask_ner):
        assert module.char_to_word_positions(1, 2, '张三丰', 'zh', [0, 20, 45, 70]) == (20, 45)
    assert normalize_text('醫學。', 'zh') == normalize_text('醫學。', '中文') == '医学'
    assert _tts_lang_prefix('zh') == 'ch'


def test_model_defaults_match_prompt_and_registered_backends():
    from casa_downstream import INITIAL_PLAN_PROMPT, REFLECTION_SYSTEM_PROMPT, TOOL_BY_NAME
    from utils.vc_parameters import VC_DEFAULTS, VC_PARAMETER_GUIDE
    from utils.voice_convert import get
    for method, defaults in VC_DEFAULTS.items():
        registered = get(method)['defaults']
        assert {k: registered[k] for k in defaults} == defaults
    assert VC_PARAMETER_GUIDE in INITIAL_PLAN_PROMPT
    assert VC_PARAMETER_GUIDE in REFLECTION_SYSTEM_PROMPT
    assert VC_PARAMETER_GUIDE == TOOL_BY_NAME['run_voice_conversion'].parameters['vc_params']


def test_downstream_scores_audio_from_the_selected_parameter_directory(tmp_path):
    from casa_downstream import CASA
    inputs = tmp_path / 'input'
    inputs.mkdir()
    data = tmp_path / 'data.json'
    original = [{'wav_path': str(inputs / 'sample.flac'), 'label': 1}]
    data.write_text(json.dumps(original))
    agent = CASA(str(inputs), str(tmp_path / 'out'), str(tmp_path / 'middle'), use_memory=False)
    agent.downstream_data_json = str(data)
    agent.ctx.select_vc('pitch', {'n_steps': -5}, with_mask=True)
    selected = Path(agent.ctx.vc_audio_dir)
    selected.mkdir()
    (selected / 'sample.wav').write_bytes(b'converted')
    agent._run_downstream_script = Mock(return_value={
        'balanced_accuracy': {'mean': 0.7}, 'f1_macro': {'mean': 0.6},
    })
    assert agent._run_downstream_for_current_vc() == {'downstream_bacc': 0.7, 'downstream_f1': 0.6}
    metadata, directory = agent._run_downstream_script.call_args.args
    assert Path(directory) == selected
    assert json.loads(Path(metadata).read_text()) == [{'wav_path': str(selected / 'sample.wav'), 'label': 1}]
    assert json.loads(data.read_text()) == original


def test_masking_failure_blocks_combined_conversion(tmp_path):
    from casa_downstream import CASA

    agent = CASA(str(tmp_path/'input'), str(tmp_path/'out'), str(tmp_path/'middle'), use_memory=False)
    agent._execute_step = Mock(side_effect=[(True, {}, 0), (False, {'error': 'missing mask'}, 0)])
    result = agent._execute_plan({'steps': [
        {'tool': 'run_transcription', 'params': {'language': 'en'}},
        {'tool': 'run_mask', 'params': {'language': 'en'}},
        {'tool': 'run_voice_conversion', 'params': {'method': 'pitch', 'language': 'en'}},
    ]})
    assert not result['success']
    assert agent._execute_step.call_count == 2
    assert result['steps_results'][-1]['step'] == 'run_voice_conversion'
    assert not result['steps_results'][-1]['ok']


@pytest.mark.parametrize('fenced', [False, True])
def test_planner_uses_complete_plan_after_parameter_object(tmp_path, fenced):
    from casa_downstream import CASA, _validate_plan

    agent = CASA(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'mid'), use_memory=False)
    snippets = ['{"method":"pitch"}', '{"steps":[{"tool":"run_voice_conversion","params":{"method":"pitch"}}]}']
    if fenced:
        snippets = [f"```json\n{snippet}\n```" for snippet in snippets]
    agent._call_llm = Mock(return_value='\n'.join(snippets))
    result = agent._call_llm_with_retry('Plan.', 'Voice anonymization.', validator=_validate_plan)
    assert result['steps'][0]['params']['vc_params'] == {'n_steps': -3.0}
    agent._call_llm.assert_called_once()


def test_demo_recall_uses_both_layers_for_related_tasks_only():
    from demo.demo import DemoCASA, ROOT
    from utils.agent_memory import TwoLayerMemory

    memory = TwoLayerMemory(root=str(ROOT/'memory'))
    unrelated = memory.recall_for_prompt({'task_type': 'other'}, focused=DemoCASA.focused_memory)
    assert unrelated == ''
    related = memory.recall_for_prompt({'task_type': 'dementia'}, focused=DemoCASA.focused_memory)
    assert 'Layer-1' in related and 'Layer-2' in related and 'ADReSS' in related
    assert 'Saar-HD' not in related


def test_planner_retries_with_distinct_shape_guidance(tmp_path, caplog):
    from casa_downstream import CASA, _validate_plan

    agent = CASA(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'mid'), use_memory=False)
    bad = '{"task_type":"other"}'
    valid = '{"steps":[{"tool":"run_voice_conversion","params":{"method":"pitch"}}]}'
    agent._call_llm = Mock(side_effect=[bad, bad, valid])
    assert agent._call_llm_with_retry('Plan.', 'Mask and anonymize.', validator=_validate_plan)['steps']
    prompts = [call.args[1] for call in agent._call_llm.call_args_list]
    assert len(set(prompts)) == 3
    assert 'nonempty steps array' in prompts[1]
    assert 'Omit examples' in prompts[2]
    assert "JSON keys: [['task_type']]" in caplog.text


def test_empty_mask_does_not_alter_audio(tmp_path):
    import numpy as np
    import soundfile as sf
    from apply_mask import add_adaptive_ssn_to_audio

    wave = tmp_path / 'sample.wav'
    output = tmp_path / 'out.wav'
    mask = tmp_path / 'sample.json'
    sf.write(wave, np.full(1600, 0.25), 16000)
    for content in ([], [{'idx': -1, 'start': 0, 'end': 0}]):
        mask.write_text(json.dumps(content))
        add_adaptive_ssn_to_audio(str(wave), str(mask), str(output), noise_method='mute', time_append=50)
        assert np.array_equal(sf.read(wave)[0], sf.read(output)[0])


def test_invalid_privacy_response_is_not_cached_as_clear(tmp_path, monkeypatch):
    from mask_llm import mask_llm

    trans = tmp_path / 'trans' / 'nested'
    trans.mkdir(parents=True)
    (trans / 'example.json').write_text(json.dumps([
        {'index': 0, 'text': 'Call Sample.', 'timestamp': [0, 100, 300]},
    ]))
    monkeypatch.setattr('utils.language_model.load_model', Mock(return_value=(None, None)))
    infer = Mock(return_value='No tagged result')
    monkeypatch.setattr('utils.language_model.generate_response', infer)
    mask_llm(str(trans.parent), str(tmp_path/'mask'), Mock(), language='en')
    assert infer.call_count == 2
    assert not (tmp_path/'mask'/'nested'/'example.json').exists()


def test_nested_content_mask_reaches_text_only_and_combined_audio(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    import casa_downstream as runtime

    source = tmp_path / 'input' / 'nested' / 'example.wav'
    source.parent.mkdir(parents=True)
    sr = 16000
    sf.write(source, 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr), sr)
    agent = runtime.CASA(str(source.parents[1]), str(tmp_path/'out'), str(tmp_path/'mid'), use_memory=False)
    transcript = Path(agent.ctx.trans_origin_dir) / 'nested' / 'example.json'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({'sentences': [
        {'index': 0, 'text': 'Call Sample today.', 'timestamp': [0, 250, 600, 1000]},
    ]}))
    monkeypatch.setattr(runtime, 'run_in_isolation', lambda func, *args, conda_env=None: func(*args))
    monkeypatch.setattr('utils.language_model.load_model', Mock(return_value=(None, None)))
    monkeypatch.setattr('utils.language_model.generate_response', Mock(return_value='<result>sample</result>'))
    monkeypatch.setattr('utils.metrics.spk_audio_extractor.extract_dir', Mock())

    result = agent._execute_plan({'steps': [
        {'tool': 'run_mask', 'params': {'language': 'en', 'mask_method': 'mute'}},
        {'tool': 'run_voice_conversion', 'params': {'method': 'pitch', 'language': 'en'}},
    ]})
    assert result['success']
    assert (Path(agent.ctx.mask_dir) / 'nested' / 'example.json').is_file()
    for directory in ('text_only', 'text+pitch'):
        audio, _ = sf.read(tmp_path/'out'/directory/'nested'/'example.wav')
        assert np.all(audio[4800:8000] == 0)
        assert np.max(np.abs(audio[:1600])) > 0.05


def test_neuralvc_reports_missing_reference_before_model_load(tmp_path, monkeypatch):
    from utils.voice_convert import seedvc

    pool = tmp_path/'pool.json'
    pool.write_text(json.dumps({'22_30': {'female': [{'subject_id': 'a', 'wav_path': 'missing.wav'}]}}))
    initialize = Mock()
    monkeypatch.setattr(seedvc, '_load_seedvc_model', initialize)
    with pytest.raises(ValueError, match='1 missing reference files out of 1'):
        seedvc.batch_convert_fn(str(tmp_path), str(tmp_path/'out'), Mock(),
                                aishell_target_pool_path=str(pool), age_gender_model_path=str(tmp_path))
    initialize.assert_not_called()


def test_text_only_plan_previews_masked_audio_without_voice_reflection(tmp_path):
    from casa_downstream import CASA

    agent = CASA(str(tmp_path/'input'), str(tmp_path/'out'), str(tmp_path/'mid'), use_memory=False)
    agent._generate_initial_plan = Mock(return_value={
        'language': 'en', 'steps': [{'tool':'run_mask','params':{'language':'en'}}],
    })
    agent.ctx.text_only_dir = str(tmp_path/'out'/'text_only')
    agent._execute_plan = Mock(return_value={'success': True, 'eval_metrics': {'cer': 0.1}})
    agent._reflect = Mock()
    result = agent.run('Mask spoken identifiers only.', max_reflections=3)
    agent._reflect.assert_not_called()
    assert result['final_vc_method'] is None
    assert result['final_audio_dir'] == agent.ctx.text_only_dir
