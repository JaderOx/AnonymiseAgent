"""
agent_memory.py — Agent 跨运行双层 Memory（v2）

由粗到细两层：
- Layer-1  <root>/experiences.json ：任务级经验，一任务一条
  （教训 lesson、推荐/避免方法、关键指标、覆盖的 run_ids）。
  由 LLM 在运行结束时自动总结写入（llm_call 回调注入，本模块零 LLM 依赖）。
- Layer-2  <root>/runs/<run_id>.json ：单次运行全细节
  （尝试序列、各方法指标、反思理由、下游 bacc、产物路径）。

recall_for_prompt() 由粗到细两段注入：先 Layer-1 全部任务经验（粗），
再 Layer-2 仅与当前任务同 task_type / 同 dataset 的运行明细（细，封顶 6 条）。

任务的 task_type（下游任务领域）/语言/说话人数/下游任务描述必须来自用户
request 原文（模板："语言为X，说话人数量为N，医疗任务为Y。我希望Z"），
提取不到的字段进 warnings，禁止猜测。

设计约束：
- 纯文件实现，零外部依赖（无向量库/数据库）
- 条目量级几十条，Layer-1 全量注入；Layer-2 按任务类型筛选
- 写入失败只 warning，绝不影响主流程（memory 是增益不是依赖）
- 写序固定「先 run 后 experience」：LLM 总结失败时 Layer-2 不丢，
  下次运行仍可读取已保存的 runs；经验在后续成功运行时更新

用法：
    from utils.agent_memory import TwoLayerMemory
    mem = TwoLayerMemory(root='memory', logger=logger)
    mem.record_run(..., task_meta=meta, llm_call=lambda s, u: agent._call_llm(s, u))
    prompt_block = mem.recall_for_prompt(task_meta=meta)

兼容：旧单文件 agent_memory.json（{"runs": [...]}）→ legacy 只读模式，
      参与召回；migrate_legacy_runs() 显式迁移成新布局，不自动改写原文件。
"""

import os
import json
import re
import time
import tempfile

from utils.vc_names import ALIASES, METHODS

# ── 常量 ─────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2

# Layer-2 明细注入条数封顶（防 prompt 膨胀）
MAX_RUNS_IN_LAYER2_PROMPT = 6

# Layer-1 全量注入的条数上限，超过则只注同 task_type + 最近条目
MAX_EXPERIENCES_IN_PROMPT = 30
DEMO_RECALL_LIMIT = 3

# 入 memory 时各文本字段的截断长度
MAX_TEXT_LEN = 300        # reason / reflection
MAX_REQUEST_LEN = 5000    # request 原文
MAX_LESSON_LEN = 2000     # Layer-1 lesson（存全文，2000 只是防极端长文的硬顶）

# 经验总结/召回 prompt 里的 VC 方法类别表（防止 LLM 犯类别性错误，
# 如把 NeuralVC 说成"TTS类方法"——它是神经声线转换，不是重合成）
METHOD_CATEGORIES = {
    'pitch':           '信号处理（F0 变换，不重合成）',
    'formant_spec':    '信号处理（共振峰变换，不重合成）',
    'formant_lpc':     '信号处理（McAdams LPC 共振峰变换）',
    'NeuralVC':        '神经声线转换 VC（保留内容与韵律结构，非 TTS 重合成）',
    'Resyn':           'ASR→替换→TTS 重合成（语音完全重新生成）',
    'combined':        '组合方法',
}

# 指标方向说明（防止 LLM 犯方向性错误——尤其 EER 是说话人验证等错误率，
# 匿名后 EER 越接近 50%（验证器随机猜水平）匿名效果越好，绝不是越低/越高越好）
METRIC_HINTS = {
    'eer':     '说话人验证等错误率(%)，【匿名指标：越接近50%越好，50%=验证器无法区分原/匿名说话人；偏离50%（无论偏低偏高）= 身份仍可被识别】',
    'snr_db':  '信噪比(dB)，越高越好',
    'mos':     '语音自然度(1-5)，越高越好',
    'cer':     '字符错误率，越低越好（可懂度）',
    'wer':     '词错误率，越低越好（可懂度）',
    'emo':     '情感相似度(0-1)，越高越好',
    'pit_l1':  '韵律L1距离，越低越好',
    'pit_pcc': '韵律相关系数，越高越好',
    'downstream_bacc': '下游任务平衡准确率，越高越好（与 baseline_bacc 对比）',
}

# 下游任务领域受控词表（初始值；<root>/task_types.json 可增改，不改代码）
TASK_TYPE_VOCAB = {
    'voice_pathology': '声道/发声病变检测（功能性发声障碍、喉炎、声带麻痹等）',
    'articulation_disorder': '构音障碍检测（腭裂等器质性构音障碍）',
    'depression': '抑郁检测',
    'dementia': '痴呆/认知障碍检测（阿尔茨海默病、MCI等）',
    'parkinson': '帕金森病检测',
    'developmental_language_disorder': '发育性/儿童语言障碍检测',
    'crisis_risk': '心理危机风险检测',
    'other': '其他（未归类）',
}

# 指标里出现过的全部 VC 方法名（lesson 校验用，宽松集合）。
# 注意规范名大小写敏感（与结果目录名一致）：Resyn / NeuralVC；
# 小写归一与旧名（fishaudio_tts/seedvc/mcadams/formant）由 _normalize 的别名表处理
KNOWN_VC_METHODS = set(METHODS)

# fallback_classify 的领域关键词 → task_type（只做归一化，不猜测缺失字段）
_TASK_TYPE_KEYWORDS = [
    ('voice_pathology', ['病变', '发声障碍', '喉炎', '声带', '嗓音', 'dysphonia',
                         'laryngitis', 'pathology', 'voice']),
    ('depression', ['抑郁', 'depression']),
    ('dementia', ['阿尔茨海默', '痴呆', '认知', 'dementia', 'alzheimer']),
    ('parkinson', ['帕金森', 'parkinson']),
    ('developmental_language_disorder', ['语言障碍', 'language disorder']),
    ('crisis_risk', ['危机', '自杀', 'crisis', 'suicide']),
]


def _truncate(text, limit=MAX_TEXT_LEN):
    if not isinstance(text, str):
        return text
    return text if len(text) <= limit else text[:limit] + '...'


# agent 自创新 task_type 的合法格式（小写 snake_case；命中即自动入表持久化）
_TASK_TYPE_RE = re.compile(r'^[a-z][a-z0-9_]{2,40}$')


def is_valid_new_task_type(task_type):
    """新领域名是否合法：小写英文 snake_case（如 dysarthria）。"""
    return isinstance(task_type, str) and bool(_TASK_TYPE_RE.match(task_type))


def infer_dataset_name(input_dir):
    """从 input_dir 推断数据集名：取最后一个非 workspace/wav/audio/data 的目录名。

    例：
        /path/to/data/DAIC-WOZ/wav                        → DAIC-WOZ
        /path/to/data_backup/crisis/audio                 → crisis
    """
    parts = [p for p in os.path.normpath(os.path.abspath(input_dir)).split(os.sep) if p]
    skip = {'wav', 'audio', 'data', 'workspace', 'wavs', 'output', 'middle'}
    for p in reversed(parts):
        if p not in skip:
            return p
    return parts[-1] if parts else 'unknown'


def fallback_classify(request):
    """规则版任务信息提取：从 request 原文正则抽取 语言/说话人数/下游任务/目标。

    只提取原文明确写出的字段；提取不到 → 字段缺省 + warnings（禁止猜测）。
    返回 {task_type, language, n_speakers, downstream_task, goal, source='rule', warnings}
    """
    meta = {'task_type': None, 'language': None, 'n_speakers': None,
            'downstream_task': None, 'goal': None,
            'source': 'rule', 'warnings': []}
    text = request or ''

    m = re.search(r'语言为(.+?)[，,。.\n]', text)
    if m:
        meta['language'] = m.group(1).strip()
    else:
        meta['warnings'].append('language 未能从请求中提取')

    m = re.search(r'说话人(?:数量)?为(\d+)', text)
    if m:
        meta['n_speakers'] = int(m.group(1))
    else:
        meta['warnings'].append('n_speakers 未能从请求中提取')

    m = re.search(r'(?:医疗|下游)?任务为(.+?)[。.\n]', text)
    if m:
        meta['downstream_task'] = m.group(1).strip()
    else:
        meta['warnings'].append('downstream_task 未能从请求中提取')

    m = re.search(r'我希望(.+)', text)
    if m:
        meta['goal'] = _truncate(m.group(1).strip().rstrip('。'), 200)

    if meta['downstream_task']:
        task_text = meta['downstream_task'].lower()
        for task_type, kws in _TASK_TYPE_KEYWORDS:
            if any(kw in task_text or kw in task_text.lower() for kw in kws):
                meta['task_type'] = task_type
                break
        else:
            meta['task_type'] = 'other'
            meta['warnings'].append(
                f"downstream_task「{meta['downstream_task']}」无法映射到词表，task_type 归为 other")
    else:
        meta['warnings'].append('task_type 无法推断（缺少下游任务描述）')
    return meta


def _extract_json(text):
    """从 LLM 原始输出提取 JSON dict：直接 parse，失败则取最后一个平衡的大括号块。"""
    if not isinstance(text, str):
        raise ValueError('输出非字符串')
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 取最后一个平衡 {...}（思考文本里出现 { 时，首块常是残缺的）
    depth, start, candidates = 0, None, []
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError('输出中找不到可解析的 JSON 对象')


# ═══════════════════════════════════════════════════════════════════════════════

class TwoLayerMemory:
    def __init__(self, root=None, logger=None):
        root = root or 'memory'
        self.logger = logger
        self.legacy_path = None      # 非 None = 旧单文件模式（只读参与召回）
        if os.path.isfile(root):     # 旧版单文件 agent_memory.json
            self.legacy_path = os.path.abspath(root)
            base = os.path.splitext(os.path.basename(root))[0]
            root = os.path.join(os.path.dirname(self.legacy_path), base + '_v2')
            self._log('warning',
                      f"检测到旧版单文件 memory: {self.legacy_path}，新数据将写入 {root}/；"
                      f"可调用 migrate_legacy_runs() 显式迁移")
        self.root = os.path.abspath(root)
        self.runs_dir = os.path.join(self.root, 'runs')
        self.exp_path = os.path.join(self.root, 'experiences.json')
        self.task_types_path = os.path.join(self.root, 'task_types.json')

        self.runs = []               # list[dict]，每条 = 一次运行（v2 schema）
        self.experiences = []        # list[dict]，每条 = 一个任务（Layer-1）
        self.task_types = dict(TASK_TYPE_VOCAB)
        self.load()

    # 兼容旧名（backfill_memory.py 等仍在用）
    def _log(self, level, msg):
        if self.logger is not None:
            getattr(self.logger, level)(f"[Memory] {msg}")

    # ── 加载 / 保存 ─────────────────────────────────────────────────

    def load(self):
        # Layer-2 runs
        if os.path.isdir(self.runs_dir):
            loaded = []
            for fn in os.listdir(self.runs_dir):
                if not fn.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(self.runs_dir, fn), 'r', encoding='utf-8') as f:
                        loaded.append(json.load(f))
                except Exception as e:
                    self._log('warning', f"run 文件读取失败（跳过）: {fn}: {e}")
            loaded.sort(key=lambda r: r.get('timestamp') or '')
            self.runs = loaded
            self._log('info', f"加载 {len(loaded)} 条历史运行记录: {self.runs_dir}")
        else:
            self._log('info', f"memory runs 目录不存在，从空开始: {self.runs_dir}")

        # Layer-1 experiences
        if os.path.exists(self.exp_path):
            try:
                with open(self.exp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.experiences = data.get('experiences', [])
                self._log('info', f"加载 {len(self.experiences)} 条任务级经验: {self.exp_path}")
            except Exception as e:
                self._log('warning', f"experiences 加载失败（按空处理）: {e}")
                self.experiences = []

        # 领域词表（文件可扩展内置词表）
        if os.path.exists(self.task_types_path):
            try:
                with open(self.task_types_path, 'r', encoding='utf-8') as f:
                    self.task_types.update(json.load(f).get('vocab', {}))
            except Exception as e:
                self._log('warning', f"task_types 读取失败（用内置词表）: {e}")

        # legacy 单文件：只读并入 runs（标记 evidence 来源），等待显式迁移
        if self.legacy_path:
            try:
                with open(self.legacy_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                legacy_runs = data.get('runs', []) if isinstance(data, dict) else data
                for r in legacy_runs:
                    r = dict(r)
                    r.setdefault('schema_version', 1)
                    r['legacy'] = True
                self.runs.extend(legacy_runs)
                self._log('info', f"legacy 单文件并入 {len(legacy_runs)} 条（只读）")
            except Exception as e:
                self._log('warning', f"legacy memory 加载失败（忽略）: {e}")

    def _atomic_write_json(self, path, obj):
        """原子写：先写临时文件再 rename，避免写一半被读到。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                                   prefix='.memory_', suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def save_run(self, run):
        self._atomic_write_json(os.path.join(self.runs_dir, f"{run['run_id']}.json"), run)
        self._log('info', f"已保存运行记录: {run['run_id']}")

    def save_experiences(self):
        self._atomic_write_json(self.exp_path, {
            'version': SCHEMA_VERSION,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'experiences': self.experiences,
        })
        self._log('info', f"已保存 {len(self.experiences)} 条任务级经验: {self.exp_path}")

    def save_task_types(self):
        self._atomic_write_json(self.task_types_path, {
            'vocab': self.task_types,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })

    def register_task_type(self, task_type, desc=None):
        """agent/LLM 遇到词表外的新领域时自动入表（受控扩展：小写 snake_case）。
        写入 task_types.json 持久化，后续运行可直接复用。返回是否新增。"""
        if task_type in self.task_types or not is_valid_new_task_type(task_type):
            return False
        self.task_types[task_type] = desc or '(agent 自动新增领域)'
        try:
            self.save_task_types()
            self._log('info', f"新任务领域已入表: {task_type} = {self.task_types[task_type]}")
        except Exception as e:
            self._log('warning', f"task_types 持久化失败（内存内仍生效）: {e}")
        return True

    # ── Layer-2 查询 ────────────────────────────────────────────────

    def load_runs(self, task_type=None, dataset=None, limit=None):
        """按 task_type / dataset 过滤（或不过滤），时间新→旧，可选截断。"""
        runs = []
        for r in reversed(self.runs):
            if task_type and r.get('task_type') != task_type:
                continue
            if dataset and r.get('dataset') != dataset:
                continue
            runs.append(r)
            if limit and len(runs) >= limit:
                break
        return runs

    # ── Layer-1 维护 ────────────────────────────────────────────────

    @staticmethod
    def exp_key(dataset, downstream_task):
        return f"{dataset or 'unknown'}::{downstream_task or 'unknown'}"

    def find_experience(self, dataset, downstream_task):
        key = self.exp_key(dataset, downstream_task)
        for exp in self.experiences:
            if self.exp_key(exp.get('dataset'), exp.get('downstream_task')) == key:
                return exp
        return None

    def upsert_experience(self, exp):
        """同 (dataset, downstream_task) 更新已有条目，否则追加。"""
        old = self.find_experience(exp.get('dataset'), exp.get('downstream_task'))
        now = time.strftime('%Y-%m-%dT%H:%M:%S')
        if old is not None:
            run_ids = list(dict.fromkeys(old.get('run_ids', []) + exp.get('run_ids', [])))
            old.update(exp)
            old['run_ids'] = run_ids
            old['n_runs'] = len(run_ids)
            old['updated_at'] = now
            return old
        exp.setdefault('created_at', now)
        exp['updated_at'] = now
        exp['n_runs'] = len(exp.get('run_ids', []))
        self.experiences.append(exp)
        return exp

    # ── 写入（run() 结束时调用）─────────────────────────────────────

    def record_run(self, input_dir, language, request, output_dir,
                   history, final_vc_method, success=True,
                   task_meta=None, baseline_bacc=None, final_metrics=None,
                   artifacts=None, llm_call=None, final_vc_params=None):
        """记一次运行：先写 Layer-2 run 文件，再让 LLM 总结更新 Layer-1。

        Args:
            history: CASA 的 history 列表
                [{"vc_method": str, "metrics": dict, "reason": str}, ...]
            task_meta: _classify_request / fallback_classify 的输出（可含 warnings）
            final_metrics: agent 最终 current_metrics（从中提取 downstream_*）
            llm_call: (sys_prompt, user_prompt) -> str 的回调；None = 规则模式
        """
        meta = dict(task_meta or {})
        warnings = list(meta.get('warnings') or [])

        vc_results, reflections = {}, []
        for h in history or []:
            method = h.get('vc_method')
            if not method:
                continue
            vc_results[method] = {k: _to_float(v) for k, v in (h.get('metrics') or {}).items()}
            if h.get('reason'):
                reflections.append(f"{method}: {_truncate(h['reason'])}")

        if final_vc_params is not None and final_vc_method and final_metrics:
            vc_results[final_vc_method] = {k: _to_float(v) for k, v in final_metrics.items()}

        final_ds = {}
        for k, v in (final_metrics or {}).items():
            if k.startswith('downstream'):
                final_ds[k] = _to_float(v)

        dataset = infer_dataset_name(input_dir)
        run_id = f"{dataset}_{time.strftime('%Y%m%d-%H%M%S')}"
        # 同秒多次 record（如测试）时保证 run_id 唯一
        existing = {r.get('run_id') for r in self.runs}
        for suffix in range(1, 100):
            if run_id not in existing:
                break
            run_id = f"{dataset}_{time.strftime('%Y%m%d-%H%M%S')}_{suffix}"
        run = {
            'run_id': run_id,
            'schema_version': SCHEMA_VERSION,
            'task_type': meta.get('task_type') or 'other',
            'task_meta': {
                'language': meta.get('language'),
                'n_speakers': meta.get('n_speakers'),
                'downstream_task': meta.get('downstream_task'),
                'goal': meta.get('goal'),
                'source': meta.get('source') or 'unknown',
                'warnings': warnings,
            },
            'dataset': dataset,
            'language': meta.get('language') or language,
            'request': _truncate(request, MAX_REQUEST_LEN),
            'output_dir': output_dir,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'attempts': len(history or []),
            'final_vc_method': final_vc_method,
            'success': success,
            'baseline_bacc': _to_float(baseline_bacc) if baseline_bacc is not None else None,
            'final_downstream': final_ds or None,
            'downstream_bacc_by_method': None,   # 回填脚本从 summary_all_methods.json 补
            'vc_results': vc_results,
            'reflections': reflections,
            'artifacts': artifacts or {},
        }
        if final_vc_params is not None:
            run['final_vc_params'] = final_vc_params
            run['vc_attempts'] = [
                {key: attempt[key] for key in ('vc_method', 'vc_params', 'metrics')}
                for attempt in history if 'vc_params' in attempt
            ]
        # Layer-2 先落盘（LLM 失败也不丢）
        self.runs.append(run)
        self.save_run(run)

        # Layer-1：LLM 总结（或规则模式）
        try:
            exp = self._build_experience(run, llm_call)
            self.upsert_experience(exp)
            self.save_experiences()
            self._log('info', f"任务经验已更新: {exp.get('exp_id')}")
        except Exception as e:
            self._log('warning',
                      f"经验总结失败（run 已保存，运行记录已保留）: {e}")

        self._log('info', f"记录运行 {run_id}: task_type={run['task_type']}, "
                          f"final={final_vc_method}, 尝试 {list(vc_results.keys())}")
        return run

    # ── Layer-1 生成（LLM / 规则）───────────────────────────────────

    def _build_experience(self, run, llm_call=None):
        """从单次 run + 已有经验生成/更新一条 Layer-1 经验。"""
        if llm_call is not None:
            sys_prompt, user_prompt = self._experience_prompts(run)
            raw = llm_call(sys_prompt, user_prompt)
            out = _extract_json(raw)
            err = _validate_experience(out)
            if err:
                raise ValueError(f"LLM 经验输出校验失败: {err}")
            task_type = out.get('task_type') or run.get('task_type') or 'other'
            lesson = _truncate(out.get('lesson') or '', MAX_LESSON_LEN)
            recommended = _clean_methods(out.get('recommended_method'))
            avoid = _clean_methods(out.get('avoid_methods'))
            source = 'llm'
        else:
            task_type = run.get('task_type') or 'other'
            lesson, recommended, avoid = self._rule_lesson(run)
            source = run.get('task_meta', {}).get('source') or 'rule'

        # 词表外的新领域：合法 snake_case 自动入表（agent 自主扩展），否则归 other
        if task_type not in self.task_types:
            if not self.register_task_type(task_type, out.get('task_type_desc') if llm_call is not None else None):
                task_type = 'other'

        dataset = run.get('dataset')
        downstream_task = run.get('task_meta', {}).get('downstream_task')
        return {
            'exp_id': self._safe_id(f"{dataset}__{task_type}"),
            'task_type': task_type,
            'task_type_source': source,
            'evidence_level': run.get('evidence_level') or 'full_run',
            'dataset': dataset,
            'downstream_task': downstream_task,
            'language': run.get('language'),
            'n_speakers': run.get('task_meta', {}).get('n_speakers'),
            'goal': run.get('task_meta', {}).get('goal'),
            'lesson': lesson,
            'recommended_method': recommended,
            'avoid_methods': avoid,
            'key_metrics': self._key_metrics(run),
            'run_ids': [run['run_id']],
        }

    @staticmethod
    def _safe_id(s):
        return re.sub(r'[^\w\-]+', '_', s)

    def _experience_prompts(self, run):
        vocab_str = "\n".join(f"  - {k}: {v}" for k, v in sorted(self.task_types.items()))
        method_cat_str = "\n".join(
            f"  - {m}: {desc}" for m, desc in METHOD_CATEGORIES.items())
        metric_hint_str = "\n".join(
            f"  - {k}: {desc}" for k, desc in METRIC_HINTS.items())
        sys_prompt = (
            "你是语音匿名化实验的经验总结助手。根据实验运行细节提炼【任务级经验】，"
            "供以后遇到同类下游任务时直接参考（选什么匿名方法、避开什么）。只输出 JSON。"
            "描述方法时类别必须准确：NeuralVC 是神经声线转换（非重合成），"
            "只有 Resyn 是 TTS 重合成——严禁把 NeuralVC 归为「TTS类」。"
            "解读指标时方向必须正确（见下方指标方向参考）：尤其 EER 是匿名指标，"
            "越接近 50% 匿名效果越好——严禁把 EER 当『错误率越低越好』。"
        )
        facts = [
            f"数据集: {run.get('dataset')}",
            f"语言: {run.get('language')}",
            f"下游任务: {run.get('task_meta', {}).get('downstream_task')}",
            f"用户目标: {run.get('task_meta', {}).get('goal')}",
            f"原始音频下游基线: baseline_bacc={run.get('baseline_bacc')}",
            f"本次尝试 {run.get('attempts')} 次，最终方法: {run.get('final_vc_method')}",
        ]
        if run.get('vc_attempts'):
            facts.append("VC attempts with parameters and metrics: " + json.dumps(run['vc_attempts'], ensure_ascii=False))
            facts.append("Selected parameters: " + json.dumps(run['final_vc_params'], ensure_ascii=False))
        if run.get('downstream_bacc_by_method'):
            per = ", ".join(f"{m}={_fmt(v)}" for m, v in
                            run['downstream_bacc_by_method'].items())
            facts.append(f"各方法下游 bacc: {per}")
        for m, ms in (run.get('vc_results') or {}).items():
            per = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted((ms or {}).items()))
            facts.append(f"方法 {m} 指标: {per}")
        if run.get('final_downstream'):
            per = ", ".join(f"{k}={_fmt(v)}" for k, v in run['final_downstream'].items())
            facts.append(f"最终下游指标: {per}")
        if run.get('secure_bacc') is not None:
            facts.append(f"baseline 匿名流水线(NER替换+TTS重合成)下游 bacc: {_fmt(run['secure_bacc'])}")
        goal_runs = run.get('goal_runs') or []
        if goal_runs:
            gr_lines = [
                f"  - 目标「{g.get('goal')}」→ 实验验证最优={g.get('target')} / "
                f"agent选择={g.get('choice')} / 达成={g.get('state')}"
                for g in goal_runs]
            facts.append(f"各目标变体实验（共 {len(goal_runs)} 个）:\n" + "\n".join(gr_lines))
        if run.get('reflections'):
            facts.append("Agent 反思理由: " + " | ".join(run['reflections'][-3:]))

        old = self.find_experience(run.get('dataset'),
                                   run.get('task_meta', {}).get('downstream_task'))
        old_block = ''
        if old:
            old_block = (f"\n【已有经验（可在新证据基础上更新或推翻）】\n"
                         f"lesson: {old.get('lesson')}\n推荐: {old.get('recommended_method')} "
                         f"| 避免: {old.get('avoid_methods')} | 历史运行数: {old.get('n_runs')}\n")

        user_prompt = (
            "【任务信息与实验细节】\n" + "\n".join(facts) +
            f"\n{old_block}\n【VC 方法类别参考（描述方法时类别必须准确，禁止张冠李戴）】\n{method_cat_str}"
            f"\n\n【指标方向参考（引用数据下结论时方向必须正确）】\n{metric_hint_str}"
            f"\n\n【任务类型词表（task_type 优先从中选；若确属词表没有的新领域，"
            "可提出新的小写英文 snake_case 名称并附 task_type_desc 中文一句话说明）】\n"
            f"{vocab_str}\n\n"
            "请输出 JSON（不要输出其他内容）：\n"
            '{"task_type": "词表之一或新的snake_case", '
            '"task_type_desc": "仅当提出新 task_type 时给中文一句话描述，否则null", '
            '"lesson": "中文总结，写完整不要截断：这个任务上什么方法为什么好/坏，'
            '引用关键数据支撑结论；方法描述注意类别准确（重合成 vs 信号处理 vs 神经VC）", '
            '"recommended_method": "该方法上表现最好的VC方法名或null", '
            '"avoid_methods": ["明确不该用的方法名（可空）"]}\n'
            "要求：方法名必须来自上面出现过的 VC 方法名；禁止编造数据；"
            "task_type 按下游任务的医学领域归类。"
        )
        return sys_prompt, user_prompt

    @staticmethod
    def _rule_lesson(run):
        """无 LLM 时的规则模板 lesson（回填兜底 / 单测）。"""
        final = run.get('final_vc_method')
        parts = []
        if final:
            m = (run.get('vc_results') or {}).get(final) or {}
            per = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(m.items())) or "无指标"
            parts.append(f"最终选择 {final}（{per}）")
        bacc_map = run.get('downstream_bacc_by_method') or {}
        if bacc_map:
            ranking = sorted(bacc_map.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else -1))
            parts.append("各方法下游 bacc: " + ", ".join(f"{m}={_fmt(v)}" for m, v in ranking))
        if run.get('baseline_bacc') is not None:
            parts.append(f"原始音频基线 bacc={_fmt(run['baseline_bacc'])}")
        refl = run.get('reflections') or []
        if refl:
            parts.append(f"反思摘要: {_truncate(refl[-1], 150)}")
        recommended = final if (final and run.get('success')) else None
        return ("；".join(parts) or "无可用细节"), recommended, None

    @staticmethod
    def _key_metrics(run):
        km = {}
        if run.get('baseline_bacc') is not None:
            km['baseline_bacc'] = run['baseline_bacc']
        if run.get('downstream_bacc_by_method'):
            km['downstream_bacc_by_method'] = run['downstream_bacc_by_method']
        if run.get('final_downstream'):
            km['final_downstream'] = run['final_downstream']
        final = run.get('final_vc_method')
        if final and final in (run.get('vc_results') or {}):
            km['final_method_metrics'] = {final: run['vc_results'][final]}
        return km or None

    # ── recall（注入 prompt）────────────────────────────────────────

    def recall_records(self, task_meta=None, exclude_dataset=None, *, focused=False):
        """Select the same experiences and runs for the decision prompt and the demo."""
        exps = [e for e in self.experiences
                if not (exclude_dataset and e.get('dataset') == exclude_dataset)]
        exps.sort(key=lambda e: e.get('updated_at') or '', reverse=True)
        truncated = False
        cur_type = (task_meta or {}).get('task_type')
        dataset = (task_meta or {}).get('dataset')
        if focused:
            def matches(record):
                if exclude_dataset and record.get('dataset') == exclude_dataset:
                    return False
                return bool(
                    (dataset and record.get('dataset') == dataset)
                    or (cur_type not in (None, '', 'other') and record.get('task_type') == cur_type)
                )

            matched_exps = [record for record in exps if matches(record)]
            matched_runs = [record for record in reversed(self.runs) if matches(record)]
            limit = DEMO_RECALL_LIMIT
            return matched_exps[:limit], matched_runs[:limit], len(matched_exps) > limit, bool(matched_runs)
        if len(exps) > MAX_EXPERIENCES_IN_PROMPT:
            keep = [e for e in exps if cur_type and e.get('task_type') == cur_type]
            keep += [e for e in exps if e not in keep][:MAX_EXPERIENCES_IN_PROMPT - len(keep)]
            exps, truncated = keep[:MAX_EXPERIENCES_IN_PROMPT], True
        runs = []
        for r in reversed(self.runs):
            if exclude_dataset and r.get('dataset') == exclude_dataset:
                continue
            if cur_type and r.get('task_type') == cur_type:
                runs.append(r)
            elif dataset and r.get('dataset') == dataset:
                runs.append(r)
            if len(runs) >= MAX_RUNS_IN_LAYER2_PROMPT:
                break
        related = bool(runs)
        if not related:
            # 无同领域记录（或无法分类）→ 最近几条兜底，明确标注非同类
            runs = [r for r in reversed(self.runs)
                    if not (exclude_dataset and r.get('dataset') == exclude_dataset)]
            runs = runs[:MAX_RUNS_IN_LAYER2_PROMPT]
        return exps, runs, truncated, related

    def recall_for_prompt(self, task_meta=None, exclude_dataset=None, *, focused=False):
        """生成注入 prompt 的「历史经验」文本块（由粗到细两段）。空 memory 返回 ''。"""
        if not self.experiences and not self.runs:
            return ''

        exps, runs, truncated, related = self.recall_records(task_meta, exclude_dataset, focused=focused)
        cur_type = (task_meta or {}).get('task_type')
        exp_lines = []
        for e in exps:
            head = (f"- [{e.get('task_type')}] {e.get('dataset')} "
                    f"{e.get('downstream_task') or ''} ({e.get('language') or '?'})")
            if e.get('goal'):
                head += f" 目标: {_truncate(e['goal'], 100)}"
            rec = e.get('recommended_method') or 'N/A'
            avoid = ",".join(e.get('avoid_methods') or []) or 'N/A'
            metrics = ''
            km = e.get('key_metrics') or {}
            if km.get('baseline_bacc') is not None:
                metrics = f" baseline_bacc={_fmt(km['baseline_bacc'])}"
            bacc_map = km.get('downstream_bacc_by_method')
            if bacc_map:
                best = sorted(bacc_map.items(),
                              key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else -1))[:3]
                metrics += " 各方法bacc: " + ",".join(f"{m}={_fmt(v)}" for m, v in best)
            lesson = _truncate(e.get('lesson'), 700) if focused else e.get('lesson')
            exp_lines.append(
                f"{head}\n  推荐={rec} | 避免={avoid} | 运行{e.get('n_runs', len(e.get('run_ids', [])))}次{metrics}"
                f"\n  经验: {lesson}")
        l1_block = ''
        if exp_lines:
            note = f"（共 {len(self.experiences)} 条{'' if not truncated else '，已按相关性裁剪'}）"
            l1_block = (f"\n【Layer-1 任务级经验：历史任务的教训（新→旧）】{note}\n"
                        + "\n".join(exp_lines))

        # ── 段2：Layer-2 同类任务运行明细（细）──
        l2_note = '' if related else '（暂无同领域记录，以下为其他领域最近运行，仅供参考）'
        run_lines = []
        for r in runs:
            final = r.get('final_vc_method') or 'N/A'
            if r.get('final_vc_params'):
                final += ' ' + json.dumps(r['final_vc_params'], ensure_ascii=False)
            per_method = "; ".join(
                f"{m}→" + ",".join(f"{k}={_fmt(v)}" for k, v in sorted((ms or {}).items()))
                for m, ms in (r.get('vc_results') or {}).items())
            bacc_map = r.get('downstream_bacc_by_method') or {}
            bacc_str = ''
            if bacc_map:
                bacc_str = " | bacc: " + ",".join(f"{m}={_fmt(v)}" for m, v in bacc_map.items())
            refl = r.get('reflections') or []
            last_refl = f" | 经验: {refl[-1]}" if refl else ''
            gr = r.get('goal_runs') or []
            gr_str = ''
            if gr:
                bacc_gr = [g for g in gr if 'BACC' in (g.get('goal') or '')]
                best = bacc_gr[0].get('target') if bacc_gr else None
                gr_str = f" | 目标变体{len(gr)}个" + (f"(BACC目标最优={best})" if best else '')
            task_desc = r.get('task_meta', {}).get('downstream_task') or ''
            run_lines.append(
                f"- [{r.get('dataset')}] {task_desc} 语言={r.get('language')} "
                f"最终选择={final} | {per_method}{bacc_str}{gr_str}{last_refl}")
        l2_block = ''
        if run_lines:
            scope = f"task_type={cur_type}" if cur_type else "最近运行"
            scope += l2_note
            l2_block = (f"\n【Layer-2 同类任务运行明细（{scope}，新→旧）】\n"
                        + "\n".join(run_lines))

        parts = [p for p in (l1_block, l2_block) if p]
        if not parts:
            return ''
        return ("以下是本 Agent 的双层历史经验（由粗到细）：Layer-1 为任务级教训，"
                "Layer-2 为同类任务每次运行的细节。当前数据集/同领域的记录参考价值最高，"
                "供决策参考，不代表当前任务一定同样适用）：\n" + "\n".join(parts))

    # ── legacy 迁移 ─────────────────────────────────────────────────

    def migrate_legacy_runs(self):
        """把 legacy 单文件条目转成 v2 run 文件（原文件不动）。返回迁移条数。"""
        if not self.legacy_path:
            return 0
        with open(self.legacy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        legacy_runs = data.get('runs', []) if isinstance(data, dict) else data
        n = 0
        for old in legacy_runs:
            run = legacy_to_v2(old)
            if run is None:
                continue
            self.runs.append(run)
            self.save_run(run)
            n += 1
        self._log('info', f"legacy 迁移完成: {n} 条 → {self.runs_dir}")
        return n


def legacy_to_v2(old):
    """旧版单条 run entry → v2 schema（fallback_classify 补 task_meta）。"""
    if not isinstance(old, dict) or not old.get('run_id'):
        return None
    meta = fallback_classify(old.get('request') or '')
    if old.get('language'):
        meta['language'] = old['language']
    meta['source'] = 'legacy_csv' if old.get('request') == '(backfill from evaluate CSVs)' else 'legacy'
    return {
        'run_id': old['run_id'],
        'schema_version': SCHEMA_VERSION,
        'task_type': meta.get('task_type') or 'other',
        'task_meta': {k: meta.get(k) for k in
                      ('language', 'n_speakers', 'downstream_task', 'goal', 'source', 'warnings')},
        'dataset': old.get('dataset'),
        'language': old.get('language') or meta.get('language'),
        'request': old.get('request'),
        'output_dir': old.get('output_dir'),
        'timestamp': old.get('timestamp'),
        'attempts': old.get('attempts'),
        'final_vc_method': old.get('final_vc_method'),
        'success': old.get('success', True),
        'baseline_bacc': None,
        'final_downstream': None,
        'downstream_bacc_by_method': None,
        'vc_results': old.get('vc_results') or {},
        'reflections': old.get('reflections') or [],
        'artifacts': {},
        'evidence_level': 'legacy_csv',
    }


def _validate_experience(out):
    """LLM 经验输出校验：返回错误字符串或 None。"""
    if not isinstance(out, dict):
        return '输出非 dict'
    lesson = out.get('lesson')
    if not isinstance(lesson, str) or len(lesson.strip()) < 10:
        return 'lesson 缺失或过短'
    task_type = out.get('task_type')
    if task_type is not None and not isinstance(task_type, str):
        return 'task_type 非字符串'
    rec = out.get('recommended_method')
    if rec is not None and not isinstance(rec, str):
        return 'recommended_method 类型错误'
    avoid = out.get('avoid_methods')
    if avoid is not None and not isinstance(avoid, list):
        return 'avoid_methods 类型错误'
    return None


# ── 兼容别名：旧代码/旧脚本 import AgentMemory ──────────────────────────────
AgentMemory = TwoLayerMemory


def _clean_methods(methods, allow_none=True):
    """把 LLM 输出的方法名列表清洗成词表规范名：小写归一 + 别名/旧名映射
    （asr_tts→Resyn，旧名 fishaudio_tts/seedvc/mcadams/formant 也映射到新名），
    丢弃非 VC 方法名（如 baseline_ner_tts）；recommended_method 单值同样处理。"""
    if methods is None:
        return None if allow_none else []
    single = isinstance(methods, str)
    seq = [methods] if single else list(methods)
    out = []
    # 小写别名（含旧方法名）→ 规范名（大小写与结果目录一致）
    _ALIAS = {'asr_tts': 'Resyn', 'asrtts': 'Resyn', 'fishaudio': 'Resyn', **ALIASES}
    _CANONICAL = {m.lower(): m for m in KNOWN_VC_METHODS}
    for m in seq:
        if not isinstance(m, str):
            continue
        low = m.strip().lower().replace('-', '_').replace(' ', '_')
        canon = _ALIAS.get(low) or _CANONICAL.get(low)
        if canon and canon not in out:
            out.append(canon)
    if single:
        return out[0] if out else None
    return out or (None if allow_none else [])


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _to_float(v):
    try:
        return round(float(v), 6)
    except (ValueError, TypeError):
        return v


def _fmt(v):
    try:
        return f"{float(v):.3g}"
    except (ValueError, TypeError):
        return str(v)
