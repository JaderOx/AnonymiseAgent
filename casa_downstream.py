"""
casa_downstream.py — CASA 语音匿名 Agent

执行 → 评估 → 反思 → 调整 → 重试，直到满足用户目标。根据评估指标动态调整 VC 方法。

依赖：agent.py（pipeline 函数）、utils/（logger、内存隔离）

用法示例：
    python casa_downstream.py -i /path/to/audio -r "匿名化，不要改变韵律，SNR尽量高，不需要内容匿名"
    python casa_downstream.py -i /path/to/audio -r "..." --max_reflections 3
"""

import os
import re
import json
import inspect
import time
import argparse
from configs.config_loader import get_asr_config, uses_character_timestamps
from utils.agent_choices import choice_error, validate_tool_choices
from utils.vc_names import METHODS as VC_METHODS, canonical_method
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from utils.agent_selection import final_attempt_from
from utils.vc_parameters import VC_DEFAULTS, VC_PARAMETER_GUIDE, resolve_vc_params, vc_output_name

from agent import (
    trans, mask, vc,
    _has_audio, _has_json,
    _ASR_CONDA_ENV, _VC_CONDA_ENV,
    SUPPORTED_FORMATS,
)
from utils.logger_example import setup_logger
from utils.manage_module_memory import run_in_isolation
from utils.agent_memory import (TwoLayerMemory, fallback_classify, _validate_experience,
                                is_valid_new_task_type)

# 指标含义（给 LLM 参考）
METRIC_DESCRIPTIONS = {
    "snr_db": "信噪比(dB)，越高越好",
    "mos": "语音自然度(1-5)，越高越好",
    "cer": "字符错误率，越低越好",
    "emo": "情感相似度(0-1)，越高越好",
    "pit_l1": "韵律L1距离，越低越好",
    "pit_pcc": "韵律相关系数(0-1)，越高越好",
    "eer": "等错误率(%)，越接近50%越好（匿名效果）",
}

DOWNSTREAM_METRIC_DESCRIPTIONS = {
    "downstream_bacc": "下游分类任务 Balanced Accuracy (0-1)，越高越好",
    "downstream_f1": "下游分类任务 Macro F1 (0-1)，越高越好",
}

# 初始计划 system prompt（规划提示词）
INITIAL_PLAN_PROMPT = f"""你是语音匿名处理 Agent。根据用户请求生成 JSON 执行计划。

【可用工具】
- run_transcription: ASR 转录。参数: language(必填，如 zh/en/ja/es), asr_model(可选: {', '.join(_ASR_CONDA_ENV)}，留空按语言选择), spk_num(可选，精确说话人数), min_speakers(可选，最少说话人数), max_speakers(可选，最多说话人数)。
- run_mask: 识别并掩蔽录音中说出的敏感信息（依赖转录）。用户要求保护内容时，将此步骤放在声线转换前，并为转换设置 with_mask=true。参数: language(必填，如 zh/en/ja/es), mask_method (noise or mute, default noise). Use mask_method="mute" when the user asks to mute identifiers or replace them with silence.
- run_voice_conversion: 声线转换。参数: method(必填, 可选: {', '.join(VC_METHODS)}), with_mask(bool), language, vc_params (optional).
{VC_PARAMETER_GUIDE}
- run_evaluation: 评估。参数: language(必填)

【步骤依赖】
- run_mask 依赖 run_transcription
- run_voice_conversion 独立，with_mask=true 时需要 mask 输出
- run_evaluation 依赖 run_transcription，且需要 mask 或 vc 输出

Follow the methods and privacy requirements in the user request. Use historical experience only for choices the request leaves open.
Write reasoning in the language of the user request. Use a language code such as zh or en for audio.
【输出格式】在适当位置输出 JSON，以 {{ 开头。
{{"language":"语言","reasoning":"分析","steps":[{{"tool":"工具名","params":{{}}}}]}}

Return exactly one JSON object with a top-level steps array. Include at most one
run_voice_conversion call. Put method settings inside that call's vc_params object.

Example request: "Mute spoken identifiers in an English recording, anonymize the voice, and evaluate."
{{"language":"en","reasoning":"Mask spoken identifiers, change the voice, then evaluate.","steps":[{{"tool":"run_transcription","params":{{"language":"en"}}}},{{"tool":"run_mask","params":{{"language":"en","mask_method":"mute"}}}},{{"tool":"run_voice_conversion","params":{{"method":"pitch","language":"en","with_mask":true,"vc_params":{{}}}}}},{{"tool":"run_evaluation","params":{{"language":"en"}}}}]}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 定义 & AgentContext
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
    dependencies: list = field(default_factory=list)


class AgentContext:
    def __init__(self, input_dir, output_dir, middle_dir, mode, logger, logger_file):
        self.input_dir   = input_dir
        self.output_dir  = output_dir
        self.middle_dir  = middle_dir
        self.mode        = mode
        self.logger      = logger
        self.logger_file = logger_file

        M = middle_dir
        self.trans_origin_dir = os.path.join(M, 'trans_origin')
        self.mask_dir         = os.path.join(M, 'mask')
        self.mask_method = 'noise'
        self.text_only_dir = None
        self.spk_audio_origin = os.path.join(M, 'spk_audio', 'origin')
        self.age_gender_model = os.path.join(
            os.environ.get('TORCH_HOME', './models'), 'w2v2-L-robust-6-age-gender'
        )

        self.asr_model_used = None
        self.vc_method_used = None
        self.vc_params = {}
        self.vc_with_mask = False

    def select_vc(self, method, params, with_mask):
        self.vc_method_used = method
        self.vc_params = resolve_vc_params(method, params)
        self.vc_with_mask = with_mask

    @property
    def vc_name(self):
        return vc_output_name(self.vc_method_used or "NeuralVC", self.vc_params)

    @property
    def vc_audio_dir(self):
        dirname = f"text+{self.vc_name}" if self.vc_with_mask else self.vc_name
        return os.path.join(self.output_dir, dirname)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具处理器
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_check_directory(ctx, directory, file_type="any"):
    ctx.logger.debug(f"[handler] _handle_check_directory: directory={directory}, file_type={file_type}")
    p = Path(directory)
    if not p.exists():
        return {"exists": False, "count": 0, "message": f"目录不存在: {directory}"}
    if file_type == "audio":
        files = [f for f in p.rglob('*') if f.suffix.lower() in SUPPORTED_FORMATS]
    elif file_type == "json":
        files = list(p.rglob('*.json'))
    else:
        files = [f for f in p.rglob('*') if f.is_file()]
    sample = [str(f.relative_to(p)) for f in files[:5]]
    return {"exists": True, "count": len(files), "sample_files": sample,
            "message": f"找到 {len(files)} 个文件"}


def _missing_audio_artifacts(input_dir, artifact_dir, suffix=None):
    root = Path(input_dir)
    paths = (audio.relative_to(root) for audio in root.rglob("*")
             if audio.is_file() and audio.suffix.lower() in SUPPORTED_FORMATS)
    if suffix:
        paths = (path.with_suffix(suffix) for path in paths)
    return sum(not (Path(artifact_dir) / path).is_file() for path in paths)


def _handle_transcription(ctx, language, asr_model="", hotwords=None, spk_num=0,
                          min_speakers=None, max_speakers=None):
    hotwords = hotwords or []
    spk_num = spk_num or None
    min_speakers = min_speakers or None
    max_speakers = max_speakers or None
    ctx.logger.debug(f"[handler] _handle_transcription: language={language}, asr_model={asr_model}, hotwords={hotwords}, spk_num={spk_num}, min_speakers={min_speakers}, max_speakers={max_speakers}")
    if not asr_model:
        asr_model = get_asr_config(language)['model']
    if asr_model not in _ASR_CONDA_ENV:
        raise ValueError(f'Unknown ASR model: {asr_model}')
    ctx.asr_model_used = asr_model
    ctx.logger.info(f"[Agent] 转录: language={language} model={asr_model}")
    run_in_isolation(
        trans,
        ctx.input_dir, asr_model, ctx.trans_origin_dir,
        hotwords, language, ctx.logger_file, ctx.mode,
        ctx.spk_audio_origin, ctx.age_gender_model, spk_num,
        min_speakers, max_speakers,
        conda_env=_ASR_CONDA_ENV.get(asr_model),
    )
    missing = _missing_audio_artifacts(ctx.input_dir, ctx.trans_origin_dir, ".json")
    if missing:
        return {"success": False, "error": f"Missing transcripts for {missing} input audio files. See the log."}
    return {"success": True, "trans_dir": ctx.trans_origin_dir, "asr_model": asr_model}


def _handle_mask(ctx, language, llm_model="", mask_method="noise",
                 group_num=60, custom_prompt="", detect_method="llm"):
    ctx.logger.debug(f"[handler] _handle_mask: language={language}, llm_model={llm_model}, mask_method={mask_method}, group_num={group_num}, detect_method={detect_method}")
    if not _has_json(ctx.trans_origin_dir):
        return {"success": False,
                "error": f"转录结果不存在于 {ctx.trans_origin_dir}，请先执行转录步骤"}
    if detect_method == 'ner':
        ctx.logger.info(f"[Agent] 内容匿名: NER 模式")
    else:
        if not llm_model:
            try:
                from configs.config_loader import get_llm_config
                cfg = get_llm_config(language)
                llm_model = cfg['model']
                if not custom_prompt:
                    custom_prompt = cfg.get('prompt', '')
            except Exception:
                llm_model = 'Qwen2.5-32B-Instruct'
        ctx.logger.info(f"[Agent] 内容匿名: llm={llm_model} method={mask_method}")
    run_in_isolation(
        mask,
        ctx.trans_origin_dir, ctx.mask_dir, ctx.logger_file,
        language, custom_prompt or None, llm_model, group_num, ctx.mode,
        detect_method,
    )
    missing = _missing_audio_artifacts(ctx.input_dir, ctx.mask_dir, ".json")
    if missing:
        return {"success": False, "error": f"Missing content mask files for {missing} input audio files. See the log."}
    noise_method = 'formant' if mask_method == 'noise' else 'mute'
    text_only_dir = os.path.join(ctx.output_dir, 'text_only')
    from apply_mask import apply_mask
    apply_mask(wav_dir=ctx.input_dir, mask_dir=ctx.mask_dir,
               output_dir=text_only_dir, logger=ctx.logger,
               noise_method=noise_method)
    missing = _missing_audio_artifacts(ctx.input_dir, text_only_dir)
    if missing:
        return {"success": False, "error": f"Missing text-masked audio for {missing} input files. See the log."}
    ctx.mask_method = mask_method
    ctx.text_only_dir = text_only_dir
    return {"success": True, "text_only_dir": text_only_dir, "mask_dir": ctx.mask_dir}


def _handle_voice_conversion(ctx, method, with_mask=False, num_workers=1,
                             whole_audio=False, language="zh", vc_params=None):
    method = canonical_method(method)
    ctx.select_vc(method, vc_params, with_mask)
    vc_output_dir = ctx.vc_audio_dir
    name = ctx.vc_name
    spk_vc_dir = os.path.join(ctx.middle_dir, 'spk_audio', f'text_{name}' if with_mask else f'{name}_only')
    if with_mask and _missing_audio_artifacts(ctx.input_dir, ctx.mask_dir, ".json"):
        return {"success": False,
                "error": "with_mask=True 但 mask 结果不存在，请先执行内容匿名步骤"}
    trans_dir_for_vc = ctx.trans_origin_dir if _has_json(ctx.trans_origin_dir) else None
    ctx.logger.info(f"[Agent] 声线转换: method={method} vc_params={ctx.vc_params} with_mask={with_mask}")
    run_in_isolation(
        vc,
        ctx.input_dir, vc_output_dir, method,
        trans_dir_for_vc, ctx.logger_file, ctx.mode,
        num_workers,
        ctx.mask_dir if with_mask else None,
        ctx.mask_method, language, whole_audio, ctx.vc_params,
        conda_env=_VC_CONDA_ENV.get(method),
    )
    missing = _missing_audio_artifacts(ctx.input_dir, vc_output_dir)
    if missing or not _has_audio(vc_output_dir):
        return {"success": False, "error": f"Converted audio is incomplete ({missing} missing) in {vc_output_dir}. See the log."}
    from utils.metrics.spk_audio_extractor import extract_dir as _ext_spk_pair
    _ext_spk_pair(ctx.input_dir, vc_output_dir, ctx.trans_origin_dir,
                  ctx.spk_audio_origin, spk_vc_dir, ctx.logger)
    return {"success": True, "vc_output_dir": vc_output_dir, "method": method, "vc_params": ctx.vc_params}


def _cached_metrics(output_dir, mode):
    import csv
    path = Path(output_dir) / f'evaluate_{mode}.csv'
    if path.is_file():
        try:
            with path.open(encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('name') == '-mean-':
                        return {k: v for k, v in row.items() if k != 'name' and v}
        except (OSError, ValueError, csv.Error):
            pass
    return {}


def _handle_evaluation(ctx, language, gt_dir="", subject_map_path="", with_mask=False):
    ctx.logger.debug(f"[handler] _handle_evaluation: language={language}, gt_dir={gt_dir}, subject_map_path={subject_map_path}, with_mask={with_mask}")
    method = ctx.vc_name
    text_vc_dir = os.path.join(ctx.output_dir, f'text+{method}')
    vc_only_dir = os.path.join(ctx.output_dir, method)
    text_only   = os.path.join(ctx.output_dir, 'text_only')

    if with_mask and _has_audio(text_vc_dir):
        eval_dir, trans_vc, mode_str = text_vc_dir, os.path.join(ctx.middle_dir, f'trans_text+{method}'), f'text_{method}'
    elif not with_mask and _has_audio(vc_only_dir):
        eval_dir, trans_vc, mode_str = vc_only_dir, os.path.join(ctx.middle_dir, f'trans_{method}'), method
    elif with_mask and ctx.vc_method_used is None and _has_audio(text_only):
        eval_dir, trans_vc, mode_str = text_only, os.path.join(ctx.middle_dir, 'trans_text'), 'text'
    else:
        return {"success": False, "error": "找不到匿名后音频，请先执行 mask 或 vc 步骤"}

    cached = _cached_metrics(ctx.output_dir, mode_str)
    if cached:
        return {"success": True, "eval_dir": eval_dir, "mode": mode_str,
                "metric_cols": list(cached), "mean_results": cached, "skipped": True}

    asr_model = ctx.asr_model_used or get_asr_config(language)['model']
    ctx.logger.info(f"[Agent] 评估: eval_dir={eval_dir}")

    # ── 先评估原始音频（用于对比基线） ──
    origin_csv = os.path.join(ctx.output_dir, 'evaluate_origin.csv')
    if not os.path.exists(origin_csv):
        ctx.logger.info("[Agent] 评估原始音频基线...")
        try:
            from evaluate import evaluate_origin
            evaluate_origin(
                input_dir_origin=ctx.input_dir,
                output_dir=ctx.output_dir,
                logger=ctx.logger,
            )
        except Exception as e:
            ctx.logger.warning(f"[Agent] 原始音频评估失败（继续匿名评估）: {e}")

    from evaluate import evaluate_all
    df = evaluate_all(
        input_dir_vc=eval_dir, trans_dir_vc=trans_vc,
        input_dir_origin=ctx.input_dir, trans_dir_origin=ctx.trans_origin_dir,
        middle_dir=ctx.middle_dir, asr_model=asr_model,
        hotwords=[], language=language, output_dir=ctx.output_dir,
        logger=ctx.logger, logger_file=ctx.logger_file,
        _trans_env=_ASR_CONDA_ENV.get(asr_model),
        gt_dir=gt_dir or None, subject_map_path=subject_map_path or None,
        mode=mode_str, wer=None if uses_character_timestamps(language) else True,
    )

    if df is None:
        return {"success": False, "error": "所有指标计算均失败，请检查日志"}

    metric_cols = [c for c in df.columns if c != 'name']
    mean_row = df[df['name'] == '-mean-']
    results = {}
    if not mean_row.empty:
        for col in metric_cols:
            results[col] = str(mean_row[col].iloc[0])

    return {
        "success": True,
        "eval_dir": eval_dir,
        "mode": mode_str,
        "metric_cols": metric_cols,
        "mean_results": results,
    }


# ── 工具注册表 ────────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="check_directory",
        description="检查目录下是否存在文件。file_type: audio/json/any",
        parameters={"directory": "string (路径)", "file_type": "string (可选, 默认 any)"},
        handler=_handle_check_directory,
    ),
    Tool(
        name="run_transcription",
        description="ASR 转录：将音频转为带时间戳的 JSON 文本",
        parameters={
            "language": "string (必填, 语言代码如 zh/en/ja/es)",
            "asr_model": f"string (可选, {', '.join(_ASR_CONDA_ENV)}, 留空按语言选择)",
            "hotwords": "list (可选, 热词列表)",
            "spk_num": "int (可选, 精确说话人数量, 0=自动)",
            "min_speakers": "int (可选, 最少说话人数)",
            "max_speakers": "int (可选, 最多说话人数)",
        },
        handler=_handle_transcription,
    ),
    Tool(
        name="run_mask",
        description="内容匿名：识别隐私词并在音频中遮蔽（依赖 run_transcription）",
        parameters={
            "language": "string (必填)",
            "llm_model": "string (可选, 留空自动选择)",
            "mask_method": "string (可选, noise 或 mute, 默认 noise)",
            "group_num": "int (可选, 每批送入LLM的句子数, 默认60)",
            "custom_prompt": "string (可选)",
            "detect_method": "string (可选, llm 或 ner, 默认 llm)",
        },
        handler=_handle_mask,
        dependencies=["run_transcription"],
    ),
    Tool(
        name="run_voice_conversion",
        description="声线转换：改变说话人声线特征",
        parameters={
            "method": f"string (required, {', '.join(VC_DEFAULTS)})",
            "vc_params": VC_PARAMETER_GUIDE,
            "with_mask": "bool (可选, 默认 false)",
            "num_workers": "int (可选, 默认1)",
            "whole_audio": "bool (可选, 默认 false)",
            "language": "string (可选, 默认 中文)",
        },
        handler=_handle_voice_conversion,
    ),
    Tool(
        name="run_evaluation",
        description="评估匿名效果：计算 SNR/MOS/CER/EMO/EER 等指标",
        parameters={
            "language": "string (必填)",
            "gt_dir": "string (可选)",
            "subject_map_path": "string (可选)",
        },
        handler=_handle_evaluation,
        dependencies=["run_transcription"],
    ),
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}


# ═══════════════════════════════════════════════════════════════════════════════
# JSON 解析工具
# ═══════════════════════════════════════════════════════════════════════════════

def _json_objects(text):
    """Yield complete JSON objects in response order, skipping their nested fields."""
    decoder = json.JSONDecoder()
    text = text.lstrip()
    # Some completion templates continue immediately after an opening brace.
    if text.startswith('"'):
        text = '{' + text
    cursor = 0
    while (start := text.find('{', cursor)) != -1:
        try:
            value, cursor = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        yield value


def _validate_plan(plan: dict) -> Optional[str]:
    """Validate tool names and method-specific parameters before execution."""
    if not isinstance(plan, dict):
        return "Expected a JSON plan"
    if plan.get("error"):
        return f"无法从 LLM 输出中提取执行计划，原始输出:\n{plan.get('raw', '')}"
    if "steps" not in plan:
        return "计划缺少 steps 字段"
    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        return "steps 为空或格式不正确"
    for i, s in enumerate(plan["steps"]):
        if not isinstance(s, dict) or not isinstance(s.get("params", {}), dict):
            return f"Invalid tool call at step {i+1}"
        name = s.get("tool", "")
        problem = validate_tool_choices(name, s.get("params", {}), TOOL_BY_NAME, _ASR_CONDA_ENV)
        if problem:
            return f"Step {i+1}: {problem}"
        if name == "run_voice_conversion":
            params = s.setdefault("params", {})
            unknown = sorted(set(params) - TOOL_BY_NAME[name].parameters.keys())
            if unknown:
                return f"Unexpected voice conversion fields: {', '.join(unknown)}. Put method settings inside vc_params."
            method = params.get("method")
            params["method"] = canonical_method(method)
            try:
                params["vc_params"] = resolve_vc_params(params["method"], params.get("vc_params"))
            except ValueError as exc:
                return str(exc)
    if sum(step["tool"] == "run_voice_conversion" for step in plan["steps"]) > 1:
        return "Plan one VC configuration at a time. Evaluate additional settings during reflection."
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 反思 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM_PROMPT = f"""你是一个语音匿名处理专家。根据评估指标和用户目标，决定是否更换声线转换(VC)方法或调整其参数。

【可用 VC 方法】
{chr(10).join(f'- {m}' for m in VC_METHODS)}
{VC_PARAMETER_GUIDE}

【指标含义】
{chr(10).join(f'- {k}: {v}' for k, v in METRIC_DESCRIPTIONS.items())}


【决策规则】
1. 若用户指定了具体方法（如“用NeuralVC”），且该方法与参数已经满足用户目标，则直接结束（done=true）。
2. 若用户只提出抽象目标（如“SNR尽量高”），则用指标判定：当前方法若满足目标即结束；否则可调整当前方法的参数，或尝试另一方法。
3. 若用户要求某指标“最高/最低”，需要对比所有方法后再决定，不可提前结束。
4. 重复选择相同方法和相同参数表示回到该次结果。若同一方法有多次尝试，用 best_vc_params 指明选中的参数。
5. 历史仅记录“已尝试”，不代表“失败”，不要因此否定某个方法。
6. 只要满足用户明确提出的目标，无需在意未提及的指标。

Write reason and summary in the language of the user request.
【输出格式】只输出 JSON，以 {{ 开头、以 }} 结尾。

满意时:
{{"done":true,"reason":"简短说明","summary":"最终结果摘要","best_vc_method":"最终认定的最优方法名（从已尝试方法中选，不一定是最后尝试的那个）"}}

不满意时:
{{"done":false,"reason":"简短说明","new_vc_method":"方法名","vc_params":{{}}}}

示例:
用户要求"用NeuralVC"，NeuralVC已执行:
{{"done":true,"reason":"用户指定NeuralVC，已成功执行","summary":"NeuralVC","best_vc_method":"NeuralVC"}}

遍历全部方法后认定 pitch 综合最优（当前最后尝试的是 formant_spec）:
{{"done":true,"reason":"pitch在匿名与下游可用性之间平衡最好","summary":"pitch: EER=42%, bacc=0.67","best_vc_method":"pitch"}}

用户希望减小变调幅度，当前 pitch 使用了 -3 个半音:
{{"done":false,"reason":"尝试较小的变调幅度","new_vc_method":"pitch","vc_params":{{"n_steps":-1.5}}}}

用户要求"SNR尽量高"，当前 pitch SNR=15dB:
{{"done":false,"reason":"pitch的SNR只有15dB，NeuralVC通常SNR更高","new_vc_method":"NeuralVC"}}
"""


def _build_context_info(ctx):
    has_trans = _has_json(ctx.trans_origin_dir)
    has_mask = _has_json(ctx.mask_dir)
    return (
        f"输入音频目录：{ctx.input_dir}\n"
        f"已有转录结果：{'是' if has_trans else '否'}\n"
        f"已有内容匿名结果：{'是' if has_mask else '否'}\n"
    )


def _metrics_to_str(metrics: dict) -> str:
    if not metrics:
        return "（无指标）"
    parts = []
    for k, v in metrics.items():
        desc = METRIC_DESCRIPTIONS.get(k, "")
        try:
            v = float(v)
        except (ValueError, TypeError):
            pass
        # 保留3位有效数字
        if isinstance(v, (int, float)):
            v_str = f"{v:.3g}"
        else:
            v_str = str(v)
        parts.append(f"  {k} = {v_str}" + (f" ({desc})" if desc else ""))
    return "\n".join(parts)


def _history_to_str(history: list) -> str:
    if not history:
        return "（无历史尝试）"
    lines = []
    for i, h in enumerate(history):
        m = h.get("metrics", {})
        m_str = ", ".join(f"{k}={v}" for k, v in m.items()) or "无指标"
        reason = h.get("reason", "")
        line = ""
        if reason:
            line += f"第{i+1}次:\n[思考]: {reason}\n"
        line += f"[结果]: VC={h['vc_method']} vc_params={json.dumps(h.get('vc_params', {}))} | {m_str}"

        lines.append(line)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 子进程推理（进程退出后显存自动归还）
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_infer_worker(model_name, sys_prompt, user_prompt, output_file, gen_kwargs=None, open_source=True):
    """在子进程中加载 LLM、推理一次、写结果到文件。进程退出后 GPU 显存完全释放。"""
    gen_kwargs = gen_kwargs or {}
    if open_source:
        from utils.language_model import load_model, generate_response
        model, tokenizer = load_model(model_name)
        # temperature > 0 时必须 do_sample=True，否则模型会忽略 temperature 并报警告
        if gen_kwargs.get('temperature', 0) > 0:
            gen_kwargs.setdefault('do_sample', True)
        else:
            gen_kwargs['do_sample'] = False
        raw = generate_response(model, tokenizer, sys_prompt=sys_prompt, prompt=user_prompt, **gen_kwargs)
    else:
        from utils.openai_gpt import load_openai, generate_response
        client = load_openai()
        raw = generate_response(model=model_name, client=client, sys_prompt=sys_prompt,
                                prompt=user_prompt, **gen_kwargs)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# CASA
# ═══════════════════════════════════════════════════════════════════════════════

class CASA:
    focused_memory = False

    def __init__(self, input_dir, output_dir, middle_dir, llm_model=None, mode='INFO',
                 temperature=0.01, max_new_tokens=2048, repetition_penalty=1.05, open_source=True,
                 memory_path=None, use_memory=True, memory_exclude_dataset=None):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = output_dir or os.path.join(os.path.dirname(self.input_dir), 'output')
        self.middle_dir = middle_dir or os.path.join(os.path.dirname(self.input_dir), 'middle')
        self.mode = mode

        if llm_model is None:
            # 默认 LLM：Qwen2.5-32B-Instruct；AGENT_LLM_MODEL 可覆盖
            self.llm_model = os.environ.get('AGENT_LLM_MODEL', 'Qwen2.5-32B-Instruct')
        else:
            self.llm_model = llm_model

        self.open_source = open_source

        self.gen_kwargs = {
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
            'enable_thinking': False,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger, self.logger_file = setup_logger(self.output_dir, 'casa', mode)
        self.ctx = AgentContext(self.input_dir, self.output_dir, self.middle_dir,
                                mode, self.logger, self.logger_file)

        # ── 下游任务配置（可选）──
        self.downstream_data_json = None
        self.downstream_script = None

        # ── 评估额外参数 ──
        self.gt_dir = ""
        self.subject_map_path = ""

        # ── 跨运行双层 memory（失败/关闭时 Agent 行为与无 memory 完全一致） ──
        from utils.agent_memory import infer_dataset_name
        self._dataset_name = infer_dataset_name(self.input_dir)
        self.task_meta = None      # run() 开头由 _classify_request 填充
        self.memory = None
        self._memory_exclude = memory_exclude_dataset  # recall 时排除某数据集的经验（LOO 消融）
        if use_memory:
            try:
                self.memory = TwoLayerMemory(root=memory_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory'), logger=self.logger)
            except Exception as e:
                self.logger.warning(f"[Memory] 初始化失败（按无 memory 运行）: {e}")
            if self._memory_exclude and self.memory is not None:
                self.logger.info(f"[Memory] LOO 模式：recall 时排除数据集 '{self._memory_exclude}' 的经验")

    def _call_llm(self, sys_prompt: str, user_prompt: str, gen_override: dict = None) -> str:
        """在子进程中调用 LLM，进程退出后显存自动释放。返回原始响应文本。"""
        fd, output_file = tempfile.mkstemp(suffix='.txt', prefix='llm_result_')
        os.close(fd)
        gen = {**self.gen_kwargs, **(gen_override or {})}
        src = "local" if self.open_source else "api"
        try:
            self.logger.debug(f"[LLM 子进程] 启动 | model={self.llm_model} | source={src} | gen={gen}")
            self.logger.debug(f"[LLM 子进程] system prompt ({len(sys_prompt)} chars):\n{sys_prompt}")
            self.logger.debug(f"[LLM 子进程] user prompt ({len(user_prompt)} chars):\n{user_prompt}")

            run_in_isolation(
                _llm_infer_worker,
                self.llm_model, sys_prompt, user_prompt, output_file, gen, self.open_source,
            )

            with open(output_file, 'r', encoding='utf-8') as f:
                raw = f.read()
            self.logger.debug(f"[LLM 子进程] 响应 ({len(raw)} chars):\n{raw}")
            return raw
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def _call_llm_with_retry(self, sys_prompt: str, user_prompt: str,
                             validator: Callable = None, max_retries: int = 3) -> dict:
        """调用 LLM 并解析 JSON，解析或验证失败时追加提示重试，最多 max_retries 次。

        Args:
            validator: 可选的验证函数，接收 dict 返回错误字符串或 None。
                       例如 _validate_plan。解析成功但验证失败时也会重试。
        """
        current_prompt = user_prompt
        error = "No complete JSON object found."
        for attempt in range(max_retries):
            override = {'temperature': 0, "max_new_tokens": 4096} if attempt else None
            raw = self._call_llm(sys_prompt, current_prompt, gen_override=override)
            candidates = list(_json_objects(raw or ''))
            error = "No complete JSON object found."
            for result in reversed(candidates):
                problem = validator(result) if validator else None
                if not problem:
                    return result
                if result is candidates[-1]:
                    error = problem
            keys = [sorted(candidate) for candidate in candidates[-5:]]
            self.logger.warning("[LLM] Attempt %d/%d failed: %s. JSON keys: %s",
                                attempt + 1, max_retries, error, keys)
            hint = "Return exactly one complete JSON object that satisfies the system instructions."
            if validator is _validate_plan:
                hint += ' Include a top-level language code and a nonempty steps array of tool calls.'
            current_prompt = f"{user_prompt}\n\nAttempt {attempt + 1} failed: {error}\n{hint}"
            if attempt >= 1:
                current_prompt += "\nReturn the final object only. Omit examples, comments, and Markdown fences."

        self.logger.error("[LLM] All %d attempts failed", max_retries)
        return {"error": True, "raw": error}

    # ── 下游任务 ────────────────────────────────────────────────────────────

    def _prepare_downstream_data_json(self, anon_audio_dir):
        """读取原始 data.json，将 wav_path 替换为匿名音频路径，保存在音频目录下。"""
        with open(self.downstream_data_json, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list) or not any(isinstance(e, dict) and "wav_path" in e for e in entries):
            # Custom downstream scripts may use their own metadata format.
            return self.downstream_data_json
        converted_dir = Path(anon_audio_dir).resolve()
        source_dir = Path(self.input_dir).resolve()
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("wav_path"):
                continue
            source = Path(entry["wav_path"])
            if source.is_absolute():
                relative = source.relative_to(source_dir) if source.is_relative_to(source_dir) else Path(source.name)
            else:
                relative = source
            converted = converted_dir / relative
            if not converted.is_file():
                converted = converted.with_suffix(".wav")
            if not converted.is_file() or not converted.resolve().is_relative_to(converted_dir):
                self.logger.warning(f"[Downstream] Missing converted audio for {source}, skipping evaluation")
                return None
            entry["wav_path"] = str(converted)
        out_path = os.path.join(anon_audio_dir, "downstream_input_data.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[Downstream] 已生成: {out_path}")
        return out_path

    def _run_downstream_script(self, data_json_path, anon_audio_dir):
        """执行下游 bash 脚本，output 放在 anon_audio_dir/downstream_output/ 下。"""
        ds_out = os.path.join(anon_audio_dir, "downstream_output")
        os.makedirs(ds_out, exist_ok=True)

        # ── 跳过：final_result.json 已存在 ──
        result_path = os.path.join(ds_out, "final_result.json")
        if os.path.exists(result_path):
            metrics = json.load(open(result_path))
            ba = metrics.get("balanced_accuracy", {})
            fm = metrics.get("f1_macro", {})
            self.logger.info(f"[跳过] downstream final_result.json 已存在: {result_path}")
            return metrics

        cmd = ["bash", self.downstream_script,
               os.path.abspath(data_json_path),
               os.path.abspath(ds_out)]
        self.logger.info(f"[Downstream] 执行: {' '.join(cmd)}")
        print(f"  [Downstream] 运行脚本...", flush=True)

        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
        except subprocess.TimeoutExpired:
            self.logger.error("[Downstream] 超时 (>24h)")
            return None

        if ret.returncode != 0:
            self.logger.error(f"[Downstream] 失败 (exit={ret.returncode}): {ret.stderr[:500]}")
            return None

        # 优先读 final_result.json
        result_path = os.path.join(ds_out, "final_result.json")
        if os.path.exists(result_path):
            metrics = json.load(open(result_path))
            ba = metrics.get("balanced_accuracy", {})
            fm = metrics.get("f1_macro", {})
            self.logger.info(f"[Downstream] 结果: bacc={ba.get('mean', 'N/A')}, f1={fm.get('mean', 'N/A')}")
            return metrics

        # 备选: 从 stdout 解析 MACHINE_READABLE_RESULT
        m = re.search(
            r'===MACHINE_READABLE_RESULT===\n(.*?)\n===END_MACHINE_READABLE_RESULT===',
            ret.stdout, re.DOTALL
        )
        if m:
            return json.loads(m.group(1))

        self.logger.warning("[Downstream] 未找到 final_result.json")
        return None

    def _run_baseline_downstream(self):
        """获取 baseline 下游指标：优先从 summary JSON 读 origin，否则跑下游脚本。"""
        if hasattr(self, '_baseline_ds_metrics') and self._baseline_ds_metrics is not None:
            return self._baseline_ds_metrics

        # ── 策略1：从 summary JSON 读 origin 条目 ──
        if self.downstream_data_json and os.path.exists(self.downstream_data_json):
            try:
                summary = json.load(open(self.downstream_data_json, "r", encoding="utf-8"))
                for key in summary:
                    if key.startswith("origin"):
                        self._baseline_ds_metrics = {
                            "downstream_bacc": summary[key].get("balanced_accuracy_mean", 0),
                            "downstream_f1": summary[key].get("f1_macro_mean", 0),
                        }
                        print(f"  [Baseline] 从 summary 读取: bacc={self._baseline_ds_metrics['downstream_bacc']:.4f}, "
                              f"f1={self._baseline_ds_metrics['downstream_f1']:.4f}")
                        return self._baseline_ds_metrics
            except Exception:
                pass

        # ── 策略2：跑下游脚本 ──
        if not self.downstream_script:
            self._baseline_ds_metrics = {}
            return {}

        baseline_dir = os.path.join(self.output_dir, "original", "downstream_output")
        result_path = os.path.join(baseline_dir, "final_result.json")
        if os.path.exists(result_path):
            metrics = json.load(open(result_path))
            ba = metrics.get("balanced_accuracy", {})
            fm = metrics.get("f1_macro", {})
            self._baseline_ds_metrics = {
                "downstream_bacc": ba.get("mean", 0),
                "downstream_f1": fm.get("mean", 0),
            }
            self.logger.info(f"[跳过] baseline downstream 已存在: {result_path}")
            return self._baseline_ds_metrics

        if not self.downstream_data_json:
            self._baseline_ds_metrics = {}
            return {}

        os.makedirs(baseline_dir, exist_ok=True)
        cmd = ["bash", self.downstream_script,
               os.path.abspath(self.downstream_data_json),
               os.path.abspath(baseline_dir)]
        self.logger.info(f"[Baseline] 执行: {' '.join(cmd)}")
        print(f"  [Baseline] 下游评估原始音频...", flush=True)

        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
        except subprocess.TimeoutExpired:
            self.logger.error("[Baseline] 超时")
            self._baseline_ds_metrics = {}
            return {}

        if ret.returncode != 0:
            self.logger.warning(f"[Baseline] 失败 (exit={ret.returncode}): {ret.stderr[:300]}")
            self._baseline_ds_metrics = {}
            return {}

        if os.path.exists(result_path):
            metrics = json.load(open(result_path))
            ba = metrics.get("balanced_accuracy", {})
            fm = metrics.get("f1_macro", {})
            self._baseline_ds_metrics = {
                "downstream_bacc": ba.get("mean", 0),
                "downstream_f1": fm.get("mean", 0),
            }
            print(f"  [Baseline] bacc={self._baseline_ds_metrics['downstream_bacc']:.4f}, "
                  f"f1={self._baseline_ds_metrics['downstream_f1']:.4f}")
            return self._baseline_ds_metrics

        self._baseline_ds_metrics = {}
        return {}

    def _run_downstream_for_current_vc(self):
        """一键：找匿名音频 → 生成 data.json → 跑下游 → 返回扁平指标。"""
        anon_dir = self.ctx.vc_audio_dir if self.ctx.vc_method_used else self.ctx.text_only_dir
        if not anon_dir or not _has_audio(anon_dir):
            self.logger.warning(f"[Downstream] No converted audio in {anon_dir}")
            return {}
        modified = self._prepare_downstream_data_json(anon_dir)
        if modified is None:
            return {}
        raw = self._run_downstream_script(modified, anon_dir)
        if raw:
            ba = raw.get("balanced_accuracy", {})
            fm = raw.get("f1_macro", {})
            return {
                "downstream_bacc": ba.get("mean", ba.get("per_fold", [0])[0]),
                "downstream_f1":   fm.get("mean", fm.get("per_fold", [0])[0]),
            }
        return {}

    # ── 计划参数注入 ─────────────────────────────────────────────────────────

    def _inject_extra_params(self, plan: dict):
        """将用户提供的额外参数（gt_dir / subject_map_path）注入评估步骤。"""
        for step in plan.get("steps", []):
            if step["tool"] == "run_evaluation":
                params = step.setdefault("params", {})
                if self.gt_dir and not params.get("gt_dir"):
                    params["gt_dir"] = self.gt_dir
                if self.subject_map_path and not params.get("subject_map_path"):
                    params["subject_map_path"] = self.subject_map_path

    # ── 任务信息提取（双层 memory 的入口）────────────────────────────────────

    def _classify_request(self, user_request: str) -> dict:
        """从用户 request 原文提取任务信息（task_type/语言/说话人数/下游任务/目标）。

        只提取原文明确写出的字段，提取不到 → 字段缺省 + warnings，禁止 LLM 猜测。
        LLM 失败时退回 fallback_classify（正则）。返回值同时喂给
        recall_for_prompt（选同领域经验）和 record_run（写 run 文件）。
        """
        if self.memory is None:
            return {}
        vocab = self.memory.task_types
        vocab_str = "\n".join(f"  - {k}: {v}" for k, v in sorted(vocab.items()))
        sys_prompt = (
            "你是任务信息提取器。从用户的语音匿名化请求原文中提取任务信息，只输出 JSON。"
        )
        user_prompt = (
            f"【用户请求原文】\n{user_request}\n\n"
            f"【下游任务领域词表（task_type 优先从中选；确属新领域时可提出新的小写英文"
            f" snake_case 名称并给 task_type_desc，都不符合才用 other）】\n{vocab_str}\n\n"
            "输出 JSON（不要输出其他内容）：\n"
            '{"task_type": "词表之一或新的snake_case", '
            '"task_type_desc": "仅当提出新 task_type 时给中文一句话描述，否则null", '
            '"language": "语言或null", '
            '"n_speakers": 说话人数量整数或null, "downstream_task": "下游任务描述或null", '
            '"goal": "用户对匿名化的目标要求或null"}\n'
            "严格要求：只从请求原文中提取，原文没写的字段输出 null，禁止猜测或补全。"
        )

        def _validate(d):
            if not isinstance(d, dict):
                return "输出非 dict"
            tt = d.get('task_type')
            if tt is not None and tt != 'other' and tt not in vocab:
                if not is_valid_new_task_type(tt):
                    return f"task_type「{tt}」不在词表且不是合法的小写 snake_case 新类型名"
            ns = d.get('n_speakers')
            if ns is not None and not isinstance(ns, int):
                return "n_speakers 必须是整数或 null"
            return None

        meta = None
        try:
            out = self._call_llm_with_retry(sys_prompt, user_prompt, validator=_validate)
            if not out.get('error'):
                tt = out.get('task_type') or None
                if tt and tt not in vocab and tt != 'other':
                    # agent 自主扩展：新领域自动入表（task_types.json），下次运行直接复用
                    self.memory.register_task_type(tt, out.get('task_type_desc') or None)
                meta = {
                    'task_type': tt,
                    'language': out.get('language') or None,
                    'n_speakers': out.get('n_speakers'),
                    'downstream_task': out.get('downstream_task') or None,
                    'goal': out.get('goal') or None,
                    'source': 'llm',
                    'warnings': [],
                }
        except Exception as e:
            self.logger.warning(f"[Memory] 任务信息 LLM 提取失败: {e}")

        if meta is None:
            meta = fallback_classify(user_request)
            self.logger.info("[Memory] 使用规则版任务信息提取（fallback_classify）")

        # 缺字段只提示，不猜测；dataset 记录在 run 文件里另存
        missing = [k for k in ('task_type', 'language', 'n_speakers', 'downstream_task')
                   if meta.get(k) in (None, '')]
        if missing:
            msg = (f"任务信息缺少字段 {missing}（用户 request 原文未提供，"
                   f"建议在 -r 中写明：语言/说话人数量/下游任务）")
            meta.setdefault('warnings', []).append(msg)
            self.logger.warning(f"[Memory] {msg}")
            print(f"[WARNING] {msg}")
        self.logger.info(f"[Memory] task_meta = {json.dumps(meta, ensure_ascii=False)}")
        return meta

    # ── 初始计划 ─────────────────────────────────────────────────────────────

    def _generate_initial_plan(self, user_request: str) -> dict:
        plan_prompt = INITIAL_PLAN_PROMPT

        user_msg = (
            f"【用户请求】\n{user_request}\n\n"
        )

        # ── 注入跨运行双层 memory：Layer-1 任务级经验 + Layer-2 同类任务运行细节 ──
        if self.memory:
            mem_block = self.memory.recall_for_prompt(
                task_meta=self.task_meta, exclude_dataset=self._memory_exclude, focused=self.focused_memory)
            if mem_block:
                user_msg += (
                    f"{mem_block}\n\n"
                    "制定计划时可参考以上经验选择初始 VC 方法：与当前任务同领域（task_type）"
                    "的任务经验参考价值最高（可直接沿用已验证的最终方案、避开已证实的失败方法），"
                    "其他领域仅供参考。\n\n"
                )

        user_msg += "如果用户要求使用多种声线匿名方法，这不是你的任务，任选一种作为计划执行即可。请输出 JSON 执行计划：\n"

        self.logger.info("向 LLM 请求初始计划...")
        plan = self._call_llm_with_retry(plan_prompt, user_msg, validator=_validate_plan)
        self.logger.debug(f"[解析结果] plan = {json.dumps(plan, ensure_ascii=False, indent=2)}")

        err = _validate_plan(plan)
        if err:
            self.logger.error(f"初始计划解析失败: {err}")
            return None

        for i, s in enumerate(plan.get("steps", [])):
            self.logger.debug(f"  计划步骤 {i+1}: {s['tool']} | params={json.dumps(s.get('params', {}), ensure_ascii=False)}")
        return plan

    # ── 步骤执行 ─────────────────────────────────────────────────────────────

    def _execute_step(self, step: dict):
        tool_name = step.get("tool", "")
        params = step.get("params", {})
        tool = TOOL_BY_NAME.get(tool_name)
        if tool is None:
            self.logger.error(f"[工具调用] 未知工具: {tool_name}")
            return False, {"error": f"未知工具: {tool_name}"}, 0

        accepted_params = {
            name for name, parameter in inspect.signature(tool.handler).parameters.items()
            if name != "ctx" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
            )
        }
        dropped_params = [name for name in params if name not in accepted_params]
        if dropped_params:
            self.logger.warning(f"[工具参数] {tool_name} 忽略未声明的参数: {', '.join(dropped_params)}")
        params = {name: value for name, value in params.items() if name in accepted_params}

        self.logger.debug(f"[工具调用] {tool_name} | params={json.dumps(params, ensure_ascii=False)}")
        t0 = time.time()
        try:
            result = tool.handler(self.ctx, **params)
        except Exception as e:
            self.logger.error(f"[工具异常] {tool_name} 抛出异常: {e}", exc_info=True)
            result = {"success": False, "error": str(e)}
        elapsed = time.time() - t0

        ok = result.get("success", False)
        if ok:
            self.logger.info(f"[OK] {tool_name} ({elapsed:.1f}s)")
            self.logger.debug(f"[工具结果] {tool_name} -> {json.dumps(result, ensure_ascii=False)}")
        else:
            self.logger.error(f"[FAIL] {tool_name}: {result.get('error', '')}")
            self.logger.debug(f"[工具结果] {tool_name} -> {json.dumps(result, ensure_ascii=False)}")
        return ok, result, elapsed

    def _execute_plan(self, plan: dict) -> dict:
        steps = plan.get("steps", [])
        self.logger.debug(f"[执行计划] 共 {len(steps)} 步: {[s['tool'] for s in steps]}")
        results = []
        eval_metrics = {}

        has_mask = any(s["tool"] == "run_mask" for s in steps)
        for step in steps:
            if step["tool"] == "run_voice_conversion":
                params = step["params"]
                has_mask = has_mask or params.get("with_mask", False)
                self.ctx.select_vc(params["method"], params.get("vc_params"), has_mask)
                mode = f"text_{self.ctx.vc_name}" if has_mask else self.ctx.vc_name
                cached = _cached_metrics(self.output_dir, mode)
                if cached and _has_audio(self.ctx.vc_audio_dir):
                    self.logger.info(f"[Cache] Reusing evaluate_{mode}.csv")
                    return {"success": True, "steps_results": [], "eval_metrics": cached}
                break

        # ── 注入 with_mask 到评估步骤 ──
        for step in steps:
            if step["tool"] in ("run_voice_conversion", "run_evaluation"):
                step.setdefault("params", {})["with_mask"] = has_mask or step.get("params", {}).get("with_mask", False)

        for i, step in enumerate(steps):
            tool_name = step["tool"]
            if step.get("_skipped"):
                self.logger.debug(f"[执行计划] 跳过步骤 {i+1}: {tool_name} (已标记跳过)")
                continue

            print(f"  [{i+1}/{len(steps)}] {tool_name} ...", flush=True)
            ok, result, elapsed = self._execute_step(step)
            results.append({"step": tool_name, "ok": ok, "elapsed": elapsed, "result": result})

            if ok:
                print(f"  [{i+1}/{len(steps)}] {tool_name}  完成 ({elapsed:.1f}s)")
            else:
                print(f"  [{i+1}/{len(steps)}] {tool_name}  失败: {result.get('error', '')}")
                for j in range(i + 1, len(steps)):
                    later = steps[j]
                    later_tool = TOOL_BY_NAME.get(later["tool"])
                    masked_conversion = (
                        tool_name in ("run_transcription", "run_mask")
                        and later["tool"] == "run_voice_conversion"
                        and later.get("params", {}).get("with_mask")
                    )
                    evaluation = later["tool"] == "run_evaluation" and tool_name in ("run_mask", "run_voice_conversion")
                    if masked_conversion or evaluation or (later_tool and tool_name in later_tool.dependencies):
                        self.logger.debug(f"[依赖跳过] 步骤 {j+1} {later['tool']} 依赖失败的 {tool_name}，跳过")
                        print(f"  → 跳过 [{j+1}/{len(steps)}] {later['tool']}")
                        results.append({"step": later["tool"], "ok": False, "elapsed": 0,
                                        "result": {"error": f"依赖步骤 {tool_name} 失败"}})
                        steps[j]["_skipped"] = True

            if ok and tool_name == "run_evaluation":
                eval_metrics = result.get("mean_results", {})
                self.logger.debug(f"[执行计划] 评估指标: {json.dumps(eval_metrics, ensure_ascii=False)}")

        all_ok = all(r["ok"] for r in results)
        return {"success": all_ok, "steps_results": results, "eval_metrics": eval_metrics}

    # ── 反思 ─────────────────────────────────────────────────────────────────

    def _reflect(self, user_request: str, current_metrics: dict, history: list,
                 baseline_ds: dict = None) -> dict:
        # Build system prompt: base + downstream descriptions if active
        sys_prompt = REFLECTION_SYSTEM_PROMPT
        has_ds = any(k.startswith("downstream_") for k in current_metrics)
        if has_ds:
            extra = chr(10).join(f'- {k}: {v}' for k, v in DOWNSTREAM_METRIC_DESCRIPTIONS.items())
            sys_prompt += f"\n\n【下游任务指标（以下指标来自匿名后音频的下游分类任务）】\n{extra}"

        # ── 读取原始音频基线指标 ──
        origin_metrics = _cached_metrics(self.output_dir, 'origin')
        origin_str = f"\n【原始音频基线指标】\n{_metrics_to_str(origin_metrics)}" if origin_metrics else ""

        # ── baseline 下游指标 ──
        baseline_ds_str = ""
        if baseline_ds:
            baseline_ds_str = (
                f"\n【原始音频下游基线指标】\n"
                f"  downstream_bacc = {baseline_ds.get('downstream_bacc', 'N/A')} (原始音频的下游分类 Balanced Accuracy)\n"
                f"  downstream_f1 = {baseline_ds.get('downstream_f1', 'N/A')} (原始音频的下游分类 Macro F1)\n"
                f"  匿名后指标不应显著低于以上基线值。"
            )

        # ── 跨运行双层 memory 经验（同领域任务优先） ──
        mem_str = ""
        if self.memory:
            mem_block = self.memory.recall_for_prompt(
                task_meta=self.task_meta, exclude_dataset=self._memory_exclude, focused=self.focused_memory)
            if mem_block:
                mem_str = f"\n{mem_block}"

        reflection_msg = f"""【用户请求】
{user_request}

【当前目录】
{_build_context_info(self.ctx)}
{origin_str}
{baseline_ds_str}
{mem_str}

【当前评估指标】
{_metrics_to_str(current_metrics)}

【历史尝试】
{_history_to_str(history)}

请根据用户目标和当前指标，决定是否更换 VC 方法或调整参数。输出 JSON:"""

        self.logger.info("向 LLM 请求反思决策...")

        def _validate_decision(d):
            if not isinstance(d, dict):
                return "Expected a JSON decision"
            for key in ("new_vc_method", "best_vc_method"):
                if key in d:
                    if isinstance(d[key], str):
                        d[key] = canonical_method(d[key])
                    problem = choice_error(key, d[key], VC_METHODS)
                    if problem:
                        return problem
            if not isinstance(d.get("done"), bool):
                return "done must be true or false"
            if d["done"] and "best_vc_method" not in d:
                return "best_vc_params requires best_vc_method" if "best_vc_params" in d else None
            method = d.get("best_vc_method" if d["done"] else "new_vc_method")
            problem = choice_error("VC method", method, VC_METHODS)
            if problem:
                return problem
            key = "best_vc_params" if d["done"] else "vc_params"
            if key in d:
                try:
                    d[key] = resolve_vc_params(method, d[key])
                except ValueError as exc:
                    return str(exc)
            if d["done"] and not any(
                h["vc_method"] == method and h.get("metrics") and
                (key not in d or h.get("vc_params") == d[key]) for h in history
            ):
                return "Select an evaluated method and parameter combination from history"
            return None

        decision = self._call_llm_with_retry(
            sys_prompt, reflection_msg, validator=_validate_decision,
        )
        self.logger.debug(f"[反思解析] decision = {json.dumps(decision, ensure_ascii=False, indent=2)}")

        if decision.get("error"):
            self.logger.warning(f"反思解析失败: {decision}")
            return {"done": True, "reason": "反思解析失败，默认完成", "summary": ""}
        return decision

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def run(self, user_request: str, max_reflections: int = 3) -> dict:
        """
        CASA 主循环：初始计划 → 执行 → 反思 → 调整 → 重试。
        """
        self.logger.debug(f"[启动] input_dir={self.input_dir}")
        self.logger.debug(f"[启动] output_dir={self.output_dir}")
        self.logger.debug(f"[启动] middle_dir={self.middle_dir}")
        self.logger.debug(f"[启动] llm_model={self.llm_model}")
        self.logger.debug(f"[启动] mode={self.mode}")
        self.logger.debug(f"[启动] max_reflections={max_reflections}")
        self.logger.debug(f"[用户输入] request={user_request}")

        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"最大反思次数: {max_reflections}")
        print(f"LLM 模型: {self.llm_model}")

        # ── 1. 生成初始计划 ─────────────────────────────────────────────────
        print(f"\n用户请求: {user_request}\n")
        print("正在分析请求...", flush=True)

        # ── 任务信息提取（双层 memory：先粗后细召回的前提）──
        if self.memory:
            self.task_meta = self._classify_request(user_request)

        plan = self._generate_initial_plan(user_request)
        if plan is None:
            self.logger.error("[主循环] 初始计划生成失败，退出")
            print("错误: 无法生成初始计划")
            return None
        self._inject_extra_params(plan)

        language = plan.get("language", "中文")
        reasoning = plan.get("reasoning", "")
        steps = plan["steps"]
        with_mask = any(step["tool"] == "run_mask" for step in steps)
        self.logger.info(f"[Agent]: {reasoning}")
        self.logger.info(f"[计划] steps={[s['tool'] for s in steps]}")

        initial_vc = None
        initial_params = {}
        for s in steps:
            if s["tool"] == "run_voice_conversion":
                initial_vc = s["params"].get("method", "NeuralVC")
                initial_params = resolve_vc_params(initial_vc, s["params"].get("vc_params"))
                with_mask = with_mask or s["params"].get("with_mask", False)
                break

        print(f"语言: {language}")
        print(f"分析: {reasoning}")
        print(f"初始计划: {' → '.join(s['tool'] for s in steps)}")
        if initial_vc:
            print(f"初始 VC 方法: {initial_vc}")

        # ── 2. 执行初始计划 ─────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print("执行初始计划...")
        print(f"{'='*55}")

        exec_result = self._execute_plan(plan)
        current_metrics = exec_result.get("eval_metrics", {})
        self.logger.debug(f"[初始执行] success={exec_result['success']}, metrics={json.dumps(current_metrics, ensure_ascii=False)}")

        if not exec_result["success"]:
            self.logger.error("[主循环] 初始计划执行失败，无法进行反思")
            print("初始计划执行失败，无法进行反思")
            return {"success": False, "attempts": 1, "history": []}

        # ── baseline 下游评估（原始音频）──
        baseline_ds = {}
        if self.downstream_data_json and self.downstream_script:
            print(f"\n{'='*55}")
            print("Baseline 下游评估（原始音频）...")
            baseline_ds = self._run_baseline_downstream()
            if baseline_ds:
                print(f"  Baseline: bacc={baseline_ds.get('downstream_bacc', 'N/A')}, "
                      f"f1={baseline_ds.get('downstream_f1', 'N/A')}")

        # ── 下游任务（初始）──
        if self.downstream_data_json and self.downstream_script:
            print(f"\n{'='*55}")
            print("下游任务（初始评估）...")
            ds_metrics = self._run_downstream_for_current_vc()
            if ds_metrics:
                current_metrics.update(ds_metrics)
                print(f"  下游: bacc={ds_metrics.get('downstream_bacc', 'N/A')}, "
                      f"f1={ds_metrics.get('downstream_f1', 'N/A')}")

        # ── 3. CASA 循环 ───────────────────────────────────────────────
        history = [{"vc_method": initial_vc, "vc_params": initial_params, "metrics": current_metrics, "reason": reasoning}]
        final_decision = None  # LLM 的最终裁决（done=true 的 best_vc_method / 回退分支选择）

        if not current_metrics:
            self.logger.info("No evaluation metrics available for reflection.")
        for reflection_round in range(max_reflections if current_metrics and initial_vc else 0):
            self.logger.debug(f"[反思循环] 第 {reflection_round + 1}/{max_reflections} 轮")
            print(f"\n{'='*55}")
            print(f"反思 #{reflection_round + 1}")
            print(f"{'='*55}")
            print(f"当前指标:\n{_metrics_to_str(current_metrics)}")

            decision = self._reflect(user_request, current_metrics, history, baseline_ds)
            self.logger.debug(f"[反思决策] round={reflection_round+1}, decision={json.dumps(decision, ensure_ascii=False)}")
            self.logger.info(f"[反思结果] {json.dumps(decision, ensure_ascii=False)}")
            print(f"决策: {json.dumps(decision, ensure_ascii=False)}")

            reason = decision.get("reason", "")

            if decision.get("done", True):
                reason = decision.get("reason", "")
                summary = decision.get("summary", "")
                final_decision = decision
                print(f"\n✅ 完成: {reason}")
                if summary:
                    print(f"摘要: {summary}")
                break

            new_vc = decision.get("new_vc_method")
            if new_vc not in VC_DEFAULTS:
                self.logger.warning(f"[反思] LLM 返回无效 VC 方法: {new_vc}，终止反思")
                print(f"LLM 返回了无效的 VC 方法: {new_vc}，终止反思")
                break

            new_params = resolve_vc_params(new_vc, decision.get("vc_params"))
            if any(h["vc_method"] == new_vc and h["vc_params"] == new_params and h.get("metrics") for h in history):
                final_decision = {"best_vc_method": new_vc, "best_vc_params": new_params}
                self.logger.info(f"[反思] 选择已评估的结果: {new_vc} {new_params}")
                break

            self.ctx.select_vc(new_vc, new_params, with_mask)
            self.logger.debug(f"[反思] VC 方法切换: {history[-1]['vc_method']} → {new_vc}")

            print(f"\n更换 VC 方法: {history[-1]['vc_method']} → {new_vc}")
            print(f"{'='*55}")

            mode = f"text_{self.ctx.vc_name}" if with_mask else self.ctx.vc_name
            cached = _cached_metrics(self.output_dir, mode)
            if cached and _has_audio(self.ctx.vc_audio_dir):
                current_metrics = cached
                self.logger.info(f"[Cache] Reusing evaluate_{mode}.csv")
                history.append({"vc_method": new_vc, "vc_params": new_params, "metrics": current_metrics, "reason": reason})
                # 下游也要读已有结果
                if self.downstream_data_json and self.downstream_script:
                    print(f"  下游任务 ...", flush=True)
                    ds_metrics = self._run_downstream_for_current_vc()
                    if ds_metrics:
                        current_metrics.update(ds_metrics)
                        print(f"  下游任务  完成: bacc={ds_metrics.get('downstream_bacc', 'N/A')}, "
                              f"f1={ds_metrics.get('downstream_f1', 'N/A')}")
                continue

            print(f"执行: vc({new_vc}) → eval")
            vc_step = {"tool": "run_voice_conversion", "params": {"method": new_vc, "vc_params": new_params, "language": language, "with_mask": with_mask}}
            eval_params = {"language": language, "with_mask": with_mask}
            if self.gt_dir:
                eval_params["gt_dir"] = self.gt_dir
            if self.subject_map_path:
                eval_params["subject_map_path"] = self.subject_map_path
            eval_step = {"tool": "run_evaluation", "params": eval_params}
            self.logger.debug(f"[重试] vc_step={json.dumps(vc_step, ensure_ascii=False)}")
            self.logger.debug(f"[重试] eval_step={json.dumps(eval_step, ensure_ascii=False)}")

            print(f"  [1/2] run_voice_conversion (method={new_vc}) ...", flush=True)
            ok, result, elapsed = self._execute_step(vc_step)
            if ok:
                print(f"  [1/2] run_voice_conversion  完成 ({elapsed:.1f}s)")
            else:
                print(f"  [1/2] run_voice_conversion  失败: {result.get('error', '')}")
                history.append({"vc_method": new_vc, "vc_params": new_params, "metrics": {}, "reason": reason})
                continue

            print(f"  [2/2] run_evaluation ...", flush=True)
            ok, result, elapsed = self._execute_step(eval_step)
            if ok:
                print(f"  [2/2] run_evaluation  完成 ({elapsed:.1f}s)")
            else:
                self.logger.error(f"[重试] 评估失败: {result.get('error', '')}")
                print(f"  [2/2] run_evaluation  失败: {result.get('error', '')}")
                history.append({"vc_method": new_vc, "vc_params": new_params, "metrics": {}, "reason": reason})
                continue

            current_metrics = result.get("mean_results", {})
            history.append({"vc_method": new_vc, "vc_params": new_params, "metrics": current_metrics, "reason": reason})
            self.ctx.vc_method_used = new_vc
            self.logger.debug(f"[重试] 新指标: {json.dumps(current_metrics, ensure_ascii=False)}")

            # ── 下游任务（每次新 VC 后）──
            if self.downstream_data_json and self.downstream_script:
                print(f"  下游任务 ...", flush=True)
                ds_metrics = self._run_downstream_for_current_vc()
                if ds_metrics:
                    current_metrics.update(ds_metrics)
                    print(f"  下游任务  完成: bacc={ds_metrics.get('downstream_bacc', 'N/A')}, "
                          f"f1={ds_metrics.get('downstream_f1', 'N/A')}")
        else:
            if current_metrics and max_reflections:
                self.logger.info(f"[主循环] 达到最大反思次数 ({max_reflections})")

        # ── 4. 最终汇总 ─────────────────────────────────────────────────────
        selected = final_attempt_from(final_decision, history)
        final_vc = selected["vc_method"]
        current_metrics = selected["metrics"]
        self.logger.info(f"[最终选择] {final_vc}")
        if final_vc:
            self.ctx.select_vc(final_vc, selected["vc_params"], with_mask)
        self.logger.info(f"[最终汇总] 共 {len(history)} 次尝试")
        for i, h in enumerate(history):
            self.logger.info(f"  第{i+1}次: VC={h['vc_method']} | metrics={json.dumps(h.get('metrics', {}), ensure_ascii=False)}")

        # print(f"\n{'='*55}")
        # print(f"执行完成 | 共 {len(history)} 次尝试")
        # print(f"{'='*55}")
        # for i, h in enumerate(history):
        #     m = h.get("metrics", {})
        #     m_str = ", ".join(f"{k}={v}" for k, v in m.items()) if m else "无指标"
        #     print(f"  第{i+1}次: VC={h['vc_method']} | {m_str}")

        # ── 写结果到 JSON 文件 ──
        result = {
            "success": True,
            "final_vc_method": final_vc,
            "final_vc_params": selected["vc_params"],
            "final_audio_dir": self.ctx.vc_audio_dir if final_vc else self.ctx.text_only_dir,
            "attempts": len(history),
            "history": history,
            "final_metrics": current_metrics,
        }
        result_path = os.path.join(self.output_dir, "agent_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[结果] 已写入: {result_path}")

        # ── 写入跨运行双层 memory（先 Layer-2 run，再 LLM 总结 Layer-1；失败只 warning） ──
        if self.memory and current_metrics:
            try:
                self.memory.record_run(
                    input_dir=self.input_dir,
                    language=language,
                    request=user_request,
                    output_dir=self.output_dir,
                    history=history,
                    final_vc_method=final_vc,
                    final_vc_params=selected["vc_params"],
                    success=bool(result.get("success")),
                    task_meta=self.task_meta,
                    baseline_bacc=(baseline_ds or {}).get('downstream_bacc'),
                    final_metrics=current_metrics,
                    artifacts={
                        'agent_result': result_path,
                        'audio_dir': result['final_audio_dir'],
                        'log': self.logger_file,
                    },
                    llm_call=lambda s, u: json.dumps(
                        self._call_llm_with_retry(s, u, validator=_validate_experience),
                        ensure_ascii=False),
                )
            except Exception as e:
                self.logger.warning(f"[Memory] record_run 失败（不影响主流程）: {e}")

        return result


# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description='CASA — 基于评估反馈自动调整 VC 方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python casa_downstream.py -i /path/to/audio -r "匿名化中文音频，SNR尽量高，不需要内容匿名"
  python casa_downstream.py -i /path/to/audio -r "匿名化，不要改变韵律" --max_reflections 5
  python casa_downstream.py -i /path/to/audio -r "..." --llm Qwen2.5-32B-Instruct
        """)
    parser.add_argument('--input_dir', '-i', required=True, help='输入音频目录')
    parser.add_argument('--request', '-r', required=True, help='自然语言任务描述')
    parser.add_argument('--llm', default=None, help='模型名称（默认自动推断）')
    parser.add_argument('--open_source', action=argparse.BooleanOptionalAction, default=True,
                        help='使用本地模型（默认）。--no-open_source 使用 OpenAI 兼容 API（读取环境变量 OPENAI_API_KEY / BASE_URL）')
    parser.add_argument('--output_dir', '-o', default=None, help='输出目录')
    parser.add_argument('--middle_dir', default=None, help='中间结果目录')
    parser.add_argument('--mode', choices=['DEBUG', 'INFO'], default='INFO', help='日志级别')
    parser.add_argument('--max_reflections', type=int, default=7, help='最大反思次数（默认7）')
    parser.add_argument('--temperature', type=float, default=0.01, help='LLM 温度（默认0.01，重试时自动降为0）')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='LLM 最大生成 token 数（默认2048）')
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help='LLM 重复惩罚（默认1.05）')
    parser.add_argument('--downstream_data_json', type=str, default=None,
                        help='下游任务原始 data.json 路径（启用下游反思路径）')
    parser.add_argument('--downstream_script', type=str, default=None,
                        help='下游任务 bash 脚本路径（与 --downstream_data_json 配对使用）')
    parser.add_argument('--memory_exclude_dataset', type=str, default=None,
                        help='recall 时排除该数据集的经验/运行记录（leave-one-out 消融实验用）；'
                             '注意 record_run 仍会正常写回（写到 --memory_path 指定的目录，'
                             '实验时建议指向副本目录以免污染主 memory）')
    parser.add_argument('--memory_path', type=str, default=None,
                        help='跨运行双层 memory 根目录（默认 ./memory；含 experiences.json 与 runs/）')
    parser.add_argument('--no_memory', action='store_true',
                        help='禁用跨运行 memory（行为与无 memory 完全一致）')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    if (args.downstream_data_json or args.downstream_script) and not (
        args.downstream_data_json and args.downstream_script
    ):
        print("错误: --downstream_data_json 和 --downstream_script 必须同时提供")
        sys.exit(1)

    if args.downstream_data_json and not os.path.exists(args.downstream_data_json):
        print(f"错误: data.json 不存在: {args.downstream_data_json}")
        sys.exit(1)

    if args.downstream_script and not os.path.exists(args.downstream_script):
        print(f"错误: 脚本不存在: {args.downstream_script}")
        sys.exit(1)

    agent = CASA(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        middle_dir=args.middle_dir,
        llm_model=args.llm,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        open_source=args.open_source,
        memory_path=args.memory_path,
        use_memory=not args.no_memory,
        memory_exclude_dataset=args.memory_exclude_dataset,
    )
    agent.downstream_data_json = args.downstream_data_json
    agent.downstream_script = args.downstream_script

    if agent.downstream_data_json:
        print(f"下游 data.json: {agent.downstream_data_json}")
        print(f"下游脚本: {agent.downstream_script}")
    result = agent.run(args.request, max_reflections=args.max_reflections)

    if result and result.get("success"):
        history = result.get("history", [])
        final_vc = result.get("final_vc_method")
        if history:
            print(f"\n最终选择: {final_vc}")
        else:
            print("\n执行完成（无需反思）。")
    else:
        print("\n执行失败，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
