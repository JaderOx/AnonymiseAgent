"""
reflexion_agent.py — Reflexion 架构的语音匿名 Agent

执行 → 评估 → 反思 → 调整 → 重试，直到满足用户目标。根据评估指标动态调整 VC 方法。

完全独立文件，不依赖 plan_and_execute.py。
依赖：agent.py（pipeline 函数）、utils/（logger、内存隔离）

用法示例：
    python reflexion_agent.py -i /path/to/audio -r "匿名化，不要改变韵律，SNR尽量高，不需要内容匿名"
    python reflexion_agent.py -i /path/to/audio -r "..." --max_reflections 3
"""

import os
import re
import json
import time
import argparse
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent import (
    trans, mask, vc,
    _has_audio, _has_json,
    _ASR_CONDA_ENV, _VC_CONDA_ENV,
    SUPPORTED_FORMATS, NO_WER_LANGUAGE,
)
from utils.logger_example import setup_logger
from utils.manage_module_memory import run_in_isolation

# 可用的 VC 方法
VC_METHODS = ["mcadams", "formant", "pitch", "combined", "seedvc", "fishaudio_tts", "cosyvoice_tts"]

# 指标含义（给 LLM 参考）
METRIC_DESCRIPTIONS = {
    "snr_db": "信噪比(dB)，越高越好，>30为优秀",
    "mos": "语音自然度(1-5)，越高越好，>2为良好",
    "cer": "字符错误率，越低越好，<10% 为良好 <30% 可以接受",
    "emo": "情感相似度(0-1)，越高越好，>0.85为良好",
    "pit_l1": "韵律L1距离，越低越好，<15为良好（衡量是否改变韵律）",
    "pit_pcc": "韵律相关系数(0-1)，越高越好，>0.9为良好（衡量是否改变韵律）",
    "eer": "等错误率(%)，越接近50%越好（匿名效果）",
}


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
        self.spk_audio_origin = os.path.join(M, 'spk_audio', 'origin')
        self.age_gender_model = os.path.join(
            os.environ.get('TORCH_HOME', './models'), 'w2v2-L-robust-6-age-gender'
        )

        self.asr_model_used = None
        self.vc_method_used = None


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


def _handle_transcription(ctx, language, asr_model="", hotwords=None, spk_num=0,
                          min_speakers=None, max_speakers=None):
    hotwords = hotwords or []
    spk_num = spk_num or None
    min_speakers = min_speakers or None
    max_speakers = max_speakers or None
    ctx.logger.debug(f"[handler] _handle_transcription: language={language}, asr_model={asr_model}, hotwords={hotwords}, spk_num={spk_num}, min_speakers={min_speakers}, max_speakers={max_speakers}")
    if not asr_model:
        try:
            from configs.config_loader import get_asr_config
            asr_model = get_asr_config(language)['model']
        except Exception:
            asr_model = 'paraformer' if '中文' in language else 'FunASRNano'
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
        language, custom_prompt, llm_model, group_num, ctx.mode,
        detect_method,
    )
    noise_method = 'formant' if mask_method == 'noise' else 'mute'
    text_only_dir = os.path.join(ctx.output_dir, 'text_only')
    from apply_mask import apply_mask
    apply_mask(wav_dir=ctx.input_dir, mask_dir=ctx.mask_dir,
               output_dir=text_only_dir, logger=ctx.logger,
               noise_method=noise_method)
    return {"success": True, "text_only_dir": text_only_dir, "mask_dir": ctx.mask_dir}


def _handle_voice_conversion(ctx, method, with_mask=False, num_workers=1,
                             whole_audio=False, language="中文"):
    ctx.logger.debug(f"[handler] _handle_voice_conversion: method={method}, with_mask={with_mask}, language={language}")
    ctx.vc_method_used = method
    vc_output_dir = (os.path.join(ctx.output_dir, f'text+{method}') if with_mask
                     else os.path.join(ctx.output_dir, method))
    spk_vc_dir = (os.path.join(ctx.middle_dir, 'spk_audio', f'text_{method}') if with_mask
                  else os.path.join(ctx.middle_dir, 'spk_audio', f'{method}_only'))
    if with_mask and not _has_json(ctx.mask_dir):
        return {"success": False,
                "error": "with_mask=True 但 mask 结果不存在，请先执行内容匿名步骤"}
    trans_dir_for_vc = ctx.trans_origin_dir if _has_json(ctx.trans_origin_dir) else None
    ctx.logger.info(f"[Agent] 声线转换: method={method} with_mask={with_mask}")
    run_in_isolation(
        vc,
        ctx.input_dir, vc_output_dir, method,
        trans_dir_for_vc, ctx.logger_file, ctx.mode,
        num_workers,
        ctx.mask_dir if with_mask else None,
        'noise', language, whole_audio,
        conda_env=_VC_CONDA_ENV.get(method),
    )
    from utils.metrics.spk_audio_extractor import extract_dir as _ext_spk_pair
    _ext_spk_pair(ctx.input_dir, vc_output_dir, ctx.trans_origin_dir,
                  ctx.spk_audio_origin, spk_vc_dir, ctx.logger)
    return {"success": True, "vc_output_dir": vc_output_dir, "method": method}


def _handle_evaluation(ctx, language, gt_dir="", subject_map_path=""):
    ctx.logger.debug(f"[handler] _handle_evaluation: language={language}, gt_dir={gt_dir}, subject_map_path={subject_map_path}")
    method = ctx.vc_method_used or 'seedvc'
    text_vc_dir = os.path.join(ctx.output_dir, f'text+{method}')
    vc_only_dir = os.path.join(ctx.output_dir, method)
    text_only   = os.path.join(ctx.output_dir, 'text_only')

    if _has_audio(text_vc_dir):
        eval_dir, trans_vc, mode_str = text_vc_dir, os.path.join(ctx.middle_dir, f'trans_text+{method}'), f'text_{method}'
    elif _has_audio(vc_only_dir):
        eval_dir, trans_vc, mode_str = vc_only_dir, os.path.join(ctx.middle_dir, f'trans_{method}'), method
    elif _has_audio(text_only):
        eval_dir, trans_vc, mode_str = text_only, os.path.join(ctx.middle_dir, 'trans_text'), 'text'
    else:
        return {"success": False, "error": "找不到匿名后音频，请先执行 mask 或 vc 步骤"}

    asr_model = ctx.asr_model_used or 'paraformer'
    ctx.logger.info(f"[Agent] 评估: eval_dir={eval_dir}")
    from evaluate import evaluate_all
    df = evaluate_all(
        input_dir_vc=eval_dir, trans_dir_vc=trans_vc,
        input_dir_origin=ctx.input_dir, trans_dir_origin=ctx.trans_origin_dir,
        middle_dir=ctx.middle_dir, asr_model=asr_model,
        hotwords=[], language=language, output_dir=ctx.output_dir,
        logger=ctx.logger, logger_file=ctx.logger_file,
        _trans_env=_ASR_CONDA_ENV.get(asr_model),
        gt_dir=gt_dir or None, subject_map_path=subject_map_path or None,
        mode=mode_str, wer=None if language in NO_WER_LANGUAGE else True,
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
            "language": "string (必填且必须为汉字, 如 中文/英文/日文/西班牙语)",
            "asr_model": "string (可选, 留空自动选择)",
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
            "method": "string (必填, mcadams/formant/pitch/combined/seedvc/fishaudio_tts/cosyvoice_tts)",
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

def _extract_json_braces(text: str) -> Optional[str]:
    """从文本中提取最外层的 {...} JSON 字符串。"""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_json_output(text: str) -> dict:
    """从 LLM 输出中提取 JSON。多策略。"""
    # 策略1: ```json ... ```
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 策略2: 括号计数匹配
    json_str = _extract_json_braces(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 策略3: 在关键字段附近搜索
    for key in ('"steps"', '"language"', '"tool"', '"done"'):
        idx = text.find(key)
        if idx != -1:
            search_start = max(0, idx - 5)
            brace_start = text.rfind('{', search_start, idx)
            if brace_start != -1:
                json_str = _extract_json_braces(text[brace_start:])
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

    return {"error": True, "raw": text[:500]}


def _validate_plan(plan: dict) -> Optional[str]:
    """验证 plan 结构。"""
    if plan.get("error"):
        return f"无法从 LLM 输出中提取执行计划，原始输出:\n{plan.get('raw', '')}"
    if "steps" not in plan:
        return "计划缺少 steps 字段"
    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        return "steps 为空或格式不正确"
    for i, s in enumerate(plan["steps"]):
        name = s.get("tool", "")
        if name not in TOOL_BY_NAME:
            return f"步骤 {i+1} 包含未知工具: {name}"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 反思 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM_PROMPT = f"""你是一个语音匿名处理专家。根据评估指标和用户目标，决定是否需要更换声线转换(VC)方法。

【可用 VC 方法】
{chr(10).join(f'- {m}' for m in VC_METHODS)}

【指标含义】
{chr(10).join(f'- {k}: {v}' for k, v in METRIC_DESCRIPTIONS.items())}

【决策规则】
先看用户请求的类型：
1. 如果用户明确指定了 VC 方法（如"用seedvc"、"做seedvc"），且该方法已在历史记录中成功执行 → 直接 done=true，不要换方法。
2. 如果用户描述的是抽象目标（如"SNR尽量高"、"不要改变韵律"），则根据指标判断。当前方法能满足目标 → done=true；不能满足 → 换一个不同的方法重试。

不要因为看到历史记录中有某个方法就认为它失败了。历史记录只是记录了已尝试过的方法和对应指标。

【输出格式】只输出 JSON，以 {{ 开头、以 }} 结尾。

满意时:
{{"done":true,"reason":"简短说明","summary":"最终结果摘要"}}

不满意时:
{{"done":false,"reason":"简短说明","new_vc_method":"方法名"}}

示例:
用户要求"用seedvc"，seedvc已执行:
{{"done":true,"reason":"用户指定seedvc，已成功执行","summary":"seedvc, SNR=47dB"}}

用户要求"SNR尽量高"，当前 pitch SNR=15dB:
{{"done":false,"reason":"pitch的SNR只有15dB，seedvc通常SNR更高","new_vc_method":"seedvc"}}

用户要求"不要改变韵律"，当前 seedvc pit_pcc=0.92:
{{"done":true,"reason":"seedvc的pit_pcc=0.92，韵律基本保持","summary":"seedvc, pit_pcc=0.92, SNR=35dB"}}"""


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
        parts.append(f"  {k} = {v}" + (f" ({desc})" if desc else ""))
    return "\n".join(parts)


def _history_to_str(history: list) -> str:
    if not history:
        return "（无历史尝试）"
    lines = []
    for i, h in enumerate(history):
        m = h.get("metrics", {})
        m_str = ", ".join(f"{k}={v}" for k, v in m.items()) if m else "无指标"
        lines.append(f"  第{i+1}次: VC={h['vc_method']} | {m_str}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 子进程推理（进程退出后显存自动归还）
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_infer_worker(model_name, sys_prompt, user_prompt, output_file, gen_kwargs=None):
    """在子进程中加载 LLM、推理一次、写结果到文件。进程退出后 GPU 显存完全释放。"""
    from utils.language_model import load_model, generate_response
    model, tokenizer = load_model(model_name)
    gen_kwargs = gen_kwargs or {}
    # temperature > 0 时必须 do_sample=True，否则模型会忽略 temperature 并报警告
    if gen_kwargs.get('temperature', 0) > 0:
        gen_kwargs.setdefault('do_sample', True)
    else:
        gen_kwargs['do_sample'] = False
    raw = generate_response(model, tokenizer, sys_prompt=sys_prompt, prompt=user_prompt, **gen_kwargs)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# ReflexionAgent
# ═══════════════════════════════════════════════════════════════════════════════

class ReflexionAgent:
    def __init__(self, input_dir, output_dir, middle_dir, llm_model=None, mode='INFO',
                 temperature=0.01, max_new_tokens=2048, repetition_penalty=1.2):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = output_dir or os.path.join(os.path.dirname(self.input_dir), 'output')
        self.middle_dir = middle_dir or os.path.join(os.path.dirname(self.input_dir), 'middle')
        self.mode = mode

        if llm_model is None:
            try:
                from configs.config_loader import get_llm_config
                self.llm_model = get_llm_config('中文')['model']
            except Exception:
                self.llm_model = 'Qwen2.5-32B-Instruct'
        else:
            self.llm_model = llm_model

        self.gen_kwargs = {
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger, self.logger_file = setup_logger(self.output_dir, 'reflexion', mode)
        self.ctx = AgentContext(self.input_dir, self.output_dir, self.middle_dir,
                                mode, self.logger, self.logger_file)

    def _call_llm(self, sys_prompt: str, user_prompt: str, gen_override: dict = None) -> str:
        """在子进程中调用 LLM，进程退出后显存自动释放。返回原始响应文本。"""
        fd, output_file = tempfile.mkstemp(suffix='.txt', prefix='llm_result_')
        os.close(fd)
        gen = {**self.gen_kwargs, **(gen_override or {})}
        try:
            self.logger.debug(f"[LLM 子进程] 启动 | model={self.llm_model} | gen={gen}")
            self.logger.debug(f"[LLM 子进程] system prompt ({len(sys_prompt)} chars):\n{sys_prompt}")
            self.logger.debug(f"[LLM 子进程] user prompt ({len(user_prompt)} chars):\n{user_prompt}")

            run_in_isolation(
                _llm_infer_worker,
                self.llm_model, sys_prompt, user_prompt, output_file, gen,
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
        _RETRY_HINT = "\n\n注意：请减少思考，尽快输出 JSON。以 { 开头，以 } 结尾。不要输出其他内容。"
        current_prompt = user_prompt

        for attempt in range(max_retries):
            # 重试时强制 temperature=0，提高确定性
            override = {'temperature': 0} if attempt > 0 else None
            raw = self._call_llm(sys_prompt, current_prompt, gen_override=override)
            if raw and raw[0] != '{':
                raw = '{' + raw

            result = _parse_json_output(raw)

            # JSON 解析失败
            if result.get("error"):
                self.logger.warning(f"[LLM 重试] 第 {attempt+1}/{max_retries} 次 JSON 解析失败")
                current_prompt = user_prompt + _RETRY_HINT
                continue

            # JSON 解析成功，但内容验证失败
            if validator:
                err = validator(result)
                if err:
                    self.logger.warning(f"[LLM 重试] 第 {attempt+1}/{max_retries} 次验证失败: {err}")
                    current_prompt = user_prompt + _RETRY_HINT
                    continue

            self.logger.debug(f"[LLM 重试] 第 {attempt+1} 次调用成功")
            return result

        self.logger.error(f"[LLM 重试] {max_retries} 次均失败")
        return {"error": True, "raw": "多次重试后仍无法获取有效输出"}

    # ── 初始计划 ─────────────────────────────────────────────────────────────

    def _generate_initial_plan(self, user_request: str) -> dict:
        context_info = _build_context_info(self.ctx)

        plan_prompt = f"""你是语音匿名处理 Agent。根据用户请求生成 JSON 执行计划。

【重要】所有工具的 language 参数必须用中文书写（"中文"/"英文"/"日文"/"西班牙语"等），不要用英文或拼音。

【可用工具】
- run_transcription: ASR 转录。参数: language(必填，必须用中文，如"中文"/"英文"/"日文"/"西班牙语"), asr_model(可选), hotwords(可选), spk_num(可选，精确说话人数), min_speakers(可选，最少说话人数), max_speakers(可选，最多说话人数)。spk_num 和 min/max_speakers 三选一，都不填则自动检测
- run_mask: 内容匿名（依赖转录）。参数: language(必填，中文), mask_method(noise/mute), detect_method(llm/ner, 默认llm), group_num(每批处理句数，30-60，不要与说话人数混淆)
- run_voice_conversion: 声线转换。参数: method(必填, 可选: {', '.join(VC_METHODS)}), with_mask(bool), language
- run_evaluation: 评估。参数: language(必填), gt_dir(可选), subject_map_path(可选)

【步骤依赖】
- run_mask 依赖 run_transcription
- run_voice_conversion 独立，with_mask=true 时需要 mask 输出
- run_evaluation 依赖 run_transcription，且需要 mask 或 vc 输出

【输出格式】在适当位置输出 JSON，以 {{ 开头。
{{"language":"语言","reasoning":"分析","steps":[{{"tool":"工具名","params":{{}}}}]}}

示例:
用户: "匿名化中文音频，不需要内容匿名"
{{"language":"中文","reasoning":"跳过mask","steps":[{{"tool":"run_transcription","params":{{"language":"中文"}}}},{{"tool":"run_voice_conversion","params":{{"method":"seedvc","language":"中文"}}}},{{"tool":"run_evaluation","params":{{"language":"中文"}}}}]}}"""

        user_msg = (
            f"【目录信息】\n{context_info}\n"
            f"【用户请求】\n{user_request}\n\n"
            "请输出 JSON 执行计划（只输出 JSON，以 { 开头）：\n{"
        )

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
                    if later_tool and tool_name in later_tool.dependencies:
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

    def _reflect(self, user_request: str, current_metrics: dict, history: list) -> dict:
        reflection_msg = f"""【用户请求】
{user_request}

【当前目录】
{_build_context_info(self.ctx)}

【当前评估指标】
{_metrics_to_str(current_metrics)}

【历史尝试】
{_history_to_str(history)}

请根据用户目标和当前指标，决定是否需要更换 VC 方法。输出 JSON:"""

        self.logger.info("向 LLM 请求反思决策...")

        def _validate_decision(d):
            if "done" not in d:
                return "缺少 'done' 字段"
            if not d["done"] and "new_vc_method" not in d:
                return "done=false 时缺少 'new_vc_method' 字段"
            return None

        decision = self._call_llm_with_retry(
            REFLECTION_SYSTEM_PROMPT, reflection_msg, validator=_validate_decision,
        )
        self.logger.debug(f"[反思解析] decision = {json.dumps(decision, ensure_ascii=False, indent=2)}")

        if decision.get("error"):
            self.logger.warning(f"反思解析失败: {decision}")
            return {"done": True, "reason": "反思解析失败，默认完成", "summary": ""}
        return decision

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def run(self, user_request: str, max_reflections: int = 3) -> dict:
        """
        Reflexion 主循环：初始计划 → 执行 → 反思 → 调整 → 重试。
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

        plan = self._generate_initial_plan(user_request)
        if plan is None:
            self.logger.error("[主循环] 初始计划生成失败，退出")
            print("错误: 无法生成初始计划")
            return None

        language = plan.get("language", "中文")
        reasoning = plan.get("reasoning", "")
        steps = plan["steps"]
        self.logger.debug(f"[初始计划] language={language}, reasoning={reasoning}")
        self.logger.debug(f"[初始计划] steps={[s['tool'] for s in steps]}")

        initial_vc = None
        for s in steps:
            if s["tool"] == "run_voice_conversion":
                initial_vc = s["params"].get("method", "seedvc")
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

        if not current_metrics:
            self.logger.warning("[主循环] 未获得评估指标，无法进行反思")
            print("未获得评估指标，无法进行反思")
            return {"success": True, "attempts": 1, "history": []}

        # ── 3. Reflexion 循环 ───────────────────────────────────────────────
        history = [{"vc_method": initial_vc, "metrics": current_metrics}]

        for reflection_round in range(max_reflections):
            self.logger.debug(f"[反思循环] 第 {reflection_round + 1}/{max_reflections} 轮")
            print(f"\n{'='*55}")
            print(f"反思 #{reflection_round + 1}")
            print(f"{'='*55}")
            print(f"当前指标:\n{_metrics_to_str(current_metrics)}")

            decision = self._reflect(user_request, current_metrics, history)
            self.logger.debug(f"[反思决策] round={reflection_round+1}, decision={json.dumps(decision, ensure_ascii=False)}")
            print(f"决策: {json.dumps(decision, ensure_ascii=False)}")

            if decision.get("done", True):
                reason = decision.get("reason", "")
                summary = decision.get("summary", "")
                print(f"\n✅ 完成: {reason}")
                if summary:
                    print(f"摘要: {summary}")
                break

            new_vc = decision.get("new_vc_method")
            if not new_vc or new_vc not in VC_METHODS:
                self.logger.warning(f"[反思] LLM 返回无效 VC 方法: {new_vc}，终止反思")
                print(f"LLM 返回了无效的 VC 方法: {new_vc}，终止反思")
                break

            tried = [h["vc_method"] for h in history]
            if new_vc in tried:
                self.logger.warning(f"[反思] VC 方法 {new_vc} 已试过 (历史: {tried})，终止反思")
                print(f"VC 方法 {new_vc} 已经试过，终止反思")
                break

            self.logger.debug(f"[反思] VC 方法切换: {history[-1]['vc_method']} → {new_vc}")

            print(f"\n更换 VC 方法: {history[-1]['vc_method']} → {new_vc}")
            print(f"{'='*55}")
            print(f"执行: vc({new_vc}) → eval")

            vc_step = {"tool": "run_voice_conversion", "params": {"method": new_vc, "language": language}}
            eval_step = {"tool": "run_evaluation", "params": {"language": language}}
            self.logger.debug(f"[重试] vc_step={json.dumps(vc_step, ensure_ascii=False)}")
            self.logger.debug(f"[重试] eval_step={json.dumps(eval_step, ensure_ascii=False)}")

            print(f"  [1/2] run_voice_conversion (method={new_vc}) ...", flush=True)
            ok, result, elapsed = self._execute_step(vc_step)
            if ok:
                print(f"  [1/2] run_voice_conversion  完成 ({elapsed:.1f}s)")
            else:
                print(f"  [1/2] run_voice_conversion  失败: {result.get('error', '')}")
                history.append({"vc_method": new_vc, "metrics": {}})
                continue

            print(f"  [2/2] run_evaluation ...", flush=True)
            ok, result, elapsed = self._execute_step(eval_step)
            if ok:
                print(f"  [2/2] run_evaluation  完成 ({elapsed:.1f}s)")
            else:
                self.logger.error(f"[重试] 评估失败: {result.get('error', '')}")
                print(f"  [2/2] run_evaluation  失败: {result.get('error', '')}")
                history.append({"vc_method": new_vc, "metrics": {}})
                continue

            current_metrics = result.get("mean_results", {})
            history.append({"vc_method": new_vc, "metrics": current_metrics})
            self.ctx.vc_method_used = new_vc
            self.logger.debug(f"[重试] 新指标: {json.dumps(current_metrics, ensure_ascii=False)}")
        else:
            self.logger.warning(f"[主循环] 达到最大反思次数 ({max_reflections})")
            print(f"\n达到最大反思次数 ({max_reflections})，使用最后的结果")

        # ── 4. 最终汇总 ─────────────────────────────────────────────────────
        self.logger.debug(f"[最终汇总] 共 {len(history)} 次尝试")
        for i, h in enumerate(history):
            self.logger.debug(f"  第{i+1}次: VC={h['vc_method']} | metrics={json.dumps(h.get('metrics', {}), ensure_ascii=False)}")

        print(f"\n{'='*55}")
        print(f"执行完成 | 共 {len(history)} 次尝试")
        print(f"{'='*55}")
        for i, h in enumerate(history):
            m = h.get("metrics", {})
            m_str = ", ".join(f"{k}={v}" for k, v in m.items()) if m else "无指标"
            print(f"  第{i+1}次: VC={h['vc_method']} | {m_str}")

        return {
            "success": True,
            "attempts": len(history),
            "history": history,
            "final_metrics": current_metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Reflexion Agent — 基于评估反馈自动调整 VC 方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python reflexion_agent.py -i /path/to/audio -r "匿名化中文音频，SNR尽量高，不需要内容匿名"
  python reflexion_agent.py -i /path/to/audio -r "匿名化，不要改变韵律" --max_reflections 5
  python reflexion_agent.py -i /path/to/audio -r "..." --llm Qwen2.5-32B-Instruct
        """)
    parser.add_argument('--input_dir', '-i', required=True, help='输入音频目录')
    parser.add_argument('--request', '-r', required=True, help='自然语言任务描述')
    parser.add_argument('--llm', default=None, help='本地模型名称（默认自动推断）')
    parser.add_argument('--output_dir', '-o', default=None, help='输出目录')
    parser.add_argument('--middle_dir', default=None, help='中间结果目录')
    parser.add_argument('--mode', choices=['DEBUG', 'INFO'], default='INFO', help='日志级别')
    parser.add_argument('--max_reflections', type=int, default=3, help='最大反思次数（默认3）')
    parser.add_argument('--temperature', type=float, default=0.01, help='LLM 温度（默认0.01，重试时自动降为0）')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='LLM 最大生成 token 数（默认2048）')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='LLM 重复惩罚（默认1.2）')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    agent = ReflexionAgent(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        middle_dir=args.middle_dir,
        llm_model=args.llm,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    result = agent.run(args.request, max_reflections=args.max_reflections)

    if result and result.get("success"):
        history = result.get("history", [])
        if history:
            print(f"\n最终使用 VC 方法: {history[-1]['vc_method']}")
        else:
            print("\n执行完成（无需反思）。")
    else:
        print("\n执行失败，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
