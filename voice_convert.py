"""
voice_convert.py — 统一 VC 接口（薄层调度）

职责：
  - 分句模式（_paste_with_crossfade / _sentence_level_convert）
  - convert_audio：单文件入口，调用注册表中的方法
  - batch_convert ：批量入口，支持多进程
  - CLI

具体 VC 实现位于 utils/voice_convert/，通过注册表统一管理：
  formant  : STFT 快速共振峰缩放
  pitch    : 相位声码器变调
  combined : formant + pitch
  mcadams  : McAdams OLA-LPC
  seedvc   : vc大模型（需要整批处理）

添加新方法（DSP 或神经网络 VC）：
  1. 在 utils/voice_convert/ 新建方法文件
  2. 在 utils/voice_convert/__init__.py 末尾添加 import + register()
"""

import json
import os
import time
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from utils.voice_convert import get as _get_method, list_methods

SUPPORTED = {'.wav', '.flac', '.mp3', '.ogg', '.aiff'}


def _load_trans_json(path):
    """Load transcription JSON, returning (sentences: list, speaker_info: dict).
    Handles both old list format and new dict format with speaker_info."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get('sentences', []), data.get('speaker_info', {})
    return data, {}


# ─────────────────────────────────────────────────────────────
# 分句模式共享工具
# ─────────────────────────────────────────────────────────────

def _paste_with_crossfade(sig: np.ndarray, core: np.ndarray,
                          start_s: int, end_s: int, cf: int) -> np.ndarray:
    """
    将 core（长度 == end_s - start_s）用线性交叉淡化贴回 sig[start_s:end_s]。
    cf 自动截短为片段长度的 1/4，避免边界重叠。
    """
    result = sig.copy()
    L  = end_s - start_s
    cf = min(cf, L // 4)

    if cf > 0:
        fade_in  = np.linspace(0.0, 1.0, cf)
        fade_out = np.linspace(1.0, 0.0, cf)
        result[start_s : start_s + cf] = (
            sig[start_s : start_s + cf] * fade_out + core[:cf] * fade_in
        )
        result[end_s - cf : end_s] = (
            core[-cf:] * fade_out + sig[end_s - cf : end_s] * fade_in
        )
        result[start_s + cf : end_s - cf] = core[cf : L - cf]
    else:
        result[start_s:end_s] = core

    return result


def _sentence_level_convert(sig: np.ndarray, sr: int, sentences: list,
                             convert_fn, speaker_info: dict = None,
                             pad_ms=150, crossfade_ms=30,
                             **kwargs) -> np.ndarray:
    """
    分句 VC：按 JSON 逐句处理，句间背景噪声保持原样。

    流程（每句）：
      1. timestamp[0] / [-1] 确定句子边界（ms）
      2. 前后各延拓 pad_ms 提供 VC 算法上下文
      3. 对延拓片段做 VC
      4. 截取核心区域（去 padding 偏移），交叉淡化贴回
    """
    result = sig.astype(np.float64).copy()
    pad_s  = int(pad_ms      * sr / 1000)
    cf_s   = int(crossfade_ms * sr / 1000)
    total  = len(sig)

    for sentence in sentences:
        ts = sentence.get("timestamp", [])
        if not ts or len(ts) < 2:
            continue

        start_s = int(ts[0]  * sr / 1000)
        end_s   = int(ts[-1] * sr / 1000)

        if end_s <= start_s or start_s >= total:
            continue
        end_s = min(end_s, total)

        pad_start = max(0, start_s - pad_s)
        pad_end   = min(total, end_s + pad_s)
        segment   = sig[pad_start:pad_end].astype(np.float64)

        # 将句子元数据透传给 convert_fn；从 speaker_info 补充 age/gender
        sent_meta = {
            k: v for k, v in sentence.items()
            if k not in ('index', 'text', 'timestamp')
        }
        if speaker_info:
            spk_key = str(sentence.get('speaker', 0))
            info = speaker_info.get(spk_key, {})
            sent_meta.setdefault('age',    info.get('age'))
            sent_meta.setdefault('gender', info.get('gender'))
        try:
            seg_processed = convert_fn(segment, sr, **kwargs, **sent_meta)
        except Exception:
            continue

        core_start = start_s - pad_start
        core_end   = end_s   - pad_start
        core       = seg_processed[core_start:core_end]

        target_len = end_s - start_s
        if len(core) < target_len:
            core = np.pad(core, (0, target_len - len(core)))
        elif len(core) > target_len:
            core = core[:target_len]

        result = _paste_with_crossfade(result, core, start_s, end_s, cf_s)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────────────────────

def convert_audio(input_path, output_path, method: str = "combined",
                  trans_json=None, pad_ms: int = 150,
                  crossfade_ms: int = 30, **kwargs):
    """
    单文件 VC。

    Args:
        method      : 注册表中的方法名（见 utils/voice_convert/__init__.py）
        trans_json  : 转录 JSON 路径。提供且存在时自动判断是否分句：
                      - 短音频(<5min)且单说话人 → 整段处理（透传 age/gender）
                      - 长音频或多说话人       → 分句处理
        pad_ms      : 分句模式：每句前后延拓时长（ms）
        crossfade_ms: 分句模式：边界交叉淡化时长（ms）
        **kwargs    : 透传给具体方法（覆盖 DEFAULTS）
    """
    entry  = _get_method(method)
    fn     = entry.get("fn")
    if fn is None:
        raise ValueError(
            f"VC 方法 '{method}' 不支持单文件 convert_audio，请使用 batch_convert"
        )
    params = {**entry["defaults"], **kwargs}   # 用户参数优先

    sig, sr = librosa.load(str(input_path), sr=None, mono=True)

    if trans_json is not None and Path(trans_json).exists():
        sentences, speaker_info = _load_trans_json(trans_json)
        if sentences:
            # 计算音频总时长（ms）和说话人数
            last_ts = sentences[-1].get("timestamp", [])
            duration_ms = last_ts[-1] if last_ts else (len(sig) / sr * 1000)
            num_speakers = len(speaker_info) if speaker_info else 1

            # 短音频（<5min）+ 单说话人 → 整段模式，只透传 age/gender
            if duration_ms < 300_000 and num_speakers <= 1:
                spk0 = speaker_info.get('0', speaker_info.get(0, {})) if speaker_info else {}
                whole_kwargs = {**params}
                if spk0.get('age') is not None:
                    whole_kwargs['age'] = spk0['age']
                if spk0.get('gender') is not None:
                    whole_kwargs['gender'] = spk0['gender']
                sig_out = fn(sig, sr, **whole_kwargs)
            else:
                sig_out = _sentence_level_convert(
                    sig, sr, sentences, fn,
                    speaker_info=speaker_info,
                    pad_ms=pad_ms, crossfade_ms=crossfade_ms, **params
                )
        else:
            # 转录为空（全音频模式）：从 speaker_info 取 spk0 的 age/gender 透传
            spk0 = speaker_info.get('0', speaker_info.get(0, {}))
            whole_kwargs = {**params}
            if spk0.get('age') is not None:
                whole_kwargs['age'] = spk0['age']
            if spk0.get('gender') is not None:
                whole_kwargs['gender'] = spk0['gender']
            sig_out = fn(sig, sr, **whole_kwargs)
    else:
        sig_out = fn(sig, sr, **params)

    peak = np.max(np.abs(sig_out))
    if peak > 1e-6:
        sig_out = sig_out / peak * 0.95

    sf.write(str(output_path), sig_out, sr)
    return True


# ── 多进程 worker（必须在模块顶层才能被 pickle）─────────────────

def _worker(args):
    fp_str, out_str, method, trans_json_str, pad_ms, crossfade_ms, mask_json_str, mask_kwargs, kwargs = args
    try:
        convert_audio(fp_str, out_str, method=method,
                      trans_json=trans_json_str,
                      pad_ms=pad_ms, crossfade_ms=crossfade_ms,
                      **kwargs)
        # 如果提供了 mask JSON，VC 完立刻 in-place 覆写
        if mask_json_str and Path(mask_json_str).exists():
            from apply_mask import add_adaptive_ssn_to_audio
            add_adaptive_ssn_to_audio(out_str, mask_json_str, out_str, **mask_kwargs)
        return fp_str, True, None
    except Exception as e:
        import traceback
        return fp_str, False, traceback.format_exc()


def batch_convert(input_dir, output_dir, logger,
                  method: str = "combined",
                  trans_dir=None, pad_ms: int = 150,
                  crossfade_ms: int = 30, num_workers: int = 1,
                  mask_dir=None, mask_method: str = 'formant',
                  mask_noise_strength: float = 1.0, mask_time_append: int = 50,
                  language=None,
                  **kwargs):
    """
    批量 VC。

    Args:
        trans_dir   : 转录 JSON 目录。提供且对应 JSON 存在则分句模式；
                      缺失时自动降级为整段模式。
        pad_ms      : 分句模式：每句前后延拓时长（ms）
        crossfade_ms: 分句模式：边界交叉淡化时长（ms）
        num_workers : 并行进程数（默认 1 串行；多文件时建议设为 CPU 核数的一半）
        language    : 流水线语言（如 agent --language），TTS 方法用于选 en/ch 参考音等
    """
    # ── 增量跳过：输出目录音频数 >= 输入音频数则跳过 ──
    _input_dir_p = Path(input_dir)
    _output_dir_p = Path(output_dir)
    _input_count = len([p for p in _input_dir_p.rglob('*') if p.suffix.lower() in SUPPORTED]) if _input_dir_p.exists() else 0
    if _output_dir_p.exists() and _input_count > 0:
        _output_count = len([p for p in _output_dir_p.rglob('*') if p.suffix.lower() in SUPPORTED])
        if _output_count >= _input_count:
            logger.info(
                f"[跳过] VC 输出已存在（{_output_count} 条 >= 输入 {_input_count} 条），不加载模型"
            )
            return

    start_time = time.time()
    entry = _get_method(method)
    mask_kwargs = dict(noise_method=mask_method, noise_strength=mask_noise_strength,
                       time_append=mask_time_append)

    vc_kwargs = dict(kwargs)
    if language is not None:
        vc_kwargs["language"] = language

    # 神经网络 VC：方法自带批量处理逻辑
    if entry.get("batch_fn") is not None:
        all_kwargs = {**entry["defaults"], **vc_kwargs}
        entry["batch_fn"](
            input_dir, output_dir, logger,
            trans_dir=trans_dir, mask_dir=mask_dir, mask_kwargs=mask_kwargs,
            **all_kwargs,
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    _input_dir_p = Path(input_dir)
    audio_files = sorted([
        p for p in _input_dir_p.rglob('*')
        if p.suffix.lower() in SUPPORTED
    ])
    if not audio_files:
        logger.warning(f"目录 {input_dir} 中无音频文件")
        return

    mode_desc = f"分句(pad={pad_ms}ms)" if trans_dir else "整段"
    mask_desc = f" + mask({mask_method})" if mask_dir else ""
    logger.info(
        f"批量 VC  方法={method}  模式={mode_desc}{mask_desc}  workers={num_workers}\n"
        f"输入: {input_dir}\n输出: {output_dir}\n"
    )

    # 构建任务列表（跳过已存在文件）
    tasks   = []
    skipped = 0
    for fp in audio_files:
        rel = fp.relative_to(_input_dir_p)
        out_fp = Path(output_dir) / rel
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        if out_fp.exists() and out_fp.stat().st_size > 0:
            logger.info(f"已存在，跳过: {rel}")
            skipped += 1
            continue

        trans_json = None
        if trans_dir:
            c = Path(trans_dir) / rel.with_suffix('.json')
            if c.exists():
                trans_json = str(c)
            else:
                logger.warning(f"未找到 {c.name}，{rel} 降级为整段模式")

        mask_json = None
        if mask_dir:
            m = Path(mask_dir) / rel.with_suffix('.json')
            if m.exists():
                mask_json = str(m)
            else:
                logger.warning(f"未找到 mask JSON {m.name}，{rel} 跳过 mask")

        tasks.append((str(fp), str(out_fp), method,
                      trans_json, pad_ms, crossfade_ms, mask_json, mask_kwargs, vc_kwargs))

    if not tasks:
        logger.info(f"所有文件已处理，无需重复计算,耗时{time.time() - start_time:.2f}s")
        return

    ok = failed = 0

    if num_workers > 1:
        from multiprocessing import get_context
        ctx = get_context("spawn")   # spawn 对 numpy/librosa 更安全
        with ctx.Pool(processes=num_workers) as pool:
            for fp_str, success, err in pool.imap_unordered(_worker, tasks):
                name = Path(fp_str).name
                if success:
                    ok += 1
                    logger.info(f"完成: {name}")
                else:
                    failed += 1
                    logger.error(f"失败: {name}\n{err}")
    else:
        for i, task in enumerate(tasks):
            name = Path(task[0]).name
            logger.info(f"[{i+1}/{len(tasks)}] {name}")
            t0 = time.time()
            _, success, err = _worker(task)
            if success:
                ok += 1
                logger.info(f"完成: {name}  ({time.time()-t0:.2f}s)")
            else:
                failed += 1
                logger.error(f"失败: {name}\n{err}")

    logger.info(
        f"批量 VC 完成：成功 {ok}，跳过 {skipped}，失败 {failed}  "
        f"总耗时 {time.time()-start_time:.2f}s"
    )


# ─────────────────────────────────────────────────────────────
# CLI（独立测试）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from utils.logger_example import setup_logger

    parser = argparse.ArgumentParser(description="语音转换独立测试")
    parser.add_argument("-i", "--input_dir",   required=True)
    parser.add_argument("-o", "--output_dir",  required=True)
    parser.add_argument("--method",       choices=list_methods(), default="combined")
    parser.add_argument("--trans_dir",    default=None,
                        help="转录 JSON 目录（提供则启用分句模式）")
    parser.add_argument("--pad_ms",       type=int,   default=150)
    parser.add_argument("--crossfade_ms", type=int,   default=30)
    parser.add_argument("--workers",      type=int,   default=1)
    # combined / formant 参数
    parser.add_argument("--formant_scale", type=float, default=1.1)
    parser.add_argument("--pitch_steps",   type=float, default=-3.0)
    # mcadams 参数
    parser.add_argument("--alpha",         type=float, default=0.8)
    parser.add_argument("--language",      default=None, help="语言（TTS 方法选参考音用）")
    parser.add_argument("--mask_dir",      default=None, help="隐私词 JSON 目录（提供则 VC 后立刻 mask）")
    parser.add_argument("--mask_method",   default="formant", help="mask 噪声方法（默认 formant）")
    args = parser.parse_args()

    logger, _ = setup_logger(args.output_dir, "voice_convert", "INFO")
    batch_convert(
        args.input_dir, args.output_dir, logger,
        method=args.method,
        trans_dir=args.trans_dir,
        pad_ms=args.pad_ms,
        crossfade_ms=args.crossfade_ms,
        num_workers=args.workers,
        formant_scale=args.formant_scale,
        pitch_steps=args.pitch_steps,
        alpha=args.alpha,
        language=args.language,
        mask_dir=args.mask_dir,
        mask_method=args.mask_method,
    )
