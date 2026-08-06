"""
按说话人提取音频片段，同时处理 origin 和 VC 音频，保证一一对应。
依赖转录 JSON 中的 speaker 字段（paraformer / paraformer-en 输出）。
JSON timestamp 格式：[word0_start, word1_start, ..., last_word_end]，单位 ms。

VC（McAdams）不改变时序，因此 origin 的转录时间戳可直接用于切割 VC 音频。
"""
import os
import json
import numpy as np
import soundfile as sf
import torchaudio

from pathlib import Path


def _load_sentences(json_path):
    """Load sentences list from JSON, handling both old list and new dict format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get('sentences', [])
    return data


def _load_mono_numpy(audio_path):
    """加载音频，转单声道 numpy，返回 (audio_np, sr)。"""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), sr


def extract_speaker_segments(
        origin_audio_path,
        vc_audio_path,
        json_path,
        origin_output_dir,
        vc_output_dir,
        logger,
        max_sentences=8,
        min_sent_s=1.0,
        file_num=None
):
    """
    用同一份转录 JSON，同时从 origin 和 VC 音频中切出每个说话人的片段，
    保证两侧片段严格一一对应。

    Args:
        origin_audio_path:  原始音频路径
        vc_audio_path:      对应的 VC 音频路径（时间戳与 origin 一致）
        json_path:          origin 的转录 JSON 路径
        origin_output_dir:  origin 片段保存目录
        vc_output_dir:      VC 片段保存目录
        logger:             日志对象
        max_sentences:      每个说话人取时长最长的前 N 句
        min_sent_s:         句子最短时长（秒），过短的句子过滤

    Returns:
        dict: {spk_id: (origin_path, vc_path)}，失败时返回 {}
    """
    sentences = _load_sentences(json_path)

    if not any('speaker' in s for s in sentences):
        logger.info(f"{json_path} 无 speaker 字段，将整个文件视为 spk0")
        for s in sentences:
            s.setdefault('speaker', 0)

    # 加载两侧音频
    origin_np, sr_o = _load_mono_numpy(origin_audio_path)
    vc_np,     sr_v = _load_mono_numpy(vc_audio_path)

    # 按说话人分组，记录 (start_ms, end_ms, duration_s)
    spk_sentences = {}
    for s in sentences:
        spk_id = s.get('speaker', None)
        if spk_id is None:
            continue
        ts = s.get('timestamp', [])
        if not ts or len(ts) < 2:
            continue
        start_ms = ts[0]
        end_ms = ts[-1]
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s < min_sent_s:
            continue
        spk_sentences.setdefault(spk_id, []).append((start_ms, end_ms, duration_s))

    if not spk_sentences:
        logger.warning(f"{json_path} 无有效说话人片段（均被过滤）")
        return {}

    os.makedirs(origin_output_dir, exist_ok=True)
    os.makedirs(vc_output_dir, exist_ok=True)
    stem = Path(origin_audio_path).stem
    result = {}

    for spk_id, segs in spk_sentences.items():
        # 取时长最长的前 max_sentences 句，再按时间顺序排列
        segs_top = sorted(segs, key=lambda x: x[2], reverse=True)[:max_sentences]
        segs_top = sorted(segs_top, key=lambda x: x[0])

        origin_chunks = []
        vc_chunks = []
        for start_ms, end_ms, _ in segs_top:
            # origin
            s_o = int(start_ms / 1000.0 * sr_o)
            e_o = min(int(end_ms / 1000.0 * sr_o), len(origin_np))
            if e_o > s_o:
                origin_chunks.append(origin_np[s_o:e_o])
            # vc（用相同时间戳，sr 可能不同）
            s_v = int(start_ms / 1000.0 * sr_v)
            e_v = min(int(end_ms / 1000.0 * sr_v), len(vc_np))
            if e_v > s_v:
                vc_chunks.append(vc_np[s_v:e_v])

        # 只有两侧都有片段时才写出，保证严格对应
        if not origin_chunks or not vc_chunks:
            continue

        origin_merged = np.concatenate(origin_chunks)
        vc_merged     = np.concatenate(vc_chunks)

        origin_out = os.path.join(origin_output_dir, f"{stem}_spk{spk_id}.wav")
        vc_out     = os.path.join(vc_output_dir,     f"{stem}_spk{spk_id}.wav")
        sf.write(origin_out, origin_merged, sr_o)
        sf.write(vc_out,     vc_merged,     sr_v)

        result[spk_id] = (origin_out, vc_out)
        prefix = f"[{file_num}] " if file_num else ""
        logger.info(
            f"{prefix}spk{spk_id}: {len(segs_top)}句 → {Path(origin_out).name} "
            f"({len(origin_merged)/sr_o:.1f}s)"
        )

    return result


def extract_dir(
        origin_audio_dir,
        vc_audio_dir,
        json_dir,
        origin_output_dir,
        vc_output_dir,
        logger,
        max_sentences=8,
        min_sent_s=1.0
):
    """
    对目录下所有音频文件，同时从 origin 和 VC 中提取说话人片段，
    已处理的文件跳过（缓存）。

    Returns:
        bool: True 表示至少一个文件成功提取
    """
    origin_audio_dir  = Path(origin_audio_dir)
    vc_audio_dir      = Path(vc_audio_dir)
    json_dir          = Path(json_dir)
    origin_output_dir = Path(origin_output_dir)
    vc_output_dir     = Path(vc_output_dir)

    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    origin_files = sorted([
        p for p in origin_audio_dir.rglob('*')
        if p.suffix.lower() in SUPPORTED_FORMATS
    ])

    total = len(origin_files)
    if not origin_files:
        logger.warning(f"extract_dir: {origin_audio_dir} 中无音频文件")
        return False

    any_success = False
    count = 0
    for origin_path in origin_files:
        count += 1
        rel = origin_path.relative_to(origin_audio_dir)
        stem = origin_path.stem
        sub_o = origin_output_dir / rel.parent
        sub_v = vc_output_dir / rel.parent

        # 缓存检查：两侧都有文件时才跳过（只有 origin 不跳过，保证配对一致）
        existing_o = list(sub_o.glob(f"{stem}_spk*.wav")) if sub_o.exists() else []
        existing_v = list(sub_v.glob(f"{stem}_spk*.wav")) if sub_v.exists() else []
        if existing_o and existing_v:
            logger.info(f"[{count}/{total}] {rel} 两侧说话人片段已存在，跳过")
            any_success = True
            continue

        # 找对应的 VC 音频（同名，扩展名可能不同）
        vc_sub_dir = vc_audio_dir / rel.parent
        vc_candidates = [
            p for p in vc_sub_dir.iterdir()
            if p.stem == stem and p.suffix.lower() in SUPPORTED_FORMATS
        ] if vc_sub_dir.exists() else []
        if not vc_candidates:
            logger.warning(f"未找到对应 VC 音频: {vc_audio_dir}/{rel.parent}/{stem}.*，跳过")
            continue
        vc_path = vc_candidates[0]

        json_path = json_dir / rel.with_suffix('.json')
        if not json_path.exists():
            logger.warning(f"未找到对应 JSON: {json_path}，跳过 {rel}")
            continue

        sub_o.mkdir(parents=True, exist_ok=True)
        sub_v.mkdir(parents=True, exist_ok=True)
        result = extract_speaker_segments(
            str(origin_path), str(vc_path), str(json_path),
            str(sub_o), str(sub_v),
            logger, max_sentences, min_sent_s,
            file_num=f"{count}/{total}"
        )
        if result:
            any_success = True

    return any_success


# ─────────────────────────────────────────────────────────────────────────────
# 单侧提取（转录阶段提取 origin、VC 阶段提取 vc）
# ─────────────────────────────────────────────────────────────────────────────

def extract_single_file(
        audio_path,
        json_path,
        output_dir,
        logger,
        max_sentences=8,
        min_sent_s=1.0,
        max_total_sec=40.0,
):
    """
    从单个音频文件中按说话人提取片段，保存到 output_dir。
    每位说话人最多取 max_sentences 句或累计 max_total_sec 秒（取先到者）。

    Returns:
        dict: {spk_id: out_path}，失败时返回 {}
    """
    logger.debug(
        f"extract_single_file: {Path(audio_path).name}  "
        f"json={Path(json_path).name}  output={output_dir}"
    )

    sentences = _load_sentences(json_path)

    if not any('speaker' in s for s in sentences):
        logger.info(f"{json_path} 无 speaker 字段，将整个文件视为 spk0")
        for s in sentences:
            s.setdefault('speaker', 0)

    audio_np, sr = _load_mono_numpy(audio_path)
    logger.debug(f"  音频时长: {len(audio_np)/sr:.1f}s  sr={sr}")

    spk_segs: dict = {}
    for s in sentences:
        spk_id = s.get('speaker')
        if spk_id is None:
            continue
        ts = s.get('timestamp', [])
        if not ts or len(ts) < 2:
            continue
        start_ms, end_ms = ts[0], ts[-1]
        dur = (end_ms - start_ms) / 1000.0
        if dur < min_sent_s:
            continue
        text_l = len(s.get('text', ''))
        spk_segs.setdefault(spk_id, []).append((start_ms, end_ms, dur, text_l))

    if not spk_segs:
        logger.warning(f"{Path(json_path).name} 无有效说话人片段（均被过滤）")
        return {}

    logger.debug(f"  {len(spk_segs)} 位说话人: {list(spk_segs)}")
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(audio_path).stem
    result = {}

    for spk_id, segs in spk_segs.items():
        segs_top = sorted(segs, key=lambda x: x[3], reverse=True)[:max_sentences]
        segs_top = sorted(segs_top, key=lambda x: x[0])

        chunks = []
        total_sec = 0.0
        for start_ms, end_ms, _, _ in segs_top:
            if total_sec >= max_total_sec:
                logger.debug(f"    spk{spk_id}: 已达 {max_total_sec}s 上限，停止")
                break
            s_s = int(start_ms / 1000.0 * sr)
            e_s = min(int(end_ms   / 1000.0 * sr), len(audio_np))
            if e_s <= s_s:
                continue
            chunk = audio_np[s_s:e_s]

            chunks.append(chunk)
            total_sec += len(chunk) / sr

        if not chunks:
            logger.debug(f"  spk{spk_id}: 无有效音频块，跳过")
            continue

        merged   = np.concatenate(chunks)
        out_path = os.path.join(output_dir, f"{stem}_spk{spk_id}.wav")
        sf.write(out_path, merged, sr)
        result[spk_id] = out_path
        logger.debug(f"  spk{spk_id}: {total_sec:.1f}s → {out_path}")

    logger.info(f"{stem}: 提取 {len(result)} 位说话人音频 → {output_dir}")
    return result


def extract_audio_dir(
        audio_dir,
        json_dir,
        output_dir,
        logger,
        max_sentences=8,
        min_sent_s=1.0,
        max_total_sec=60.0,
):
    """
    批量从音频目录提取每位说话人片段（单侧）。
    已有 {stem}_spk*.wav 的文件跳过（缓存）。

    Returns:
        bool: 至少一个文件成功提取
    """
    audio_dir_p = Path(audio_dir)
    json_dir_p  = Path(json_dir)
    output_p    = Path(output_dir)
    SUPPORTED   = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}

    files = sorted(p for p in audio_dir_p.rglob('*') if p.suffix.lower() in SUPPORTED)
    total = len(files)
    logger.debug(f"extract_audio_dir: {audio_dir}  共 {total} 个文件")

    if not files:
        logger.warning(f"extract_audio_dir: {audio_dir} 无音频文件")
        return False

    any_success = False
    count = 0
    for fp in files:
        count += 1
        rel = fp.relative_to(audio_dir_p)
        sub_output_p = output_p / rel.parent
        existing = list(sub_output_p.glob(f"{fp.stem}_spk*.wav")) if sub_output_p.exists() else []
        if existing:
            logger.info(f"[{count}/{total}] {rel} 已有 {len(existing)} 条说话人片段，跳过")
            any_success = True
            continue

        json_path = json_dir_p / rel.with_suffix('.json')
        if not json_path.exists():
            logger.warning(f"[{count}/{total}] 未找到 JSON {json_path}，跳过 {rel}")
            continue

        sub_output_p.mkdir(parents=True, exist_ok=True)
        result = extract_single_file(
            str(fp), str(json_path), str(sub_output_p),
            logger, max_sentences, min_sent_s, max_total_sec,
        )
        if result:
            any_success = True
            logger.info(f"[{count}/{total}] 已保存 {len(result)} 位说话人音频 → {sub_output_p}")

    return any_success


# ======================== 独立测试 ========================
if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from logger_example import setup_logger

    # 修改为实际路径后运行
    origin_audio_dir  = "/path/to/origin_audio"
    vc_audio_dir      = "/path/to/vc_audio"
    json_dir          = "/path/to/trans_origin"
    origin_output_dir = "/path/to/spk_audio/origin"
    vc_output_dir     = "/path/to/spk_audio/vc"

    logger, _ = setup_logger(origin_output_dir, "spk_extractor", "DEBUG")
    success = extract_dir(
        origin_audio_dir, vc_audio_dir, json_dir,
        origin_output_dir, vc_output_dir, logger
    )
    print(f"提取结果: {'成功' if success else '失败（无有效片段）'}")
