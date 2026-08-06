#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
import warnings
import tempfile
import shutil
import sys
import time
import subprocess
import os

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger

DIR_BEFORE = "/path/to/data/crisis/small_ver"
DIR_AFTER = "/path/to/data/crisis/Exp3/output/vc"
out_dir = "/path/to/data/crisis/Exp3/output/"

# 转码/比较参数
TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "s16"   # 16-bit PCM
TIME_STEP = 0.01        # uniform grid step for diagnostics (10 ms)

def extract_pitch_contour(path, pitch_floor=75.0, pitch_ceiling=600.0, hop_length=None):
    """
    使用librosa提取基频轮廓
    hop_length: 帧移（默认根据TIME_STEP计算）
    """
    # 加载音频
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    
    # 计算hop_length
    if hop_length is None:
        hop_length = int(TIME_STEP * sr)
    
    # 使用pyin算法提取基频（更准确）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=pitch_floor, 
        fmax=pitch_ceiling,
        sr=sr,
        hop_length=hop_length
    )
    
    # 计算时间戳
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    # 将未 voiced 的部分设为 NaN
    f0 = f0.astype(np.float64)
    f0[~voiced_flag] = np.nan
    
    return times, f0

def interp_with_nan(x_old, y_old, x_new):
    """
    Interpolate y_old (may contain np.nan) from x_old -> x_new.
    - Interpolate only between valid (non-NaN) samples.
    - For x_new outside the span of valid x_old -> set NaN (no extrapolation).
    """
    x_old = np.asarray(x_old, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)

    valid = ~np.isnan(y_old)
    if valid.sum() < 2:
        return np.full_like(x_new, np.nan, dtype=np.float64)

    x_valid = x_old[valid]
    y_valid = y_old[valid]

    if not np.all(np.diff(x_valid) > 0):
        order = np.argsort(x_valid)
        x_valid = x_valid[order]
        y_valid = y_valid[order]

    y_new = np.interp(x_new, x_valid, y_valid)
    left_bound = x_valid.min()
    right_bound = x_valid.max()
    outside = (x_new < left_bound) | (x_new > right_bound)
    y_new[outside] = np.nan
    return y_new

def hz_to_semitones(freq):
    freq = np.asarray(freq, dtype=np.float64)
    out = np.full_like(freq, np.nan, dtype=np.float64)
    valid = np.isfinite(freq) & (freq > 0.0)
    if np.any(valid):
        out[valid] = 12.0 * np.log2(freq[valid] / 440.0)
    return out


# ── 评估接口（与其他 metric 模块风格一致） ───────────────────────────────────

def _compute_pair(origin_path, vc_path, base_dir_origin=None, pitch_floor=75, pitch_ceiling=600):
    """
    对一对文件提取基频并计算 L1/PCC，每条音频仅提取一次。
    librosa.load 内置重采样，无需 ffmpeg 预处理。
    """
    from pathlib import Path
    if base_dir_origin:
        name = str(Path(origin_path).relative_to(base_dir_origin).with_suffix(''))
    else:
        name = Path(origin_path).stem
    try:
        times1, f1 = extract_pitch_contour(str(origin_path), pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        times2, f2 = extract_pitch_contour(str(vc_path),     pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    except Exception:
        return {"name": name, "pit_l1": np.nan, "pit_pcc": np.nan}

    f1 = f1.astype(np.float64); f1[f1 == 0] = np.nan
    f2 = f2.astype(np.float64); f2[f2 == 0] = np.nan

    t_min = max(np.nanmin(times1), np.nanmin(times2))
    t_max = min(np.nanmax(times1), np.nanmax(times2))
    if t_max <= t_min:
        return {"name": name, "pit_l1": np.nan, "pit_pcc": np.nan}

    mask1 = (times1 >= t_min) & (times1 <= t_max)
    mask2 = (times2 >= t_min) & (times2 <= t_max)
    t2_r, f2_r = times2[mask2], f2[mask2]
    if len(t2_r) == 0:
        return {"name": name, "pit_l1": np.nan, "pit_pcc": np.nan}

    # 将 f1 插值到 f2 的时间轴上
    f1_on_t2 = interp_with_nan(times1[mask1], f1[mask1], t2_r)
    valid = ~np.isnan(f1_on_t2) & ~np.isnan(f2_r)
    if valid.sum() < 2:
        return {"name": name, "pit_l1": np.nan, "pit_pcc": np.nan}

    s1 = hz_to_semitones(f1_on_t2[valid])
    s2 = hz_to_semitones(f2_r[valid])
    finite = np.isfinite(s1) & np.isfinite(s2)
    if finite.sum() < 2:
        return {"name": name, "pit_l1": np.nan, "pit_pcc": np.nan}

    s1, s2 = s1[finite], s2[finite]
    l1 = float(np.mean(np.abs(s2 - s1)))
    std1, std2 = float(np.std(s1, ddof=0)), float(np.std(s2, ddof=0))
    if std1 == 0.0 or std2 == 0.0:
        r = float(np.nan)
    else:
        try:
            r = float(np.corrcoef(s1, s2)[0, 1])
        except Exception:
            r = float(np.nan)

    return {"name": name, "pit_l1": l1, "pit_pcc": r}


def _pair_worker(args):
    """ProcessPoolExecutor 顶层包装，用于 pickle。"""
    return _compute_pair(*args)


def compute_directory_pit(origin_files, vc_files, logger, base_dir_origin=None, base_dir_vc=None, n_workers=None):
    """
    对已按说话人切分的音频文件对，并行计算基频 L1/PCC。

    Args:
        origin_files: list[Path]，原始说话人片段
        vc_files:     list[Path]，对应 VC 说话人片段（同名）
        logger:       日志对象
        n_workers:    并行进程数，默认使用所有 CPU 核

    Returns:
        (df, mean_std_str)，df 含 name/pit_l1/pit_pcc 列；失败返回 (None, "")
    """
    from concurrent.futures import ProcessPoolExecutor

    # 用相对路径（不含扩展名）作为 key 进行配对
    vc_by_relpath = {}
    if base_dir_vc:
        for p in vc_files:
            rel = str(Path(p).relative_to(base_dir_vc).with_suffix(''))
            vc_by_relpath[rel] = p
    else:
        vc_by_relpath = {p.stem: p for p in vc_files}

    pairs = []
    for op in origin_files:
        if base_dir_origin:
            rel_key = str(Path(op).relative_to(base_dir_origin).with_suffix(''))
        else:
            rel_key = Path(op).stem

        vp = vc_by_relpath.get(rel_key)
        if vp is None:
            logger.warning(f"pit_calc: 未找到 VC 对应文件: {rel_key}，跳过")
            continue
        pairs.append((str(op), str(vp), base_dir_origin))

    if not pairs:
        logger.warning("pit_calc: 无有效文件对，跳过")
        return None, ""

    n_workers = n_workers or min(os.cpu_count() or 4, len(pairs))
    logger.info(f"Pitch 计算开始，共 {len(pairs)} 对，并行进程数: {n_workers}")

    # # 串行处理避免 pickle 问题
    # results = []
    # for op, vp, base_dir in pairs:
    #     results.append(_compute_pair(op, vp, base_dir_origin=base_dir))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_pair_worker, pairs))

    df = pd.DataFrame(results)
    mean_l1  = df["pit_l1"].mean(skipna=True)
    std_l1   = df["pit_l1"].std(skipna=True)
    mean_pcc = df["pit_pcc"].mean(skipna=True)
    std_pcc  = df["pit_pcc"].std(skipna=True)

    df = pd.concat([
        df,
        pd.DataFrame([{"name": "-mean-", "pit_l1": mean_l1, "pit_pcc": mean_pcc}]),
        pd.DataFrame([{"name": "-std-",  "pit_l1": std_l1,  "pit_pcc": std_pcc}]),
    ], ignore_index=True)

    mean_std_str = (
        f"L1={mean_l1:.3f}±{std_l1:.3f}(semitone), "
        f"PCC={mean_pcc:.4f}±{std_pcc:.4f}"
    )
    return df, mean_std_str

if __name__ == "__main__":

    logger, _ = setup_logger("/path/to/data/example/agent/metrics/", "pit", "DEBUG")

    # 使用示例: (需要提前像这样对目录做处理)
    audio_dir1 = Path(DIR_BEFORE)
    if not audio_dir1.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir1}")
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    wav_files1 = sorted([p for p in audio_dir1.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])

    audio_dir2 = Path(DIR_AFTER)
    if not audio_dir2.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir2}")

    wav_files2 = sorted([p for p in audio_dir2.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    logger.info(f"开始计算pitch:\n音频目录1：{audio_dir1}，文件数量：{len(wav_files1)}\n音频目录2：{audio_dir2}，文件数量：{len(wav_files2)}\n")

    compute_directory_pit(wav_files1, wav_files2, logger)