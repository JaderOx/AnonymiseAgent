#!/usr/bin/env python3
from pathlib import Path

import sys
import time
import warnings
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger

input_dir = ""   # 直接运行时填入；经 evaluate.py 调用时由参数传入

# 参数
FRAME_MS = 20         # 帧长，毫秒
HOP_MS = 10           # 步长，毫秒
NOISE_PERCENT = 0.10  # 将能量最低的 NOISE_PERCENT 比例的帧视为噪声帧
NOISE_MULT = 1.5      # 噪声阈 multiplier：信号帧的阈值 = noise_power * NOISE_MULT
EPS = 1e-12           # 防止除 0

def read_wav(path):
    """读取 wav 文件，返回 float32 型的一维 numpy 数组（mono）和采样率 sr"""
    data, sr = sf.read(str(path), always_2d=False)
    # soundfile 返回 float 或 int，但如果是 int，会被转换为整数数组——统一转为 float32 并归一化（如果是整数）
    if np.issubdtype(data.dtype, np.integer):
        # 根据 dtype 推算最大值
        maxv = np.iinfo(data.dtype).max
        data = data.astype("float32") / maxv
    else:
        data = data.astype("float32")

    # 如果是多通道，转为 mono（均值）
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr

def frame_signal(x, sr, frame_ms=20, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0:
        raise ValueError("frame_ms 太小导致 frame_len <= 0")
    if hop_len <= 0:
        hop_len = frame_len
    n = len(x)
    frames = []
    for start in range(0, n - frame_len + 1, hop_len):
        frames.append(x[start:start + frame_len])
    if len(frames) == 0 and n > 0:
        # 音频很短：把整个音频当一帧
        frames = [x]
    return np.stack(frames) if len(frames) > 0 else np.empty((0, frame_len), dtype=x.dtype)

def estimate_snr_from_wave(x, sr, frame_ms=20, hop_ms=10, noise_percent=0.1, noise_mult=1.5):
    """
    基于帧能量估算 SNR（dB）。
    返回 snr_db（float 或 np.nan）
    """
    if x is None or len(x) == 0:
        return np.nan

    frames = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if frames.size == 0:
        return np.nan

    # 每帧功率（均方）
    powers = np.mean(frames.astype("float64") ** 2, axis=1)  # shape (n_frames,)
    # 若全部为 0（静音），直接返回 NaN
    if np.all(powers <= EPS):
        return np.nan

    # 估计噪声功率：取最小的若干帧
    n_frames = len(powers)
    noise_count = max(1, int(np.ceil(n_frames * noise_percent)))
    sorted_p = np.sort(powers)
    noise_power = float(np.mean(sorted_p[:noise_count]))

    # 若 noise_power 太小（接近 0），避免除 0，返回 nan 并告警
    if noise_power < EPS:
        warnings.warn("估计到的噪声功率接近 0，无法稳健计算 SNR -> 返回 NaN")
        return np.nan

    # 选取信号帧：功率大于 noise_power * noise_mult 的帧
    threshold = noise_power * noise_mult
    signal_frames = powers[powers > threshold]

    if signal_frames.size == 0:
        # 没有高于阈值的帧：改用比 noise_power 高一点的帧（比如排除最低 noise_count 帧后剩余帧的均值）
        remaining = sorted_p[noise_count:]
        if remaining.size == 0:
            total_mean = float(np.mean(powers))
            signal_power = max(total_mean - noise_power, EPS)
        else:
            signal_power = float(np.mean(remaining))
            if signal_power <= noise_power:
                signal_power = noise_power + EPS
    else:
        signal_power = float(np.mean(signal_frames))

    # SNR 计算
    snr_linear = max(signal_power / (noise_power + EPS), EPS)
    snr_db = 10.0 * np.log10(snr_linear)
    return float(snr_db)

def compute_directory_snr(wav_files, logger=None, base_dir=None):
    # start_time = time.time()

    logger.info("开始计算SNR")
    results = []
    length = len(wav_files)
    for i, wav_file in enumerate(wav_files):
        try:
            file_time = time.time()
            data, sr = read_wav(wav_file)
            snr_db = estimate_snr_from_wave(data, sr,
                                            frame_ms=FRAME_MS,
                                            hop_ms=HOP_MS,
                                            noise_percent=NOISE_PERCENT,
                                            noise_mult=NOISE_MULT)
            if not np.isnan(snr_db):
                logger.info(f"[{i+1}/{length}]SNR:{snr_db:.2f} dB\tcost:{time.time() - file_time:.2f}s\t--{wav_file.name}")
            else:
                logger.info(f"[{i+1}/{length}]SNR:NaN\t\t\tcost:{time.time() - file_time:.2f}s\t--{wav_file.name}")

            from pathlib import Path
            if base_dir:
                name = str(Path(wav_file).relative_to(base_dir).with_suffix(''))
            else:
                name = wav_file.stem
            results.append((name, snr_db))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"处理{wav_file.name}时报错：{e}\n{error_trace}")

    df = pd.DataFrame(results, columns=["name", "snr_db"])
    
    # 统计 mean ± std（忽略 NaN 与 inf）
    vals = df["snr_db"].to_numpy(dtype="float64")
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        mean = np.nan
        std = np.nan
        mean_std_str = "NaN ± NaN dB (no finite SNR values)"
    else:
        mean = float(np.mean(finite_vals))
        std = float(np.std(finite_vals, ddof=0))
        mean_std_str = f"{mean:.2f}±{std:.2f} dB"
    
    # 把mean和std加到df的最后两行，方便excel读取
    df.loc[len(df)] = ['-mean-', mean]
    df.loc[-1] = ['-std-', std]
    # logger.info(f"SNR处理完毕, 耗时 {time.time() - start_time:.2f} 秒")
    logger.debug(f"SNR结果: \n{df.to_string(index=False)}")

    return df, mean_std_str

if __name__ == "__main__":
    
    logger = setup_logger("/path/to/data/example/agent/metrics/", "snr", "DEBUG")


    # 使用示例: (需要提前像这样对目录做处理)
    audio_dir = Path(input_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir}")
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    wav_files = sorted([p for p in audio_dir.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    logger.info(f"开始计算SNR:\n音频目录：{audio_dir}，文件数量：{len(wav_files)}\n")

    compute_directory_snr(wav_files, logger)