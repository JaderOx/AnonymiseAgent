"""
utils/voice_convert/formant.py — STFT 快速共振峰缩放

原理：对每帧幅度谱做频率轴线性插值，等效于改变声道长度。
  scale > 1 → 频谱向高频拉伸（共振峰上移，声音更亮）—— 推荐
  scale < 1 → 频谱向低频压缩（共振峰下移，声音变闷，不推荐）

"""

import numpy as np
import librosa

DESCRIPTION = "STFT 频率轴插值缩放（快速共振峰变换，推荐范围 scale 1.05~1.20）"
DEFAULTS    = {"scale": 1.1, "n_fft": 2048}


def convert(sig: np.ndarray, sr: int,
            scale: float = 1.1,
            n_fft: int   = 2048,
            hop_length: int = None,
            **_) -> np.ndarray:
    """
    Args:
        sig   : 单声道浮点音频
        sr    : 采样率
        scale : 频率轴缩放因子（>1 上移共振峰，<1 下移）
        n_fft : STFT 窗长（默认 2048）
    """
    if hop_length is None:
        hop_length = n_fft // 4

    S      = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)
    mag    = np.abs(S)
    n_bins = mag.shape[0]   # n_fft // 2 + 1

    # 每个输出 bin k 读取源谱位置 k / scale
    src_pos = np.arange(n_bins, dtype=np.float32) / scale
    src_pos = np.clip(src_pos, 0.0, n_bins - 1.0)

    lo   = src_pos.astype(np.int32)
    hi   = np.minimum(lo + 1, n_bins - 1)
    frac = (src_pos - lo)[:, np.newaxis]      # (n_bins, 1)

    mag_scaled = mag[lo] * (1.0 - frac) + mag[hi] * frac   # (n_bins, n_frames)
    S_new      = mag_scaled * np.exp(1j * np.angle(S))

    return librosa.istft(S_new, hop_length=hop_length, length=len(sig))
