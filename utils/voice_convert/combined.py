"""
utils/voice_convert/combined.py — 共振峰缩放 + 变调叠加（推荐）

先做 formant 缩放，再做 pitch 变调，两个维度同时攻击 speaker embedding。
支持 gender-adaptive 变调：句子 JSON 含 gender 字段时自动翻转方向，
使男声降调、女声升调，最大化与原始嗓音的距离。
"""

import numpy as np
from .formant import convert as formant_convert
from .pitch   import convert as pitch_convert

DESCRIPTION = "共振峰缩放 + 变调叠加（推荐），支持性别自适应变调方向"
DEFAULTS    = {
    "formant_scale": 1.05,
    "pitch_steps":  -2.0,   # gender 字段存在时自动决定正负
    "n_fft":        2048,
}


def convert(sig: np.ndarray, sr: int,
            formant_scale: float = 1.1,
            pitch_steps:   float = -3.0,
            n_fft:         int   = 2048,
            hop_length:    int   = None,
            gender:        str   = None,
            **_) -> np.ndarray:
    """
    Args:
        sig           : 单声道浮点音频
        sr            : 采样率
        formant_scale : 共振峰缩放因子（>1 上移，推荐 1.05~1.20）
        pitch_steps   : 变调量（半音）；gender 存在时按性别决定方向
        n_fft         : STFT 窗长（传给 formant）
        gender        : 'male' / 'female'（来自转录 JSON）；None 时使用固定 pitch_steps
    """

    sig_f = formant_convert(sig, sr, scale=formant_scale,
                            n_fft=n_fft, hop_length=hop_length)
    return pitch_convert(sig_f, sr, n_steps=pitch_steps, gender=gender)
