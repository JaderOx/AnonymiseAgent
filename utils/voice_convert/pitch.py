"""
utils/voice_convert/pitch.py — 相位声码器变调

使用 librosa.effects.pitch_shift，保持时长不变。
"""

import warnings
import numpy as np
import librosa

DESCRIPTION = "相位声码器变调（librosa），n_steps > 0 升调，< 0 降调（单位：半音）"
DEFAULTS    = {"n_steps": -3.0}


def convert(sig: np.ndarray, sr: int,
            n_steps: float = -3.0,
            **kwargs) -> np.ndarray:
    """
    Args:
        sig     : 单声道浮点音频
        sr      : 采样率
        n_steps : 变调量（半音），负数降调，正数升调
    """
    age = kwargs.get('age', 'N/A')
    gender = kwargs.get('gender', None)
    if gender in ("", "N/A"):
        gender = None
    if gender == "female":
        n_steps = - abs(n_steps)
    elif gender == "male":
        n_steps = abs(n_steps)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.effects.pitch_shift(sig, sr=sr, n_steps=n_steps)
