"""
utils/voice_convert/mcadams.py — McAdams 幂次角度变换（OLA-LPC）

原理：对 LPC 极点角度做幂次变换 θ_new = sign(θ) * |θ|^alpha，
     改变共振峰频率分布。alpha < 1 时变化较弱，EER 改善有限。
     保留供对比和向后兼容，新实验推荐使用 combined 方法。
"""

import numpy as np
import librosa
from scipy.signal import tf2zpk, lfilter

DESCRIPTION = "McAdams 幂次角度变换（OLA-LPC，原方法，变化幅度弱）"
DEFAULTS    = {"lp_order": 20, "alpha": 0.8, "win_ms": 20, "shift_ms": 10}


def _ola_transform(sig: np.ndarray, sr: int, frame_fn, win_ms=20, shift_ms=10) -> np.ndarray:
    """通用 OLA 框架，对每帧调用 frame_fn(frame, sr)，异常时保留原帧。"""
    eps    = np.finfo(np.float32).eps
    sig    = sig + eps
    winlen = int(win_ms   * 1e-3 * sr)
    shift  = int(shift_ms * 1e-3 * sr)

    wPR = np.hanning(winlen)
    K   = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)

    Nframes = 1 + int((len(sig) - winlen) / shift)
    sig_rec = np.zeros(len(sig))

    for m in range(Nframes):
        s   = m * shift
        e   = min(s + winlen, len(sig))
        idx = np.arange(s, e)
        if len(idx) < winlen // 2:
            continue
        frame = sig[idx] * win[:len(idx)]
        try:
            frame_out = frame_fn(frame, sr)
        except Exception:
            frame_out = frame.copy()
        frame_out = frame_out * win[:len(frame_out)]
        sig_rec[idx] += frame_out

    return sig_rec


def convert(sig: np.ndarray, sr: int,
            lp_order: int   = 20,
            alpha:    float = 0.6,
            win_ms:   int   = 20,
            shift_ms: int   = 10,
            **_) -> np.ndarray:
    """
    Args:
        sig      : 单声道浮点音频
        sr       : 采样率
        lp_order : LPC 阶数
        alpha    : 幂次系数（< 1 时共振峰轻微下移）
        win_ms   : 分析窗长（ms）
        shift_ms : 帧移（ms）
    """
    eps = np.finfo(np.float32).eps

    def frame_fn(frame, sr):
        a = librosa.core.lpc(y=frame + eps, order=lp_order)
        _, poles, _ = tf2zpk([1], a)
        new_poles = poles.copy()
        for i, p in enumerate(poles):
            if not np.isreal(p):
                θ     = np.angle(p)
                new_θ = np.sign(θ) * (np.abs(θ) ** alpha)
                new_poles[i] = np.abs(p) * np.exp(1j * np.clip(new_θ, -np.pi, np.pi))
        a_new = np.real(np.poly(new_poles))
        res   = lfilter(a, [1], frame)
        return lfilter([1], a_new, res)

    return _ola_transform(sig, sr, frame_fn, win_ms, shift_ms)
