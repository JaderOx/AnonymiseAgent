"""
utils/voice_convert/__init__.py — VC 方法注册表

每个 VC 方法文件需导出：
  convert(sig, sr, **kwargs) -> np.ndarray   # 核心转换函数
  DEFAULTS: dict                              # 默认参数
  DESCRIPTION: str                            # 简短描述

注册新方法（DSP 或神经网络）只需：
  1. 在本目录新建方法文件（参照 formant.py）
  2. 在本文件末尾添加 import + register()
"""

from utils.vc_names import canonical_method

_REGISTRY: dict = {}


def register(name: str, fn=None, defaults: dict = None, description: str = "",
             batch_fn=None):
    """
    将 VC 方法注册到全局注册表。

    Args:
        fn       : convert(sig, sr, **kwargs) -> np.ndarray（DSP 方法）
        batch_fn : batch_convert_fn(input_dir, output_dir, logger, **kwargs)
                   （神经网络 VC，需要整批处理；设置后 fn 可为 None）
    """
    _REGISTRY[name] = {
        "fn":          fn,
        "batch_fn":    batch_fn,
        "defaults":    defaults or {},
        "description": description,
    }


def get(name: str) -> dict:
    """获取已注册方法的 entry dict，包含 fn / defaults / description。"""
    name = canonical_method(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"未知 VC 方法 '{name}'。当前可用：{list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_methods() -> list:
    """返回当前所有已注册方法名。"""
    return list(_REGISTRY.keys())


# ── DSP 方法注册 ─────────────────────────────────────────────
# 注意：register 名 = 论文方法名（与 output/middle 结果目录名一致）；
# 模块文件名仍是引擎实现身份（formant.py / mcadams.py / seedvc.py / fishaudio_tts.py）
from . import formant, pitch, mcadams, combined   # noqa: E402

register("formant_spec", formant.convert,  formant.DEFAULTS,  formant.DESCRIPTION)
register("pitch",        pitch.convert,    pitch.DEFAULTS,    pitch.DESCRIPTION)
register("formant_lpc",  mcadams.convert,  mcadams.DEFAULTS,  mcadams.DESCRIPTION)
register("combined",     combined.convert, combined.DEFAULTS, combined.DESCRIPTION)

# ── 神经网络 VC ──────────────────────────────────────────────
from . import seedvc   # noqa: E402
register("NeuralVC", batch_fn=seedvc.batch_convert_fn,
         defaults=seedvc.DEFAULTS, description=seedvc.DESCRIPTION)

# ── TTS 方法注册 ─────────────────────────────────────────────
from . import fishaudio_tts
register("Resyn", batch_fn=fishaudio_tts.batch_convert_fn,
         defaults=fishaudio_tts.DEFAULTS, description=fishaudio_tts.DESCRIPTION)

# ── 其他神经网络 VC 示例（按需取消注释）────────────────────────
# from . import rvc
# register("rvc", fn=rvc.convert, defaults=rvc.DEFAULTS, description=rvc.DESCRIPTION)
