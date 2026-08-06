"""
FunASR-MLT-Nano ASR 模块（多语言，字级时间戳转句级）

接口：
    load_model(language, hotwords, spk_num, logger) -> (model, generate_kwargs)
    recognize(model, generate_kwargs, audio_path, logger) -> List[dict]

recognize 返回统一格式：
    [{"index": int, "text": str, "timestamp": [ms, ms, ...]}, ...]
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

MODEL_PATH  = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
SYS_PATH    = os.path.join(_PROJECT_ROOT, "models/FunAudioLLM/Fun-ASR-MLT-Nano-2512")
VAD_MODEL   = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def load_model(language, hotwords, spk_num, logger, **kwargs):
    if SYS_PATH not in sys.path:
        sys.path.append(SYS_PATH)

    from funasr import AutoModel

    logger.info("加载 FunASRMLTNano 模型...")
    model = AutoModel(
        model=MODEL_PATH,
        vad_model=VAD_MODEL,
        vad_kwargs={'max_single_segment_time': 30000},
        disable_update=True,
        trust_remote_code=True,
        device="cuda",
    )
    generate_kwargs = {
        'cache': {},
        'language': language,
        'hotwords': hotwords if hotwords else [],
        'batch_size_s': 0,
    }
    return model, generate_kwargs


def recognize(model, generate_kwargs, audio_path, logger):
    from utils.word2sentence import convert_words_to_sentences

    res = model.generate(input=str(audio_path), **generate_kwargs)[0]
    logger.debug(f"FunASRMLTNano 原始结果:\n{res}\n")

    result = convert_words_to_sentences(
        res.get("timestamps", []),
        res.get("text", ""),
    )
    logger.debug(f"转换为句子级格式后的结果:\n{result}\n")
    return result
