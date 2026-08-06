"""
WhisperX ASR 模块（多语言，含字级对齐 + 说话人分离）

接口：
    load_model(language, hotwords, spk_num, logger) -> (model, generate_kwargs)
    recognize(model, generate_kwargs, audio_path, logger) -> List[dict]

recognize 返回统一格式：
    [{"index": int, "text": str, "timestamp": [ms, ms, ...], "speaker": int}, ...]

timestamp 单位：毫秒（与其他模块保持一致）

说话人分离依赖 pyannote.audio，需要：
  1. pip install pyannote.audio（whisperx 环境）
  2. 设置环境变量 HF_TOKEN（HuggingFace access token）
  3. 在 HuggingFace 上同意 pyannote/speaker-diarization-community-1 的使用协议
  首次运行会自动下载 diarization 模型到 DIARIZE_MODEL_PATH。
"""




import os
import re
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# 离线模式和镜像端点由 .env 控制，不在代码中硬编码
# 用户可在 .env 中设置：
#   HF_HUB_OFFLINE=1          — 离线模式（已有模型时启用）
#   TRANSFORMERS_OFFLINE=1    — Transformers 离线模式
#   HF_ENDPOINT=https://hf-mirror.com  — HuggingFace 镜像（国内用户）

MODEL_PATH       = os.path.join(_PROJECT_ROOT, "models/whisperx")
ALIGN_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/whisperx/align")
DIARIZE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/whisperx/")  # pyannote diarization 模型缓存
WHISPER_MODEL    = "large-v3"
BATCH_SIZE       = 16
COMPUTE_TYPE     = "float16"

TO_LANGUAGE_CODE = {
    # 中文名
    "中文": "zh", "英语": "en", "德语": "de", "西班牙语": "es",
    "俄语": "ru", "韩语": "ko", "法语": "fr", "日文": "ja", "日语": "ja",
    "葡萄牙语": "pt", "土耳其语": "tr", "波兰语": "pl", "加泰罗尼亚语": "ca",
    "荷兰语": "nl", "阿拉伯语": "ar", "瑞典语": "sv", "意大利语": "it",
    "印度尼西亚语": "id", "印地语": "hi", "芬兰语": "fi", "越南语": "vi",
    "希伯来语": "he", "乌克兰语": "uk", "希腊语": "el", "马来语": "ms",
    "捷克语": "cs", "罗马尼亚语": "ro", "丹麦语": "da", "匈牙利语": "hu",
    "泰米尔语": "ta", "挪威语": "no", "泰语": "th", "乌尔都语": "ur",
    "克罗地亚语": "hr", "保加利亚语": "bg", "立陶宛语": "lt", "拉丁语": "la",
    "毛利语": "mi", "马拉雅拉姆语": "ml", "威尔士语": "cy", "斯洛伐克语": "sk",
    "泰卢固语": "te", "波斯语": "fa", "拉脱维亚语": "lv", "孟加拉语": "bn",
    "塞尔维亚语": "sr", "阿塞拜疆语": "az", "斯洛文尼亚语": "sl",
    "卡纳达语": "kn", "爱沙尼亚语": "et", "马其顿语": "mk", "布列塔尼语": "br",
    "巴斯克语": "eu", "冰岛语": "is", "亚美尼亚语": "hy", "尼泊尔语": "ne",
    "蒙古语": "mn", "波斯尼亚语": "bs", "哈萨克语": "kk", "阿尔巴尼亚语": "sq",
    "斯瓦希里语": "sw", "加利西亚语": "gl", "马拉地语": "mr", "旁遮普语": "pa",
    "僧伽罗语": "si", "高棉语": "km", "绍纳语": "sn", "约鲁巴语": "yo",
    "索马里语": "so", "南非荷兰语": "af", "奥克语": "oc", "格鲁吉亚语": "ka",
    "白俄罗斯语": "be", "塔吉克语": "tg", "信德语": "sd", "古吉拉特语": "gu",
    "阿姆哈拉语": "am", "意第绪语": "yi", "老挝语": "lo", "乌兹别克语": "uz",
    "法罗语": "fo", "海地克里奥尔语": "ht", "普什图语": "ps", "土库曼语": "tk",
    "挪威尼诺斯克语": "nn", "马耳他语": "mt", "梵语": "sa", "卢森堡语": "lb",
    "缅甸语": "my", "藏语": "bo", "他加禄语": "tl", "马达加斯加语": "mg",
    "阿萨姆语": "as", "鞑靼语": "tt", "夏威夷语": "haw", "林加拉语": "ln",
    "豪萨语": "ha", "巴什基尔语": "ba", "爪哇语": "jw", "巽他语": "su",
    "粤语": "yue",
    # 鲁棒的中文名
    "汉语" : "zh", "国语" : "zh", "普通话": "zh", "英文": "en", "德文": "de",
    # 英文名（LLM 可能输出英文语言名）
    "chinese": "zh", "english": "en", "german": "de", "spanish": "es",
    "russian": "ru", "korean": "ko", "french": "fr", "japanese": "ja",
    "portuguese": "pt", "turkish": "tr", "polish": "pl", "dutch": "nl",
    "arabic": "ar", "swedish": "sv", "italian": "it", "indonesian": "id",
    "hindi": "hi", "finnish": "fi", "vietnamese": "vi", "hebrew": "he",
    "ukrainian": "uk", "greek": "el", "malay": "ms", "czech": "cs",
    "romanian": "ro", "danish": "da", "hungarian": "hu", "tamil": "ta",
    "norwegian": "no", "thai": "th", "urdu": "ur", "croatian": "hr",
    "bulgarian": "bg", "lithuanian": "lt", "latin": "la",
    # ISO 639-1 语言代码（直接透传）
    "zh": "zh", "en": "en", "de": "de", "es": "es", "ru": "ru",
    "ko": "ko", "fr": "fr", "ja": "ja", "pt": "pt", "tr": "tr",
    "pl": "pl", "nl": "nl", "ar": "ar", "sv": "sv", "it": "it",
    "id": "id", "hi": "hi", "fi": "fi", "vi": "vi", "he": "he",
    "uk": "uk", "el": "el", "ms": "ms", "cs": "cs", "ro": "ro",
    "da": "da", "hu": "hu", "ta": "ta", "no": "no", "th": "th",
    "ur": "ur", "hr": "hr", "bg": "bg", "lt": "lt", "la": "la",
    "yue": "yue",
}


def load_model(language, hotwords, spk_num, logger, min_speakers=None, max_speakers=None):
    import torch
    import whisperx as _whisperx

    lang_code = TO_LANGUAGE_CODE.get(language, language)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 优先用 min/max_speakers；若未指定则用 spk_num 同时设 min 和 max
    if min_speakers is None and max_speakers is None and spk_num is not None:
        min_speakers = spk_num
        max_speakers = spk_num

    logger.info(f"加载 WhisperX 模型（{WHISPER_MODEL}，language={lang_code}）...")
    asr_model = _whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_PATH,
    )

    logger.info("加载 WhisperX align 模型...")
    model_align, metadata = _whisperx.load_align_model(
        language_code=lang_code,
        device=asr_model.device,
        model_dir=ALIGN_MODEL_PATH,
    )

    # ── 加载说话人分离模型 ─────────────────────────────────────────────────
    diarize_model = None
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            from whisperx.diarize import DiarizationPipeline
            os.makedirs(DIARIZE_MODEL_PATH, exist_ok=True)
            # pyannote 模型需要联网下载，临时关闭离线模式
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ.pop("HF_HUB_OFFLINE", None)
            logger.info("加载 WhisperX diarization 模型...")
            diarize_model = DiarizationPipeline(
                token=hf_token,
                device=device,
                cache_dir=DIARIZE_MODEL_PATH,
            )
            # 恢复离线模式
            if old_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = old_offline
            logger.info("diarization 模型加载完成")
        except Exception as e:
            logger.warning(f"diarization 模型加载失败（将不进行说话人分离）: {e}")
    else:
        logger.info("未设置 HF_TOKEN，跳过说话人分离模型加载")

    generate_kwargs = {
        'whisperx':    _whisperx,
        'model_align': model_align,
        'metadata':    metadata,
        'lang_code':   lang_code,
        'batch_size':  BATCH_SIZE,
        'hotwords':    hotwords,
        'diarize_model': diarize_model,
        'min_speakers': min_speakers,
        'max_speakers': max_speakers,
    }
    return asr_model, generate_kwargs


def recognize(model, generate_kwargs, audio_path, logger):
    wx          = generate_kwargs['whisperx']
    model_align = generate_kwargs['model_align']
    metadata    = generate_kwargs['metadata']
    lang_code   = generate_kwargs['lang_code']
    batch_size  = generate_kwargs['batch_size']
    diarize_model = generate_kwargs.get('diarize_model')
    min_speakers = generate_kwargs.get('min_speakers')
    max_speakers = generate_kwargs.get('max_speakers')

    wav = wx.load_audio(str(audio_path))

    raw = model.transcribe(wav, batch_size=batch_size, language=lang_code)
    logger.debug(f"WhisperX 原始结果:\n{raw}\n")

    aligned = wx.align(
        raw["segments"], model_align, metadata, wav,
        model.device, return_char_alignments=False,
    )
    logger.debug(f"WhisperX align 结果:\n{aligned}\n")

    # ── 说话人分离 ─────────────────────────────────────────────────────────
    if diarize_model is not None:
        try:
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_segments = diarize_model(wav, **diarize_kwargs)
            aligned = wx.assign_word_speakers(diarize_segments, aligned)
            logger.debug(f"WhisperX diarize 结果:\n{aligned}\n")
        except Exception as e:
            logger.warning(f"说话人分离失败（继续无 speaker 输出）: {e}")

    # ── 组装输出 ───────────────────────────────────────────────────────────
    result = []
    for index, seg in enumerate(aligned.get("segments", [])):
        words = seg.get("words", [])
        if words:
            start_ms = [int(word["start"] * 1000) for word in words]
            start_ms.append(int(words[-1]["end"] * 1000))
        else:
            # alignment 失败（静音/方言/无法对齐），退回到 segment 级时间戳
            start_ms = [int(seg["start"] * 1000), int(seg["end"] * 1000)]
            logger.debug(f"  segment[{index}] words 为空，退回 segment 时间戳: {seg['text']!r}")
        # speaker 标签转 int（"SPEAKER_00" → 0），无标签时默认 0
        spk = seg.get("speaker", "SPEAKER_00")
        if isinstance(spk, str) and spk.startswith("SPEAKER_"):
            try:
                speaker = int(spk.replace("SPEAKER_", ""))
            except ValueError:
                speaker = 0
        else:
            speaker = int(spk) if spk is not None else 0

        result.append({
            "index":     index,
            "text":      seg["text"],
            "timestamp": start_ms,
            "speaker":   speaker,
        })

    return result


if __name__ == "__main__":
    # 简单测试接口
    import sys
    import json
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from logger_example import setup_logger
    out_dir = '/path/to/Anonymise_Agent/demo/audio/output'
    input_wav = '/path/to/Anonymise_Agent/demo/audio/input/S001.wav'
    logger, _ = setup_logger(out_dir, "whisperx", "DEBUG")

    asr_model, generate_kwargs = load_model("英文", hotwords=None, spk_num=None, logger=logger)
    result = recognize(asr_model, generate_kwargs, input_wav, logger)
    save_path = os.path.join(out_dir, "example.json")
    with open(save_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    logger.debug(f"JSON 已保存: {save_path}")
