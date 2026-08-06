"""
Paraformer ASR 模块（中文，含说话人分离）

接口：
    load_model(language, hotwords, spk_num, logger) -> (model, generate_kwargs)
    recognize(model, generate_kwargs, audio_path, logger) -> List[dict]

recognize 返回统一格式：
    [{"index": int, "text": str, "timestamp": [ms, ms, ...], "speaker": int}, ...]
"""

MODEL_ID       = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_REVISION = "v2.0.4"
VAD_MODEL      = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_REVISION   = "v2.0.4"
PUNC_MODEL     = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
PUNC_REVISION  = "v2.0.4"
SPK_MODEL      = "iic/speech_campplus_sv_zh-cn_16k-common"
SPK_REVISION   = "v2.0.2"


def load_model(language, hotwords, spk_num, logger, min_speakers=None, max_speakers=None):
    from funasr import AutoModel

    # paraformer 只接受单个 spk_num（钉死后不可变）
    # 优先用显式 spk_num；若 min==max 则取该值；否则不设置（让模型自动判断）
    if spk_num is None:
        if min_speakers is not None and max_speakers is not None and min_speakers == max_speakers:
            spk_num = min_speakers
        # min != max 或只指定了一个 → 不设 spk_num，避免钉死错误人数

    logger.info("加载 Paraformer 模型...")
    model = AutoModel(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        vad_model=VAD_MODEL,
        vad_model_revision=VAD_REVISION,
        punc_model=PUNC_MODEL,
        punc_model_revision=PUNC_REVISION,
        spk_model=SPK_MODEL,
        spk_model_revision=SPK_REVISION,
        disable_update=True,
        device="cuda",
    )
    generate_kwargs = {
        'cache': {},
        'language': language,
        'hotwords': hotwords if hotwords else [],
        # 不加这两个会 AssertionError: assert len(segments) == len(labels)
        'batch_size_s': 300,
        'batch_size_token_threshold_s': 40,

        'sentence_timestamp': True,
        'use_itn': True,
    }
    if spk_num is not None:
        generate_kwargs['preset_spk_num'] = spk_num
    else:
        logger.info("未指定精确说话人数，paraformer 将自动判断说话人数")
    return model, generate_kwargs


def recognize(model, generate_kwargs, audio_path, logger):
    res = model.generate(input=str(audio_path), **generate_kwargs)[0]
    logger.debug(f"Paraformer 原始结果:\n{res}\n")

    result = []
    for index, sentence in enumerate(res.get("sentence_info", [])):
        if sentence.get("timestamp") is None:
            continue
        ts = sentence["timestamp"]
        start_times = [pair[0] for pair in ts]
        start_times.append(ts[-1][-1])
        result.append({
            "index": index,
            "text": sentence["text"],
            "speaker": sentence["spk"],
            "timestamp": start_times,
        })
    return result
