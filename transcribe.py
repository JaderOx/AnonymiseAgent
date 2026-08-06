import os
import json
import time
import importlib
from pathlib import Path
from utils.logger_example import setup_logger

# ── 直接运行时的默认参数 ──────────────────────────────────────────
wav_dir              = ''   # 直接 python transcribe.py 运行时填入；经 agent.py 调用时由参数传入
trans_dir            = ''
asr_model            = 'whisperx'   # paraformer(zh) / FunASRNano / FunASRMLTNano / whisperx
language             = 'en'
hotwords             = []
spk_audio_origin_dir = ''
_age_gender_model    = './models/w2v2-L-robust-6-age-gender'
spk_num              = 1
mode                 = 'DEBUG'

# 已知模型名 → 子模块映射（utils/transcribe/<module>.py）
_MODEL_MODULES = {
    'paraformer':    'utils.transcribe.paraformer',
    'FunASRNano':    'utils.transcribe.FunASRNano',
    'FunASRMLTNano': 'utils.transcribe.FunASRMLTNano',
    'whisperx':      'utils.transcribe.whisperx_',
}


def _post_transcribe_spk(wav_path, result_save, spk_audio_dir, age_gender_model_path, logger, rel_path=None):
    """
    转录后处理：
      1. 按 speaker 字段提取各说话人音频（最多 60s）→ spk_audio_dir
      2. 若提供 age_gender_model_path，预测年龄/性别

    Returns:
        dict: {str(spk_id): {'age': float, 'gender': str}}，无信息时返回 {}
    """
    import numpy as np
    import soundfile as sf
    from collections import defaultdict

    if not result_save:
        logger.debug("_post_transcribe_spk: 无转录结果，对全音频预测年龄/性别")
        if not (age_gender_model_path and os.path.isdir(age_gender_model_path)):
            logger.debug("  age_gender_model_path 不可用，跳过")
            return {}
        try:
            import numpy as np
            import soundfile as sf
            sig, sr = sf.read(wav_path)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            sig = sig.astype(np.float32)
            from utils.age_gender import AgeGenderPredictor
            predictor = AgeGenderPredictor(age_gender_model_path, device='cpu')
            age_val, logits = predictor.predict(sig, sr)
            logits = np.asarray(logits, dtype=np.float32)
            gender_str = 'female' if logits[0] >= logits[1] else 'male'
            stem = Path(wav_path).stem
            logger.info(f"{stem}: 转录为空，全音频预测 → age={age_val:.1f} gender={gender_str}")
            return {'0': {'age': round(float(age_val), 1), 'gender': gender_str}}
        except Exception as e:
            logger.warning(f"_post_transcribe_spk: 全音频年龄/性别预测失败: {e}")
            return {}

    stem = Path(wav_path).stem
    logger.debug(f"_post_transcribe_spk: {stem}  spk_audio_dir={spk_audio_dir}")

    # ── 按说话人分组句子 ─────────────────────────────────────────────────────────
    spk_segs = defaultdict(list)
    for sent in result_save:
        spk_id = sent.get('speaker', 0)
        ts     = sent.get('timestamp', [])
        if spk_id is None or len(ts) < 2:
            continue
        start_ms, end_ms = ts[0], ts[-1]
        dur = (end_ms - start_ms) / 1000.0
        if dur < 1.0:
            continue
        spk_segs[spk_id].append((start_ms, end_ms, dur))

    if not spk_segs:
        logger.debug("_post_transcribe_spk: 无有效分组，跳过")
        return {}

    logger.debug(f"  识别到 {len(spk_segs)} 位说话人: {list(spk_segs)}")

    # ── 加载音频 ──────────────────────────────────────────────────────────────
    try:
        sig, sr = sf.read(wav_path)
    except Exception as e:
        logger.warning(f"_post_transcribe_spk: 读取音频失败 {e}")
        return {}
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)
    logger.debug(f"  音频时长: {len(sig)/sr:.1f}s  sr={sr}")

    # ── 初始化年龄/性别预测器 ────────────────────────────────────────────────
    predictor = None
    if age_gender_model_path and os.path.isdir(age_gender_model_path):
        try:
            from utils.age_gender import AgeGenderPredictor
            predictor = AgeGenderPredictor(age_gender_model_path, device='cpu')
            logger.debug("  年龄/性别模型加载完成（cpu）")
        except Exception as e:
            logger.warning(f"  年龄/性别模型加载失败: {e}，跳过预测")
    else:
        logger.debug("  age_gender_model_path 不可用，跳过年龄/性别预测")

    # 确定输出子目录（spk_audio_dir 为 None 时跳过写盘，仅做预测）
    sub_dir = None
    if spk_audio_dir is not None:
        if rel_path:
            sub_dir = os.path.join(spk_audio_dir, os.path.dirname(rel_path))
        else:
            sub_dir = spk_audio_dir
        os.makedirs(sub_dir, exist_ok=True)

    spk_ag = {}

    # ── 逐说话人：提取音频 + 预测 ──────────────────────────────────────────────
    for spk_id, segs in spk_segs.items():
        segs_top = sorted(segs, key=lambda x: x[2], reverse=True)[:20]
        segs_top = sorted(segs_top, key=lambda x: x[0])

        chunks = []
        total_sec = 0.0
        for start_ms, end_ms, _ in segs_top:
            if total_sec >= 60.0:
                break
            s_s = int(start_ms / 1000.0 * sr)
            e_s = min(int(end_ms / 1000.0 * sr), len(sig))
            if e_s <= s_s:
                continue
            chunk = sig[s_s:e_s]
            chunks.append(chunk)
            total_sec += len(chunk) / sr

        if not chunks:
            logger.debug(f"  spk{spk_id}: 无有效块")
            continue

        # 写盘（仅当 spk_audio_dir 可用时）
        if sub_dir is not None:
            out_path = os.path.join(sub_dir, f"{stem}_spk{spk_id}.wav")
            merged = np.concatenate(chunks)
            if not os.path.exists(out_path):
                sf.write(out_path, merged, sr)
                logger.debug(f"  spk{spk_id}: {total_sec:.1f}s → {out_path}")
            else:
                logger.debug(f"  spk{spk_id}: 已存在 {out_path}，跳过写入")

        if predictor is not None:
            ages, logit_list = [], []
            for chunk in chunks:
                if len(chunk) / sr < 1.0:
                    continue
                try:
                    age_i, logits_i = predictor.predict(chunk, sr)
                    ages.append(float(age_i))
                    logit_list.append(np.asarray(logits_i, dtype=np.float32))
                except Exception as e:
                    logger.debug(f"  spk{spk_id} 某句预测失败: {e}")
            if ages:
                avg_age    = float(np.mean(ages))
                avg_logits = np.mean(logit_list, axis=0)
                gender_str = 'female' if avg_logits[0] >= avg_logits[1] else 'male'
                spk_ag[spk_id] = {'age': round(avg_age, 1), 'gender': gender_str}
                logger.debug(f"  spk{spk_id}: {len(ages)}句平均 → age={avg_age:.1f}  gender={gender_str}")
            else:
                logger.warning(f"  spk{spk_id}: 所有句子预测失败")

    spk_ag_str = {str(k): v for k, v in spk_ag.items()}
    if spk_ag_str:
        logger.info(
            f"{stem}: 已为 {len(spk_ag_str)} 位说话人预测年龄/性别 → "
            + ", ".join(f"spk{k}={v['gender']}/{v['age']}岁" for k, v in spk_ag_str.items())
        )
    return spk_ag_str


def transcribe(
        wav_dir=None,
        asr_model=None,
        trans_dir=None,
        hotwords=None,
        language=None,
        logger=None,
        spk_audio_dir=None,
        age_gender_model_path=None,
        spk_num=None,
        min_speakers=None,
        max_speakers=None,
        num_files=None,
):
    if asr_model is not None and asr_model in _MODEL_MODULES:
        logger.info(f"使用指定的 ASR 模型: {asr_model}")
    else:
        # 根据 language 查配置表自动推断 ASR 模型
        from configs.config_loader import get_asr_config
        asr_cfg = get_asr_config(args.language)
        asr_model = asr_cfg['model']
        if asr_cfg['is_default']:
            logger.warning(
                f"语言 '{args.language}' 未在 configs/language_asr_map.yaml 中配置，"
                f"使用默认模型: {asr_model}"
            )
    # 防止 LLM 误传文件路径（如 .json 或单个音频）作为输入目录
    wav_dir_p = Path(wav_dir)
    supported = {'.wav', '.mp3', '.flac', '.ogg'}
    if wav_dir_p.is_file():
        logger.warning(f"输入路径是文件而非目录，只转录该文件: {wav_dir}")
        audio_files = [wav_dir_p]
        wav_dir_p = wav_dir_p.parent
    else:
        audio_files = sorted(p for p in wav_dir_p.rglob('*') if p.suffix.lower() in supported)

    # 防止 LLM 误传 .json 文件路径作为输出目录
    if trans_dir.endswith('.json') or trans_dir.endswith('.txt'):
        logger.warning(f"输出路径是 .json 或 .txt 文件，自动修正为所在目录: {os.path.dirname(trans_dir)}")
        trans_dir = os.path.dirname(trans_dir)

    # ── 增量跳过：输出目录 JSON 数量 >= 输入音频数量则跳过 ──
    trans_dir_p = Path(trans_dir)
    if trans_dir_p.exists():
        existing_json = [f for f in trans_dir_p.rglob('*.json') if f.name != 'args.json']
        if len(existing_json) >= len(audio_files):
            logger.info(
                f"[跳过] 转录结果已存在（{len(existing_json)} 条 >= 输入 {len(audio_files)} 条），不加载模型"
            )
            return

    start_time = time.time()
    try:
        os.makedirs(trans_dir, exist_ok=True)

        # ── 动态加载对应模型子模块 ────────────────────────────────────────────
        module_name = _MODEL_MODULES.get(asr_model)
        if module_name is None:
            raise ValueError(
                f"未知 ASR 模型: '{asr_model}'，已知模型: {list(_MODEL_MODULES)}"
            )
        asr_module = importlib.import_module(module_name)

        model, generate_kwargs = asr_module.load_model(language, hotwords, spk_num, logger,
                                                         min_speakers=min_speakers, max_speakers=max_speakers)
        logger.info(
            f"转录模型加载完毕，耗时: {time.time() - start_time:.2f} 秒\n"
            f"处理语音目录: {wav_dir}\n"
            f"转录结果目录: {trans_dir}\n"
        )
        logger.debug(f"generate_kwargs: {generate_kwargs}\n")
        if num_files is not None and num_files > 0:
            audio_files = audio_files[:num_files]
        all_len = len(audio_files)
        if all_len == 0:
            logger.warning(f"在目录 {wav_dir} 中未找到合法音频文件，支持格式: {supported}")
            return
        count = 0

    except Exception as e:
        import traceback
        logger.error(f"trans步骤: 初始化失败: {e}\n{traceback.format_exc()}\n")
        return

    for audio_path in audio_files:
        rel      = audio_path.relative_to(wav_dir_p)
        sub      = str(rel)
        wav_name = str(audio_path)

        file_start = time.time()
        count += 1
        logger.info(f"转录文件[{count}/{all_len}]: {sub}")

        save_path = Path(trans_dir) / rel.with_suffix('.json')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.is_file():
            logger.info(f"{sub} 已存在，跳过")
            continue

        try:
            result_save = asr_module.recognize(model, generate_kwargs, audio_path, logger)

            # ── 说话人音频提取 + 年龄/性别预测 ───────────────────────────────
            # 不论是否有 spk_audio_dir，只要有 age_gender_model_path 就会尝试预测。
            # 转录结果为空时走全音频预测分支（speaker_info={'0': {age, gender}}）。
            # 转录正常时按句子分段提取 spk_audio 并预测各说话人。
            speaker_info = {}
            if age_gender_model_path or spk_audio_dir is not None:
                speaker_info = _post_transcribe_spk(
                    wav_name, result_save,
                    spk_audio_dir, age_gender_model_path, logger,
                    rel_path=str(rel),
                )

            # ── 写 JSON ───────────────────────────────────────────────────────
            if speaker_info:
                save_data = {"speaker_info": speaker_info, "sentences": result_save}
            else:
                save_data = result_save
            with open(save_path, 'w') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4)
            logger.debug(f"JSON 已保存: {save_path}")
            logger.info(f"文件 {sub} 转录完毕，耗时 {time.time() - file_start:.2f}秒")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            fail_log = Path(trans_dir) / "logs" / "fail_list.txt"
            fail_log.parent.mkdir(parents=True, exist_ok=True)
            logger.error(f"转录文件 \"{sub}\" 失败: {e}\n{error_trace}\n")
            with open(fail_log, 'a') as f:
                f.write(sub + '\n')

    logger.info(f"所有文件转录完毕,耗时{time.time() - start_time:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASR 转录")
    parser.add_argument("--input_dir", "-i", required=True, help="输入音频目录")
    parser.add_argument("--output_dir", "-o", required=True, help="转录结果输出目录")
    parser.add_argument("--asr_model", default="paraformer",
                        help="ASR 模型: paraformer/FunASRNano/FunASRMLTNano/whisperx")
    parser.add_argument("--language", default="中文", help="语言（中文/英文/日文 等）")
    parser.add_argument("--hotwords", default=None, help="热词，逗号分隔")
    parser.add_argument("--spk_audio_dir", default=None, help="说话人音频提取目录")
    parser.add_argument("--age_gender_model", default="./models/w2v2-L-robust-6-age-gender",
                        help="年龄/性别模型路径")
    parser.add_argument("--spk_num", type=int, default=None, help="精确说话人数（0=自动）")
    parser.add_argument("--min_speakers", type=int, default=None, help="最少说话人数")
    parser.add_argument("--max_speakers", type=int, default=None, help="最多说话人数")
    parser.add_argument("--num_files", type=int, default=None, help="只转录前 N 个文件（默认全部）")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO", help="日志级别")
    args = parser.parse_args()

    hotwords_list = args.hotwords.split(",") if args.hotwords else []
    logger, _ = setup_logger(args.output_dir, "transcribe", args.mode)
    transcribe(
        wav_dir=args.input_dir,
        asr_model=args.asr_model,
        trans_dir=args.output_dir,
        language=args.language,
        hotwords=hotwords_list,
        logger=logger,
        spk_audio_dir=args.spk_audio_dir,
        age_gender_model_path=args.age_gender_model,
        spk_num=args.spk_num,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_files=args.num_files,
    )
