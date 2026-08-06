# 各种评估指标的计算。
# SNR MOS L1f0 PCCf0 CER Emo EER
import os
import time

from pathlib import Path

def _safe_merge(df_all, df_new):
    """将 df_new 合并到 df_all，任一为 None 则直接返回另一个。"""
    if df_new is None:
        return df_all
    if df_all is None:
        return df_new
    return df_all.merge(df_new, on="name", how="outer")




from utils.logger_example import setup_logger

def evaluate_all(
        input_dir_vc = None,
        trans_dir_vc = None,
        input_dir_origin = None,
        trans_dir_origin = None,
        middle_dir = None,
        asr_model = None,
        hotwords = None,
        language = None,
        output_dir = None,
        logger = None,
        logger_file = None,
        _trans_env = None,
        gt_dir = None,       # 隐私词 GT JSON 目录（可选，提供时计算 mask 准确率）
        subject_map_path = None,  # EER 计算时的 subject 映射文件（可选，提供时按 subject 级别计算 EER）
        mode = None,
        wer = None
        ):
    start_time = time.time()
    audio_dir = Path(input_dir_vc)
    if not audio_dir.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir}")
    
    audio_dir_origin = Path(input_dir_origin)
    if not audio_dir_origin.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir_origin}")
    
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    wav_files = sorted([p for p in audio_dir.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    wav_files_origin = sorted([p for p in audio_dir_origin.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    
    logger.info(f"开始评估指标\n原始音频目录：{audio_dir_origin}，文件数量：{len(wav_files_origin)}\n处理音频目录：{audio_dir}，文件数量：{len(wav_files)}\n")
    

    df_all = None

    df_snr = None
    try:
        snr_time0 = time.time()
        from utils.metrics.snr_calc import compute_directory_snr
        df_snr, mean_std_str = compute_directory_snr(wav_files, logger, base_dir=audio_dir)
        df_all = _safe_merge(df_all, df_snr)
        logger.info(f"SNR结果{mean_std_str}, 耗时：{time.time() - snr_time0:.2f}秒\n")
    except Exception as e:
        import traceback
        logger.error(f"SNR模块计算失败：{e}\n{traceback.format_exc()}")

    try:
        mos_time0 = time.time()
        from utils.metrics.mos_calc import compute_directory_mos
        df_mos, mean_std_str = compute_directory_mos(wav_files, logger, base_dir=audio_dir)
        df_all = _safe_merge(df_all, df_mos)
        logger.info(f"MOS结果{mean_std_str}, 耗时：{time.time() - mos_time0:.2f}秒\n")
    except Exception as e:
        import traceback
        logger.error(f"MOS模块计算失败：{e}\n{traceback.format_exc()}")

    try:
        cer_time0 = time.time()
        from utils.metrics.cer_calc import compute_directory_cer
        if not trans_dir_vc or not trans_dir_origin:
            logger.info("未提供转录目录，跳过 CER/WER 计算")
            df_cer = None
            result_str = "跳过（无转录）"
        else:
            df_cer, result_str = compute_directory_cer(
                vc_wav_dir = audio_dir,
                ori_files = wav_files_origin,
                ori_trans = trans_dir_origin,
                vc_files = wav_files,
                vc_trans = trans_dir_vc,
                asr_model = asr_model,
                hotwords = hotwords,
                language = language,
                logger = logger,
                logger_file = logger_file,
                _trans_env = _trans_env,
                wer = wer
            )
        df_all = _safe_merge(df_all, df_cer)
        logger.info(f"result:{result_str}, 耗时：{time.time() - cer_time0:.2f}秒\n")
        # 清理一下模型占用显存（尤其是多个模型时）
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        logger.error(f"CER模块计算失败：{e}\n{traceback.format_exc()}")

    # 说话人音频分割（EMO，EER，pit 依赖此步）
    seg0 = time.time()
    logger.info("检查说话人片段是否已存在")
    if middle_dir is None and trans_dir_origin:
        middle_dir = os.path.dirname(trans_dir_origin)

    # 确定说话人片段目录：优先使用 agent.py 生成的 origin/ 和 text_{vc}/
    spk_origin_dir = os.path.join(middle_dir, 'spk_audio', 'origin') if middle_dir else None

    # 根据 mode 确定 vc 侧目录
    if mode and mode.startswith('text_'):
        # 评估 text_{vc}，使用 agent.py 生成的 text_{vc}/
        spk_vc_dir = os.path.join(middle_dir, 'spk_audio', f'{mode}')
    elif mode == 'text':
        # 评估 text_only，需要单独提取
        spk_vc_dir = os.path.join(middle_dir, 'spk_audio', 'text_only')
    else:
        # 评估 vc_only，使用 agent.py 生成的 {vc}/ 进行提取
        spk_vc_dir = os.path.join(middle_dir, 'spk_audio', f'{mode}_only')

    has_spk = False
    spk_files_origin = []
    spk_files_vc = []

    if trans_dir_origin and os.path.exists(trans_dir_origin) and spk_origin_dir:
        try:
            from utils.metrics.spk_audio_extractor import extract_dir, extract_audio_dir
            SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}

            # 检查已存在的片段
            origin_existing = (
                sorted(p for p in Path(spk_origin_dir).rglob('*')
                       if p.suffix.lower() in SUPPORTED_FORMATS)
                if os.path.exists(spk_origin_dir) else []
            )
            vc_existing = (
                sorted(p for p in Path(spk_vc_dir).rglob('*')
                       if p.suffix.lower() in SUPPORTED_FORMATS)
                if os.path.exists(spk_vc_dir) and spk_vc_dir else []
            )

            # 都已存在，直接使用
            if origin_existing and vc_existing:
                logger.info(f"说话人片段已存在：origin={len(origin_existing)}, vc={len(vc_existing)}")
                spk_files_origin = origin_existing
                spk_files_vc = vc_existing
            # 都不存在，配对提取（确保时间戳对应）
            elif not origin_existing and not vc_existing and spk_vc_dir:
                logger.info(f"配对提取说话人片段：origin + {mode}")
                extract_dir(
                    input_dir_origin, input_dir_vc, trans_dir_origin,
                    spk_origin_dir, spk_vc_dir, logger
                )
                spk_files_origin = sorted(
                    p for p in Path(spk_origin_dir).rglob('*')
                    if p.suffix.lower() in SUPPORTED_FORMATS
                ) if os.path.exists(spk_origin_dir) else []
                spk_files_vc = sorted(
                    p for p in Path(spk_vc_dir).rglob('*')
                    if p.suffix.lower() in SUPPORTED_FORMATS
                ) if os.path.exists(spk_vc_dir) else []
            # 部分存在，补齐缺失
            else:
                if not origin_existing:
                    logger.info(f"提取 origin 说话人片段 → {spk_origin_dir}")
                    extract_audio_dir(input_dir_origin, trans_dir_origin, spk_origin_dir, logger)
                    spk_files_origin = sorted(
                        p for p in Path(spk_origin_dir).rglob('*')
                        if p.suffix.lower() in SUPPORTED_FORMATS
                    ) if os.path.exists(spk_origin_dir) else []
                else:
                    spk_files_origin = origin_existing

                if not vc_existing and spk_vc_dir:
                    logger.info(f"提取 {mode} 说话人片段 → {spk_vc_dir}")
                    extract_audio_dir(input_dir_vc, trans_dir_origin, spk_vc_dir, logger)
                    spk_files_vc = sorted(
                        p for p in Path(spk_vc_dir).rglob('*')
                        if p.suffix.lower() in SUPPORTED_FORMATS
                    ) if os.path.exists(spk_vc_dir) else []
                else:
                    spk_files_vc = vc_existing

            has_spk = bool(spk_files_origin and spk_files_vc)
            logger.info(
                f"说话人片段：origin={len(spk_files_origin)} 条，"
                f"vc={len(spk_files_vc)} 条  耗时 {time.time()-seg0:.2f}s\n"
            )
        except Exception as e:
            import traceback
            logger.error(f"说话人片段提取失败：{e}\n{traceback.format_exc()}")

    if has_spk:
        try:
            emo_time0 = time.time()
            from utils.metrics.emo_calc import compute_directory_emo
            df_emo, mean_std_str = compute_directory_emo(spk_files_vc, spk_files_origin,
                                                          spk_vc_dir, spk_origin_dir, logger)
            df_all = _safe_merge(df_all, df_emo)
            logger.info(f"EMO结果{mean_std_str}, 耗时：{time.time() - emo_time0:.2f}秒\n")
        except Exception as e:
            import traceback
            logger.error(f"EMO模块计算失败：{e}\n{traceback.format_exc()}")
    else:
        logger.warning("无说话人标注（trans_dir_origin 不存在或无 speaker 字段），跳过 EMO 计算\n")


    if has_spk:
        try:
            pit_t0 = time.time()
            from utils.metrics.pit_calc import compute_directory_pit
            df_pit, mean_std_str = compute_directory_pit(spk_files_origin, spk_files_vc, logger,
                                                          base_dir_origin=spk_origin_dir, base_dir_vc=spk_vc_dir)
            df_all = _safe_merge(df_all, df_pit)
            logger.info(f"Pitch结果{mean_std_str}, 耗时：{time.time() - pit_t0:.2f}秒\n")
        except Exception as e:
            import traceback
            logger.error(f"Pitch模块计算失败：{e}\n{traceback.format_exc()}")
    else:
        logger.warning("无说话人标注（trans_dir_origin 不存在或无 speaker 字段），跳过 Pitch 计算\n")

    if has_spk:
        try:
            eer0_time = time.time()
            from utils.metrics.eer_calc import compute_directory_eer
            eer, cil, cih = compute_directory_eer(spk_files_origin, spk_files_vc,
                                                    spk_origin_dir, spk_vc_dir, logger,
                                                    subject_map_path)
            if df_all is not None:
                df_all['eer'] = ''
                df_all.loc[df_all['name'] == '-mean-', 'eer'] = eer
                df_all.loc[df_all['name'] == '-std-', 'eer'] = f"[{cil:.2f}% - {cih:.2f}%]"
            logger.info(f"EER结果{eer:.2f}, 耗时：{time.time() - eer0_time:.2f}秒\n")
        except Exception as e:
            import traceback
            logger.error(f"EER模块计算失败：{e}\n{traceback.format_exc()}")
    else:
        logger.warning("无说话人标注（trans_dir_origin 不存在或无 speaker 字段），跳过 EER 计算\n")

    # ── Mask 准确率（需要 gt_dir） ───────────────────────────────────────────
    if gt_dir and os.path.isdir(gt_dir):
        try:
            acc_time0 = time.time()
            mask_dir_path = os.path.join(middle_dir, 'mask') if middle_dir else None
            if mask_dir_path and os.path.isdir(mask_dir_path):
                from utils.metrics.accuracy_calc import compute_directory_accuracy
                acc_output_dir = os.path.join(output_dir, 'mask_accuracy')
                df_acc, acc_str = compute_directory_accuracy(
                    gt_dir, mask_dir_path, acc_output_dir, logger
                )
                df_all = _safe_merge(df_all, df_acc)
                logger.info(f"Mask准确率{acc_str}, 耗时：{time.time() - acc_time0:.2f}秒\n")
            else:
                logger.warning(f"mask 目录不存在（{mask_dir_path}），跳过准确率计算")
        except Exception as e:
            import traceback
            logger.error(f"Mask准确率模块计算失败：{e}\n{traceback.format_exc()}")
    else:
        if gt_dir:
            logger.warning(f"gt_dir 不存在或不是目录：{gt_dir}，跳过准确率计算")
        else:
            logger.info("未提供 --gt_dir，跳过 mask 准确率计算")

    if df_all is None:
        logger.warning(f"所有指标均计算失败，无结果可保存, 耗时{time.time() - start_time:.2f}")
        return None

    csvname = f"evaluate_{mode}" if mode else logger.name if logger and logger.name else "evaluate_result"
    output_file = os.path.join(output_dir, f"{csvname}.csv")
    df_all.to_csv(output_file, index=False)
    logger.info(f"评估结果\n{df_all.to_string(index=False)}")
    logger.info(f"结果已保存到：{output_file},耗时{time.time() - start_time:.2f}")

    return df_all

def evaluate_origin(
        input_dir_origin = None,
        output_dir = None,
        logger = None
):

    audio_dir_origin = Path(input_dir_origin)
    if not audio_dir_origin.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir_origin}")
    
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    wav_files_origin = sorted([p for p in audio_dir_origin.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    
    logger.info(f"开始评估指标\n原始音频目录：{audio_dir_origin}，文件数量：{len(wav_files_origin)}\n")
    

    df_all = None

    df_snr = None
    try:
        snr_time0 = time.time()
        from utils.metrics.snr_calc import compute_directory_snr
        df_snr, mean_std_str = compute_directory_snr(wav_files_origin, logger, base_dir=audio_dir_origin)
        df_all = _safe_merge(df_all, df_snr)
        logger.info(f"SNR结果{mean_std_str}, 耗时：{time.time() - snr_time0:.2f}秒\n")
    except Exception as e:
        import traceback
        logger.error(f"SNR模块计算失败：{e}\n{traceback.format_exc()}")

    try:
        mos_time0 = time.time()
        from utils.metrics.mos_calc import compute_directory_mos
        df_mos, mean_std_str = compute_directory_mos(wav_files_origin, logger, base_dir=audio_dir_origin)
        df_all = _safe_merge(df_all, df_mos)
        logger.info(f"MOS结果{mean_std_str}, 耗时：{time.time() - mos_time0:.2f}秒\n")
    except Exception as e:
        import traceback
        logger.error(f"MOS模块计算失败：{e}\n{traceback.format_exc()}")

    csvname = f"evaluate_origin"
    output_file = os.path.join(output_dir, f"{csvname}.csv")
    df_all.to_csv(output_file, index=False)
    logger.info(f"评估结果\n{df_all.to_string(index=False)}")
    logger.info(f"结果已保存到：{output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="匿名效果评估")
    parser.add_argument("--input_dir_vc", required=True, help="匿名后音频目录")
    parser.add_argument("--input_dir_origin", required=True, help="原始音频目录")
    parser.add_argument("--trans_dir_vc", default=None, help="匿名后转录目录（可选，无转录时跳过CER）")
    parser.add_argument("--trans_dir_origin", default=None, help="原始转录目录（可选，无转录时跳过CER）")
    parser.add_argument("--middle_dir", default=None, help="中间结果目录（默认 input_dir_vc 的父目录/middle）")
    parser.add_argument("--asr_model", default="paraformer", help="ASR 模型名")
    parser.add_argument("--language", default="中文", help="语言")
    parser.add_argument("--output_dir", "-o", required=True, help="评估结果输出目录")
    parser.add_argument("--hotwords", default=None, help="热词，逗号分隔")
    parser.add_argument("--gt_dir", default=None, help="隐私词 GT JSON 目录（可选）")
    parser.add_argument("--subject_map_path", default=None, help="EER subject 映射文件（可选）")
    parser.add_argument("--mode_str", default="", help="评估模式标识（如 text_seedvc）")
    parser.add_argument("--wer", action="store_true", help="是否计算 WER")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO", help="日志级别")
    args = parser.parse_args()

    hotwords_list = args.hotwords.split(",") if args.hotwords else []
    middle_dir = args.middle_dir or os.path.join(os.path.dirname(os.path.abspath(args.input_dir_vc)), "middle")

    logger, logger_file = setup_logger(args.output_dir, "evaluate", args.mode)
    evaluate_all(
        input_dir_vc=args.input_dir_vc,
        trans_dir_vc=args.trans_dir_vc,
        input_dir_origin=args.input_dir_origin,
        trans_dir_origin=args.trans_dir_origin,
        middle_dir=middle_dir,
        asr_model=args.asr_model,
        hotwords=hotwords_list,
        language=args.language,
        output_dir=args.output_dir,
        logger=logger,
        logger_file=logger_file,
        _trans_env=None,
        gt_dir=args.gt_dir,
        subject_map_path=args.subject_map_path,
        mode=args.mode_str or None,
        wer=True if args.wer else None,
    )
