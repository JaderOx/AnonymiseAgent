import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # 读取 .env 文件（如存在）

from utils.logger_example import setup_logger, get_logger
from utils.manage_module_memory import run_in_isolation

# 从环境变量读取，未设置时使用默认值
os.environ['MODELSCOPE_CACHE'] = os.environ.get('MODELSCOPE_CACHE', '.')
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', './models')

SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
NO_WER_LANGUAGE = {"中文", "日文"}

# ── 模型 → conda 环境映射 ────────────────────────────────────────
# None 表示使用当前环境（Agent）；字符串表示切换到对应 conda 环境
# 新增模型/方法时在此处添加一行即可
_ASR_CONDA_ENV = {
    'paraformer':    None,
    'FunASRNano':    None,
    'FunASRMLTNano': None,
    'whisperx':      'whisperx',
}

_VC_CONDA_ENV = {
    'mcadams':      None,
    'formant':      None,
    'pitch':        None,
    'combined':     None,
    'seedvc':       None,
    # 子进程 conda 名：与 run_in_isolation 一致；若 CosyVoice 装在 fishaudio 环境里可改为 'fishaudio'
    'cosyvoice_tts': 'fishaudio',
    'fishaudio_tts': 'fishaudio',
}


# ── 工具函数 ────────────────────────────────────────────────────

def _has_audio(directory):
    """递归检查目录下是否存在音频文件"""
    p = Path(directory)
    if not p.exists():
        return False
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_FORMATS:
                return True
    return False

def _has_json(directory):
    """递归检查目录下是否存在 JSON 文件"""
    p = Path(directory)
    if not p.exists():
        return False
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix == '.json':
                return True
    return False

def _require_audio(directory, desc, logger):
    if not _has_audio(directory):
        logger.error(f"{desc}：目录不存在或无音频文件 → {directory}")
        sys.exit(1)

def _require_json(directory, desc, logger):
    if not _has_json(directory):
        logger.error(f"{desc}：目录不存在或无 JSON 文件 → {directory}")
        sys.exit(1)

# ── 子进程任务（LLM 内存隔离） ──────────────────────────────────

def mask(trans_dir, mask_dir, logger_file, language, prompt, llm_model, group_num, mode, mask_method='llm'):
    logger = get_logger(logger_file, 'mask', mode)

    # ── 增量跳过：输出目录 JSON 数量 >= 输入 JSON 数量则跳过 ──
    from pathlib import Path as _Path
    _mask_dir_p = _Path(mask_dir)
    _trans_dir_p = _Path(trans_dir)
    if _mask_dir_p.exists() and _trans_dir_p.exists():
        _input_jsons = [f for f in _trans_dir_p.rglob('*.json') if f.name != 'args.json']
        _output_jsons = [f for f in _mask_dir_p.rglob('*.json') if f.name != 'args.json']
        if len(_output_jsons) >= len(_input_jsons) > 0:
            logger.info(
                f"[跳过] 内容匿名结果已存在（{len(_output_jsons)} 条 >= 输入 {len(_input_jsons)} 条），不加载 LLM"
            )
            return
    if mask_method == 'ner':
        from mask_ner import mask_ner
        mask_ner(
            trans_dir=trans_dir,
            mask_dir=mask_dir,
            logger=logger,
            language=language,
            group_size=group_num,
        )
    else:
        from mask_llm import mask_llm
        mask_llm(
            trans_dir=trans_dir,
            mask_dir=mask_dir,
            logger=logger,
            language=language,
            prompt=prompt,
            llm_model=llm_model,
            group_size=group_num
        )

def trans(wav_dir, asr_model, trans_dir, hotwords, language, logger_file, mode, spk_audio_dir=None, age_gender_model_path=None, spk_num=None, min_speakers=None, max_speakers=None):
    logger = get_logger(logger_file, 'trans', mode)
    from transcribe import transcribe
    transcribe(
        wav_dir=wav_dir,
        asr_model=asr_model,
        trans_dir=trans_dir,
        hotwords=hotwords,
        language=language,
        logger=logger,
        spk_audio_dir=spk_audio_dir,
        age_gender_model_path=age_gender_model_path,
        spk_num=spk_num,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def vc(input_dir, output_dir, method, trans_dir, logger_file, mode,
       num_workers, mask_dir, mask_method, language, whole_audio=False):
    logger = get_logger(logger_file, 'vc', mode)
    from voice_convert import batch_convert

    # TTS 类方法（有 batch_fn，无 fn）不支持全音频整段模式
    _TTS_METHODS = {'fishaudio_tts', 'cosyvoice_tts'}
    if whole_audio and method in _TTS_METHODS:
        logger.error(
            f"VC 方法 '{method}' 是 TTS 方法，不支持全音频模式（--vc_whole），已跳过"
        )
        return
        
    batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        logger=logger,
        method=method,
        trans_dir=None if whole_audio else trans_dir,
        num_workers=num_workers,
        mask_dir=mask_dir,
        mask_method=mask_method,
        language=language,
    )


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='语音匿名 Agent')
    parser.add_argument('--input_dir', '-i', required=True,
                        help='输入音频目录')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='输出根目录，默认: input_dir 父目录/output')
    parser.add_argument('--middle_dir', default=None,
                        help='中间结果目录，默认: input_dir 父目录/middle')
    # 步骤开关
    parser.add_argument('--trans', action='store_true',
                        help='步骤1：ASR 转录')
    parser.add_argument('--mask', action='store_true',
                        help='步骤2：文本内容匿名（依赖 trans_origin）')
    parser.add_argument('--vc', action='store_true',
                        help='步骤3：声线转换（依赖 text_only，或直接用 input_dir）')
    parser.add_argument('--vc_method',
                        choices=['mcadams', 'formant', 'pitch', 'combined', 'seedvc',
                                 'fishaudio_tts', 'cosyvoice_tts'],
                        default='seedvc',
                        help='VC 方法（combined = 共振峰缩放+变调；seedvc = 神经网络，需 GPU, 默认seedvc）')
    parser.add_argument('--vc_whole', action='store_true',
                        help='VC 全音频模式：整段送入处理，不按转录分句（适合短音频；不支持 fishaudio_tts/cosyvoice_tts）')
    parser.add_argument('--vc_workers', type=int, default=1,
                        help='VC 并行进程数（默认 1，多文件时可设为 CPU 核数的一半）')
    parser.add_argument('--eval', action='store_true',
                        help='步骤4：指标评估（依赖匿名后音频）')
    parser.add_argument('--gt_dir', default=None,
                        help='隐私词 GT JSON 目录（提供时在 eval 阶段计算 mask 准确率）')
    parser.add_argument('--subject_map_path', default=None,
                        help='EER 计算时的 subject 映射文件（提供时按 subject 级别计算 EER）格式见README.md')
    # ASR 参数
    parser.add_argument('--hotwords', type=str, default=None,
                        help='ASR 热词，逗号分隔')
    parser.add_argument('--asr_model', type=str, default=None,
                        help='ASR 模型名称，默认根据语言自动选择')
    parser.add_argument('--language', type=str, default='中文',
                        help='ASR 语言（中文/中文方言/英文/日文）。中文→paraformer(有说话人识别)，中文方言/英文/日文→FunASRNano')
    parser.add_argument('--spk_num', type=int, default=None,
                        help='ASR 发言人数（精确值），默认无，让模型自己判断')
    parser.add_argument('--min_speakers', type=int, default=None,
                        help='最少发言人数（用于说话人分离）')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='最多发言人数（用于说话人分离）')
    # mask 参数
    parser.add_argument('--group_num', type=int, default=60,
                        help='LLM 分组句数，默认 60')
    parser.add_argument('--llm', type=str, default='Qwen2.5-32B-Instruct',
                        help='LLM 模型名称，默认 qwen2.5-32b-instruct（根据语言自动选择）')
    parser.add_argument('--prompt', type=str, default=None,
                        help='自定义 LLM 提示词')
    parser.add_argument('--method', choices=['mute', 'noise'], default='noise',
                        help='掩蔽方式：noise（加噪，默认）或 mute（静音）')
    parser.add_argument('--mask_method', choices=['llm', 'ner'], default='llm',
                        help='隐私词检测方式：llm（大模型，默认）或 ner（命名实体识别）')
    # 其他
    parser.add_argument('--mode', choices=['DEBUG', 'INFO'], default='INFO',
                        help='日志级别，默认 INFO')
    args = parser.parse_args()

    hotwords = args.hotwords.split(',') if args.hotwords else []

    # ── 路径定义 ─────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(args.input_dir))
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, 'output')
    if args.middle_dir is None:
        args.middle_dir = os.path.join(base_dir, 'middle')

    M = args.middle_dir
    O = args.output_dir

    # middle/ 子目录
    trans_origin_dir  = os.path.join(M, 'trans_origin')                 # ASR 原始转录
    mask_dir          = os.path.join(M, 'mask')                          # 隐私词集合
    
    # 以下仅 --eval 时使用
    trans_text_dir    = os.path.join(M, 'trans_text')                    # text_only 的转录
    trans_text_vc_dir = os.path.join(M, f'trans_text+{args.vc_method}')  # text_vc 的转录
    trans_vc_dir       = os.path.join(M, f'trans_{args.vc_method}')      # vc_only 的转录
    spk_audio_origin_dir = os.path.join(M, 'spk_audio', 'origin')        # 原始说话人片段
    spk_audio_vc_dir     = os.path.join(M, 'spk_audio', f'text_{args.vc_method}') if args.mask else os.path.join(M, 'spk_audio', f'{args.vc_method}_only')  # VC 说话人片段（统一命名)

    # 年龄/性别模型路径（用于转录后自动预测）
    _age_gender_model = os.path.join(
        os.environ.get('TORCH_HOME', os.path.join(M, '..', 'models')),
        'w2v2-L-robust-6-age-gender'
    )

    # output/ 子目录
    text_only_dir = os.path.join(O, 'text_only')                       # 文本匿名后音频
    text_vc_dir   = os.path.join(O, f'text+{args.vc_method}')          # +VC 后音频
    vc_only_dir        = os.path.join(O, f'{args.vc_method}')               # 仅VC 后的音频
    # ── 日志 ─────────────────────────────────────────────────────
    logger, logger_file = setup_logger(O, 'agent', args.mode)
    logger.info("语音匿名 Agent 启动")
    logger.info(f"输入目录 : {args.input_dir}")
    logger.info(f"输出目录 : {O}")
    logger.info(f"中间目录 : {M}")
    logger.info(f"启用步骤 : trans={args.trans} mask={args.mask} vc={args.vc} eval={args.eval}\n")

    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)

    start_time = time.time()

    try:
        # ── 步骤1: ASR 转录 ────────────────────────────────────
        if args.trans:
            if args.asr_model is not None:
                asr_model = args.asr_model
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

            logger.info(f"语言={args.language} → ASR 模型={asr_model}")

            # 能力警告（不论是指定还是自动选择的模型）
            from configs.config_loader import get_asr_config as _get_asr_cfg
            _asr_cfg = _get_asr_cfg(args.language) if args.asr_model is None else {}
            if hotwords and not _asr_cfg.get('supports_hotwords', True):
                logger.warning(
                    f"ASR 模型 '{asr_model}' 不支持热词，--hotwords 参数将被忽略"
                )
            if args.spk_num is not None and not _asr_cfg.get('supports_speaker', False):
                logger.warning(
                    f"ASR 模型 '{asr_model}' 不支持说话人识别，--spk_num 参数将被忽略"
                )

            # 转录顺带提取说话人音频 + 年龄/性别
            logger.debug( f"(vc={args.vc} eval={args.eval} model={asr_model})")
            _trans_env = _ASR_CONDA_ENV.get(asr_model)
            if _trans_env:
                logger.info(f"ASR 模型 '{asr_model}' 使用 conda 环境: {_trans_env}")
            run_in_isolation(trans, args.input_dir, asr_model, trans_origin_dir, hotwords, args.language, logger_file, args.mode,
                             spk_audio_origin_dir,
                             _age_gender_model,
                             args.spk_num,
                             args.min_speakers,
                             args.max_speakers,
                             conda_env=_trans_env)
            logger.info("ASR 内存已释放\n")
            
        else:
            logger.info("跳过转录步骤\n")

        # trans_time = time.time()
        # logger.debug(f"转录耗时: {trans_time - start_time:.2f} 秒\n")

        # ── 步骤2: 文本内容匿名 ────────────────────────────────
        if args.mask:
            # 依赖检查：trans_origin 必须有 JSON
            _require_json(trans_origin_dir,
                          '文本匿名依赖转录结果 trans_origin（先运行 --trans 或确认目录已存在）',
                          logger)

            # 推断 LLM 模型 + prompt（从配置文件读取）
            from configs.config_loader import get_llm_config
            llm_cfg = get_llm_config(args.language)

            # --llm 可覆盖自动推断的模型
            llm_model = args.llm if args.llm else llm_cfg['model']
            logger.info(f"语言={args.language} → LLM 模型={llm_model}")
            if llm_cfg['is_default']:
                logger.warning(
                    f"语言 '{args.language}' 未在 configs/language_llm_map.yaml 中配置，"
                    f"使用默认配置（模型: {llm_model}）"
                )

            # --prompt 可覆盖配置文件的 prompt；未指定时从配置文件读取
            mask_prompt = args.prompt if args.prompt else llm_cfg.get('prompt', '')
            if not args.prompt:
                logger.info("使用 configs/language_llm_map.yaml 中的默认 prompt")

            noise_method = 'formant' if args.method == 'noise' else 'mute'

            run_in_isolation(mask, trans_origin_dir, mask_dir, logger_file,
                             args.language, mask_prompt, llm_model, args.group_num, args.mode,
                             args.mask_method)
                             
            logger.info("LLM 内存已释放\n")

            # from apply_mask import apply_mask
            # apply_mask(
            #     wav_dir=args.input_dir,
            #     mask_dir=mask_dir,
            #     output_dir=text_only_dir,
            #     logger=logger,
            #     noise_method=noise_method
            # )
        else:
            logger.info("跳过内容匿名步骤\n")

        # mask_time = time.time()
        # logger.debug(f"内容匿名耗时: {mask_time - trans_time:.2f} 秒\n")

        # ── 步骤3: 声线转换 ────────────────────────────────────
        # 始终对原始音频做 VC；若同时选了 --mask，VC 完每个文件立刻 in-place mask。
        # --vc only    → 输出 vc_only/
        # --mask --vc  → 输出 text_{vc}/（已含 mask，不落盘中间文件）
        if args.vc:
            vc_output_dir = text_vc_dir if args.mask else vc_only_dir

            _vc_env = _VC_CONDA_ENV.get(args.vc_method)
            if _vc_env:
                logger.info(f"VC 方法 '{args.vc_method}' 使用 conda 环境: {_vc_env}")

            _vc_mode_desc = "全音频模式" if args.vc_whole else "分句模式"
            logger.info(f"VC 处理模式: {_vc_mode_desc}")

            run_in_isolation(vc,
                             args.input_dir, vc_output_dir, args.vc_method,
                             trans_origin_dir, logger_file, args.mode,
                             args.vc_workers,
                             mask_dir if args.mask else None,
                             args.method,
                             args.language,
                             args.vc_whole,
                             conda_env=_vc_env)
            logger.info("VC 内存已释放\n")

            from utils.metrics.spk_audio_extractor import extract_dir as _ext_spk_pair

            # VC 完成后配对提取 origin + vc 说话人片段（同一循环，保证严格对应）
            logger.info("VC 完成，配对提取说话人片段 → spk_audio/origin/ & spk_audio/vc/")
            _ext_spk_pair(
                args.input_dir, vc_output_dir, trans_origin_dir,
                spk_audio_origin_dir, spk_audio_vc_dir, logger,
            )

            
        else:
            logger.info("跳过声线转换步骤\n")

        # vc_time = time.time()
        # logger.info(f"声线转换耗时: {vc_time - mask_time:.2f} 秒\n")

        # ── 步骤4: 指标评估 ────────────────────────────────────
        if args.eval:
            # 自动推断评估目标：优先 text_vc，其次 vc_only，再次text_only。
            if (args.vc and args.mask) or _has_audio(text_vc_dir):
                eval_audio_dir = text_vc_dir
                eval_trans_dir = trans_text_vc_dir
                mode = f'text_{args.vc_method}'
            elif args.vc or _has_audio(vc_only_dir):
                eval_audio_dir = vc_only_dir
                eval_trans_dir = trans_vc_dir
                mode = f'{args.vc_method}'
            elif args.mask or _has_audio(text_only_dir):
                eval_audio_dir = text_only_dir
                eval_trans_dir = trans_text_dir
                mode = 'text'
            else:
                logger.error(
                    "评估找不到可用的匿名音频目录，请先运行 --mask 或 --vc，"
                    f"或确认 {text_only_dir} / {text_vc_dir} 已存在"
                )
                sys.exit(1)

            _require_audio(eval_audio_dir,
                           f'评估目标音频目录',
                           logger)

            from evaluate import evaluate_all
            evaluate_all(
                input_dir_vc     = eval_audio_dir,
                trans_dir_vc     = eval_trans_dir,
                input_dir_origin = args.input_dir,
                trans_dir_origin = trans_origin_dir,
                middle_dir       = M,
                asr_model        = asr_model,
                hotwords         = hotwords,
                language         = args.language,
                output_dir       = O,
                logger           = logger,
                logger_file      = logger_file,
                _trans_env       = _ASR_CONDA_ENV.get(asr_model),
                gt_dir           = args.gt_dir,
                subject_map_path = args.subject_map_path,
                mode             = mode,
                wer              = None if args.language in NO_WER_LANGUAGE else True
            )
        else:
            logger.info("跳过指标评估步骤\n")
        # eval_time = time.time()
        # logger.info(f"指标评估耗时: {eval_time - vc_time:.2f} 秒\n")

        elapsed_time = time.time() - start_time
        logger.info(f"全部处理结束，总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"结果保存在: {O}")

    except SystemExit:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Agent 运行失败: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
