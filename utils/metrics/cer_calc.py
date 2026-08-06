import os
import sys
import json
import time
import jiwer
import zhconv # type: ignore
import regex
import unicodedata
import re
import pandas as pd

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger, get_logger

from manage_module_memory import run_in_isolation




def trans(wav_dir, asr_model, trans_dir, hotwords, language, logger_file, mode, spk_audio_dir=None, age_gender_model_path=None, spk_num=None):
    logger = get_logger(logger_file, 'trans', mode)
    sys.path.append(str(Path(__file__).parent.parent.parent))
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
    )

def normalize_zh(s: str) -> str:
    s = zhconv.convert(s, 'zh-hans')
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"\p{Cf}+", "", s)
    s = re.sub(r"[^\u4e00-\u9fa5]", "", s)
    s = regex.sub(r"\s+", "", s)
    return s


def normalize_en(s: str) -> str:
    """英文 CER 规范化：小写 + 去标点 + 合并空白，保留字母数字空格。"""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text(s: str, language: str) -> str:
    """根据语言选择规范化函数。"""
    lang = (language or "").strip().lower()
    if lang == "zh":
        return normalize_zh(s)
    elif lang in ["en", "es"]:
        return normalize_en(s)
    else:
        # 其他语言：小写 + 去标点 + 合并空白
        s = s.lower()
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

def compute_directory_cer(
        vc_wav_dir,
        ori_files,
        ori_trans, 
        vc_files,
        vc_trans,
        asr_model,
        hotwords,
        language,
        logger,
        logger_file,
        _trans_env,
        gt_dir = None,
        wer = None,
        skip_vc_asr: bool = False,
        ):
    logger.info(f"开始计算CER")
    from pathlib import Path
    
    # 构建文件相对路径到完整路径的映射
    # 使用相对路径作为唯一标识，避免不同子目录同名文件冲突
    vc_wav_dir_path = Path(vc_wav_dir)
    ori_base_dir = Path(ori_files[0]).parent if ori_files else None
    
    # 确定原始音频的基础目录（用于计算相对路径）
    # 假设 ori_files 来自同一个根目录，取所有文件的公共父目录
    if ori_files:
        ori_base_dir = Path(os.path.commonpath([str(p.parent) for p in ori_files]))
    else:
        ori_base_dir = None
    
    # 构建映射：相对路径 -> 完整路径
    vc_file_map = {}
    for f in vc_files:
        f_path = Path(f)
        try:
            # 获取相对于 vc_wav_dir 的相对路径
            rel_path = f_path.relative_to(vc_wav_dir_path)
            vc_file_map[str(rel_path)] = f_path
        except ValueError:
            # 如果不在 vc_wav_dir 下，使用完整路径
            vc_file_map[str(f_path)] = f_path
    
    ori_file_map = {}
    if ori_base_dir:
        for f in ori_files:
            f_path = Path(f)
            try:
                rel_path = f_path.relative_to(ori_base_dir)
                ori_file_map[str(rel_path)] = f_path
            except ValueError:
                ori_file_map[str(f_path)] = f_path
    else:
        for f in ori_files:
            ori_file_map[str(Path(f))] = Path(f)
    
    # 找到共同的相对路径
    common_rel_paths = set(ori_file_map.keys()) & set(vc_file_map.keys())
    logger.info(f"找到 {len(common_rel_paths)} 个共同的音频文件")
    
    if not common_rel_paths:
        logger.warning("没有共同的音频文件，无法计算 CER")
        return None, "N/A"
    
    # 开始对 vc 侧转录（可跳过：当 trans_* 下 JSON 已齐全，避免整目录一次 ASR 导致 OOM）
    if skip_vc_asr:
        logger.info("skip_vc_asr=True：跳过 VC 侧 ASR，直接使用已有 trans JSON 计算 CER/WER")
    else:
        try:
            run_in_isolation(
                trans, str(vc_wav_dir), asr_model, str(vc_trans), hotwords, language,
                logger_file, "INFO", None, None, None, conda_env=_trans_env,
            )
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"CER转录阶段报错：{e}\n{error_trace}")

    reference_text = ""
    hypothesis_text = ""

    results = []
    length = len(common_rel_paths)
    
    # 处理每个JSON文件
    for i, rel_path_str in enumerate(sorted(common_rel_paths)):
        try:
            file_time = time.time()
            rel_path = Path(rel_path_str)
            
            # 获取不带扩展名的文件名
            base = rel_path.stem
            json_file = base + ".json"
            
            # 获取原始音频的完整路径
            ori_audio_path = ori_file_map[rel_path_str]
            vc_audio_path = vc_file_map[rel_path_str]
            
            # 获取相对路径的目录部分（用于定位 JSON 文件）
            rel_dir = rel_path.parent if rel_path.parent != Path('.') else Path('.')
            
            # 构建 JSON 文件的完整路径（保持子目录结构）
            ori_trans_path = Path(ori_trans)
            vc_trans_path = Path(vc_trans)
            
            ori_json_path = ori_trans_path / rel_dir / json_file
            vc_json_path = vc_trans_path / rel_dir / json_file
            
            logger.debug(f"处理文件: {rel_path_str}")
            logger.debug(f"  JSON 路径: {ori_json_path} | {vc_json_path}")
            
            # 读取原始转录文件
            if not ori_json_path.exists():
                logger.warning(f"原始转录文件不存在: {ori_json_path}")
                continue
                
            with open(ori_json_path, 'r', encoding='utf-8') as f:
                ori_data = json.load(f)
            
            # 读取匿名化后的文件
            if not vc_json_path.exists():
                logger.warning(f"匿名化转录文件不存在: {vc_json_path}")
                continue
                
            with open(vc_json_path, 'r', encoding='utf-8') as f:
                anonymised_data = json.load(f)
            
            # 提取并拼接所有text字段
            ori_texts = []
            anonymised_texts = []
            
            # 根据JSON结构提取text字段
            def extract_texts(data, texts_list):
                if isinstance(data, dict):
                    if 'text' in data:
                        texts_list.append(data['text'])
                    for key, value in data.items():
                        extract_texts(value, texts_list)
                elif isinstance(data, list):
                    for item in data:
                        extract_texts(item, texts_list)
            
            extract_texts(ori_data, ori_texts)
            extract_texts(anonymised_data, anonymised_texts)
            
            # 拼接所有text（句子间加空格，避免词粘连）
            ori_combined = ' '.join(ori_texts)
            anonymised_combined = ' '.join(anonymised_texts)
            if gt_dir:
                txt_file = Path(gt_dir) / rel_dir / (base + ".txt")
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        gt_text = f.read()
                    ori_combined = gt_text
            
            # 规范化文本
            ori_normalized = normalize_text(ori_combined, language)
            anonymised_normalized = normalize_text(anonymised_combined, language)
            logger.debug(f"\n原文本: {ori_normalized}\n后文本: {anonymised_normalized}")

            if wer:
                wer_file = jiwer.wer(ori_normalized, anonymised_normalized)
                logger.info(f"[{i+1}/{length}]wer: {wer_file*100:.2f}%\tcost:{time.time() - file_time:.2f}s\t--{json_file} (in {rel_dir})")

            cer_file = jiwer.cer(ori_normalized, anonymised_normalized)
            logger.info(f"[{i+1}/{length}]cer: {cer_file*100:.2f}%\tcost:{time.time() - file_time:.2f}s\t--{json_file} (in {rel_dir})")
            
            # 使用相对路径（不含扩展名）作为唯一标识
            from pathlib import Path
            name = str(Path(rel_path).with_suffix(''))
            if wer:
                results.append((name, cer_file, wer_file))
            else:
                results.append((name, cer_file))
            
            # 添加到总字符串
            reference_text += ori_normalized
            hypothesis_text += anonymised_normalized
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"CER_{rel_path_str}报错：{e}\n{error_trace}")

    # 计算CER
    result_str = "N/A"
    if reference_text and hypothesis_text:
        cer = jiwer.cer(reference_text, hypothesis_text)
        if wer:
            wer_r = jiwer.wer(reference_text, hypothesis_text)
            results.append(("-mean-", cer, wer_r))
            result_str = f"CER:{cer*100:.2f}%, WER:{wer_r*100:.2f}%"
        else:
            results.append(("-mean-", cer))
            result_str = f"CER:{cer*100:.2f}%"
        logger.info(f"result: {cer*100:.2f}%")
    else:
        logger.info("没有找到有效的文本内容")
    if wer:
        df = pd.DataFrame(results, columns=["name", "cer", "wer"]) if results else None
    else:
        df = pd.DataFrame(results, columns=["name", "cer"]) if results else None
    return df, result_str



if __name__ == "__main__":
    vc_methods = ["pitch", "formant", "combined", "mcadams", "seedvc", "fishaudio_tts"]
    for vc_method in vc_methods:
        ori_dir = f'/path/to/data/NCMMSC_AD/input'
        ori_trans = f'/path/to/data/NCMMSC_AD/middle/trans_origin'

        vc_dir = f'/path/to/data/NCMMSC_AD/output/{vc_method}'
        vc_trans = f'/path/to/data/NCMMSC_AD/middle/trans_{vc_method}'

        # 注释掉的是算groundtruth时使用的。
        # vc_dir = ori_dir
        # vc_trans = ori_trans
        # gt_dir = "/path/to/data/NeuroVoz/transcriptions"
        log_dir = "/path/to/data/NCMMSC_AD/output"
        logname = f"{vc_method}_cer"

        logger, logger_file = setup_logger(log_dir, logname, "DEBUG")

        # 使用示例: (需要提前像这样对目录做处理)
        # 使用递归查找所有音频文件
        audio_dir1 = Path(ori_dir)
        if not audio_dir1.exists():
            raise FileNotFoundError(f"指定的目录不存在：{audio_dir1}")
        
        SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
        # 递归查找所有音频文件
        wav_files1 = sorted([p for p in audio_dir1.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])

        audio_dir2 = Path(vc_dir)
        if not audio_dir2.exists():
            raise FileNotFoundError(f"指定的目录不存在：{audio_dir2}")
        
        wav_files2 = sorted([p for p in audio_dir2.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])
        logger.info(f"开始计算CER")
        logger.info(f"转录语音路径:\n原始语音:{ori_dir}\n匿名语音:{vc_dir}")

        df, result_str = compute_directory_cer(
            vc_wav_dir = vc_dir,
            ori_files = wav_files1,
            ori_trans = ori_trans, 
            vc_files = wav_files2,
            vc_trans = vc_trans,
            asr_model = 'whisperx',
            hotwords = None,
            language = 'it',
            logger = logger,
            logger_file = logger_file,
            _trans_env = 'whisperx',
            # gt_dir = gt_dir if gt_dir else None,
            wer = True
            )
        
        logger.debug(f"CER计算结果:\n{df.to_string(index=False)}")
        