import os
import sys
import time

import torch
import numpy as np
import pandas as pd

from pathlib import Path
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger


device = "cuda"
model = load_silero_vad().to(device)

# MOS模型
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", 
                          trust_repo=True).to(device)

input_dir = "/path/to/data/example/agent/input"

def process_long_audio(wav_path, max_segment_duration=60):
    """
    处理长音频：分段处理并聚合结果
    max_segment_duration: 每段最大时长（秒）
    """
    wav = read_audio(wav_path)
    
    # 获取语音段
    speech_timestamps = get_speech_timestamps(wav.to(device), model)
    
    if not speech_timestamps:
        return None
    
    # 收集所有语音段
    all_segments = []
    for seg in speech_timestamps:
        segment = wav[seg["start"]:seg["end"]]
        all_segments.append(segment)
    
    # 方案1.1：如果总语音很短，直接拼接
    total_speech = sum(len(s) for s in all_segments) / 16000
    if total_speech <= max_segment_duration:
        combined = torch.cat(all_segments)
        with torch.no_grad():
            score = predictor(combined.float().unsqueeze(0).to(device), 16000)
        return [score.item()]  # 返回单个分数
    
    # 方案1.2：长语音分段评分
    scores = []
    current_segment = []
    current_duration = 0
    
    for seg in all_segments:
        seg_duration = len(seg) / 16000
        current_segment.append(seg)
        current_duration += seg_duration
        
        # 当累积时长达到阈值，评分这一段
        if current_duration >= max_segment_duration:
            combined = torch.cat(current_segment)
            with torch.no_grad():
                score = predictor(combined.float().unsqueeze(0).to(device), 16000)
                scores.append(score.item())
            
            # 重置
            current_segment = []
            current_duration = 0
    
    # 处理最后一段
    if current_segment:
        combined = torch.cat(current_segment)
        with torch.no_grad():
            score = predictor(combined.float().unsqueeze(0).to(device), 16000)
            scores.append(score.item())
    
    return scores

def compute_directory_mos(wav_files = None, logger = None, base_dir=None):
    # start_time = time.time()

    logger.info("开始计算MOS")
    results = []
    length = len(wav_files)
    for i, wav_file in enumerate(wav_files):
        try:
            file_time = time.time()
            # 处理长音频（不限时长）
            scores = process_long_audio(wav_file, max_segment_duration=60)

            if scores:
                # 也可以保存文件所有片段的平均分
                score_avg = np.mean(scores)
                logger.info(f"[{i+1}/{length}]score:{score_avg:.2f}\tcost:{time.time() - file_time:.2f}s\t--{wav_file.name}")

                from pathlib import Path
                if base_dir:
                    name = str(Path(wav_file).relative_to(base_dir).with_suffix(''))
                else:
                    name = wav_file.stem
                results.append((name, score_avg))
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"处理{wav_file.name}报错：{e}\n{error_trace}")
    try:
        if results:
            df = pd.DataFrame(results, columns=["name", "mos"])
            mean = df["mos"].mean()
            std = df["mos"].std()
            df.loc[len(df)] = ['-mean-', mean]
            df.loc[-1] = ['-std-', std]
            mean_std_str = f"{mean:.2f}±{std:.2f} dB"

            # logger.info(f"MOS处理完毕, 耗时 {time.time() - start_time:.2f} 秒")
            logger.debug(f"MOS结果: \n{df.to_string(index=False)}")
            return df, mean_std_str
    except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"MOS最终阶段报错：{e}\n{error_trace}")
    
if __name__ == "__main__":

    logger, _ = setup_logger("/path/to/data/example/agent/metrics/", "mos", "DEBUG")

    # 使用示例: (需要提前像这样对目录做处理)
    audio_dir = Path(input_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir}")
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    wav_files = sorted([p for p in audio_dir.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS])
    logger.info(f"开始计算MOS:\n音频目录：{audio_dir}，文件数量：{len(wav_files)}\n")

    compute_directory_mos(wav_files, logger)