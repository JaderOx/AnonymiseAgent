import os
import sys
import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
import torchaudio

model_path = os.path.join(os.environ.get("MODELSCOPE_CACHE", "."), "models/iic/emotion2vec_plus_seed")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger


input_dir1 = "/path/to/data/example/agent/input"
input_dir2 = "/path/to/data/example/agent/output/vc"
MAX_DURATION = 3*60  # 三分钟

def softmax(z):
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z)

def get_audio_pairs(files1, files2, base_dir1, base_dir2):
    """
    构建音频文件配对，使用相对路径作为唯一标识
    参数：
        files1: 第一个目录的音频文件列表（Path 对象）
        files2: 第二个目录的音频文件列表（Path 对象）
        base_dir1: 第一个目录的基础路径（用于计算相对路径）
        base_dir2: 第二个目录的基础路径（用于计算相对路径）
    """
    audio_pairs = {}
    
    # 构建第一个目录的映射：相对路径 -> 文件路径
    files1_map = {}
    for file in files1:
        try:
            rel_path = file.relative_to(base_dir1)
            files1_map[str(rel_path)] = file
        except ValueError:
            # 如果文件不在 base_dir1 下，使用完整路径
            files1_map[str(file)] = file
    
    # 构建第二个目录的映射：相对路径 -> 文件路径
    files2_map = {}
    for file in files2:
        try:
            rel_path = file.relative_to(base_dir2)
            files2_map[str(rel_path)] = file
        except ValueError:
            files2_map[str(file)] = file
    
    # 找到共同的相对路径
    common_rel_paths = set(files1_map.keys()) & set(files2_map.keys())
    
    # 构建配对
    for rel_path_str in common_rel_paths:
        audio_pairs[rel_path_str] = [files1_map[rel_path_str], files2_map[rel_path_str]]
    
    return audio_pairs

def calculate_similarity(embed1, embed2):
    return cosine_similarity([embed1], [embed2])[0][0]

def process_long_audio(path, max_duration, logger, inference_pipeline, sr=16000):
    """
    处理长音频，分段提取情感embedding
    """
    try:
        audio, sr = torchaudio.load(path)
        audio = audio.mean(dim=0)  # 转为单声道
        
        # 计算最大样本数
        max_samples = sr * max_duration
        total_samples = len(audio)

        num_segments = math.ceil(total_samples / max_samples)
        probabilities = []

        for i in range(num_segments):
            start_idx = i * max_samples
            end_idx = min((i + 1) * max_samples, total_samples)
            
            # 截取当前片段
            audio_segment = audio[start_idx:end_idx]
            
            # 如果片段太短（小于1秒）则跳过
            if len(audio_segment) < sr:
                logger.debug(f"第{i+1}个片段过短（{len(audio_segment)/sr:.2f}s），跳过")
                continue
            
            # 提取embedding
            result = inference_pipeline(audio_segment, granularity="utterance", extract_embedding=False)
            embed = result[0]["scores"]
            prob = softmax(embed)
            probabilities.append(prob)
            
            logger.debug(f"处理第{i+1}/{num_segments}个片段，长度：{len(audio_segment)/sr:.2f}s")
        
        return probabilities
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"处理{path}报错：{e}\n{error_trace}")
        return []

def compute_directory_emo(wav_files1, wav_files2, base_dir1, base_dir2, logger):
    """
    计算 EMO（情感相似度）
    参数：
        wav_files1: 第一个目录的音频文件列表（Path 对象）
        wav_files2: 第二个目录的音频文件列表（Path 对象）
        base_dir1: 第一个目录的基础路径
        base_dir2: 第二个目录的基础路径
        logger: 日志对象
    """
    start_time = time.time()
    logger.info(f"开始计算EMO")

    # 初始化 emotion pipeline
    logger.info("加载 emotion2vec 模型...")
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=model_path,
        device_map="auto",
        disable_update=True
    )
    logger.info("模型加载完成")

    # 获取音频配对（使用相对路径匹配）
    audio_pairs = get_audio_pairs(wav_files1, wav_files2, base_dir1, base_dir2)
    logger.info(f"找到 {len(audio_pairs)} 对匹配音频")

    if not audio_pairs:
        logger.warning("没有找到匹配的音频对，无法计算 EMO")
        return None, "N/A"

    results = []
    count = 0
    length = len(audio_pairs)
    
    for rel_path_str, (path1, path2) in audio_pairs.items():
        try:
            file_time = time.time()
            count += 1
            logger.debug(f"开始处理 {rel_path_str} 两条音频")
            
            # 处理长音频，获取情感概率分布列表
            prob1_list = process_long_audio(path1, MAX_DURATION, logger, inference_pipeline)
            prob2_list = process_long_audio(path2, MAX_DURATION, logger, inference_pipeline)
            
            # 确保两个列表长度一致
            min_len = min(len(prob1_list), len(prob2_list))
            if min_len == 0:
                logger.warning(f"{rel_path_str} 没有有效的音频片段，跳过")
                continue
            
            similarities = []
            for j in range(min_len):
                prob1 = prob1_list[j]
                prob2 = prob2_list[j]
                sim = calculate_similarity(prob1, prob2)
                similarities.append(sim)
            
            # 计算平均相似度
            avg_similarity = np.mean(similarities)
            # 去掉扩展名统一命名格式
            from pathlib import Path
            name = str(Path(rel_path_str).with_suffix(''))
            results.append((name, avg_similarity))

            logger.info(f"[{count}/{length}] {name}: similarity={avg_similarity:.4f}\tcost:{time.time() - file_time:.2f}s")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"处理{rel_path_str}报错：{e}\n{error_trace}")
            continue

    # 保存结果
    try:
        if results:
            df = pd.DataFrame(results, columns=["name", "emo"])
            mean = df["emo"].mean()
            std = df["emo"].std()
            
            # 添加统计行
            mean_row = pd.DataFrame([["-mean-", mean]], columns=["name", "emo"])
            std_row = pd.DataFrame([["-std-", std]], columns=["name", "emo"])
            df = pd.concat([df, mean_row, std_row], ignore_index=True)
            
            mean_std_str = f"{mean:.3f}±{std:.3f}"
            
            logger.info(f"EMO处理完毕, 耗时 {time.time() - start_time:.2f} 秒")
            logger.info(f"EMO结果: {mean_std_str}")
            logger.debug(f"详细结果:\n{df.to_string(index=False)}")

            # 释放模型显存
            del inference_pipeline
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("已释放 emotion2vec 模型显存")

            return df, mean_std_str
        else:
            logger.warning("没有成功处理的音频对")
            # 释放模型显存
            del inference_pipeline
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None, "N/A"

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"EMO最终阶段报错：{e}\n{error_trace}")
        # 释放模型显存
        try:
            del inference_pipeline
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        return None, "N/A"


if __name__ == "__main__":

    logger, _ = setup_logger("/path/to/data/example/agent/metrics/", "emo", "DEBUG")

    # 使用递归查找所有音频文件
    audio_dir1 = Path(input_dir1)
    if not audio_dir1.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir1}")
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    # 递归查找所有音频文件
    wav_files1 = sorted([p for p in audio_dir1.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])

    audio_dir2 = Path(input_dir2)
    if not audio_dir2.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir2}")
    
    wav_files2 = sorted([p for p in audio_dir2.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])
    
    logger.info(f"开始计算EMO:")
    logger.info(f"音频目录1：{audio_dir1}，文件数量：{len(wav_files1)}")
    logger.info(f"音频目录2：{audio_dir2}，文件数量：{len(wav_files2)}\n")

    compute_directory_emo(wav_files1, wav_files2, audio_dir1, audio_dir2, logger)