import os
import sys
import json
import time
import torch
import torchaudio
import numpy as np
import gc
import resource
import random
import psutil

from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve
from scipy.stats import bootstrap
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger

# ======================== 配置参数 ========================


# OUTPUT_JSON = "/path/to/data/NCMMSC_AD/eer_f.json"
MAX_MEMORY_MB = 8192  # 设置8GB内存限制[1](@ref)
MAX_AUDIO_SECONDS = 60  # 限制音频处理时长
SAMPLE_PAIRS_PER_SPEAKER = 5  # 每个说话人随机抽样配对数量[7](@ref)
MODEL_PATH = os.path.join(os.environ.get("TORCH_HOME", "./models"), "wavlm-base-plus-sv")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================== 内存管理 ========================
def set_memory_limit(max_mem_mb):
    """设置进程内存上限防止OOM[1](@ref)"""
    max_bytes = max_mem_mb * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
        print(f"✅ 内存限制设置为: {max_mem_mb}MB")
    except (ValueError, resource.error) as e:
        print(f"⚠️ 内存限制设置失败: {str(e)}")

# ======================== 核心函数 ========================
def load_model():
    """加载预训练模型和特征提取器"""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
    model = WavLMForXVector.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(device)
    return feature_extractor, model

def extract_speaker_embedding(audio_path, feature_extractor, model, logger):
    """
    提取全局说话人嵌入向量（内存优化版）
    """
    try:
        # 1. 加载并预处理音频（限制时长）
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # 单声道转换
        max_samples = MAX_AUDIO_SECONDS * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # 2. 特征提取
        inputs = feature_extractor(
            audio.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # 3. 模型推理
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        
        # 4. 特征聚合与归一化
        global_embedding = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # 5. 显式释放内存
        del audio, inputs, embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return global_embedding.squeeze().cpu().numpy()
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"处理{audio_path}报错：{e}\n{error_trace}")
        return None

def load_subject_map(json_path):
    """
    从 JSON 文件加载 subject-segment 映射，返回 {segment_stem: subject_id} 的反向映射。

    JSON 格式（subject → 片段列表）：
        {"subject_001": ["seg1.wav", "seg2.wav"], "subject_002": ["seg3.wav"]}
    片段名可带或不带扩展名，匹配时均按文件名 stem 进行。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segment_to_subject = {}
    for subject_id, segments in data.items():
        for seg in segments:
            stem = Path(seg).stem
            segment_to_subject[stem] = subject_id
    return segment_to_subject

def get_speaker_id_from_path(file_path, base_dir):
    """
    从文件路径中提取 speaker_id。

    优先解析文件名中的说话人信息：
      文件名格式 taukdial-XXX-Y_spkZ.wav → speaker_id = "taukdial-XXX_spkZ"
      （取受试者编号 + 说话人角色，忽略段落编号 Y）

    若文件名不符合上述格式，则回退到：
      相对于 base_dir 的第一级子目录（若存在），否则使用文件名。
    """
    stem = file_path.stem  # e.g. "taukdial-062-3_spk1"
    if '_spk' in stem:
        # taukdial-XXX-Y_spkZ → taukdial-XXX_spkZ
        session_part, spk_part = stem.rsplit('_', 1)   # spk_part = "spkZ"
        dash_parts = session_part.split('-')            # ['taukdial', 'XXX', 'Y']
        subject_id = '-'.join(dash_parts[:2])           # "taukdial-XXX"
        return f"{subject_id}_{spk_part}"               # "taukdial-XXX_spkZ"

    # 回退：使用第一级子目录或文件名
    try:
        rel_path = file_path.relative_to(base_dir)
        parts = rel_path.parts
        return parts[0] if len(parts) > 1 else stem
    except ValueError:
        return stem

def build_speaker_embeddings(audio_files, base_dir, feature_extractor, model, logger, segment_to_subject=None):
    """
    构建说话人嵌入字典（流式处理版）
    参数：
        audio_files: 音频文件路径列表（Path 对象）
        base_dir: 音频文件的基础目录，用于提取相对路径作为 speaker_id
        feature_extractor, model: 模型对象
        logger: 日志对象
        segment_to_subject: 可选，{segment_stem: subject_id} 映射。
            提供时按 subject 级别聚合嵌入；否则每个文件视为独立说话人。
    """
    speaker_embeddings = defaultdict(list)

    for i, file in enumerate(audio_files):
        if segment_to_subject is not None:
            speaker_id = segment_to_subject.get(file.stem)
            if speaker_id is None:
                logger.warning(f"文件 {file.name} 在 subject 映射中未找到，跳过")
                continue
        else:
            speaker_id = file.stem

        embedding = extract_speaker_embedding(file, feature_extractor, model, logger)
        if embedding is not None:
            speaker_embeddings[speaker_id].append(embedding)

        # 每处理5个文件强制回收内存
        if i % 5 == 0:
            gc.collect()
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            logger.info(f"已处理 {i+1} 个文件 | 当前内存: {mem:.2f}MB")

    logger.info(f"完成 {len(speaker_embeddings)} 个说话人的特征提取\n")
    return speaker_embeddings

def compute_eer(positive_scores, negative_scores, n_bootstrap=1000):
    """计算EER和置信区间（带Bootstrap抽样）"""
    y_true = np.concatenate([np.ones_like(positive_scores), 
                             np.zeros_like(negative_scores)])
    y_scores = np.concatenate([positive_scores, negative_scores])
    
    # 计算原始EER
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - frr))
    eer = (fpr[eer_idx] + frr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    # Bootstrap计算EER的置信区间
    eer_bootstrap = []
    n_samples = len(y_scores)
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        
        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot, pos_label=1)
            frr_boot = 1 - tpr_boot
            eer_idx_boot = np.nanargmin(np.abs(fpr_boot - frr_boot))
            eer_boot = (fpr_boot[eer_idx_boot] + frr_boot[eer_idx_boot]) / 2
            eer_bootstrap.append(eer_boot)
        except:
            continue
    
    if len(eer_bootstrap) > 0:
        ci_low = np.percentile(eer_bootstrap, 2.5)
        ci_high = np.percentile(eer_bootstrap, 97.5)
    else:
        ci_low = ci_high = eer
    
    return eer, threshold, ci_low, ci_high

def evaluate_scenario(orig_embeddings, anon_embeddings, logger):
    """
    评估场景（抽样优化版）
    """
    positive_scores = []
    negative_scores = []
    speaker_ids = list(orig_embeddings.keys())
    
    logger.info(f"说话人数量: {len(speaker_ids)}")
    logger.debug(f"说话人ID示例: {speaker_ids[:5] if len(speaker_ids) > 5 else speaker_ids}")
    
    # 正样本：同一说话人的原始-匿名配对
    for spk_id in speaker_ids:
        if spk_id not in anon_embeddings or not anon_embeddings[spk_id]:
            logger.debug(f"说话人 {spk_id} 在匿名数据中不存在或无嵌入")
            continue
            
        orig_list = orig_embeddings[spk_id]
        anon_list = anon_embeddings[spk_id]
        
        # 每个说话人随机抽样N对组合
        n_pairs = min(SAMPLE_PAIRS_PER_SPEAKER, len(orig_list), len(anon_list))
        for _ in range(n_pairs):
            orig_emb = random.choice(orig_list)
            anon_emb = random.choice(anon_list)
            score = cosine_similarity([orig_emb], [anon_emb])[0][0]
            positive_scores.append(score)
    
    # 负样本：不同说话人的随机配对
    num_positive = len(positive_scores)
    logger.info(f"正样本数量: {num_positive} | 开始生成负样本...")
    
    for _ in range(num_positive):
        if len(speaker_ids) < 2:
            break
        spk1, spk2 = random.sample(speaker_ids, 2)
        if spk1 == spk2 or spk1 not in orig_embeddings or spk2 not in anon_embeddings:
            continue
            
        orig_emb = random.choice(orig_embeddings[spk1])
        anon_emb = random.choice(anon_embeddings[spk2])
        score = cosine_similarity([orig_emb], [anon_emb])[0][0]
        negative_scores.append(score)
    
    logger.info(f"负样本数量: {len(negative_scores)}")
    return compute_eer(np.array(positive_scores), np.array(negative_scores))

def compute_directory_eer(orig_files, anon_files, orig_base_dir, anon_base_dir, logger, subject_map_path=None):
    """
    计算 EER
    参数：
        orig_files: 原始音频文件路径列表（Path 对象）
        anon_files: 匿名化音频文件路径列表（Path 对象）
        orig_base_dir: 原始音频的基础目录（用于提取相对路径）
        anon_base_dir: 匿名化音频的基础目录（用于提取相对路径）
        logger: 日志对象
        subject_map_path: 可选，JSON 文件路径，包含 {subject_id: [segment, ...]} 映射。
            提供时按 subject 级别计算 EER；否则每个文件视为独立说话人。
    """
    logger.info(f"开始计算EER")

    # 加载 subject 映射（可选）
    segment_to_subject = None
    if subject_map_path is not None:
        segment_to_subject = load_subject_map(subject_map_path)
        logger.info(f"已加载 subject 映射：{subject_map_path}，共 {len(set(segment_to_subject.values()))} 个 subject，{len(segment_to_subject)} 个片段")
    else:
        logger.info("未提供 subject 映射，每个文件视为独立说话人")

    # 1. 加载模型
    feature_extractor, model = load_model()
    logger.info(f"模型加载完成，使用设备: {device}")

    # 2. 构建嵌入字典
    logger.info("开始提取原始音频嵌入...")
    orig_embeddings = build_speaker_embeddings(orig_files, orig_base_dir, feature_extractor, model, logger, segment_to_subject)

    logger.info("开始提取匿名化音频嵌入...")
    anon_embeddings = build_speaker_embeddings(anon_files, anon_base_dir, feature_extractor, model, logger, segment_to_subject)

    # 3. 保留共有说话人
    common_speakers = set(orig_embeddings.keys()) & set(anon_embeddings.keys())
    logger.info(f"原始说话人数量: {len(orig_embeddings)}")
    logger.info(f"匿名化说话人数量: {len(anon_embeddings)}")
    logger.info(f"共有说话人数量: {len(common_speakers)}")
    
    if len(common_speakers) == 0:
        logger.error("没有共同的说话人，无法计算 EER")
        return None, None, None
    
    orig_embeddings = {k: v for k, v in orig_embeddings.items() if k in common_speakers}
    anon_embeddings = {k: v for k, v in anon_embeddings.items() if k in common_speakers}

    # 4. 计算EER
    eer, threshold, ci_low, ci_high = evaluate_scenario(orig_embeddings, anon_embeddings, logger)
    
    logger.info(f"EER: {eer*100:.2f}% [95% CI: {ci_low*100:.2f}% - {ci_high*100:.2f}%]")
    
    return eer*100, ci_low*100, ci_high*100

# ======================== 主执行流程 ========================
if __name__ == "__main__":
    origin_dirs = ["/path/to/data_backup/crisis/audio",
                   "/path/to/data/pumch/wav",
                   "/path/to/data_backup/modma/audio",
                   "/path/to/data_backup/ADReSS-IS2020-data/audio",
                   "/path/to/data_backup/NeuroVoz/audios"]
    secure_dirs = ["/path/to/others/baseline/crisis/tts_out/trans_origin",
                   "/path/to/others/baseline/pumch/result",
                   "/path/to/others/baseline/modma/result",
                   "/path/to/others/baseline/adress/result",
                   "/path/to/others/baseline/neurovoz/result"]
    
    for i in [2, 3]:
        input_dir_origin = origin_dirs[i]
        input_dir_vc = secure_dirs[i]
        MODEL_PATH = "/path/to/Anonymise_Agent/models/wavlm-base-plus-sv"
        subject_map_path = None  # 可选，提供时按 subject 级别计算 EER；否则每个文件视为独立说话人
        logger, _ = setup_logger("/path/to/others/baseline/datasets", f"eer_{i}", "INFO")

        # 使用递归查找所有音频文件
        audio_dir1 = Path(input_dir_origin)
        if not audio_dir1.exists():
            raise FileNotFoundError(f"指定的目录不存在：{audio_dir1}")
        
        SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
        # 递归查找所有音频文件
        wav_files1 = sorted([p for p in audio_dir1.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])

        audio_dir2 = Path(input_dir_vc)
        if not audio_dir2.exists():
            raise FileNotFoundError(f"指定的目录不存在：{audio_dir2}")
        
        wav_files2 = sorted([p for p in audio_dir2.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])
        
        logger.info(f"开始计算EER:")
        logger.info(f"原始音频目录：{audio_dir1}，文件数量：{len(wav_files1)}")
        logger.info(f"匿名化音频目录：{audio_dir2}，文件数量：{len(wav_files2)}\n")

        compute_directory_eer(wav_files1, wav_files2, audio_dir1, audio_dir2, logger, subject_map_path)
