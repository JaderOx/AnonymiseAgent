import os
import json
import time
import librosa
import numpy as np
import soundfile as sf

from utils.logger_example import setup_logger
from scipy.signal import lfilter, firwin, find_peaks
import warnings
warnings.filterwarnings('ignore')

# 直接运行该文件：
# 会根据mask_root路径下的json文件，将wav_root中的对应音频进行mask
# 输出到output_root下

wav_dir = ""     # 音频路径（直接运行时填入；经 agent.py / reflexion_agent 调用时由参数传入）
mask_dir = ""     # 掩码信息保存路径
output_dir = ""             # 保护后音频保存路径

# 超参数
time_append = 50                     # 保护区间在原有基础上向两边分别扩充50ms  
noise_method = 'formant'              # white, orginal, spectral_match, formant, mute方法。详见下方代码注释
noise_strength = 1                    # 噪声强度(完全替代为噪声)


# ============== 语音状噪声生成器 ==============

class SpeechShapedNoiseGenerator:
    """根据输入语音段落(待掩盖的段落)特性生成语音状噪声"""
    
    def __init__(self, sr=16000, segment=None):
        self.sr = sr
        self.segment = segment          # 当前分析的语音段落
        self.formant_freqs = None       # 共振峰频率
        
    
    def _estimate_formants(self, segment):
        """估计共振峰"""
        try:
                
            # 计算频谱包络
            spectrum = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/self.sr)
            
            # 找到频谱峰值（简化的共振峰估计）
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.1, distance=50)
            
            if len(peaks) >= 3:
                # 获取所有峰值和对应的频率、幅度
                peak_freqs = freqs[peaks]
                peak_values = spectrum[peaks]
                
                # 按幅度降序排序
                sorted_indices = np.argsort(peak_values)[::-1]
                sorted_freqs = peak_freqs[sorted_indices]
                
                # 选择共振峰：设置频率间隔
                selected_freqs = []
                selected_indices = []
                
                for i in range(len(sorted_freqs)):
                    current_freq = sorted_freqs[i]
                    
                    # 检查是否与已选共振峰距离足够
                    too_close = False
                    for selected_freq in selected_freqs:
                        if abs(current_freq - selected_freq) < 100:  # 最小间距100Hz
                            too_close = True
                            break
                    
                    if not too_close:
                        selected_freqs.append(current_freq)
                        selected_indices.append(sorted_indices[i])
                        
                        # 已找到3个共振峰就停止
                        if len(selected_freqs) == 3:
                            break
                
                # 如果没找到足够的共振峰，用最大的几个峰值（即使距离较近）
                if len(selected_freqs) < 3:
                    # 取前3个最大的峰值，不管距离
                    selected_freqs = sorted_freqs[:3]
                    selected_indices = sorted_indices[:3]
                
                # 按频率排序（F1 < F2 < F3）
                sort_order = np.argsort(selected_freqs)
                self.formant_freqs = np.array(selected_freqs)[sort_order]
                
            else:
                # 如果没有找到足够的峰值，使用典型值
                self.formant_freqs = np.array([500, 1500, 2500])

        except:
            self.formant_freqs = np.array([500, 1500, 2500])
        
    
    def generate_noise(self, length_samples, method):
        """
        生成与该段落匹配的语音状噪声
        
        参数:
            length_samples: 噪声长度（样本数）
            method: 
                'origin' - 简单模拟
                'spectral_match' - 频谱匹配
                'formant' - 基于共振峰
                'mute' - 直接静音
                'white' - 直接使用白噪声
        """
        
        # 确保长度是正数
        if length_samples <= 0:
            return np.array([])
        
        # 基础白噪声
        white_noise = np.random.randn(length_samples)
        
        if method == 'simple':
            # 简单模拟语音频谱: 对白噪声过3000Hz低通滤波，然后带通强调100-1000Hz
            noise = self._apply_speech_filter(white_noise)
            
        elif method == 'spectral_match':
            # 频谱匹配方法：让噪声的频谱形状匹配原始语音
            noise = self._spectral_matching(white_noise)
            
        elif method == 'formant':
            # 基于共振峰的方法
            noise = self._formant_based_noise(white_noise)

        elif method == 'mute':
            # 直接静音
            noise = np.zeros(length_samples)

        elif method == 'white':
            # 直接使用白噪声
            noise = white_noise
            
        else:
            # 默认回退方法
            noise = self._apply_speech_filter(white_noise)
        
        # 归一化
        if len(noise) > 0:
            noise = noise / (np.max(np.abs(noise)) + 1e-10)
        
        return noise
    
    def _apply_speech_filter(self, noise):
        """应用语音滤波器"""
        sr = self.sr
        nyquist = sr / 2
        
        # 设计语音状滤波器：增强100-1000Hz，衰减高频
        # 低通部分（3000Hz截止）
        cutoff_normalized = 3000 / nyquist
        b_low = firwin(101, cutoff_normalized)
        
        # 带通强调语音主要频段
        low_freq = 100 / nyquist
        high_freq = 1000 / nyquist
        b_band = firwin(101, [low_freq, high_freq], pass_zero=False)
        
        # 应用滤波器
        filtered = lfilter(b_low, 1.0, noise)
        filtered = lfilter(b_band, 1.0, filtered)
        
        return filtered
    
    def _spectral_matching(self, noise):
        """频谱匹配：让噪声的频谱包络匹配语音"""

        D = librosa.stft(self.segment, n_fft=1024, hop_length=256)
        magnitude = np.abs(D)
        avg_spectrum = np.mean(magnitude, axis=1)

        # 将噪声分帧处理
        frame_size = 1024
        hop_size = 256
        
        # 确保噪声长度足够
        if len(noise) < frame_size:
            # 如果太短，直接应用滤波器
            return self._apply_speech_filter(noise)
        
        # 对噪声进行STFT
        D_noise = librosa.stft(noise, n_fft=frame_size, hop_length=hop_size)
        
        # 将语音的平均频谱扩展到噪声的所有帧
        target_spectrum = avg_spectrum[:frame_size//2 + 1]
        
        # 对每个时间帧应用目标频谱形状
        for i in range(D_noise.shape[1]):
            frame_mag = np.abs(D_noise[:, i])
            
            # 用语音频谱形状重新调整噪声帧
            # 保持原始相位，只改变幅度
            min_len = min(len(frame_mag), len(target_spectrum))
            if min_len > 0:
                adjusted_mag = frame_mag[:min_len] * target_spectrum[:min_len] / (np.mean(frame_mag[:min_len]) + 1e-10)
                
                # 重建复数频谱
                D_noise[:min_len, i] = adjusted_mag * np.exp(1j * np.angle(D_noise[:min_len, i]))
        
        # 逆STFT
        matched_noise = librosa.istft(D_noise, hop_length=hop_size)
        
        # 确保长度一致
        if len(matched_noise) > len(noise):
            matched_noise = matched_noise[:len(noise)]
        elif len(matched_noise) < len(noise):
            # 补零
            matched_noise = np.pad(matched_noise, (0, len(noise) - len(matched_noise)))
        
        return matched_noise
    
    def _formant_based_noise(self, noise):
        """基于共振峰的噪声生成"""
        # 估计共振峰
        self._estimate_formants(self.segment)
                
        sr = self.sr
        
        # 对每个共振峰应用带通滤波器
        filtered = noise.copy()

        
        for formant_freq in self.formant_freqs[:3]:  # 只处理前3个共振峰
            # 设计带通滤波器
            bandwidth = formant_freq * 0.15  # 15%带宽
            low = (formant_freq - bandwidth/2) / (sr/2)
            high = (formant_freq + bandwidth/2) / (sr/2)
            
            if low > 0 and high < 1:
                b_band = firwin(51, [low, high], pass_zero=False)
                filtered = lfilter(b_band, 1.0, filtered)
        
        return filtered

# ============== 单条处理函数 ==============

def add_adaptive_ssn_to_audio(audio_path, mask_path, output_path="./output", time_append = 0, 
                            noise_strength=1, noise_method='formant'):
    """
    给音频的指定时间段添加自适应语音状噪声
    
    参数:
        audio_path: 音频文件路径
        mask_path: 掩码文件路径（JSON格式，包含时间段信息）
        noise_strength: 0-1，噪声强度
        noise_method: white, orginal, spectral_match, formant, mute
        output_path: 输出文件路径
    """
    # 1. 加载音频
    audio, sr = librosa.load(audio_path, sr=None)
    
    
    
    # 2. 创建保护后的音频副本
    protected = audio.copy()
    
    # 3. 处理每个时间段
    with open(mask_path, 'r', encoding='utf-8') as f:
        time_ranges = json.load(f)
    idx = 0
    for item in time_ranges:
        idx += 1
        start_time = (item["start"] - time_append)/1000
        end_time = (item["end"] + time_append)/1000

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 确保索引在有效范围内
        start_sample = max(0, min(start_sample, len(audio)-1))
        end_sample = max(start_sample+1, min(end_sample, len(audio)))
        
        if end_sample - start_sample <= 0:
            print(f"⚠️  跳过无效时间段: {start_sample}-{end_sample}")
            continue
        
        # 提取原始片段
        original_segment = audio[start_sample:end_sample]
        segment_length = len(original_segment)

        # 初始化语音状噪声生成器
        ssn_gen = SpeechShapedNoiseGenerator(sr=sr, segment=original_segment)

        # 生成语音状噪声
        ssn = ssn_gen.generate_noise(segment_length, method=noise_method)
        
        if len(ssn) != segment_length:
            # 调整噪声长度
            if len(ssn) > segment_length:
                ssn = ssn[:segment_length]
            else:
                ssn = np.pad(ssn, (0, segment_length - len(ssn)))
        
        # 调整噪声能量与原始片段匹配
        if len(original_segment) > 0 and len(ssn) > 0:
            orig_energy = np.mean(np.abs(original_segment))
            noise_energy = np.mean(np.abs(ssn))
            if noise_energy > 0:
                ssn = ssn * (orig_energy / noise_energy)
        
        # 混合：原始语音 + 语音状噪声
        protected_segment = original_segment * (1 - noise_strength) + ssn * noise_strength
        
        # 淡入淡出避免爆音
        fade_len = min(100, segment_length // 10)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            protected_segment[:fade_len] *= fade_in
            protected_segment[-fade_len:] *= fade_out
        
        # 替换原片段
        protected[start_sample:end_sample] = protected_segment
        
    
    # 5. 保存结果
    sf.write(output_path, protected, sr)
    
    return audio, protected, sr, ssn_gen if ssn_gen.formant_freqs is not None else None

# ============== 批量处理函数 ==============
def apply_mask(
    wav_dir = None,
    mask_dir = None,
    output_dir = None,
    logger = None,
    noise_strength = noise_strength,
    noise_method = noise_method,
    time_append = time_append
    ):

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"开始处理语音内容隐私\n音频目录: {wav_dir}\n隐私目录: {mask_dir}\n输出目录: {output_dir}\n")
    
    from pathlib import Path as _Path
    _wav_dir_p = _Path(wav_dir)
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = sorted([p for p in _wav_dir_p.rglob('*')
                           if p.suffix.lower() in supported_formats])
    if not audio_files:
        logger.warning(f"在目录 {wav_dir} 中未找到合法的音频文件，支持的格式包括: {supported_formats}")
        return

    all_len = len(audio_files)

    for count, audio_file in enumerate(audio_files):
        rel = audio_file.relative_to(_wav_dir_p)
        sub = str(rel)
        wav_path = str(audio_file)
        mask_path = str(_Path(mask_dir) / rel.with_suffix('.json'))
        output_path_p = _Path(output_dir) / rel
        output_path_p.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(output_path_p)


        file_start = time.time()    
        logger.info(f"处理语音[{count + 1}/{all_len}]: {sub}")

        if os.path.isfile(output_path):
            logger.info(f"{sub}已存在")
            continue

        
        try: 
            # 执行自适应保护
            original, protected, sr, ssn_gen = add_adaptive_ssn_to_audio(
                audio_path=wav_path,
                mask_path=mask_path,
                output_path=output_path,
                noise_strength=noise_strength,
                noise_method=noise_method,
                time_append=time_append
            )
            file_time = time.time() - file_start
            logger.info(f"处理完成: {sub}, 耗时: {file_time:.2f} 秒")
            
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()  # 获取完整的traceback字符串
            os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
            with open(os.path.join(output_dir, "logs", "fail_list.txt"), 'a') as f:
                logger.error(f"处理文件名\" {sub}\" 失败: {e}\n{error_trace}\n")
                f.write(sub + '\n')\
    
    logger.info("所有语音内容处理完毕")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="音频内容遮蔽（语音状噪声替换）")
    parser.add_argument("--wav_dir", "-i", required=True, help="原始音频目录")
    parser.add_argument("--mask_dir", "-m", required=True, help="隐私词 JSON 目录")
    parser.add_argument("--output_dir", "-o", required=True, help="遮蔽后音频输出目录")
    parser.add_argument("--noise_method", default="formant",
                        choices=["formant", "spectral_match", "white", "simple", "mute"],
                        help="噪声生成方法（默认 formant）")
    parser.add_argument("--noise_strength", type=float, default=1.0, help="噪声强度（默认 1.0）")
    parser.add_argument("--time_append", type=int, default=50, help="遮蔽区间前后扩充 ms（默认 50）")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO", help="日志级别")
    args = parser.parse_args()

    logger, _ = setup_logger(args.output_dir, "apply_mask", args.mode)
    apply_mask(
        wav_dir=args.wav_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        logger=logger,
        noise_strength=args.noise_strength,
        noise_method=args.noise_method,
        time_append=args.time_append,
    )