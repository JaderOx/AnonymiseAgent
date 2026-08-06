import os
import numpy as np
import soundfile as sf
from pathlib import Path

def get_audio_duration(file_path):
    """获取单个音频文件的时长（秒）"""
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate

def calculate_total_duration(audio_dir):
    """计算目录下所有音频文件总时长"""
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"指定的目录不存在：{audio_dir}")
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.mp3'}
    # 递归查找所有音频文件
    audio_files = sorted([p for p in audio_dir.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS])

    print(f"找到 {len(audio_files)} 个音频文件，开始计算时长...")

    total_seconds = 0
    for file_path in audio_files:
        try:
            duration = get_audio_duration(file_path)
            total_seconds += duration
            print(f"{file_path.name}: {duration:.2f}s")
        except Exception as e:
            print(f"无法读取 {file_path.name}: {e}")
    
    
    print(f"\n总时长: {total_seconds:.2f} 秒")
    
    return total_seconds



def merge_audio_files(source_dir, target_dir):
    """
    将 source_dir 下每个子文件夹中的音频文件合并成一个文件
    
    Args:
        source_dir: 源目录路径，例如 'xxx/audio'
        target_dir: 目标目录路径，例如 'xxx/audio_merge'
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 遍历源目录下的所有子文件夹
    for subfolder in sorted(source_path.iterdir()):
        if not subfolder.is_dir():
            continue
            
        # 获取该子文件夹下所有 wav 文件，按文件名排序
        wav_files = sorted(subfolder.glob("*.wav"))
        
        if not wav_files:
            print(f"警告: {subfolder.name} 中没有找到 wav 文件")
            continue
        
        # 存储所有音频数据和采样率
        audio_data = []
        sample_rate = None
        
        # 逐个读取音频文件
        for wav_file in wav_files:
            try:
                data, sr = sf.read(wav_file)
                
                # 确保所有文件采样率一致
                if sample_rate is None:
                    sample_rate = sr
                elif sample_rate != sr:
                    print(f"警告: {wav_file} 采样率 ({sr}) 与第一个文件 ({sample_rate}) 不一致")
                    # 可选：重采样或继续使用第一个文件的采样率
                    continue
                
                # 如果是单声道，直接添加；如果是多声道，可能需要处理
                audio_data.append(data)
                
            except Exception as e:
                print(f"读取 {wav_file} 失败: {e}")
                continue
        
        if not audio_data:
            print(f"错误: {subfolder.name} 没有成功读取任何音频文件")
            continue
        
        # 合并音频
        merged_audio = np.concatenate(audio_data)
        
        # 输出文件路径
        output_file = target_path / f"{subfolder.name}.wav"
        
        # 保存合并后的音频
        sf.write(output_file, merged_audio, sample_rate)
        print(f"已合并: {subfolder.name} ({len(wav_files)} 个文件) -> {output_file}")
    
    print(f"\n合并完成！结果保存在: {target_path}")

import os
import shutil
from pathlib import Path

def rename_and_collect_audio(source_dir, output_dir):
    """
    将 source_dir 下所有子文件夹中的文件重命名并收集到 output_dir
    
    Args:
        source_dir: 源目录路径，例如 'audio'
        output_dir: 输出目录路径，例如 'output'
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有子目录
    for subfolder in sorted(source_path.iterdir()):
        if not subfolder.is_dir():
            continue
        
        # 获取子目录名，如 "001"
        folder_name = subfolder.name
        
        # 获取该子目录下所有 wav 文件，按文件名排序
        audio_files = sorted(subfolder.rglob("*.wav"))
        
        if not audio_files:
            print(f"警告: {folder_name} 中没有找到 wav 文件")
            continue
        
        # 处理每个音频文件
        counter = 0
        for audio_file in audio_files:
            # 获取原始文件名（不含扩展名），如 "01"
            original_name = audio_file.stem
            if original_name == '01_CF56_1':
                counter += 1
                if counter > 1:
                    # 新文件名：{文件夹名}_{原始文件名}.json，如 "001_01.json"
                    new_name = f"{original_name}_{folder_name}.wav"
                    new_path = output_path / new_name
                    
                    # 复制文件（如果不想保留原文件，改为 shutil.move）
                    shutil.copy2(audio_file, new_path)
                    
                    print(f"已复制: {audio_file} -> {new_path}")
                    break
    
    print(f"\n完成！共处理了 {len(list(output_path.glob('*.wav')))} 个文件，保存在: {output_path}")

def rename_audio_files(audio_dir):
    """将 audio_dir 下所有 wav 文件中的 '-' 替换为 '_'"""
    from pathlib import Path
    for f in Path(audio_dir).rglob('*.wav'):
        if '-' in f.name:
            f.rename(f.with_name(f.name.replace('-', '_')))


if __name__ == "__main__":
    # audio_dir = "/path/to/data/lanzhou_2015/merged/audio"
    # calculate_total_duration(audio_dir)

    # source_dir = "/path/to/data/lanzhou_2015/audio"
    # target_dir = "/path/to/data/lanzhou_2015/audio_merge"
    
    # merge_audio_files(source_dir, target_dir)

    source_dir = "/path/to/data/Androids-Corpus/workspace/output"
    output_dir = "/path/to/data/example/Androids-Corpus_Italian/Reading-Task/HC"
    rename_and_collect_audio(source_dir, output_dir)
    # audio_dir = "/path/to/data/modma/downstram/segments"
    # rename_audio_files(audio_dir)