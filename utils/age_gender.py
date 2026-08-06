"""
utils/age_gender.py — 年龄/性别预测（audonnx w2v2-L-robust-6）

AgeGenderPredictor  : 对单段音频推理年龄（0~100 岁）和性别（female/male/child logits）
build_speaker_database: 批量处理说话人目录，生成 aishell_metadata.json 格式的 JSON

依赖：
  pip install audonnx audeer soundfile numpy

模型目录（model.onnx + model.yaml）：
  默认放在项目 models/w2v2-L-robust-6-age-gender/
  或通过 model_root 参数指定
"""

import os
import json
import numpy as np
import soundfile as sf


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x


class AgeGenderPredictor:
    """
    封装 audonnx w2v2-L-robust-6-age-gender 模型。

    Args:
        model_root  : 模型目录（含 model.onnx / model.yaml）
        device      : 'cpu' 或 'cuda:N'
        num_workers : ONNX Runtime 并行线程数
    """

    def __init__(self, model_root: str, device: str = 'cpu', num_workers: int = 4):
        import audonnx
        self.model = audonnx.load(model_root, device=device, num_workers=num_workers)

    def predict(self, wav: np.ndarray, sr: int):
        """
        对单段音频推理年龄和性别。

        Returns:
            age           : float，预测年龄（岁）
            gender_logits : np.ndarray shape (3,)，[female, male, child] logits
        """
        wav = _ensure_float32(_to_mono(wav))
        out = self.model(wav, sr)
        age = float(out['logits_age']) * 100.0
        gender_logits = np.array(out['logits_gender'], dtype=np.float32).reshape(-1)
        return age, gender_logits


def build_speaker_database(
    audio_root: str,
    model_root: str,
    output_json: str,
    device: str = 'cpu',
    max_files_per_speaker: int = 10,
) -> dict:
    """
    遍历 audio_root（子目录 = 说话人 ID），每位说话人取前
    max_files_per_speaker 个 .wav 文件推理年龄/性别，取均值，
    写入 output_json。

    输出格式（与 aishell_metadata.json 兼容）：
      { "S0001": {"age": 28.5, "gender": [0.02, 0.97, 0.01]}, ... }

    Args:
        audio_root             : 根目录，子目录名即说话人 ID
        model_root             : audonnx 模型目录
        output_json            : 输出 JSON 路径
        device                 : 推理设备
        max_files_per_speaker  : 每位说话人最多使用的文件数

    Returns:
        metadata dict
    """
    predictor = AgeGenderPredictor(model_root, device=device)
    metadata: dict = {}

    speakers = sorted(
        s for s in os.listdir(audio_root)
        if os.path.isdir(os.path.join(audio_root, s))
    )

    for speaker in speakers:
        spk_path = os.path.join(audio_root, speaker)
        print(f"  [{speaker}]", end=" ", flush=True)

        all_ages: list = []
        all_genders: list = []

        for i, fname in enumerate(sorted(os.listdir(spk_path))):
            if not fname.lower().endswith('.wav'):
                continue
            signal, sr = sf.read(os.path.join(spk_path, fname))
            age, gender_logits = predictor.predict(signal, sr)
            all_ages.append(age)
            all_genders.append(gender_logits)
            if i + 1 >= max_files_per_speaker:
                break

        if not all_ages:
            print("(no wav files, skipped)")
            continue

        metadata[speaker] = {
            'age':    float(np.mean(all_ages)),
            'gender': np.mean(np.stack(all_genders, axis=0), axis=0).tolist(),
        }
        print(f"age={metadata[speaker]['age']:.1f}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(metadata)} speakers → {output_json}")
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    # 默认模型路径：项目 models/ 目录
    _default_model = str(
        Path(__file__).resolve().parent.parent / 'models' / 'w2v2-L-robust-6-age-gender'
    )

    parser = argparse.ArgumentParser(description='批量年龄/性别预测，输出说话人数据库 JSON')
    parser.add_argument('--audio_root',  required=True,  help='音频根目录（子目录=说话人）')
    parser.add_argument('--output_json', required=True,  help='输出 JSON 路径')
    parser.add_argument('--model_root',  default=_default_model, help='audonnx 模型目录')
    parser.add_argument('--device',      default='cpu',  help='推理设备，如 cuda:0')
    parser.add_argument('--max_files',   type=int, default=10, help='每位说话人最多使用文件数')
    args = parser.parse_args()

    build_speaker_database(
        audio_root=args.audio_root,
        model_root=args.model_root,
        output_json=args.output_json,
        device=args.device,
        max_files_per_speaker=args.max_files,
    )
