import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set

from hydra.utils import instantiate
from omegaconf import DictConfig
import yaml

import numpy as np
import soundfile as sf
import librosa
import torch
from modules.commons import str2bool
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# -----------------------------
# Utilities
# -----------------------------
def ms_to_samples(ms: int, sr: int) -> int:
    return int(round(ms * sr / 1000.0))

def safe_slice_audio(x: np.ndarray, start_samp: int, end_samp: int) -> np.ndarray:
    start_samp = max(0, min(len(x), start_samp))
    end_samp = max(0, min(len(x), end_samp))
    if end_samp <= start_samp:
        return np.zeros((0,), dtype=x.dtype)
    return x[start_samp:end_samp]

def to_mono(x: np.ndarray) -> np.ndarray:
    # soundfile may return (T,) or (T, C)
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-9)

def pad_or_truncate(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    pad = np.zeros((target_len - len(x),), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

# -----------------------------
# Target Speaker
# -----------------------------
# def age_to_group(age: float) -> str:
#     if age < 22:
#         return "younger_than_22"
#     if age < 30:
#         return "22_30"
#     if age < 40:
#         return "30_40"
#     return "older_than_40"

# AGE_GROUP_ORDER = ["younger_than_22", "22_30", "30_40", "older_than_40"]

# def gender_from_logits(logits3: np.ndarray) -> str:
#     logits3 = np.asarray(logits3).reshape(-1)
#     # 只在 female vs male 里取 argmax；child 不参与匹配
#     return "female" if logits3[0] >= logits3[1] else "male"

# def _group_distance(g1: str, g2: str) -> int:
#     i1 = AGE_GROUP_ORDER.index(g1)
#     i2 = AGE_GROUP_ORDER.index(g2)
#     return abs(i1 - i2)

# def pick_target_from_manual_pool(
#     source_age: float,
#     source_gender_logits: np.ndarray,
#     pool_json: Dict[str, Any],          # 你的 aishell_target_speaker.json
#     used_subject_ids: Set[str],         # 全局已分配的 target subject_id
# ) -> Tuple[str, str]:
#     """
#     返回 (target_subject_id, target_wav_path)
#     规则：
#       1) 优先：同 age group + 同 gender，且 target 未被占用
#       2) 如果该类候选用完（比如 source speaker 多于 4），则选择：同 gender + 相邻 age group（距离从 1 开始扩展）
#       3) 若还不够，再继续扩展到更远 age group（仍保持同 gender）
#       4) 若同性别所有 group 都耗尽：最后才允许跨 gender（按 age group 距离从小到大）兜底
#     """
#     sg = gender_from_logits(source_gender_logits)
#     ag = age_to_group(source_age)

#     def iter_candidates(group: str, gender: str) -> List[Dict[str, Any]]:
#         return pool_json.get(group, {}).get(gender, []) or []

#     # Step 1: 同 group 同 gender
#     for cand in iter_candidates(ag, sg):
#         sid = cand["subject_id"]
#         if sid not in used_subject_ids:
#             return sid, cand["wav_path"]

#     # Step 2/3: 同 gender，相邻->更远 group
#     for dist in range(1, len(AGE_GROUP_ORDER)):
#         # 按“相邻”含义：距离 dist 的 group 都算；dist=1 即相邻
#         for g in AGE_GROUP_ORDER:
#             if _group_distance(ag, g) != dist:
#                 continue
#             for cand in iter_candidates(g, sg):
#                 sid = cand["subject_id"]
#                 if sid not in used_subject_ids:
#                     return sid, cand["wav_path"]

#     # Step 4: 跨 gender 兜底（只有在同 gender 全部耗尽时）
#     other_g = "male" if sg == "female" else "female"
#     for dist in range(0, len(AGE_GROUP_ORDER)):
#         for g in AGE_GROUP_ORDER:
#             if _group_distance(ag, g) != dist:
#                 continue
#             for cand in iter_candidates(g, other_g):
#                 sid = cand["subject_id"]
#                 if sid not in used_subject_ids:
#                     return sid, cand["wav_path"]

#     raise RuntimeError("No available target speaker left in aishell_target_speaker.json (all used).")

# -----------------------------
# Diarization processing
# -----------------------------
@dataclass
class DiarItem:
    index: int
    text: str
    speaker: int
    start_ms: int
    end_ms: int

def parse_diarization(diar_list: List[Dict[str, Any]]) -> List[DiarItem]:
    items: List[DiarItem] = []
    for d in diar_list:
        ts = d["timestamp"]
        items.append(
            DiarItem(
                index=int(d["index"]),
                text=str(d.get("text", "")),
                speaker=int(d["speaker"]),
                start_ms=int(ts[0]),
                end_ms=int(ts[-1]),
            )
        )
    # keep original order
    items.sort(key=lambda x: x.index)
    return items

def group_consecutive_by_speaker(items: List[DiarItem]) -> List[Tuple[int, int, int]]:
    """
    Returns list of (speaker_id, start_ms, end_ms) by merging consecutive diar items
    with identical speaker id in the diarization sequence.
    """
    if not items:
        return []
    groups: List[Tuple[int, int, int]] = []
    cur_spk = items[0].speaker
    cur_start = items[0].start_ms
    cur_end = items[0].end_ms
    for it in items[1:]:
        if it.speaker == cur_spk:
            cur_end = max(cur_end, it.end_ms)
        else:
            groups.append((cur_spk, cur_start, cur_end))
            cur_spk = it.speaker
            cur_start = it.start_ms
            cur_end = it.end_ms
    groups.append((cur_spk, cur_start, cur_end))
    return groups

def collect_per_speaker_segments(items: List[DiarItem]) -> Dict[int, List[Tuple[int, int]]]:
    spk2segs: Dict[int, List[Tuple[int, int]]] = {}
    for it in items:
        spk2segs.setdefault(it.speaker, []).append((it.start_ms, it.end_ms))
    return spk2segs

# -----------------------------
# Age/Gender inference (audonnx model wrapper)
# -----------------------------
class AgeGenderPredictor:
    """
    Expects an audonnx model like:
      output = model(signal_float32, sr)
      age = output['logits_age'] * 100
      gender = output['logits_gender']  # logits, shape (3,)
    """
    def __init__(self, audonnx_model):
        self.model = audonnx_model

    def predict(self, wav: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        wav = ensure_float32(to_mono(wav))
        out = self.model(wav, sr)
        age = float(out["logits_age"]) * 100.0
        gender_logits = np.array(out["logits_gender"], dtype=np.float32).reshape(-1)
        return age, gender_logits

# -----------------------------
# Matching to AISHELL speakers
# -----------------------------
def match_target_speaker(
    age: float,
    gender_logits: np.ndarray,
    aishell_pred: Dict[str, Dict[str, Any]],
    w_age: float = 1.0,
    w_gender: float = 1.0,
    age_scale: float = 30.0,
) -> str:
    """
    aishell_pred format:
      { "S0121": {"age": 29.35, "gender": [..3..]}, ... }
    Similarity: normalized age absolute diff + L2 diff of gender probabilities.
    """
    g_p = softmax(gender_logits.astype(np.float64))
    best_spk = None
    best_dist = float("inf")

    for spk, v in aishell_pred.items():
        a2 = float(v["age"])
        g2 = np.array(v["gender"], dtype=np.float64).reshape(-1)
        if g2.size != g_p.size:
            continue
        g2_p = softmax(g2)
        d_age = abs(age - a2) / (age_scale + 1e-9)
        d_g = float(np.linalg.norm(g_p - g2_p, ord=2))
        dist = w_age * d_age + w_gender * d_g
        if dist < best_dist:
            best_dist = dist
            best_spk = spk

    if best_spk is None:
        raise ValueError("No valid AISHELL speaker found for matching (check aishell_pred content).")
    return best_spk

def build_speaker_profile_audio(
    x: np.ndarray,
    sr: int,
    segs_ms: List[Tuple[int, int]],
    inter_silence_sec: float = 0.2,
    max_total_sec: Optional[float] = 60.0,
) -> np.ndarray:
    """
    Concatenate all segments for one speaker into a single waveform for age/gender prediction.
    Inserts a short silence between segments. Optionally cap total duration.
    """
    silence = np.zeros((int(round(inter_silence_sec * sr)),), dtype=np.float32)
    chunks: List[np.ndarray] = []
    total = 0
    cap = None if max_total_sec is None else int(round(max_total_sec * sr))

    for (s_ms, e_ms) in segs_ms:
        s = ms_to_samples(s_ms, sr)
        e = ms_to_samples(e_ms, sr)
        seg = safe_slice_audio(x, s, e)
        seg = ensure_float32(to_mono(seg))
        if seg.size == 0:
            continue
        if cap is not None and total + seg.size > cap:
            seg = seg[: max(0, cap - total)]
        if seg.size == 0:
            break
        chunks.append(seg)
        total += seg.size
        if cap is not None and total >= cap:
            break
        chunks.append(silence)
        total += silence.size
        if cap is not None and total >= cap:
            break

    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)

# -----------------------------
# Main pipeline
# -----------------------------
# def anonymize_multispeaker_seedvc(
#     source_wav_path: str,
#     seedvc_sr: int,
#     diarization_result: List[Dict[str, Any]],
#     aishell_pred: Dict[str, Dict[str, Any]],
#     aishell_train_root: str,  # e.g. "aishell/data_aishell/wav/train"
#     predictor: AgeGenderPredictor,
#     convert_voice_func,  # function: convert_voice(source_wav: np.ndarray, target_wav: np.ndarray, args) -> np.ndarray
#     convert_voice_model,
#     vc_args: Any,
#     silence_fill: bool = True,
#     target_cache: bool = True,
# ) -> Tuple[np.ndarray, int, Dict[int, str]]:
#     """
#     Returns:
#       out_wav (np.ndarray float32 mono),
#       sr (int),
#       spk_id_to_aishell_spk (Dict[int, str])
#     Pipeline:
#       1) build per-speaker profile audio -> age/gender -> match to AISHELL speaker
#       2) group consecutive diar items by speaker -> VC each group -> place into output timeline
#       3) gaps are silence (output initialized as zeros)
#     """
#     x = librosa.load(source_wav_path, sr=seedvc_sr, mono=True)[0].astype(np.float32)
#     sr = seedvc_sr

#     items = parse_diarization(diarization_result)
#     spk2segs = collect_per_speaker_segments(items)

#     # # index AISHELL speaker wavs once
#     # spk2paths = build_aishell_speaker_index(aishell_train_root)

#     # 1) determine target speaker for each diar speaker
#     spk_id_to_aishell_spk: Dict[int, str] = {}
#     aishell_spk_to_targetwav: Dict[str, np.ndarray] = {}  # cache loaded target wavs (mono float32)

#     for spk_id, segs_ms in spk2segs.items():
#         profile_wav = build_speaker_profile_audio(x, sr, segs_ms)
#         if profile_wav.size == 0:
#             # fallback: arbitrary target (first in dict)
#             fallback = next(iter(aishell_pred.keys()))
#             spk_id_to_aishell_spk[spk_id] = fallback
#             continue

#         age, gender_logits = predictor.predict(profile_wav, sr)
#         matched = match_target_speaker(age, gender_logits, aishell_pred)
#         spk_id_to_aishell_spk[spk_id] = matched

#         if target_cache and matched not in aishell_spk_to_targetwav:
#             p = pick_target_wav_path(matched, spk2paths)
#             tgt_wav = librosa.load(p, sr=sr, mono=True)[0].astype(np.float32)

#     # 2) VC per consecutive speaker block
#     groups = group_consecutive_by_speaker(items)

#     # output: silence by default
#     out = np.zeros_like(x) if silence_fill else x.copy()

#     for (spk_id, start_ms, end_ms) in groups:
#         start_s = ms_to_samples(start_ms, sr)
#         end_s = ms_to_samples(end_ms, sr)
#         src_seg = safe_slice_audio(x, start_s, end_s)
#         src_seg = ensure_float32(to_mono(src_seg))
#         if src_seg.size == 0:
#             continue

#         ais_spk = spk_id_to_aishell_spk[spk_id]

#         if target_cache and ais_spk in aishell_spk_to_targetwav:
#             tgt_wav = aishell_spk_to_targetwav[ais_spk]
#         else:
#             p = pick_target_wav_path(ais_spk, spk2paths)
#             tgt_wav = librosa.load(p, sr=sr, mono=True)[0].astype(np.float32)
#             if target_cache:
#                 aishell_spk_to_targetwav[ais_spk] = tgt_wav

#         # SeedVC convert
#         conv = convert_voice_func(convert_voice_model, src_seg, tgt_wav, vc_args)
#         conv = ensure_float32(to_mono(np.asarray(conv)))

#         # place into timeline (pad/truncate to original segment length)
#         seg_len = end_s - start_s
#         conv_fit = pad_or_truncate(conv, seg_len)
#         out[start_s:end_s] = conv_fit

#     return out, sr, spk_id_to_aishell_spk


def anonymize_multispeaker_seedvc(
    source_wav_path: str,
    seedvc_sr: int,
    diarization_result: List[Dict[str, Any]],
    aishell_target_pool: Dict[str, Any],  # <-- NEW: your aishell_target_speaker.json loaded as dict
    predictor: AgeGenderPredictor,
    convert_voice_func,  # function: convert_voice(model, source_wav: np.ndarray, target_wav: np.ndarray, args) -> np.ndarray
    convert_voice_model,
    vc_args: Any,
    silence_fill: bool = True,
    target_cache: bool = True,
) -> Tuple[np.ndarray, int, Dict[int, str]]:
    """
    Returns:
      out_wav (np.ndarray float32 mono),
      sr (int),
      spk_id_to_target_subject_id (Dict[int, str])

    New matching logic:
      - predict source speaker age/gender (using only first minute of that speaker's audio, if your
        build_speaker_profile_audio() already enforces max_total_sec=60; otherwise we pass it here)
      - choose target from aishell_target_pool by (age_group, gender), ensuring NO reuse across speakers
      - if exhausted in that bucket, choose same gender from adjacent age groups (then further), finally cross-gender fallback
    """
    # ---- audio load aligned with SeedVC
    x = librosa.load(source_wav_path, sr=seedvc_sr, mono=True)[0].astype(np.float32)
    sr = seedvc_sr

    items = parse_diarization(diarization_result)
    spk2segs = collect_per_speaker_segments(items)

    # ---- helper: age group + gender
    AGE_GROUP_ORDER = ["younger_than_22", "22_30", "30_40", "older_than_40"]

    def age_to_group(age: float) -> str:
        if age < 22:
            return "younger_than_22"
        if age < 30:
            return "22_30"
        if age < 40:
            return "30_40"
        return "older_than_40"

    def gender_from_logits(logits3: np.ndarray) -> str:
        logits3 = np.asarray(logits3).reshape(-1)
        # female vs male (ignore child)
        return "female" if logits3[0] >= logits3[1] else "male"

    def group_distance(g1: str, g2: str) -> int:
        return abs(AGE_GROUP_ORDER.index(g1) - AGE_GROUP_ORDER.index(g2))

    def iter_candidates(group: str, gender: str):
        return aishell_target_pool.get(group, {}).get(gender, []) or []

    def pick_target(source_age: float, source_gender_logits: np.ndarray, used: set) -> Tuple[str, str]:
        sg = gender_from_logits(source_gender_logits)
        ag = age_to_group(source_age)

        # 1) same group + same gender
        for cand in iter_candidates(ag, sg):
            sid = cand["subject_id"]
            if sid not in used:
                return sid, cand["wav_path"]

        # 2) same gender + adjacent groups (then further)
        for dist in range(1, len(AGE_GROUP_ORDER)):
            for g in AGE_GROUP_ORDER:
                if group_distance(ag, g) != dist:
                    continue
                for cand in iter_candidates(g, sg):
                    sid = cand["subject_id"]
                    if sid not in used:
                        return sid, cand["wav_path"]

        # 3) last resort: cross gender, closest age group first
        og = "male" if sg == "female" else "female"
        for dist in range(0, len(AGE_GROUP_ORDER)):
            for g in AGE_GROUP_ORDER:
                if group_distance(ag, g) != dist:
                    continue
                for cand in iter_candidates(g, og):
                    sid = cand["subject_id"]
                    if sid not in used:
                        return sid, cand["wav_path"]

        raise RuntimeError("No available target speaker left in aishell_target_pool (all used).")

    # ---- 1) determine target for each diar speaker (no repetition)
    spk_id_to_target_sid: Dict[int, str] = {}
    target_sid_to_wav: Dict[str, np.ndarray] = {}
    used_target_sids = set()

    for spk_id, segs_ms in spk2segs.items():
        # take only first minute for prediction
        profile_wav = build_speaker_profile_audio(x, sr, segs_ms, max_total_sec=30.0)
        if profile_wav.size == 0:
            # fallback: pick any unused target from pool, prefer female younger_than_22 then scan
            picked = None
            for g in AGE_GROUP_ORDER:
                for gender in ["female", "male"]:
                    for cand in iter_candidates(g, gender):
                        sid = cand["subject_id"]
                        if sid not in used_target_sids:
                            picked = (sid, cand["wav_path"])
                            break
                    if picked:
                        break
                if picked:
                    break
            if not picked:
                raise RuntimeError("Fallback failed: no unused target speaker left.")
            t_sid, t_path = picked
        else:
            age, gender_logits = predictor.predict(profile_wav, sr)
            t_sid, t_path = pick_target(age, gender_logits, used_target_sids)

        used_target_sids.add(t_sid)
        spk_id_to_target_sid[spk_id] = t_sid

        if target_cache and t_sid not in target_sid_to_wav:
            target_sid_to_wav[t_sid] = librosa.load(t_path, sr=sr, mono=True)[0].astype(np.float32)

    # ---- 2) VC per consecutive speaker block
    groups = group_consecutive_by_speaker(items)
    out = np.zeros_like(x) if silence_fill else x.copy()

    for (spk_id, start_ms, end_ms) in groups:
        start_s = ms_to_samples(start_ms, sr)
        end_s = ms_to_samples(end_ms, sr)

        src_seg = safe_slice_audio(x, start_s, end_s)
        src_seg = ensure_float32(to_mono(src_seg))
        if src_seg.size == 0:
            continue

        t_sid = spk_id_to_target_sid[spk_id]
        if target_cache and t_sid in target_sid_to_wav:
            tgt_wav = target_sid_to_wav[t_sid]
        else:
            # load from pool on-demand (need wav_path)
            # find wav_path by scanning (small pool, acceptable). If you want O(1), build sid->path map once outside.
            t_path = None
            for g in AGE_GROUP_ORDER:
                for gender in ["female", "male"]:
                    for cand in iter_candidates(g, gender):
                        if cand["subject_id"] == t_sid:
                            t_path = cand["wav_path"]
                            break
                    if t_path:
                        break
                if t_path:
                    break
            if t_path is None:
                raise RuntimeError(f"Cannot find wav_path for target subject_id={t_sid} in aishell_target_pool.")
            tgt_wav = librosa.load(t_path, sr=sr, mono=True)[0].astype(np.float32)
            if target_cache:
                target_sid_to_wav[t_sid] = tgt_wav

        conv = convert_voice_func(convert_voice_model, src_seg, tgt_wav, vc_args)
        conv = ensure_float32(to_mono(np.asarray(conv)))

        seg_len = end_s - start_s
        conv_fit = pad_or_truncate(conv, seg_len)
        out[start_s:end_s] = conv_fit

    return out, sr, spk_id_to_target_sid


def load_v2_models(args):
    """Load V2 models using the wrapper from app.py"""
    
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


def convert_voice_v2(vc_wrapper_v2, source_wave, target_wave, args):

    # Use the generator function but collect all outputs
    generator = vc_wrapper_v2.convert_voice_with_streaming_wave(
        source_wave=source_wave,
        target_wave=target_wave,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=True
    )

    # Collect all outputs from the generator
    for output in generator:
        _, full_audio = output
    sr, signal = full_audio
    return signal


def get_parser():

    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument("--diffusion-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)")
    parser.add_argument("--compile", type=bool, default=False,
                        help="Whether to compile the model for faster inference")

    # V2 specific arguments
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7,
                        help="Intelligibility CFG rate for V2 model")
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7,
                        help="Similarity CFG rate for V2 model")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter for V2 model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature sampling parameter for V2 model")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for V2 model")
    parser.add_argument("--convert-style", type=str2bool, default=False,
                        help="Convert style/emotion/accent for V2 model")
    parser.add_argument("--anonymization-only", type=str2bool, default=False,
                        help="Anonymization only mode for V2 model")

    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    
    # Anonymise paths
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--diarization_path", type=str, required=True)
    
    return parser.parse_args()

# -----------------------------
# Example usage (fill in your objects)
# -----------------------------
if __name__ == "__main__":
    args = get_parser()
    # 1) Load your audonnx model outside, pass into predictor:
    import audonnx
    model_root = "/path/to/cuizy/Anonymise_Agent/w2v2-L-robust-6-age-gender"
    ag_model = audonnx.load(model_root, device='cuda:0', num_workers=16)
    predictor = AgeGenderPredictor(ag_model)

    # Load seedvc
    vc_wrapper_v2 = load_v2_models(args)
    seedvc_sr = vc_wrapper_v2.sr

    # Load paths
    # source_wav_path = "201901038290.wav"
    # diarization_json_path = "/path/to/data/crisis/Exp1/middle/trans/201901038290.json"
    # diarization_result = json.load(open(diarization_json_path, "r"))
    if os.path.isfile(args.source_path):
        source_wav_path_list = [args.source_path]
    elif os.path.isdir(args.source_path):
        # source_wav_path_list = [os.path.join(args.source_path, wav_name) for wav_name in os.listdir(args.source_path)]
        source_wav_path_list = []
        for wav_name in os.listdir(args.source_path):
            if wav_name.endswith(".wav"):
                source_wav_path_list.append(os.path.join(args.source_path, wav_name))
    else:
        print(f"Source path not exist.")
    os.makedirs(args.save_path, exist_ok=True)

    aishell_pred_path = "/path/to/cuizy/Anonymise_Agent/aishell_target_speaker.json"
    aishell_target_pool = json.load(open(aishell_pred_path, "r"))


    # 2) Provide:
    # - source_wav_path
    # - diarization_result (list of dicts)
    # - aishell_pred (dict)
    # - aishell_train_root: "aishell/data_aishell/wav/train"
    # - convert_voice function and vc_args

    for source_wav_path in source_wav_path_list:
        print(source_wav_path)
        id = source_wav_path.split('/')[-1].split('.')[0]
        save_wav_path = os.path.join(args.save_path, f"{id}.wav")
        if os.path.exists(save_wav_path):
            continue
        if os.path.isfile(args.diarization_path):
            diarization_json_path = args.diarization_path
        else:
            diarization_json_path = os.path.join(args.diarization_path, f"{id}.json")
        diarization_result = json.load(open(diarization_json_path, "r"))
        try:
            out, sr, mapping = anonymize_multispeaker_seedvc(
                source_wav_path=source_wav_path,
                diarization_result=diarization_result,
                aishell_target_pool=aishell_target_pool,
                # aishell_pred=aishell_pred_dict,
                # aishell_train_root="/path/to/aishell/data_aishell/wav/train",
                predictor=predictor,
                convert_voice_model=vc_wrapper_v2,
                convert_voice_func=convert_voice_v2,
                vc_args=args,
                seedvc_sr=seedvc_sr
            )
            # sf.write(f"output/aishell_anonymized_{source_wav_path.split('/')[-1]}", out, sr)
            wav_resample = librosa.resample(out, orig_sr=sr, target_sr=16000)
            sf.write(save_wav_path, wav_resample, samplerate=16000)
            print("speaker->target:", mapping)
        except Exception as e:
            with open(os.path.join(args.save_path, "fail_log.log"), 'a+') as f:
                f.write(f"[ERROR] {id} {type(e).__name__}: {e}\n")
