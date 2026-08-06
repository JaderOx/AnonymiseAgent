"""
utils/voice_convert/seedvc.py — SeedVC v2 神经网络声线转换

流程：
  对每个音频文件：
    1. 读取转录 JSON（含 speaker 字段）获取说话人分段
    2. 为每位说话人拼接音频 → 预测年龄/性别 → 匹配 AISHELL 目标说话人
    3. 按连续说话人块逐块调用 SeedVC 转换
    4. 拼回原始时间轴并保存

依赖：
  pip install hydra-core omegaconf audonnx audeer
  SeedVC 代码位于项目根目录 seedvc/ 子目录
  audonnx 模型位于 models/w2v2-L-robust-6-age-gender/
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# ── SeedVC 模块路径 ───────────────────────────────────────────────────────────
_SEEDVC_DIR = Path(__file__).resolve().parent.parent.parent / 'seedvc'
if str(_SEEDVC_DIR) not in sys.path:
    sys.path.insert(0, str(_SEEDVC_DIR))

# ── 注册表元数据 ──────────────────────────────────────────────────────────────
_DEFAULT_AGE_GENDER_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / 'models' / 'w2v2-L-robust-6-age-gender'
)
_DEFAULT_TARGET_POOL = str(
    Path(__file__).resolve().parent / 'aishell' / 'aishell_target_speaker.json'
)

DESCRIPTION = "SeedVC v2 神经网络声线转换（按说话人匹配年龄/性别目标音色，需 GPU）"
DEFAULTS = {
    "aishell_target_pool_path": _DEFAULT_TARGET_POOL,
    "age_gender_model_path":    _DEFAULT_AGE_GENDER_MODEL,
    "age_gender_device":        "cpu",         # 年龄/性别模型推理设备
    "ar_checkpoint_path":       None,          # SeedVC AR 权重，None = 用 yaml 默认
    "cfm_checkpoint_path":      None,          # SeedVC CFM 权重，None = 用 yaml 默认
    "diffusion_steps":          30,
    "length_adjust":            1.0,
    "intelligibility_cfg_rate": 0.7,
    "similarity_cfg_rate":      0.7,
    "top_p":                    0.9,
    "temperature":              1.0,
    "repetition_penalty":       1.0,
    "convert_style":            False,
    "anonymization_only":       True,
    "output_sr":                16000,
    "pad_ms":                   200,   # 每块前后延拓上下文（ms）
    "crossfade_ms":             50,    # 块边界交叉淡化（ms）
    "max_chunk_s":              180.0,  # 同说话人合并块最大时长（秒）
}


# ─────────────────────────────────────────────────────────────────────────────
# 音频辅助
# ─────────────────────────────────────────────────────────────────────────────

def _ms_to_samples(ms: int, sr: int) -> int:
    return int(round(ms * sr / 1000.0))


def _safe_slice(x: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(len(x), start))
    end   = max(0, min(len(x), end))
    return x[start:end] if end > start else np.zeros(0, dtype=x.dtype)


def _to_mono_f32(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _pad_or_truncate(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def _match_rms(ref: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """将 target 的 RMS 缩放到与 ref 相同，保持波形形状不变。"""
    rms_ref = np.sqrt(np.mean(ref ** 2) + eps)
    rms_tgt = np.sqrt(np.mean(target ** 2) + eps)
    return target * (rms_ref / rms_tgt)


def _paste_with_crossfade(sig: np.ndarray, core: np.ndarray,
                          start_s: int, end_s: int, cf: int) -> np.ndarray:
    """线性交叉淡化将 core 贴回 sig[start_s:end_s]，其余区域保持原样。"""
    result = sig.copy()
    L  = end_s - start_s
    cf = min(cf, L // 4)
    if cf > 0:
        fade_in  = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        result[start_s : start_s + cf] = (
            sig[start_s : start_s + cf] * fade_out + core[:cf] * fade_in
        )
        result[end_s - cf : end_s] = (
            core[-cf:] * fade_out + sig[end_s - cf : end_s] * fade_in
        )
        result[start_s + cf : end_s - cf] = core[cf : L - cf]
    else:
        result[start_s:end_s] = core
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 转录/说话人分段解析
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DiarItem:
    index:    int
    speaker:  int
    start_ms: int
    end_ms:   int
    age:      Optional[float] = None    # 转录阶段写入，None = 需推理
    gender:   Optional[str]   = None    # 'male' / 'female' / None


def _parse_diarization(diar_list: list) -> list:
    items = []
    for d in diar_list:
        ts = d['timestamp']
        items.append(_DiarItem(
            index=int(d['index']),
            speaker=int(d['speaker']) if 'speaker' in d else 0,  # 无说话人字段时视为同一人
            start_ms=int(ts[0]),
            end_ms=int(ts[-1]),
            age=d.get('age'),
            gender=d.get('gender'),
        ))
    items.sort(key=lambda x: x.index)
    return items


def _collect_per_speaker(items: list) -> dict:
    spk2segs: dict = {}
    for it in items:
        spk2segs.setdefault(it.speaker, []).append((it.start_ms, it.end_ms))
    return spk2segs


def _build_profile_audio(x: np.ndarray, sr: int,
                          segs_ms: list, max_total_sec: float = 30.0) -> np.ndarray:
    """拼接说话人各段音频（加短静音间隔），供年龄/性别预测使用，最多 max_total_sec 秒。"""
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    cap = int(max_total_sec * sr)
    chunks: list = []
    total = 0
    for (s_ms, e_ms) in segs_ms:
        seg = _safe_slice(x, _ms_to_samples(s_ms, sr), _ms_to_samples(e_ms, sr))
        seg = _to_mono_f32(seg)
        if seg.size == 0:
            continue
        if total + seg.size > cap:
            seg = seg[:max(0, cap - total)]
        if seg.size == 0:
            break
        chunks.append(seg)
        total += seg.size
        if total >= cap:
            break
        chunks.append(silence)
        total += silence.size
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 目标说话人匹配（年龄/性别 → AISHELL pool）
# ─────────────────────────────────────────────────────────────────────────────

_AGE_GROUPS = ['younger_than_22', '22_30', '30_40', 'older_than_40']


def _age_to_group(age: float) -> str:
    if age < 22:
        return 'younger_than_22'
    if age < 30:
        return '22_30'
    if age < 40:
        return '30_40'
    return 'older_than_40'


def _gender_from_logits(logits3: np.ndarray) -> str:
    logits3 = np.asarray(logits3).reshape(-1)
    return 'female' if logits3[0] >= logits3[1] else 'male'


def _pick_target(age: float, gender_logits: np.ndarray,
                 pool: dict, used: set) -> Tuple[str, str]:
    """
    从 aishell_target_pool 按年龄组+性别优先匹配，避免重用。
    返回 (subject_id, wav_path)。
    """
    sg = _gender_from_logits(gender_logits)
    ag = _age_to_group(age)

    def cands(group, gender):
        return pool.get(group, {}).get(gender, []) or []

    def dist(g1, g2):
        return abs(_AGE_GROUPS.index(g1) - _AGE_GROUPS.index(g2))

    # 1) 同年龄组 + 同性别
    for c in cands(ag, sg):
        if c['subject_id'] not in used:
            return c['subject_id'], c['wav_path']

    # 2) 同性别 + 相邻年龄组（从近到远）
    for d in range(1, len(_AGE_GROUPS)):
        for g in _AGE_GROUPS:
            if dist(ag, g) != d:
                continue
            for c in cands(g, sg):
                if c['subject_id'] not in used:
                    return c['subject_id'], c['wav_path']

    # 3) 跨性别兜底
    og = 'male' if sg == 'female' else 'female'
    for d in range(len(_AGE_GROUPS)):
        for g in _AGE_GROUPS:
            if dist(ag, g) != d:
                continue
            for c in cands(g, og):
                if c['subject_id'] not in used:
                    return c['subject_id'], c['wav_path']

    raise RuntimeError('No unused target speaker left in aishell_target_pool.')


def _first_available(pool: dict, used: set):
    """兜底：返回第一个未被使用的目标说话人 (subject_id, wav_path)。"""
    for grp in _AGE_GROUPS:
        for gender in ['female', 'male']:
            for c in (pool.get(grp, {}).get(gender, []) or []):
                if c['subject_id'] not in used:
                    return c['subject_id'], c['wav_path']
    raise RuntimeError('No available target speaker in aishell_target_pool (all used).')


# ─────────────────────────────────────────────────────────────────────────────
# SeedVC 模型加载 / 推理
# ─────────────────────────────────────────────────────────────────────────────

def _load_seedvc_model(ar_checkpoint_path=None, cfm_checkpoint_path=None, device=None):
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    import yaml

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg_path = _SEEDVC_DIR / 'configs' / 'v2' / 'vc_wrapper.yaml'
    cfg = DictConfig(yaml.safe_load(open(str(cfg_path), 'r')))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=ar_checkpoint_path,
        cfm_checkpoint_path=cfm_checkpoint_path,
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(
        max_batch_size=1, max_seq_len=4096,
        dtype=torch.float16, device=device,
    )
    return vc_wrapper


def _convert_voice(vc_wrapper, source_wave: np.ndarray, target_wave: np.ndarray,
                   diffusion_steps=30, length_adjust=1.0,
                   intelligibility_cfg_rate=0.7, similarity_cfg_rate=0.7,
                   top_p=0.9, temperature=1.0, repetition_penalty=1.0,
                   convert_style=False, anonymization_only=True,
                   device=None, **_):
    if device is None:
        device = next(vc_wrapper.parameters()).device

    gen = vc_wrapper.convert_voice_with_streaming_wave(
        source_wave=source_wave,
        target_wave=target_wave,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligebility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
        device=device,
        dtype=torch.float16,
        stream_output=True,
    )
    for _, full_audio in gen:
        pass
    sr_out, signal = full_audio
    return signal, sr_out


# ─────────────────────────────────────────────────────────────────────────────
# 单文件匿名
# ─────────────────────────────────────────────────────────────────────────────

def _anonymize_file(source_wav_path: str, output_path: str, trans_json_path: str,
                    vc_wrapper, seedvc_sr: int, predictor,
                    aishell_target_pool: dict, output_sr: int = 16000,
                    pad_ms: int = 150, crossfade_ms: int = 30,
                    max_chunk_s: float = 15.0,
                    **vc_kwargs):
    """单文件多说话人 SeedVC 匿名：逐句处理，交叉淡化贴回，间隙保留原始音频。
    trans_json_path 为 None 时走全音频模式：整段送入 VC，不拆句，不贴回。
    """
    import logging
    _log = logging.getLogger(__name__)

    x = librosa.load(source_wav_path, sr=seedvc_sr, mono=True)[0].astype(np.float32)
    stem = Path(source_wav_path).stem

    # ── 全音频模式（无转录 JSON）────────────────────────────────────────────────
    if trans_json_path == "None":
        _log.debug(f"  {stem}: 全音频模式（无转录 JSON），整段 VC")
        if predictor is not None:
            age, gender_logits = predictor.predict(x, seedvc_sr)
            _log.debug(f"  {stem}: 推理 age={age:.1f}")
            t_sid, t_path = _pick_target(age, gender_logits, aishell_target_pool, set())
        else:
            t_sid, t_path = _first_available(aishell_target_pool, set())
        _log.debug(f"  {stem}: 全音频 → 目标 {t_sid}")

        tgt_wav = librosa.load(t_path, sr=seedvc_sr, mono=True)[0].astype(np.float32)
        conv, _ = _convert_voice(vc_wrapper, x, tgt_wav, **vc_kwargs)
        out = _to_mono_f32(np.asarray(conv))
        out = _match_rms(x, out)
        if output_sr != seedvc_sr:
            out = librosa.resample(out, orig_sr=seedvc_sr, target_sr=output_sr)
        sf.write(output_path, out, output_sr)
        return
    else:
        # ── 分句模式（有转录 JSON）──────────────────────────────────────────────────
        raw = json.load(open(trans_json_path, 'r', encoding='utf-8'))
        if isinstance(raw, dict):
            sentences    = raw.get('sentences', [])
            speaker_info = raw.get('speaker_info', {})
        else:
            sentences    = raw
            speaker_info = {}

        items    = _parse_diarization(sentences)
        spk2segs = _collect_per_speaker(items)

        # 每个说话人的 JSON 年龄/性别（优先 speaker_info，回退旧格式）
        spk_from_json: Dict[int, dict] = {}
        for spk_id in spk2segs:
            key = str(spk_id)
            if key in speaker_info:
                info = speaker_info[key]
                if info.get('age') is not None or info.get('gender') is not None:
                    spk_from_json[spk_id] = {'age': info.get('age'), 'gender': info.get('gender')}
        if not spk_from_json:
            for it in items:
                if it.speaker not in spk_from_json and (it.age is not None or it.gender is not None):
                    spk_from_json[it.speaker] = {'age': it.age, 'gender': it.gender}

        # ── 1) 为每位说话人匹配目标音色（每个说话人一个固定目标） ──────────────────
        used_targets: set = set()
        spk_to_target: Dict[int, str]          = {}
        target_to_wav: Dict[str, np.ndarray]   = {}

        for spk_id, segs_ms in spk2segs.items():
            json_info = spk_from_json.get(spk_id, {})
            age_j     = json_info.get('age')
            gender_j  = json_info.get('gender')

            if age_j is not None and gender_j is not None:
                age           = float(age_j)
                gender_str    = str(gender_j)
                gender_logits = np.array(
                    [1.0, 0.0, 0.0] if gender_str == 'female' else [0.0, 1.0, 0.0],
                    dtype=np.float32,
                )
                _log.debug(f"  {stem} spk{spk_id}: JSON age={age:.1f} gender={gender_str}")
                t_sid, t_path = _pick_target(age, gender_logits, aishell_target_pool, used_targets)
            elif predictor is not None:
                profile = _build_profile_audio(x, seedvc_sr, segs_ms)
                if profile.size > 0:
                    age, gender_logits = predictor.predict(profile, seedvc_sr)
                    _log.debug(f"  {stem} spk{spk_id}: 推理 age={age:.1f}")
                    t_sid, t_path = _pick_target(age, gender_logits, aishell_target_pool, used_targets)
                else:
                    t_sid, t_path = _first_available(aishell_target_pool, used_targets)
            else:
                t_sid, t_path = _first_available(aishell_target_pool, used_targets)

            used_targets.add(t_sid)
            spk_to_target[spk_id] = t_sid
            if t_sid not in target_to_wav:
                target_to_wav[t_sid] = librosa.load(t_path, sr=seedvc_sr, mono=True)[0].astype(np.float32)
            _log.debug(f"  {stem} spk{spk_id} → 目标 {t_sid}")

        # ── 2) 合并同说话人相邻句（含中间沉默）→ chunks ─────────────────────────
        max_chunk_ms = int(max_chunk_s * 1000)
        chunks: List[tuple] = []   # (speaker, start_ms, end_ms, first_index)
        if items:
            cur_spk    = items[0].speaker
            cur_start  = items[0].start_ms
            cur_end    = items[0].end_ms
            cur_idx    = items[0].index
            for it in items[1:]:
                merged_end = it.end_ms
                if it.speaker == cur_spk and (merged_end - cur_start) <= max_chunk_ms:
                    cur_end = merged_end          # 合并：延伸终点（中间沉默自然包含）
                else:
                    chunks.append((cur_spk, cur_start, cur_end, cur_idx))
                    cur_spk   = it.speaker
                    cur_start = it.start_ms
                    cur_end   = it.end_ms
                    cur_idx   = it.index
            chunks.append((cur_spk, cur_start, cur_end, cur_idx))
        _log.debug(f"  {stem}: {len(items)} 句 → {len(chunks)} 块")

        # ── 3) 逐块 VC + 交叉淡化贴回（间隙保留原始音频） ────────────────────────
        pad_s = int(pad_ms      * seedvc_sr / 1000)
        cf_s  = int(crossfade_ms * seedvc_sr / 1000)
        out   = x.copy()   # 从原始音频起步，只替换有句子的区域

        for (spk_id, start_ms, end_ms, first_idx) in chunks:
            start_s = _ms_to_samples(start_ms, seedvc_sr)
            end_s   = _ms_to_samples(end_ms,   seedvc_sr)
            if end_s <= start_s or start_s >= len(x):
                continue
            end_s = min(end_s, len(x))

            # 带 padding 的上下文片段
            pad_start = max(0, start_s - pad_s)
            pad_end   = min(len(x), end_s + pad_s)
            segment   = _to_mono_f32(x[pad_start:pad_end])

            tgt_wav = target_to_wav[spk_to_target[spk_id]]
            try:
                conv, _ = _convert_voice(vc_wrapper, segment, tgt_wav, **vc_kwargs)
            except Exception as e:
                _log.warning(f"  {stem} 块(首句{first_idx}) VC 失败，保留原始: {e}")
                continue

            conv = _to_mono_f32(np.asarray(conv))

            # 去掉 padding，取核心区域
            core_start = start_s - pad_start
            core_end   = end_s   - pad_start
            core       = conv[core_start:core_end]
            target_len = end_s - start_s
            if len(core) < target_len:
                core = np.pad(core, (0, target_len - len(core)))
            else:
                core = core[:target_len]

            # RMS 对齐：把 VC 结果的响度拉回原始片段水平
            orig_seg = x[start_s:end_s]
            core = _match_rms(orig_seg, core)

            out = _paste_with_crossfade(out, core, start_s, end_s, cf_s)
            _log.debug(f"  块(首句{first_idx}) spk{spk_id}: {start_ms}~{end_ms}ms VC完成")

        # ── 3) 重采样并保存 ───────────────────────────────────────────────────────
        if output_sr != seedvc_sr:
            out = librosa.resample(out, orig_sr=seedvc_sr, target_sr=output_sr)
        sf.write(output_path, out, output_sr)


# ─────────────────────────────────────────────────────────────────────────────
# 批量接口（注册表 batch_fn）
# ─────────────────────────────────────────────────────────────────────────────

def batch_convert_fn(input_dir: str, output_dir: str, logger,
                     trans_dir: str = None,
                     aishell_target_pool_path: str = None,
                     age_gender_model_path: str = None,
                     age_gender_device: str = 'cpu',
                     ar_checkpoint_path: str = None,
                     cfm_checkpoint_path: str = None,
                     output_sr: int = 16000,
                     pad_ms: int = 200,
                     crossfade_ms: int = 50,
                     max_chunk_s: float = 15.0,
                     mask_dir: str = None,
                     mask_kwargs: dict = None,
                     **vc_kwargs):
    """
    批量 SeedVC 声线匿名（注册表 batch_fn 入口）。

    Args:
        input_dir               : 待匿名音频目录
        output_dir              : 输出目录
        logger                  : 日志对象
        trans_dir               : 转录 JSON 目录（必须含 speaker 字段）
        aishell_target_pool_path: aishell_target_speaker.json 路径
        age_gender_model_path   : audonnx w2v2-L-robust-6 模型目录
        age_gender_device       : 年龄/性别模型推理设备（默认 cpu）
        ar_checkpoint_path      : SeedVC AR 模型权重（None = 使用 yaml 默认）
        cfm_checkpoint_path     : SeedVC CFM 模型权重（None = 使用 yaml 默认）
        output_sr               : 输出采样率（默认 16000）
        **vc_kwargs             : 透传给 SeedVC 推理参数
    """
    # if not trans_dir or not os.path.isdir(trans_dir):
    #     raise ValueError("seedvc 需要有效的 trans_dir（含 speaker 字段的转录 JSON）")
    if not aishell_target_pool_path or not os.path.exists(aishell_target_pool_path):
        raise ValueError(f"seedvc 需要有效的 aishell_target_pool_path: {aishell_target_pool_path}")
    if not age_gender_model_path or not os.path.isdir(age_gender_model_path):
        raise ValueError(f"seedvc 需要有效的 age_gender_model_path: {age_gender_model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    predictor = None
    if age_gender_model_path and os.path.isdir(age_gender_model_path):
        logger.info("SeedVC: 加载年龄/性别模型...")
        from utils.age_gender import AgeGenderPredictor
        predictor = AgeGenderPredictor(age_gender_model_path, device=age_gender_device)
        logger.debug(f"SeedVC: 年龄/性别模型加载完成 device={age_gender_device}")
    else:
        logger.info(
            "SeedVC: 未提供 age_gender_model_path，将优先从 JSON 读取 age/gender，"
            "无法读取时随机选取目标说话人"
        )

    logger.info("SeedVC: 加载 SeedVC v2 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vc_wrapper = _load_seedvc_model(ar_checkpoint_path, cfm_checkpoint_path, device)
    seedvc_sr  = vc_wrapper.sr
    logger.info(f"SeedVC: 模型加载完成，内部采样率={seedvc_sr}，输出={output_sr}，设备={device}")

    aishell_pool = json.load(open(aishell_target_pool_path, 'r', encoding='utf-8'))

    SUPPORTED = {'.wav', '.flac', '.mp3', '.ogg', '.aiff'}
    _input_dir_p = Path(input_dir)
    audio_files = sorted(p for p in _input_dir_p.rglob('*')
                         if p.suffix.lower() in SUPPORTED)
    if not audio_files:
        logger.warning(f"目录 {input_dir} 中无音频文件")
        return

    ok = failed = skipped = 0
    t_start = time.time()
    length = len(audio_files)

    for i, fp in enumerate(audio_files):
        rel = fp.relative_to(_input_dir_p)
        out_fp = Path(output_dir) / rel
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        if out_fp.exists() and out_fp.stat().st_size > 0:
            logger.info(f"已存在，跳过: {rel}")
            skipped += 1
            continue
        json_path = None
        if trans_dir:
            json_path = Path(trans_dir) / rel.with_suffix('.json')
            if not json_path.exists():
                logger.warning(f"未找到转录 JSON {json_path}，跳过 {rel}")
                failed += 1
                continue

        logger.debug(f"SeedVC 处理: {rel}  json={json_path}")
        t0 = time.time()
        try:
            _anonymize_file(
                str(fp), str(out_fp), str(json_path),
                vc_wrapper, seedvc_sr, predictor, aishell_pool,
                output_sr=output_sr, pad_ms=pad_ms, crossfade_ms=crossfade_ms,
                max_chunk_s=max_chunk_s, **vc_kwargs,
            )
            # VC 完立刻 in-place mask
            if mask_dir:
                mask_json = Path(mask_dir) / rel.with_suffix('.json')
                if mask_json.exists():
                    from apply_mask import add_adaptive_ssn_to_audio
                    add_adaptive_ssn_to_audio(str(out_fp), str(mask_json), str(out_fp),
                                              **(mask_kwargs or {}))
                else:
                    logger.warning(f"未找到 mask JSON {mask_json}，跳过 mask")
            ok += 1
            logger.info(f"[{i+1}/{length}] 完成: {rel}  ({time.time()-t0:.2f}s)")
        except Exception:
            import traceback
            failed += 1
            logger.error(f"失败: {fp.name}\n{traceback.format_exc()}")

    logger.info(
        f"SeedVC 批量完成：成功 {ok}，跳过 {skipped}，失败 {failed}  "
        f"总耗时 {time.time()-t_start:.1f}s"
    )
