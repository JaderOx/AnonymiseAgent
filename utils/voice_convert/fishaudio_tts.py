"""
utils/voice_convert/fishaudio_tts.py

以 TTS 方式替代 VC（匿名音色，不沿用原录音时间轴）：
  1) 读取 trans_dir 句级转录（index/text/speaker）
  2) 可选：读取 mask_dir 隐私词结果，把命中的词替换为“某”；未提供 mask_dir 或缺少对应 JSON 时按转录原文合成
  3) 参考音色：按「流水线语言」与转录 JSON 的 speaker_info.gender，从 tts_ref_data_root 下
     audios/texts 选取 {lang}_{m|w}_{0|1}（支持多语言前缀，常见如 en/ch/es/ja...）。
     语言来源顺序：batch_convert 的 language 参数 → 转录 JSON 顶层的 language / asr_language /
     pipeline_language / tts_language → 环境变量 ANONYMISE_AGENT_LANGUAGE 或 AGENT_LANGUAGE；
     皆无则按中文 ref。fishaudio_ref_map_json 可覆盖同名 spk。
     未启用动态选参考或无 speaker_info / 选文件失败时回退 fishaudio_ref_wav / _2。
  4) 可选：将同一说话人连续多句合并为单次合成（fishaudio_merge_max_chars 限制单段最大字数）
  5) 每条送入 Fish 的片段均加情感标签（默认 <Suffocating>）；空 fishaudio_emotion 则不加。
     dialogue JSON 中仍记录无标签的脱敏正文。
  6) 直接保存拼接后的长音频（不写中间 segments 目录）；并将合并段元数据写入 *_tts_dialogue.json
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch


DESCRIPTION = (
    "FishAudio TTS（可选按语言+speaker_info 选多语言参考音色，敏感词替换为”某”）"
)

# ── 默认路径（基于项目根目录，可通过环境变量覆盖）────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_TTS_REF_ROOT = os.path.join(_PROJECT_ROOT, "tts_reference")

DEFAULTS = {
    "fishaudio_model_path": os.path.join(_PROJECT_ROOT, "models/fish_speech_models/s2-pro"),
    "fish_speech_root": os.path.join(_PROJECT_ROOT, "models/fish-speech"),
    "fishaudio_device": "cuda",
    "output_sr": 16000,
    # 第一说话人（单人音频只会用到这一对；双人时 spk0 用这组）
    "fishaudio_ref_wav": os.path.join(_TTS_REF_ROOT, "audios/ch_w_0.wav"),
    "fishaudio_ref_prompt_txt": os.path.join(_TTS_REF_ROOT, "texts/ch_w_0.txt"),
    # 第二说话人（仅当转录里出现 spk1 及更高 id 时使用；未配置则与第一组相同）
    "fishaudio_ref_wav_2": os.path.join(_TTS_REF_ROOT, "audios/ch_m_0.wav"),
    "fishaudio_ref_prompt_txt_2": os.path.join(_TTS_REF_ROOT, "texts/ch_m_0.txt"),
    # 可选：JSON，按 spk 精确覆盖（优先级高于上面两组默认）
    "fishaudio_ref_map_json": None,
    # 同说话人连续句合并为一次 TTS；0 表示仅按说话人切换切段、不限制单段长度
    "fishaudio_merge_max_chars": 400,
    # 合并时句与句之间的连接符（如 "，" 或 ""）
    "fishaudio_merge_joiner": "",
    # 为 False 时退化为「一句一合成」（与旧行为一致）
    "fishaudio_merge_same_speaker": True,
    # 是否在输出 wav 同目录写入 *_tts_dialogue.json
    "fishaudio_save_dialogue_json": True,
    # 每条 run_batch 文本前加 <emotion>；空字符串则不加
    "fishaudio_emotion": "Suffocating",
    # 按语言+speaker_info 选参考：audios/{lang}_{m|w}_{slot}.wav，texts 同名 .txt
    "tts_ref_data_root": _TTS_REF_ROOT,
    "tts_use_speaker_info_refs": True,
}


def _fish_speech_inference_modules(fish_speech_root: str):
    root = str(Path(fish_speech_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        encode_audio,
        generate_long,
        init_model,
        load_codec_model,
    )

    return decode_to_audio, encode_audio, generate_long, init_model, load_codec_model


class _FishAudioTTSRunner:
    """Fish Speech 推理封装"""

    def __init__(
        self,
        model_id_or_path: str,
        output_dir: str,
        device: str,
        logger,
        fish_speech_root: str,
    ):
        self.device = device
        self.output_dir = output_dir
        self.model_path = Path(model_id_or_path)
        self.logger = logger
        self.precision = torch.half

        decode_to_audio, encode_audio, generate_long, init_model, load_codec_model = (
            _fish_speech_inference_modules(fish_speech_root)
        )
        self._decode_to_audio = decode_to_audio
        self._encode_audio = encode_audio
        self._generate_long = generate_long

        self.logger.info("FishAudioTTS: 加载 Text2Semantic 模型...")
        self.llm_model, self.decode_one_token = init_model(
            checkpoint_path=self.model_path,
            device=self.device,
            precision=self.precision,
            compile=True,
        )
        with torch.device(self.device):
            self.llm_model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.llm_model.config.max_seq_len,
                dtype=next(self.llm_model.parameters()).dtype,
            )

        self.logger.info("FishAudioTTS: 加载 Codec...")
        codec_checkpoint = self.model_path / "codec.pth"
        self.codec_model = load_codec_model(codec_checkpoint, self.device, self.precision)
        self.sample_rate = self.codec_model.sample_rate
        self.speaker_cache = {}

    def run_batch(
        self,
        texts: List[str],
        speaker_sequence: List[str],
        prompts: List[str],
        wavs: List[str],
        output_filename: str,
    ):
        if len(texts) != len(speaker_sequence):
            raise ValueError("texts 与 speaker_sequence 长度必须一致")

        unique_speakers = sorted(set(speaker_sequence))
        speaker_to_prompt_map = {
            name: {"prompt_text": p, "wav_path": w}
            for name, p, w in zip(unique_speakers, prompts, wavs)
        }

        all_waveforms = []

        for i, text in enumerate(texts):
            speaker_name = speaker_sequence[i]

            if speaker_name not in speaker_to_prompt_map:
                self.logger.warning(f"找不到说话人 '{speaker_name}' 的参考信息，跳过此句。")
                continue

            prompt_info = speaker_to_prompt_map[speaker_name]
            prompt_text = prompt_info["prompt_text"]
            prompt_wav_path = prompt_info["wav_path"]

            self.logger.info(
                f"[{i+1}/{len(texts)}] 合成 | {speaker_name} | {text[:48]}..."
            )

            try:
                if speaker_name not in self.speaker_cache:
                    prompt_tokens = self._encode_audio(
                        prompt_wav_path, self.codec_model, self.device
                    ).cpu()
                    self.speaker_cache[speaker_name] = prompt_tokens
                else:
                    prompt_tokens = self.speaker_cache[speaker_name]

                generator = self._generate_long(
                    model=self.llm_model,
                    device=self.device,
                    decode_one_token=self.decode_one_token,
                    text=text,
                    num_samples=1,
                    max_new_tokens=0,
                    top_p=0.9,
                    top_k=30,
                    temperature=1.0,
                    compile=True,
                    iterative_prompt=True,
                    chunk_length=300,
                    prompt_text=[prompt_text],
                    prompt_tokens=[prompt_tokens],
                )

                codes = []
                final_waveform = None

                for response in generator:
                    if response.action == "sample":
                        codes.append(response.codes)
                    elif response.action == "next":
                        if codes:
                            merged_codes = torch.cat(codes, dim=1)
                            audio_tensor = self._decode_to_audio(
                                merged_codes.to(self.device), self.codec_model
                            )
                            final_waveform = audio_tensor.cpu().float().numpy()
                        codes = []

                if final_waveform is None:
                    raise RuntimeError("生成过程未返回有效音频")

                all_waveforms.append(final_waveform)

            except Exception as e:
                self.logger.error(f"合成失败 ({speaker_name}): {e}")
                estimated_duration_ms = len(text) * 150
                silence = np.zeros(
                    int(self.sample_rate * estimated_duration_ms / 1000), dtype=np.float32
                )
                all_waveforms.append(silence)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_waveforms or all(w.size == 0 for w in all_waveforms):
            self.logger.warning("批量生成未产生有效波形，跳过拼接")
            legacy_seg = Path(self.output_dir) / "segments"
            if legacy_seg.is_dir():
                shutil.rmtree(legacy_seg, ignore_errors=True)
            return

        concat_list = []
        for w in all_waveforms:
            concat_list.append(w)
            concat_list.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))

        full_conversation_waveform = np.concatenate(concat_list)
        output_path = os.path.join(self.output_dir, output_filename)

        full_waveform_clipped = np.clip(full_conversation_waveform, -1.0, 1.0)
        full_waveform_int16 = (full_waveform_clipped * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, rate=self.sample_rate, data=full_waveform_int16)

        # 旧版本曾写入 segments/；不再保留片段文件，并清理遗留目录
        legacy_seg = Path(self.output_dir) / "segments"
        if legacy_seg.is_dir():
            shutil.rmtree(legacy_seg, ignore_errors=True)


def _load_trans_document(path: Path) -> dict:
    """读取完整转录 JSON（含可选 speaker_info）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {"sentences": raw}
    if isinstance(raw, dict):
        return raw
    return {"sentences": []}


def _sentences_from_doc(doc: dict) -> List[dict]:
    s = doc.get("sentences")
    return s if isinstance(s, list) else []


def _load_trans(path: Path) -> List[dict]:
    return _sentences_from_doc(_load_trans_document(path))


def _load_mask(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    return raw


def _tts_mask_dir_resolved(
    mask_dir: Optional[str], logger, pipeline_name: str
) -> Optional[str]:
    """
    返回 mask JSON 根目录；未配置或路径无效时返回 None（不做敏感词替换，仅按转录文本合成）。
    """
    if not mask_dir or not str(mask_dir).strip():
        logger.info(
            f"{pipeline_name}: 未提供 mask_dir，按转录原文合成（不做敏感词替换）"
        )
        return None
    md = str(mask_dir).strip()
    if not os.path.isdir(md):
        logger.warning(
            f"{pipeline_name}: mask_dir 不是有效目录 {mask_dir!r}，按转录原文合成（不做敏感词替换）"
        )
        return None
    return md


def _mask_char(word: str) -> str:
    # 按需求统一替换成“某”，并保持原词长度。
    return "某" * max(1, len(word))


def _build_sentence_mask_words(mask_items: List[dict]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for item in mask_items:
        try:
            idx = int(item.get("idx"))
        except Exception:
            continue
        word = str(item.get("text", "")).strip()
        if not word or word == "None" or word.startswith("识别到隐私词"):
            continue
        out.setdefault(idx, []).append(word)
    return out


def _replace_sensitive(text: str, words: List[str]) -> str:
    if not words:
        return text
    # 先替换更长词，避免短词先替换破坏长词匹配
    for w in sorted(set(words), key=len, reverse=True):
        text = re.sub(re.escape(w), _mask_char(w), text)
    return text


def _read_prompt_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _resolve_prompt_text(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if s and os.path.isfile(s):
        return _read_prompt_txt(s)
    return s


def _speaker_index_from_label(spk: str) -> int:
    """从 spk0 / spk1 或 ASR 的 speaker 字段还原整数 id。"""
    if not isinstance(spk, str):
        return 0
    if spk.startswith("spk"):
        try:
            return int(spk[3:])
        except ValueError:
            return 0
    try:
        return int(spk)
    except ValueError:
        return 0


def _tts_lang_prefix(language: Optional[str]) -> str:
    """
    参考资源前缀：把流水线语言规范化为“英文名首两字母”风格（你当前资源命名习惯）。
    常见映射：
      英文/english -> en
      中文/chinese -> ch
      荷兰语/dutch -> du
      西语/spanish -> sp
      德语/german -> ge
      法语/french -> fr
      意大利语/italian -> it
    未识别时回退 ch。
    """
    if language is None:
        return "ch"
    full = str(language).strip()
    s = full.lower()

    # 显式中文别名优先用 ch（兼容你已有 ch_* 资源命名）
    if any(k in full for k in ("中文", "汉语", "普通话")) or "chinese" in s:
        return "ch"

    # 常见语言别名映射
    alias_map = {
        "英文": "en",
        "英语": "en",
        "english": "en",
        "中文": "ch",
        "汉语": "ch",
        "普通话": "ch",
        "chinese": "ch",
        "荷兰语": "du",
        "荷兰文": "du",
        "dutch": "du",
        "西语": "sp",
        "西班牙语": "sp",
        "spanish": "sp",
        "espanol": "sp",
        "español": "sp",
        "德语": "ge",
        "german": "ge",
        "法语": "fr",
        "french": "fr",
        "意大利语": "it",
        "italian": "it",
        # 兼容历史语言码输入，统一映射到你当前文件前缀
        "es": "sp",
        "de": "ge",
        "nl": "du",
    }
    for k, v in alias_map.items():
        if k in full or k in s:
            return v

    # 标准语言码（如 en / en-US / pt_BR）归一化到两位前缀
    token = s.replace("_", "-").split("-")[0]
    if re.fullmatch(r"[a-z]{2}", token):
        return token

    return "ch"


def _resolve_tts_ref_lang_prefix(
    language: Optional[str], doc: Optional[dict]
) -> str:
    """
    合成参考音色的语言前缀（lang code）：优先显式参数，其次转录 JSON 元数据，再次环境变量。
    用于在 agent 未向 VC 层传入 --language 时仍能一致选用 {lang}_* 资源目录。
    """
    cand: Optional[str] = None
    if language is not None and str(language).strip():
        cand = str(language).strip()
    elif isinstance(doc, dict):
        for key in (
            "language",
            "asr_language",
            "pipeline_language",
            "tts_language",
        ):
            v = doc.get(key)
            if v is not None and str(v).strip():
                cand = str(v).strip()
                break
    if cand is None:
        for envk in ("ANONYMISE_AGENT_LANGUAGE", "AGENT_LANGUAGE"):
            v = os.environ.get(envk)
            if v is not None and str(v).strip():
                cand = str(v).strip()
                break
    return _tts_lang_prefix(cand)


def _tts_gender_suffix(gender: Optional[str]) -> str:
    """male -> m，female -> w；未知默认 m。"""
    if gender is None:
        return "m"
    g = str(gender).strip().lower()
    if g in ("female", "f", "woman", "w", "女", "女性"):
        return "w"
    return "m"


def _tts_ref_slot(speaker_idx: int) -> int:
    """资源仅有 _0 / _1 两套时，更高 speaker id 在 0/1 间轮转。"""
    if speaker_idx <= 0:
        return 0
    if speaker_idx == 1:
        return 1
    return speaker_idx % 2


def _tts_ref_pair_paths(ref_root: str, stem: str) -> Tuple[str, str]:
    root = str(Path(ref_root).resolve())
    wav = os.path.join(root, "audios", f"{stem}.wav")
    txt = os.path.join(root, "texts", f"{stem}.txt")
    return wav, txt


def _pick_tts_ref_paths(
    ref_root: str,
    lang: str,
    speaker_idx: int,
    gender: Optional[str],
    logger,
    pipeline: str,
    spk_label: str,
) -> Tuple[str, str]:
    """
    在 ref_root 下查找存在的 wav+txt；多候选按性别、另一槽位、语言回退尝试。
    """
    g = _tts_gender_suffix(gender)
    og = "w" if g == "m" else "m"
    slot = _tts_ref_slot(speaker_idx)
    other = 1 - slot
    # 语言回退链：优先目标语言，再尝试 ch 与 en（去重后顺序保留）
    langs: List[str] = []
    for cand in (lang, "ch", "en"):
        if cand and cand not in langs:
            langs.append(cand)

    stems: List[str] = []
    for L in langs:
        stems.extend(
            [
                f"{L}_{g}_{slot}",
                f"{L}_{og}_{slot}",
                f"{L}_{g}_{other}",
                f"{L}_{og}_{other}",
            ]
        )
    seen = set()
    for stem in stems:
        if stem in seen:
            continue
        seen.add(stem)
        wav, txt = _tts_ref_pair_paths(ref_root, stem)
        if os.path.isfile(wav) and os.path.isfile(txt):
            logger.info(
                f"{pipeline}: 说话人 {spk_label} 使用参考 {stem} "
                f"(lang={lang}, gender_hint={gender!r})"
            )
            return wav, txt

    raise FileNotFoundError(
        f"{pipeline}: 在 {ref_root} 下未找到说话人 {spk_label} 的参考 "
        f"(lang={lang}, gender={gender!r}, 已尝试 stems 如 {stems[:4]}...)"
    )


def _build_dynamic_ref_map(
    unique_speakers: List[str],
    speaker_info: dict,
    ref_lang: str,
    ref_root: str,
    logger,
    pipeline: str,
) -> Dict[str, dict]:
    """为每个 spk* 生成 ref_map 条目（wav + prompt_txt 路径）。"""
    lang = _tts_lang_prefix(ref_lang)
    out: Dict[str, dict] = {}
    for spk in unique_speakers:
        idx = _speaker_index_from_label(spk)
        g: Optional[str] = None
        if isinstance(speaker_info, dict) and speaker_info:
            entry = speaker_info.get(str(idx))
            if entry is None:
                entry = speaker_info.get(idx)
            if isinstance(entry, dict):
                g = entry.get("gender") or entry.get("Gender")
        wav, txt = _pick_tts_ref_paths(
            ref_root, lang, idx, g, logger, pipeline, spk
        )
        out[spk] = {"wav": wav, "prompt_txt": txt}
    return out


def _fishaudio_emotion_tts_texts(texts: List[str], emotion: Optional[str]) -> List[str]:
    """每条合成文本均为 <Emotion>正文（与 batch_generate_fishaudio_nested 一致）。"""
    em = (emotion or "").strip()
    if not em:
        return list(texts)
    return [f"<{em}>{t}" for t in texts]


def _resolve_ref_pair(
    spk: str,
    wav1: str,
    prompt1: str,
    wav2: Optional[str],
    prompt2: Optional[str],
    ref_map: Dict[str, dict],
) -> Tuple[str, str]:
    """ref_map 命中则优先；否则 spk0→第一组，spk1+→第二组（第二组 wav 无效则回退第一组）。"""
    entry = ref_map.get(spk) if ref_map else None
    if entry:
        wav = entry.get("wav") or wav1
        raw_pt = entry.get("prompt_txt", entry.get("prompt", prompt1))
    else:
        idx = _speaker_index_from_label(spk)
        if idx == 0:
            wav, raw_pt = wav1, prompt1
        else:
            if wav2 and os.path.isfile(wav2):
                wav, raw_pt = wav2, (prompt2 if prompt2 is not None else prompt1)
            else:
                wav, raw_pt = wav1, prompt1
    if not wav or not os.path.isfile(wav):
        raise FileNotFoundError(f"FishAudio 参考 wav 不存在: {wav} (speaker={spk})")
    prompt_text = _resolve_prompt_text(raw_pt)
    if not prompt_text:
        raise ValueError(f"FishAudio 参考 prompt 为空 (speaker={spk})")
    return wav, prompt_text


def _build_unique_prompt_lists(
    unique_speakers: List[str],
    wav1: str,
    prompt1: str,
    wav2: Optional[str],
    prompt2: Optional[str],
    ref_map: Dict[str, dict],
) -> Tuple[List[str], List[str]]:
    prompt_texts: List[str] = []
    wavs: List[str] = []
    for spk in unique_speakers:
        w, t = _resolve_ref_pair(spk, wav1, prompt1, wav2, prompt2, ref_map)
        wavs.append(w)
        prompt_texts.append(t)
    return prompt_texts, wavs


def _speaker_of(sent: dict) -> str:
    spk = sent.get("speaker")
    if spk is None:
        return "spk0"
    return f"spk{spk}"


def _build_tts_units(
    sentences: List[dict],
    sent2words: Dict[int, List[str]],
    merge_same_speaker: bool,
    merge_max_chars: int,
    merge_joiner: str,
) -> Tuple[List[str], List[str], List[dict]]:
    """
    生成送入 run_batch 的 texts / speaker_sequence，以及用于落盘的 segments 元数据。

    merge_same_speaker=False：一句一条，与旧逻辑一致。
    merge_same_speaker=True：同一说话人连续句合并；merge_max_chars>0 时单段总长（含 joiner）超限则同说话人也会拆成多段。
    """
    if not merge_same_speaker:
        texts: List[str] = []
        spk_seq: List[str] = []
        segments: List[dict] = []
        for s in sentences:
            try:
                idx = int(s.get("index", len(texts)))
            except Exception:
                idx = len(texts)
            raw = str(s.get("text", ""))
            piece = _replace_sensitive(raw, sent2words.get(idx, [])).strip()
            if not piece:
                continue
            spk = _speaker_of(s)
            texts.append(piece)
            spk_seq.append(spk)
            segments.append(
                {
                    "speaker": spk,
                    "text": piece,
                    "sentence_indices": [idx],
                }
            )
        return texts, spk_seq, segments

    texts_out: List[str] = []
    spk_out: List[str] = []
    segments_out: List[dict] = []

    cur_spk: Optional[str] = None
    cur_parts: List[str] = []
    cur_indices: List[int] = []
    cur_len = 0
    joiner = merge_joiner or ""

    def _flush():
        nonlocal cur_spk, cur_parts, cur_indices, cur_len
        if not cur_parts or cur_spk is None:
            cur_parts = []
            cur_indices = []
            cur_len = 0
            cur_spk = None
            return
        merged = joiner.join(cur_parts)
        texts_out.append(merged)
        spk_out.append(cur_spk)
        segments_out.append(
            {
                "speaker": cur_spk,
                "text": merged,
                "sentence_indices": list(cur_indices),
            }
        )
        cur_parts = []
        cur_indices = []
        cur_len = 0
        cur_spk = None

    for s in sentences:
        try:
            idx = int(s.get("index", -1))
        except Exception:
            idx = -1
        raw = str(s.get("text", ""))
        piece = _replace_sensitive(raw, sent2words.get(idx, [])).strip()
        if not piece:
            continue
        spk = _speaker_of(s)

        if cur_spk is None:
            cur_spk = spk
            cur_parts = [piece]
            cur_indices = [idx]
            cur_len = len(piece)
            continue

        would_len = cur_len + len(joiner) + len(piece)

        if spk != cur_spk:
            _flush()
            cur_spk = spk
            cur_parts = [piece]
            cur_indices = [idx]
            cur_len = len(piece)
            continue

        if merge_max_chars > 0 and cur_parts and would_len > merge_max_chars:
            _flush()
            cur_spk = spk
            cur_parts = [piece]
            cur_indices = [idx]
            cur_len = len(piece)
            continue

        cur_parts.append(piece)
        cur_indices.append(idx)
        cur_len = would_len

    _flush()
    return texts_out, spk_out, segments_out


def _gather_fishaudio_jobs(
    input_dir: str,
    trans_dir: str,
    supported: set,
) -> Tuple[str, List[Tuple[Path, Path]]]:
    """
    返回 (mode, jobs)。
    mode == "audio"：jobs 为 (rel 相对 input_dir，该 rel 指向原音频路径)。
    mode == "trans_only"：input_dir 下无音频时，按 trans_dir 内 **/*.json 枚举；
      jobs 为 (rel 相对 trans_dir 的 .json 路径, 占位 Path 仅用于统一循环)。
    """
    input_p = Path(input_dir)
    trans_p = Path(trans_dir)
    audio_files = sorted(
        p for p in input_p.rglob("*") if p.suffix.lower() in supported
    )
    if audio_files:
        return "audio", [(fp.relative_to(input_p), fp) for fp in audio_files]

    json_files = sorted(trans_p.rglob("*.json"))
    jobs = []
    for jp in json_files:
        jobs.append((jp.relative_to(trans_p), jp))
    return "trans_only", jobs


def batch_convert_fn(
    input_dir: str,
    output_dir: str,
    logger,
    trans_dir: str = None,
    mask_dir: str = None,
    fishaudio_model_path: str = None,
    fish_speech_root: str = None,
    fishaudio_device: str = "cuda",
    output_sr: int = 16000,
    fishaudio_ref_wav: str = None,
    fishaudio_ref_prompt_txt: str = None,
    fishaudio_ref_wav_2: str = None,
    fishaudio_ref_prompt_txt_2: str = None,
    fishaudio_ref_map_json: str = None,
    fishaudio_merge_same_speaker: Optional[bool] = None,
    fishaudio_merge_max_chars: Optional[int] = None,
    fishaudio_merge_joiner: Optional[str] = None,
    fishaudio_save_dialogue_json: Optional[bool] = None,
    fishaudio_emotion: Optional[str] = None,
    language: Optional[str] = None,
    tts_ref_data_root: Optional[str] = None,
    tts_use_speaker_info_refs: Optional[bool] = None,
    **_,
):
    if not trans_dir or not os.path.isdir(trans_dir):
        raise ValueError("fishaudio_tts 需要有效的 trans_dir（句级转录 JSON）")

    os.makedirs(output_dir, exist_ok=True)
    mask_root = _tts_mask_dir_resolved(mask_dir, logger, "FishAudioTTS")
    model_path = fishaudio_model_path or DEFAULTS["fishaudio_model_path"]
    _fs_root = fish_speech_root or DEFAULTS["fish_speech_root"]
    wav1 = fishaudio_ref_wav or DEFAULTS["fishaudio_ref_wav"]
    ptxt1 = fishaudio_ref_prompt_txt or DEFAULTS["fishaudio_ref_prompt_txt"]
    wav2 = fishaudio_ref_wav_2 if fishaudio_ref_wav_2 is not None else DEFAULTS.get("fishaudio_ref_wav_2")
    ptxt2 = fishaudio_ref_prompt_txt_2 if fishaudio_ref_prompt_txt_2 is not None else DEFAULTS.get("fishaudio_ref_prompt_txt_2")
    ref_map_path = fishaudio_ref_map_json if fishaudio_ref_map_json is not None else DEFAULTS.get("fishaudio_ref_map_json")
    ref_map: Dict[str, dict] = {}
    if ref_map_path and str(ref_map_path).strip():
        mp = Path(ref_map_path)
        if mp.is_file():
            with open(mp, "r", encoding="utf-8") as f:
                ref_map = json.load(f)
            if not isinstance(ref_map, dict):
                ref_map = {}
        else:
            logger.warning(f"fishaudio_ref_map_json 不是有效文件，忽略: {ref_map_path}")

    merge_same = (
        DEFAULTS["fishaudio_merge_same_speaker"]
        if fishaudio_merge_same_speaker is None
        else fishaudio_merge_same_speaker
    )
    merge_max = (
        DEFAULTS["fishaudio_merge_max_chars"]
        if fishaudio_merge_max_chars is None
        else int(fishaudio_merge_max_chars)
    )
    merge_join = (
        DEFAULTS["fishaudio_merge_joiner"]
        if fishaudio_merge_joiner is None
        else fishaudio_merge_joiner
    )
    save_dlg = (
        DEFAULTS["fishaudio_save_dialogue_json"]
        if fishaudio_save_dialogue_json is None
        else fishaudio_save_dialogue_json
    )
    emotion_tag = (
        DEFAULTS["fishaudio_emotion"]
        if fishaudio_emotion is None
        else fishaudio_emotion
    )
    use_dyn_refs = (
        DEFAULTS["tts_use_speaker_info_refs"]
        if tts_use_speaker_info_refs is None
        else bool(tts_use_speaker_info_refs)
    )
    ref_root = (
        tts_ref_data_root
        if tts_ref_data_root is not None
        else DEFAULTS.get("tts_ref_data_root")
    )

    tts = _FishAudioTTSRunner(
        model_id_or_path=model_path,
        output_dir=output_dir,
        device=fishaudio_device,
        logger=logger,
        fish_speech_root=_fs_root,
    )
    logger.info("FishAudioTTS: 模型加载完成")
    logger.info(
        f"FishAudioTTS: merge_same_speaker={merge_same} merge_max_chars={merge_max} "
        f"joiner={merge_join!r} save_dialogue_json={save_dlg} "
        f"emotion={emotion_tag!r} language_arg={language!r} "
        f"use_speaker_info_refs={use_dyn_refs} "
        f"(每条 ref语言见日志：参数 / JSON 语言字段 / $ANONYMISE_AGENT_LANGUAGE，未知默认 ch)"
    )

    supported = {".wav", ".flac", ".mp3", ".ogg", ".aiff"}
    mode, jobs = _gather_fishaudio_jobs(input_dir, trans_dir, supported)
    if not jobs:
        logger.warning(
            f"无可处理条目：input_dir={input_dir} 无音频，且 trans_dir={trans_dir} 无 JSON"
        )
        return
    if mode == "trans_only":
        logger.info(
            f"input_dir 中无音频文件，改为按 trans_dir 中的 JSON 处理（共 {len(jobs)} 条）"
        )

    ok = failed = skipped = 0
    t_start = time.time()

    for i, (rel, _) in enumerate(jobs):
        if mode == "audio":
            out_fp = Path(output_dir) / rel
            trans_path = Path(trans_dir) / rel.with_suffix(".json")
            mask_path = (
                Path(mask_root) / rel.with_suffix(".json") if mask_root else None
            )
            rel_log = rel
            stem_for_seg = rel.as_posix().replace("/", "__")
        else:
            out_fp = Path(output_dir) / rel.with_suffix(".wav")
            trans_path = Path(trans_dir) / rel
            mask_path = Path(mask_root) / rel if mask_root else None
            rel_log = rel.with_suffix(".wav")
            stem_for_seg = str(rel.with_suffix("")).replace("/", "__")

        out_fp.parent.mkdir(parents=True, exist_ok=True)
        if out_fp.exists() and out_fp.stat().st_size > 0:
            logger.info(f"已存在，跳过: {rel_log}")
            skipped += 1
            continue

        if not trans_path.exists():
            logger.warning(f"未找到转录 JSON：{trans_path}，跳过 {rel_log}")
            failed += 1
            continue

        file_t0 = time.time()
        try:
            doc = _load_trans_document(trans_path)
            ref_lang = _resolve_tts_ref_lang_prefix(language, doc)
            sentences = sorted(
                _sentences_from_doc(doc), key=lambda x: x.get("index", 0)
            )
            if not sentences:
                raise RuntimeError("转录句子为空")
            mask_items = _load_mask(mask_path) if mask_path is not None else []
            sent2words = _build_sentence_mask_words(mask_items)

            texts, spk_seq, dialogue_segments = _build_tts_units(
                sentences,
                sent2words,
                merge_same_speaker=merge_same,
                merge_max_chars=max(0, merge_max),
                merge_joiner=merge_join,
            )
            if not texts:
                raise RuntimeError("有效合成文本为空，无法合成（转录是否全为空句）")

            unique_speakers = sorted(set(spk_seq))
            ref_map_eff: Dict[str, dict] = dict(ref_map) if ref_map else {}
            if (
                use_dyn_refs
                and ref_root
                and str(ref_root).strip()
                and os.path.isdir(str(ref_root))
                and unique_speakers
            ):
                si = doc.get("speaker_info")
                if not isinstance(si, dict):
                    si = {}
                try:
                    dyn = _build_dynamic_ref_map(
                        unique_speakers,
                        si,
                        ref_lang,
                        str(ref_root).strip(),
                        logger,
                        "FishAudioTTS",
                    )
                    ref_map_eff = {**dyn, **ref_map_eff}
                except FileNotFoundError as e:
                    logger.warning(
                        f"FishAudioTTS: 按 speaker_info 选参考失败，使用默认 ref wav: {e}"
                    )

            prompt_texts, prompt_wavs = _build_unique_prompt_lists(
                unique_speakers, wav1, ptxt1, wav2, ptxt2, ref_map_eff
            )

            seg_name = f"{stem_for_seg}__tts.wav"
            tts.speaker_cache.clear()
            texts_for_model = _fishaudio_emotion_tts_texts(texts, emotion_tag)
            tts.run_batch(
                texts=texts_for_model,
                speaker_sequence=spk_seq,
                prompts=prompt_texts,
                wavs=prompt_wavs,
                output_filename=seg_name,
            )

            tts_out = Path(output_dir) / seg_name
            if not tts_out.exists():
                raise RuntimeError(f"TTS 输出不存在: {tts_out}")
            tts_audio, _ = librosa.load(str(tts_out), sr=output_sr, mono=True)
            sf.write(str(out_fp), tts_audio, output_sr)
            if tts_out.exists():
                tts_out.unlink()

            if save_dlg:
                dlg_path = out_fp.parent / f"{out_fp.stem}_tts_dialogue.json"
                payload = {
                    "output_wav": out_fp.name,
                    "merge_same_speaker": merge_same,
                    "merge_max_chars": merge_max,
                    "merge_joiner": merge_join,
                    "num_tts_segments": len(texts),
                    "segments": dialogue_segments,
                }
                with open(dlg_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

            ok += 1
            logger.info(f"[{i+1}/{len(jobs)}] 完成: {rel_log} ({time.time()-file_t0:.2f}s)")
        except Exception:
            import traceback
            failed += 1
            logger.error(f"失败: {rel_log}\n{traceback.format_exc()}")

    logger.info(
        f"FishAudioTTS 批量完成：成功 {ok}，跳过 {skipped}，失败 {failed}  "
        f"总耗时 {time.time()-t_start:.2f}s"
    )
