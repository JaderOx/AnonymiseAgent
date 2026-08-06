"""
utils/voice_convert/cosyvoice_tts.py

CosyVoice TTS 流水线，与 fishaudio_tts 对齐：
  trans +（可选）mask → 脱敏或原文 →（可选）同说话人句合并 → CosyVoice zero-shot → 重采样保存 wav
  参考 tts_methods/tts/cosyTTS.py；依赖本机已安装 CosyVoice、ffmpeg（拼接片段）。
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import soundfile as sf
import torch

from .fishaudio_tts import (
    _build_dynamic_ref_map,
    _build_sentence_mask_words,
    _build_tts_units,
    _gather_fishaudio_jobs,
    _load_mask,
    _load_trans_document,
    _resolve_ref_pair,
    _resolve_tts_ref_lang_prefix,
    _sentences_from_doc,
    _tts_mask_dir_resolved,
)

DESCRIPTION = (
    "CosyVoice TTS（可按语言+speaker_info 选多语言参考音色，敏感词替换为「某」）"
)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_TTS_REF_ROOT = os.path.join(_PROJECT_ROOT, "tts_reference")

DEFAULTS = {
    "cosyvoice_model_path": os.path.join(_PROJECT_ROOT, "models/cosyvoice2-0.5B"),
    "tts_methods_root": os.path.join(_PROJECT_ROOT, "tts_methods"),
    "output_sr": 16000,
    "cosyvoice_ref_wav": os.path.join(_TTS_REF_ROOT, "audios/ch_w_0.wav"),
    "cosyvoice_ref_prompt_txt": os.path.join(_TTS_REF_ROOT, "texts/ch_w_0.txt"),
    "cosyvoice_ref_wav_2": os.path.join(_TTS_REF_ROOT, "audios/ch_m_0.wav"),
    "cosyvoice_ref_prompt_txt_2": os.path.join(_TTS_REF_ROOT, "texts/ch_m_0.txt"),
    "cosyvoice_ref_map_json": None,
    "cosyvoice_merge_max_chars": 400,
    "cosyvoice_merge_joiner": "",
    "cosyvoice_merge_same_speaker": True,
    "cosyvoice_save_dialogue_json": True,
    "tts_ref_data_root": _TTS_REF_ROOT,
    "tts_use_speaker_info_refs": True,
}


def _import_cosyvoice_agent(tts_methods_root: str):
    root = str(Path(tts_methods_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from tts.cosyTTS import cosyvoice_agent

    return cosyvoice_agent


def _ensure_cosy_ref_end_token(ref_text: str) -> str:
    """CosyVoice 3 zero-shot 要求参考文本以 <|endofprompt|> 结尾（与 cosyTTS.py 一致）。"""
    s = str(ref_text).strip()
    tok = "<|endofprompt|>"
    if not s.endswith(tok):
        s = s + tok
    return s


def _cosy_line_prompts_wavs(
    spk_seq: List[str],
    wav1: str,
    prompt1: str,
    wav2: Optional[str],
    prompt2: Optional[str],
    ref_map: Dict[str, dict],
) -> Tuple[List[str], List[str]]:
    """CosyVoice 需与 text_batches 等长的逐条 prompts / wavs。"""
    prompts: List[str] = []
    wavs: List[str] = []
    for spk in spk_seq:
        w, t = _resolve_ref_pair(spk, wav1, prompt1, wav2, prompt2, ref_map)
        prompts.append(_ensure_cosy_ref_end_token(t))
        wavs.append(w)
    return prompts, wavs


def batch_convert_fn(
    input_dir: str,
    output_dir: str,
    logger,
    trans_dir: str = None,
    mask_dir: str = None,
    cosyvoice_model_path: str = None,
    tts_methods_root: str = None,
    output_sr: int = None,
    cosyvoice_ref_wav: str = None,
    cosyvoice_ref_prompt_txt: str = None,
    cosyvoice_ref_wav_2: str = None,
    cosyvoice_ref_prompt_txt_2: str = None,
    cosyvoice_ref_map_json: str = None,
    cosyvoice_merge_same_speaker: Optional[bool] = None,
    cosyvoice_merge_max_chars: Optional[int] = None,
    cosyvoice_merge_joiner: Optional[str] = None,
    cosyvoice_save_dialogue_json: Optional[bool] = None,
    language: Optional[str] = None,
    tts_ref_data_root: Optional[str] = None,
    tts_use_speaker_info_refs: Optional[bool] = None,
    **_,
):
    if not trans_dir or not os.path.isdir(trans_dir):
        raise ValueError("cosyvoice_tts 需要有效的 trans_dir（句级转录 JSON）")

    os.makedirs(output_dir, exist_ok=True)
    mask_root = _tts_mask_dir_resolved(mask_dir, logger, "CosyVoiceTTS")
    model_path = cosyvoice_model_path or DEFAULTS["cosyvoice_model_path"]
    tm_root = tts_methods_root or DEFAULTS["tts_methods_root"]
    out_sr = int(output_sr if output_sr is not None else DEFAULTS["output_sr"])

    wav1 = cosyvoice_ref_wav or DEFAULTS["cosyvoice_ref_wav"]
    ptxt1 = cosyvoice_ref_prompt_txt or DEFAULTS["cosyvoice_ref_prompt_txt"]
    wav2 = (
        cosyvoice_ref_wav_2
        if cosyvoice_ref_wav_2 is not None
        else DEFAULTS.get("cosyvoice_ref_wav_2")
    )
    ptxt2 = (
        cosyvoice_ref_prompt_txt_2
        if cosyvoice_ref_prompt_txt_2 is not None
        else DEFAULTS.get("cosyvoice_ref_prompt_txt_2")
    )
    ref_map_path = (
        cosyvoice_ref_map_json
        if cosyvoice_ref_map_json is not None
        else DEFAULTS.get("cosyvoice_ref_map_json")
    )
    ref_map: Dict[str, dict] = {}
    if ref_map_path and str(ref_map_path).strip():
        mp = Path(ref_map_path)
        if mp.is_file():
            with open(mp, "r", encoding="utf-8") as f:
                ref_map = json.load(f)
            if not isinstance(ref_map, dict):
                ref_map = {}
        else:
            logger.warning(f"cosyvoice_ref_map_json 不是有效文件，忽略: {ref_map_path}")

    merge_same = (
        DEFAULTS["cosyvoice_merge_same_speaker"]
        if cosyvoice_merge_same_speaker is None
        else cosyvoice_merge_same_speaker
    )
    merge_max = (
        DEFAULTS["cosyvoice_merge_max_chars"]
        if cosyvoice_merge_max_chars is None
        else int(cosyvoice_merge_max_chars)
    )
    merge_join = (
        DEFAULTS["cosyvoice_merge_joiner"]
        if cosyvoice_merge_joiner is None
        else cosyvoice_merge_joiner
    )
    save_dlg = (
        DEFAULTS["cosyvoice_save_dialogue_json"]
        if cosyvoice_save_dialogue_json is None
        else cosyvoice_save_dialogue_json
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

    CosyVoiceAgent = _import_cosyvoice_agent(tm_root)
    logger.info(f"CosyVoiceTTS: 加载模型 {model_path}")
    logger.info(
        f"CosyVoiceTTS: merge_same_speaker={merge_same} merge_max_chars={merge_max} "
        f"joiner={merge_join!r} save_dialogue_json={save_dlg} output_sr={out_sr} "
        f"language_arg={language!r} use_speaker_info_refs={use_dyn_refs} "
        f"(每条 ref语言：kwargs / 转录JSON语言字段 / $ANONYMISE_AGENT_LANGUAGE，未知默认 ch)"
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

    work_root = Path(output_dir) / "_cosy_work"
    work_root.mkdir(parents=True, exist_ok=True)
    model_home = work_root / "_model_home"
    model_home.mkdir(parents=True, exist_ok=True)
    tts = CosyVoiceAgent(model_path=model_path, output_dir=str(model_home))

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
        job_dir = work_root / stem_for_seg
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            job_dir.mkdir(parents=True, exist_ok=True)

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
                        "CosyVoiceTTS",
                    )
                    ref_map_eff = {**dyn, **ref_map_eff}
                except FileNotFoundError as e:
                    logger.warning(
                        f"CosyVoiceTTS: 按 speaker_info 选参考失败，使用默认 ref wav: {e}"
                    )

            line_prompts, line_wavs = _cosy_line_prompts_wavs(
                spk_seq, wav1, ptxt1, wav2, ptxt2, ref_map_eff
            )

            out_fn = "full.wav"
            tts.output_dir = str(job_dir)
            tts.run_batch(
                text_batches=texts,
                speakers=spk_seq,
                prompts=line_prompts,
                wavs=line_wavs,
                output_filename=out_fn,
                emotion_instructs=[None] * len(texts),
            )

            gen_wav = job_dir / "full" / "full.wav"
            if not gen_wav.is_file():
                raise RuntimeError(f"CosyVoice 输出不存在: {gen_wav}")

            audio, _ = librosa.load(str(gen_wav), sr=out_sr, mono=True)
            sf.write(str(out_fp), audio, out_sr)

            if save_dlg:
                dlg_path = out_fp.parent / f"{out_fp.stem}_tts_dialogue.json"
                payload = {
                    "engine": "cosyvoice",
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
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(
        f"CosyVoiceTTS 批量完成：成功 {ok}，跳过 {skipped}，失败 {failed}  "
        f"总耗时 {time.time()-t_start:.2f}s"
    )
