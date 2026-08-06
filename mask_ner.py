"""
mask_ner.py — 基于 NER 的隐私词检测（替代 LLM 方案）

输出格式与 mask_llm.py 完全一致，可直接被 apply_mask.py 使用。

用法：
    from mask_ner import mask_ner
    mask_ner(trans_dir, mask_dir, logger, language='中文')
"""

import os
import re
import json
import time
from pathlib import Path


# ── 语言 → 默认 NER 模型 ───────────────────────────────────────────────────
DEFAULT_NER_MODELS = {
    "中文": "uer/roberta-base-finetuned-cluener2020-chinese",
    "en": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "英文": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "es": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "西班牙语": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "nl": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "荷兰语": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "de": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "德语": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "fr": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "法语": "Davlan/bert-base-multilingual-cased-ner-hrl",
}

# 多语言模型的隐私相关实体标签
MULTILINGUAL_PRIVACY_LABELS = {"PER", "LOC", "ORG"}
# 中文 CLUENER 模型的隐私相关标签
CHINESE_PRIVACY_LABELS = {"address", "name", "company", "government", "organization", "school"}

NO_WER_LANGUAGE = {"中文", "日文"}


def _get_privacy_labels(model_name: str) -> set:
    """根据模型名返回隐私相关实体标签集合。"""
    model_lower = model_name.lower()
    if "chinese" in model_lower or "cluener" in model_lower:
        return CHINESE_PRIVACY_LABELS
    return MULTILINGUAL_PRIVACY_LABELS


def _is_chinese_model(model_name: str) -> bool:
    return "chinese" in model_name.lower() or "cluener" in model_name.lower()


# ── 复用 mask_llm 的字符→时间戳转换 ─────────────────────────────────────────
def char_to_word_positions(
        char_start: int,
        char_end: int,
        text: str,
        language: str,
        word_timestamps: list
        ) -> tuple:
    """将字符位置范围转换为时间戳范围（与 mask_llm.py 逻辑一致）。"""
    if language in NO_WER_LANGUAGE:
        word_start_timestamp = word_timestamps[char_start] if char_start < len(word_timestamps) else word_timestamps[-1]
        word_end_timestamp = word_timestamps[char_end] if char_end < len(word_timestamps) else word_timestamps[-1] + 100
    else:
        words = text.split()
        char_to_word = [-1] * len(text)
        current_char_pos = 0
        for word_idx, word in enumerate(words):
            for i in range(len(word)):
                if current_char_pos < len(char_to_word):
                    char_to_word[current_char_pos] = word_idx
                current_char_pos += 1
            if word_idx < len(words) - 1 and current_char_pos < len(char_to_word):
                char_to_word[current_char_pos] = word_idx
                current_char_pos += 1

        char_start = max(0, min(char_start, len(char_to_word) - 1))
        char_end = max(0, min(char_end, len(char_to_word) - 1))

        word_indices = set()
        for pos in range(char_start, char_end):
            if pos < len(char_to_word) and char_to_word[pos] != -1:
                word_indices.add(char_to_word[pos])

        if not word_indices:
            return word_timestamps[0], word_timestamps[-1] + 100

        start_word_idx = min(word_indices)
        end_word_idx = max(word_indices)
        word_start_timestamp = word_timestamps[start_word_idx]
        if end_word_idx + 1 < len(word_timestamps):
            word_end_timestamp = word_timestamps[end_word_idx + 1]
        else:
            word_end_timestamp = word_timestamps[-1] + 100

    return word_start_timestamp, word_end_timestamp


# ── 按 token 长度切分文本 ────────────────────────────────────────────────────

def _split_by_tokens(tokenizer, text: str, max_len: int) -> list:
    """
    将文本按 token 长度切分为多段，每段不超过 max_len token。
    在句子边界（。.!?！？）处切分，避免切断实体。
    """
    # 快速检查：直接 tokenize 看是否超长
    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    if token_count <= max_len:
        return [text]

    # 按句子边界切分
    # 中文和英文的句子分隔符
    parts = re.split(r'([。.!?！？；;\n])', text)

    chunks = []
    current = ""
    for part in parts:
        candidate = current + part
        if len(tokenizer.encode(candidate, add_special_tokens=True)) > max_len:
            if current:
                chunks.append(current)
            current = part
        else:
            current = candidate

    if current:
        chunks.append(current)

    # 兜底：如果单句就超长，强制截断
    final = []
    for chunk in chunks:
        if len(tokenizer.encode(chunk, add_special_tokens=True)) > max_len:
            # 二分法找到合适长度
            lo, hi = 0, len(chunk)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(tokenizer.encode(chunk[:mid], add_special_tokens=True)) <= max_len:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 0:
                final.append(chunk[:lo])
            # 剩余部分继续处理
            remaining = chunk[lo:]
            if remaining:
                final.extend(_split_by_tokens(tokenizer, remaining, max_len))
        else:
            final.append(chunk)

    return final


# ── 主函数 ────────────────────────────────────────────────────────────────────

def mask_ner(
    trans_dir: str,
    mask_dir: str,
    logger,
    language: str = '中文',
    ner_model: str = None,
    group_size: int = 50,
):
    """
    基于 NER 的隐私词检测。按 group_size 句分组拼接后整体推理，再映射回各句。

    Args:
        trans_dir:  转录 JSON 目录
        mask_dir:   输出 mask JSON 目录
        logger:     日志对象
        language:   语言
        ner_model:  HuggingFace NER 模型名（None 时按语言自动选择）
        group_size: 每组最大句数（默认 50）
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    if group_size > 30:
        group_size = 30  # 避免过大导致内存暴增，尤其是中文模型
        logger.warning(f"[NER] group_size 设置过大，已自动调整为 30")
    # ── 确定模型 ──────────────────────────────────────────────────────────
    if ner_model is None:
        lang_key = language.strip().lower() if language else ""
        ner_model = DEFAULT_NER_MODELS.get(language, DEFAULT_NER_MODELS.get(lang_key, "Davlan/bert-base-multilingual-cased-ner-hrl"))

    privacy_labels = _get_privacy_labels(ner_model)
    logger.info(f"[NER] 模型: {ner_model}")
    logger.info(f"[NER] 隐私实体标签: {privacy_labels}")

    os.makedirs(mask_dir, exist_ok=True)

    # ── 加载模型 ──────────────────────────────────────────────────────────
    start_time = time.time()
    logger.info(f"[NER] 开始加载模型: {ner_model}")

    tokenizer = AutoTokenizer.from_pretrained(ner_model)
    model = AutoModelForTokenClassification.from_pretrained(ner_model, device_map="auto")
    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU，避免与下游 GPU 工具冲突
    )
    logger.info(f"[NER] 模型加载完成，耗时 {time.time() - start_time:.1f}s")

    # ── 遍历转录文件 ──────────────────────────────────────────────────────
    trans_dir_p = Path(trans_dir)
    trans_files = sorted(trans_dir_p.rglob('*.json'))
    all_len = len(trans_files)
    if all_len == 0:
        logger.warning(f"[NER] 在 {trans_dir} 中未找到 JSON 文件")
        return

    logger.info(f"[NER] 共 {all_len} 个文件待处理")

    for count, trans_file in enumerate(trans_files):
        try:
            rel = trans_file.relative_to(trans_dir_p)
            mask_path_p = Path(mask_dir) / rel
            mask_path_p.parent.mkdir(parents=True, exist_ok=True)
            mask_path = str(mask_path_p)

            if os.path.isfile(mask_path):
                logger.info(f"[NER] {mask_path} 已存在，跳过")
                continue

            with open(trans_file, 'r', encoding='utf-8') as f:
                _raw = json.load(f)
            sentences = _raw.get('sentences', _raw) if isinstance(_raw, dict) else _raw

            file_start = time.time()
            num_sentences = len(sentences)
            logger.info(f"[NER] 处理 [{count+1}/{all_len}]: {rel}, 共 {num_sentences} 句")

            mask_save = []

            # ── 分组拼接 ──────────────────────────────────────────────────
            groups = []  # [(meta_list, joined_text)]
            buf_texts = []
            buf_meta = []   # [(idx, text, timestamps, char_offset)]
            buf_offset = 0

            for idx, sent in enumerate(sentences):
                text = sent.get("text", "")
                timestamps = sent.get("timestamp", [])
                if not text.strip() or not timestamps:
                    continue

                if buf_texts and len(buf_texts) >= group_size:
                    groups.append((buf_meta, "".join(buf_texts)))
                    buf_texts, buf_meta, buf_offset = [], [], 0

                buf_meta.append((idx, text, timestamps, buf_offset))
                buf_texts.append(text)
                buf_offset += len(text)

            if buf_texts:
                groups.append((buf_meta, "".join(buf_texts)))

            logger.debug(f"[NER] 分 {len(groups)} 组处理")

            # ── 逐组 NER 推理（自动分 chunk 避免超 512 token）──────────
            max_len = getattr(tokenizer, 'model_max_length', 512) or 512

            for g_idx, (meta_list, joined_text) in enumerate(groups):
                # 按 token 长度切 chunk，避免 BERT 512 上限
                entities = []
                text_chunks = _split_by_tokens(tokenizer, joined_text, max_len)
                char_offset = 0
                for chunk_text in text_chunks:
                    chunk_entities = nlp(chunk_text)
                    for ent in chunk_entities:
                        ent["start"] += char_offset
                        ent["end"] += char_offset
                    entities.extend(chunk_entities)
                    char_offset += len(chunk_text)

                for ent in entities:
                    label = ent.get("entity_group", ent.get("entity", ""))
                    if label not in privacy_labels:
                        continue

                    abs_start = ent["start"]
                    abs_end = ent["end"]

                    # 直接从拼接原文截取（避免 tokenizer 加空格）
                    ent_text = joined_text[abs_start:abs_end].strip()
                    if not ent_text:
                        continue

                    # 映射回所属句子
                    for (idx, sent_text, timestamps, offset) in meta_list:
                        local_start = abs_start - offset
                        local_end = abs_end - offset

                        if 0 <= local_start < len(sent_text) and 0 < local_end <= len(sent_text):
                            # 实体完全在当前句内
                            verify = sent_text[local_start:local_end]
                            if verify.strip() != ent_text:
                                continue
                            start_ts, end_ts = char_to_word_positions(
                                local_start, local_end, sent_text, language, timestamps
                            )
                            mask_save.append({
                                "idx": idx,
                                "sentence": sent_text,
                                "start": start_ts,
                                "end": end_ts,
                                "text": ent_text,
                            })
                            logger.debug(f"[NER] 句{idx}: '{ent_text}' ({label}) → [{start_ts}:{end_ts}]")
                            break
                        elif local_start < len(sent_text) and local_end > 0:
                            # 实体跨越句边界，取交集
                            clip_start = max(0, local_start)
                            clip_end = min(len(sent_text), local_end)
                            clipped = sent_text[clip_start:clip_end].strip()
                            if not clipped:
                                continue
                            start_ts, end_ts = char_to_word_positions(
                                clip_start, clip_end, sent_text, language, timestamps
                            )
                            mask_save.append({
                                "idx": idx,
                                "sentence": sent_text,
                                "start": start_ts,
                                "end": end_ts,
                                "text": clipped,
                            })
                            logger.debug(f"[NER] 句{idx} (跨句): '{clipped}' ({label}) → [{start_ts}:{end_ts}]")

            if not mask_save:
                mask_save.append({
                    "idx": -1,
                    "sentence": "",
                    "start": 0,
                    "end": 0,
                    "text": "未识别到隐私词",
                })

            with open(mask_path, 'w', encoding='utf-8') as f:
                json.dump(mask_save, f, ensure_ascii=False, indent=4)

            file_time = time.time() - file_start
            privacy_count = sum(1 for m in mask_save if m["idx"] >= 0)
            logger.info(f"[NER] {rel} 完成: {privacy_count} 个隐私词, 耗时 {file_time:.1f}s")

        except Exception as e:
            import traceback
            logger.error(f"[NER] 处理 {trans_file} 失败: {e}\n{traceback.format_exc()}")
            os.makedirs(os.path.join(mask_dir, "logs"), exist_ok=True)
            with open(os.path.join(mask_dir, "logs", "fail_list.txt"), 'a') as f:
                f.write(str(trans_file) + '\n')

    logger.info(f"[NER] 全部处理完毕，耗时 {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    import argparse
    from utils.logger_example import setup_logger

    parser = argparse.ArgumentParser(description="NER 隐私词检测")
    parser.add_argument("--trans_dir", "-t", required=True, help="转录 JSON 目录")
    parser.add_argument("--mask_dir", "-m", required=True, help="隐私词结果输出目录")
    parser.add_argument("--language", default="中文", help="语言")
    parser.add_argument("--ner_model", default=None, help="NER 模型名称（不指定则自动选择）")
    parser.add_argument("--group_num", type=int, default=50, help="每批处理句数")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO", help="日志级别")
    args = parser.parse_args()

    logger, _ = setup_logger(args.mask_dir, "mask_ner", args.mode)
    mask_ner(
        trans_dir=args.trans_dir,
        mask_dir=args.mask_dir,
        logger=logger,
        language=args.language,
        ner_model=args.ner_model,
        group_size=args.group_num,
    )
