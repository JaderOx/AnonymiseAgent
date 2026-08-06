import os
import re
import json
import time
from utils.logger_example import setup_logger

# 直接运行该文件：
# 会根据trans_root中的json文件(音频的转录内容)，提取文本以group_size句为一组,前面加上prompt喂给大模型。
# 对生成的response后处理成json文件保存到mask_root中。后处理时关键词会在原句中上下search_tolerance之间进行查找，避免大模型数数幻觉。
llm_model = "Qwen2.5-32B-Instruct"   # LLM 模型名称，None 时按语言自动推断
trans_dir = "/path/to/data_backup/crisis/mask_accuracy/biyuan/trans_clear"
mask_dir = "/path/to/data_backup/crisis/mask_accuracy/biyuan/Qwen2.5-32B-Instruct-gp=20-default"
log_dir = "/path/to/data_backup/crisis/mask_accuracy/biyuan/Qwen2.5-32B-Instruct-gp=20-default"
logmode = "DEBUG"
NO_WER_LANGUAGE = {"中文", "日文"}
# 超参数
group_size = 20              # 每group_size句分一组处理


def mask_llm(
    trans_dir = None,
    mask_dir = None,
    logger = None,
    language = '中文',
    prompt = None,
    llm_model = None,           # LLM 模型名称，None 时按语言自动推断
    group_size = group_size
):
    from utils.language_model import load_model, generate_response
    from configs.config_loader import get_llm_config

    # ── 从配置文件读取 LLM 配置（模型、prompt、格式后缀、分组前缀） ──────────
    llm_cfg = get_llm_config(language)

    # 模型：参数传入 > 配置文件
    if llm_model is None:
        llm_model = llm_cfg['model']

    # prompt 主体：参数传入 > 配置文件
    if prompt is None:
        prompt = llm_cfg.get('prompt', '')

    # prompt_suffix 和 group_prefix 始终从配置文件读取
    prompt_suffix = llm_cfg.get('prompt_suffix', '')
    group_prefix  = llm_cfg.get('group_prefix', '')

    full_prompt = (prompt or '') + '\n' + (prompt_suffix or '')

    os.makedirs(mask_dir, exist_ok=True)

    start_time = time.time()
    logger.info(f"开始加载LLM:{llm_model}")

    model, tokenizer = load_model(llm_model)
    time_cost = time.time() - start_time

    logger.info(f"LLM加载完毕, 耗时: {time_cost:.2f} 秒")

    logger.info(f"开始使用LLM筛选转录文本中的隐私信息，以{group_size}句为一组\n处理文本目录:{trans_dir}\n隐私结果保存目录:{mask_dir}\nLLM:{llm_model}\n")

    from pathlib import Path as _Path
    _trans_dir_p = _Path(trans_dir)
    trans_files = sorted(_trans_dir_p.rglob('*.json'))
    all_len = len(trans_files)
    if all_len == 0:
        logger.warning(f"在目录 {trans_dir} 中未找到合法的转录文本文件（*.json）")
        return
    count = 0

    for trans_file in trans_files:
        try:
            rel = trans_file.relative_to(_trans_dir_p)
            sub = str(rel)
            trans_path = str(trans_file)
            mask_path_p = _Path(mask_dir) / rel
            mask_path_p.parent.mkdir(parents=True, exist_ok=True)
            mask_path = str(mask_path_p)

            mask_save = []
            with open(trans_path, 'r', encoding='utf-8') as f:
                _raw = json.load(f)
            trans = _raw.get('sentences', _raw) if isinstance(_raw, dict) else _raw

            num_sentences = len(trans)
            file_start = time.time()
            # 找出分组数，向上取整
            num_group = (num_sentences + group_size - 1) // group_size

            logger.info(f"处理文本[{count + 1}/{all_len}]: {sub}, 共{num_sentences}句，分{num_group}组处理")
            count += 1

            if os.path.isfile(mask_path):
                logger.info(f"{mask_path}已存在")
                continue
            if os.path.exists(mask_path):
                logger.info(f"{mask_path} already exists")
                continue

            for idx_g in range(num_group):
                start_sentence_idx = idx_g * group_size
                end_sentence_idx = min(start_sentence_idx+group_size, num_sentences)

                logger.debug(f"处理{idx_g+1}/{num_group}: 第{start_sentence_idx+1}句~第{end_sentence_idx}句")

                # 将json文件文本整合
                text = group_prefix
                for i in range(start_sentence_idx, end_sentence_idx):
                    sentence = trans[i]
                    text += sentence["text"]
                logger.debug(f"输入LLM的文本: \n{text}\n")

                res = generate_response(model, tokenizer, prompt=full_prompt + text)

                logger.debug(f"LLM输出: \n{res}")

                final_result = extract_result_from_response(res)
                logger.debug(f"提取结果: \n{final_result}")

                # 把res转为json格式:
                res = text2list(final_result)
                logger.debug(f"处理结果: \n{res}")

                # 循环处理每个隐私词
                for word in res:
                    found_in_context = False
                    # 对每个隐私词，查找当前组的全文
                    for idx in range(start_sentence_idx, end_sentence_idx):
                        sentence = trans[idx]

                        # 使用正则表达式查找所有匹配位置
                        pattern = re.escape(word)  # 转义特殊字符
                        matches = [(m.start(), m.end()) for m in re.finditer(pattern, sentence["text"])]
                        
                        if matches:  # 在当前句找到了
                            logger.debug(f"在句子 '{idx}' : '{sentence['text']}'找到隐私词 '{word}', 位置信息：'{matches}'")
                            # 可能一句中有多个匹配
                            for start_idx, end_idx in matches:
                                start, end = char_to_word_positions(start_idx, end_idx, sentence["text"], language, sentence["timestamp"])

                                mask_save.append({
                                    "idx": idx,
                                    "sentence": sentence["text"],
                                    "start": start,
                                    "end": end,
                                    "text": word
                                })
                            found_in_context = True
                    # 如果没找到
                    if not found_in_context:
                        mask_save.append({
                            "idx": idx,
                            "sentence": sentence["text"],
                            "start": 0,
                            "end": 0,
                            "text": f"识别到隐私词'{word}'但原文并没有"
                        })
            if not mask_save:
                mask_save.append({
                        "idx": -1,
                        "sentence": "",
                        "start": 0,
                        "end": 0,
                        "text": "未识别到隐私词"
                    })
            with open(mask_path, 'w') as f:
                file_time = time.time() - file_start
                logger.info(f"文件 {sub} 文本处理完成, 耗时: {file_time:.2f}秒")

                json.dump(mask_save, f, ensure_ascii=False, indent=4)

        except Exception as e:
                import traceback
                error_trace = traceback.format_exc()  # 获取完整的traceback字符串
                os.makedirs(os.path.join(mask_dir, "logs"), exist_ok=True)
                with open(os.path.join(mask_dir, "logs", "fail_list.txt"), 'a') as f:
                    logger.error(f"处理文件\" {trans_file}\" 失败: {e}\n{error_trace}\n")
                    f.write(str(trans_file) + '\n')\
                    
    logger.info(f"所有文本处理完毕，耗时{time.time() - start_time:.2f}s")

def text2list(data_str):
    """
    解析隐私数据字符串，转换为列表
    
    Args:
        data_str: 格式为"word1,word2,..."的字符串
    
    Returns:
        list: ["word1", "word2", ...]
    """
    if not data_str or data_str.strip() == "":
        return []
    if data_str.strip().lower() == "none":
        return []
    
    result = []
    
    # 按'|'分割不同的块
    blocks = data_str.split(',')
    for block in blocks:
        block = block.strip()
        if block:
            result.append(block)

    return result

def char_to_word_positions(
        char_start: int,
        char_end: int,
        text: str,
        language: str,
        word_timestamps: list[int]
        ) -> tuple[int, int]:
    """
    将字符位置范围转换为词位置范围
    
    Args:
        char_start: 字符起始位置
        char_end: 字符结束位置
        text: 完整文本
        word_timestamps: 每个词的起始时间戳列表（长度等于词数）
    
    Returns:
        (word_start_timestamp, word_end_timestamp) 词级别的时间戳范围
    """
    if language in NO_WER_LANGUAGE:
        # 按字处理，每个字就是一个词
        word_start_timestamp = word_timestamps[char_start] if char_start < len(word_timestamps) else word_timestamps[-1]
        word_end_timestamp = word_timestamps[char_end] if char_end < len(word_timestamps) else word_timestamps[-1] + 100

    else:
        # 将文本按词分割
        words = text.split()
        
        # 构建字符到词的映射数组
        # 数组长度等于文本长度，每个位置记录该字符属于哪个词
        char_to_word = [-1] * len(text)
        
        current_word_idx = 0
        current_char_pos = 0
        
        for word_idx, word in enumerate(words):
            # 标记这个词的每个字符
            for i in range(len(word)):
                if current_char_pos < len(char_to_word):
                    char_to_word[current_char_pos] = word_idx
                current_char_pos += 1
            
            # 处理词后面的空格（除了最后一个词）
            if word_idx < len(words) - 1 and current_char_pos < len(char_to_word):
                char_to_word[current_char_pos] = word_idx  # 空格属于前一个词
                current_char_pos += 1
        
        # 确保字符位置在有效范围内
        char_start = max(0, min(char_start, len(char_to_word) - 1))
        char_end = max(0, min(char_end, len(char_to_word) - 1))
        
        # 找到覆盖字符范围的所有词索引
        word_indices = set()
        for pos in range(char_start, char_end):
            if pos < len(char_to_word) and char_to_word[pos] != -1:
                word_indices.add(char_to_word[pos])
        
        if not word_indices:
            # 如果没有找到任何词，返回默认值
            return word_timestamps[0], word_timestamps[-1] + 100
        
        # 获取词索引范围
        start_word_idx = min(word_indices)
        end_word_idx = max(word_indices)
        # 转换为时间戳
        word_start_timestamp = word_timestamps[start_word_idx]
        
        # 结束时间戳：如果是最后一个词，用最后一个时间戳+100，否则用下一个词的开始时间戳
        if end_word_idx + 1 < len(word_timestamps):
            word_end_timestamp = word_timestamps[end_word_idx + 1]
        else:
            word_end_timestamp = word_timestamps[-1] + 100

    return word_start_timestamp, word_end_timestamp       

def extract_result_from_response(response):
    import re
    # 优先提取标签内的内容
    tag_matches = re.findall(r'<result>(.*?)</result>', response, re.DOTALL)
    if tag_matches:
        return tag_matches[-1].strip()  # 取最后一个
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM 隐私词检测")
    parser.add_argument("--trans_dir", "-t", required=True, help="转录 JSON 目录")
    parser.add_argument("--mask_dir", "-m", required=True, help="隐私词结果输出目录")
    parser.add_argument("--language", default="中文", help="语言")
    parser.add_argument("--llm_model", default="Qwen2.5-32B-Instruct", help="LLM 模型名称")
    parser.add_argument("--group_num", type=int, default=60, help="每批处理句数")
    parser.add_argument("--prompt", default=None, help="自定义 prompt（不指定则用配置文件默认）")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO", help="日志级别")
    args = parser.parse_args()

    logger, _ = setup_logger(args.mask_dir, "mask_llm", args.mode)
    mask_llm(
        trans_dir=args.trans_dir,
        mask_dir=args.mask_dir,
        logger=logger,
        language=args.language,
        llm_model=args.llm_model,
        group_size=args.group_num,
        prompt=args.prompt,
    )