import json

# 给不支持自动分句的ASR使用。

def convert_words_to_sentences(timestamps, text):
    """
    将单词级时间戳转换为句子级格式
    
    Args:
        timestamps: 包含每个token时间戳的列表
                [ {'token': '你', 'start_time': 0.54, 'end_time': 0.6, ...}, ... ]
    
    Returns:
        句子级格式的列表
        [
            {
                "index": 0,
                "text": "你好，",
                "timestamp": [
                    540,
                    720,
                    840
                ]
            },...

    """
    sentences = []
    current_sentence = []
    current_timestamps = []
    sentence_index = 0
    
    # 标点符号集合（中英文标点）
    punctuation = set('，。！？；：,.!?;:')
    
    for token_info in timestamps:
        token = token_info['token']
        start_time = token_info['start_time']
        
        # 将秒转换为毫秒
        start_ms = int(start_time * 1000)
        
        # 添加到当前句子
        current_sentence.append(token)
        current_timestamps.append(start_ms)
        
        # 如果当前token是标点符号，结束当前句子
        if token in punctuation:
            # 如果句子不为空，保存
            if current_sentence:
                sentence_text = ''.join(current_sentence)
                
                sentences.append({
                    "index": sentence_index,
                    "text": sentence_text,
                    "timestamp": current_timestamps.copy()  # 如果只需要开始时间
                })
                
                sentence_index += 1
                current_sentence = []
                current_timestamps = []
    
    # 处理最后可能没有标点结尾的句子
    if current_sentence:
        sentence_text = ''.join(current_sentence)
        sentences.append({
            "index": sentence_index,
            "text": sentence_text,
            "timestamp": current_timestamps
        })
    
    return sentences


# # 原始数据
# timestamps = [
#     {'token': '你', 'start_time': 0.54, 'end_time': 0.6, 'score': 0.73, 0: 0.54, 1: 0.6},
#       {'token': '好', 'start_time': 0.72, 'end_time': 0.78, 'score': 0.997, 0: 0.72, 1: 0.78}, {'token': '，', 'start_time': 0.84, 'end_time': 0.9, 'score': 0.0, 0: 0.84, 1: 0.9}, {'token': '心', 'start_time': 0.96, 'end_time': 1.02, 'score': 0.886, 0: 0.96, 1: 1.02}, {'token': '理', 'start_time': 1.14, 'end_time': 1.2, 'score': 0.988, 0: 1.14, 1: 1.2}, {'token': '援', 'start_time': 1.26, 'end_time': 1.32, 'score': 0.595, 0: 1.26, 1: 1.32}, {'token': '助', 'start_time': 1.38, 'end_time': 1.44, 'score': 0.471, 0: 1.38, 1: 1.44}, {'token': '热', 'start_time': 1.62, 'end_time': 1.68, 'score': 0.987, 0: 1.62, 1: 1.68}, {'token': '线', 'start_time': 1.8, 'end_time': 1.86, 'score': 0.994, 0: 1.8, 1: 1.86}, {'token': '。', 'start_time': 1.92, 'end_time': 1.98, 'score': 0.0, 0: 1.92, 1: 1.98}, {'token': '呃', 'start_time': 0.36, 'end_time': 0.42, 'score': 0.913, 0: 2960.36, 1: 2960.42}, {'token': '，', 'start_time': 0.72, 'end_time': 0.78, 'score': 0.0, 0: 2960.72, 1: 2960.78}, {'token': '我', 'start_time': 0.9, 'end_time': 0.96, 'score': 0.999, 0: 2960.9, 1: 2960.96}, {'token': '那', 'start_time': 1.14, 'end_time': 1.2, 'score': 0.999, 0: 2961.14, 1: 2961.2}, {'token': '个', 'start_time': 1.26, 'end_time': 1.32, 'score': 0.997, 0: 2961.26, 1: 2961.32}]

# # 转换
# result = convert_words_to_sentences(timestamps)

# # 打印结果

# print(json.dumps(result, ensure_ascii=False, indent=2))