import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from logger_example import setup_logger


def compute_directory_accuracy(gt_dir: str, mask_dir: str,
                                accuracy_output_dir: str, logger) -> tuple:
    """
    批量计算隐私词识别准确率，逐文件保存详细 JSON 报告，返回汇总 DataFrame。

    Args:
        gt_dir               : GT JSON 目录（每个文件对应一个 JSON，空文件表示无隐私）
        mask_dir             : 模型输出 mask JSON 目录（middle_dir/mask/）
        accuracy_output_dir  : 单文件报告保存目录（output_dir/mask_accuracy/）
        logger               : 日志对象

    Returns:
        (df, summary_str)
        df 列: name, precision, recall, f1_score
        最后两行为 -total-（累计 TP/FP/FN 重算的整体指标）和 -mean-（各文件均值）
    """
    os.makedirs(accuracy_output_dir, exist_ok=True)

    gt_dir_p   = Path(gt_dir)
    mask_dir_p = Path(mask_dir)

    gt_files = sorted(gt_dir_p.rglob("*.json"))
    if not gt_files:
        logger.warning(f"accuracy_calc: {gt_dir} 中无 GT JSON 文件")
        return None, ""

    rows = []
    total_gt_t = total_model_t = total_tp_t = total_fp_t = total_fn_t = 0
    total_gt = total_tp = total_fp = total_fn = 0

    for gt_path in gt_files:
        rel = gt_path.relative_to(gt_dir_p)
        stem = gt_path.stem
        mask_path = mask_dir_p / rel   # mirror subdir structure
        output_subdir = Path(accuracy_output_dir) / ("accuracy") / rel.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            logger.warning(f"accuracy_calc: 未找到 mask JSON {mask_path}，跳过 {rel}")
            continue

        try:
            result = calculate_accuracy(str(gt_path), str(mask_path))
        except Exception as e:
            import traceback
            logger.warning(f"accuracy_calc: {stem} 计算失败: {e}\n{traceback.format_exc()}")
            continue
        report_path = output_subdir / (stem + ".json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        p  = result['time_based_precision']
        r  = result['time_based_recall']
        f1 = result['time_based_f1_score']
        rows.append((stem, p, r, f1))

        total_gt_t    += result['tp_duration'] + result['fn_duration']
        total_model_t += result['tp_duration'] + result['fp_duration']
        total_tp_t    += result['tp_duration']
        total_fp_t    += result['fp_duration']
        total_fn_t    += result['fn_duration']
        total_gt      += result['total_groundtruth']
        total_tp       += result['true_positives']
        total_fp       += result['false_positives']
        total_fn       += result['false_negatives']

        logger.info(f"  {stem}: P={p:.1f}% R={r:.1f}% F1={f1:.1f}%")

    if not rows:
        logger.warning("accuracy_calc: 无有效文件完成计算")
        return None, ""

    df = pd.DataFrame(rows, columns=["name", "precision", "recall", "f1_score"])

    # 整体指标（按累计 TP/FP/FN 重算）
    agg_p  = total_tp_t / total_model_t * 100 if total_model_t > 0 else 0.0
    agg_r  = total_tp_t / total_gt_t    * 100 if total_gt_t    > 0 else 0.0
    agg_f1 = (2 * agg_p * agg_r / (agg_p + agg_r)) if (agg_p + agg_r) > 0 else 0.0


    df.loc[len(df)] = ['-mean-', round(agg_p, 2),  round(agg_r, 2),  round(agg_f1, 2)]

    summary_str = (
        f" 整体(按时间计算): P={agg_p:.1f}% R={agg_r:.1f}% F1={agg_f1:.1f}%"
        f"（GT={total_gt} 条，TP={total_tp}，FP={total_fp}，FN={total_fn}）"
 
    )
    logger.info(f"accuracy_calc 汇总:{summary_str}")
    return df, summary_str

def calculate_accuracy(groundtruth_file: str, model_output_file: str) -> Dict:
    """
    计算模型输出的准确率
    
    Args:
        groundtruth_file: groundtruth JSON文件路径
        model_output_file: 模型输出 JSON文件路径
        
    Returns:
        包含各项指标的字典
    """
    # 读取数据
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        groundtruth = json.load(f)
    
    with open(model_output_file, 'r', encoding='utf-8') as f:
        model_output = json.load(f)
    
    # 将groundtruth转换为更容易查询的格式
    gt_dict = {}  # 使用(idx, start, end)作为键，因为同一个句子可能有多个标注
    for item in groundtruth:
        key = (item['idx'], item['start'], item['end'])
        gt_dict[key] = item
    
    # 将模型输出也转换为字典格式
    model_dict = {}
    for item in model_output:
        if 'idx_g' in item:
            continue  # 跳过idx_g字段的条目: 这些是无敏感的组
        if item['idx'] == -1:
            continue  # 跳过idx为-1的条目: 这些是无效的条目
        else:
            key = (item['idx'], item['start'], item['end'])
            model_dict[key] = item
        
    # 统计指标
    total_gt = len(gt_dict)
    total_model = len(model_dict)

    if total_gt == 0 and total_model == 0:
        results = {
            'total_groundtruth': 0,
            'total_model_output': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'repeated_correct_items': 0,
            'tp_duration': 0,
            'fn_duration': 0,
            'fp_duration': 0,
            'time_based_precision': 100.0,
            'time_based_recall': 100.0,
            'time_based_f1_score': 100.0,
            'precision': 100.0,  # 完美准确率
            'recall': 100.0,     # 完美召回率
            'f1_score': 100.0,   # 完美F1分数
            'false_negatives_analysis': {
                'total': 0,
                'completely_missed': [],
                'text_mismatch': [],
                'completely_missed_count': 0,
                'text_mismatch_count': 0
            },
            'false_positives_analysis': {
                'total': 0,
                'non_private_info': [],
                'text_mismatch': [],
                'non_private_info_count': 0,
                'text_mismatch_count': 0
            },
            'correct_matches': []
        }
        return results
    
    # 找出正确识别的条目
    correct_items = []
    repeated_correct_items = [] # 记录重复识别的正确条目（同一groundtruth被多个模型输出匹配）
    false_positives = []  # 模型多识别出来的
    false_negatives = []  # 模型漏掉的
    
    # 复制一份用于处理
    temp_model_dict = model_dict.copy()
    
    # 检查每个groundtruth是否被正确识别
    for gt_key, gt_item in gt_dict.items():
        gt_text = gt_item['text']
        found = False
        keys_to_delete = []  # 收集要删除的key
        
        # 在模型输出中查找匹配
        for model_key, model_item in temp_model_dict.items():
            # 检查是否在同一句（idx相同）
            if model_item['idx'] == gt_item['idx']:
                model_text = model_item['text']

                # 两者start和end区间重叠超过model的50%
                gt_start, gt_end = gt_item['start'], gt_item['end']
                model_start, model_end = model_item['start'], model_item['end']
                overlap_start = max(gt_start, model_start)
                overlap_end = min(gt_end, model_end)
                overlap_length = max(0, overlap_end - overlap_start)
                model_length = model_end - model_start
                if model_length == 0:
                    continue  # 避免除以零
                gt_length = gt_end - gt_start

                if (overlap_length / model_length >= 0.5) or (overlap_length / gt_length >= 0.5):
                    # 如果模型输出的text包含groundtruth的text
                    if (gt_text in model_text) or (model_text in gt_text):
                        if found == True:
                            # 已经找到过匹配了，说明有重复识别的情况
                            repeated_correct_items.append({
                                'groundtruth': gt_item,
                                'model': model_item
                            })
                            
                        else:
                            correct_items.append({
                                'groundtruth': gt_item,
                                'model': model_item
                            })
                        found = True
                        # 从模型输出中移除匹配项
                        keys_to_delete.append(model_key)
                else:
                    continue

        for key in keys_to_delete:
            if key in temp_model_dict:
                del temp_model_dict[key]
        if not found:
            false_negatives.append(gt_item)
    
    # 找出false positives（模型识别了但groundtruth中没有的）
    # 创建一个集合，包含所有已匹配的模型输出
    matched_model_keys = set()
    for correct in correct_items:
        model_item = correct['model']
        matched_model_keys.add((model_item['idx'], model_item['start'], model_item['end']))
    
    for model_key, model_item in temp_model_dict.items():
        if model_key not in matched_model_keys:
            false_positives.append(model_item)
    
    # 计算准确率、召回率和F1分数
    true_positives = len(correct_items)
    
    precision = (true_positives + len(repeated_correct_items)) / total_model if total_model > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 计算基于时间长度的指标
    time_based_results = calculate_time_based_metrics(gt_dict, model_dict)
    
    # 详细分析漏掉的条目
    fn_analysis = analyze_false_negatives(false_negatives, temp_model_dict)
    
    # 详细分析多余的条目
    fp_analysis = analyze_false_positives(false_positives, gt_dict)
    
    results = {
        'total_groundtruth': total_gt,
        'total_model_output': total_model,
        'true_positives': true_positives,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'repeated_correct_items': len(repeated_correct_items),
        # 基于时间长度的指标
        'tp_duration': time_based_results['tp_duration'],
        'fn_duration': time_based_results['fn_duration'],
        'fp_duration': time_based_results['fp_duration'],
        'time_based_precision': time_based_results['precision'],
        'time_based_recall': time_based_results['recall'],
        'time_based_f1_score': time_based_results['f1_score'],
        # 基于个数的指标
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'false_negatives_analysis': fn_analysis,
        'false_positives_analysis': fp_analysis,
        'correct_matches': correct_items
    }
    
    return results

def analyze_false_negatives(false_negatives: List[Dict], model_dict: Dict) -> Dict:
    """
    分析漏掉的条目：检查是完全没有识别，还是识别了但文本不匹配
    
    Args:
        false_negatives: 漏识别的groundtruth列表
        model_dict: 处理后的模型输出字典（已过滤idx_g）
    """
    analysis = {
        'total': len(false_negatives),
        'completely_missed': [],  # 完全没有识别到该句子中的任何隐私信息
        'text_mismatch': []  # 识别到了但文本不匹配
    }
    
    # 将模型输出按句子索引分组
    model_by_idx = {}
    for model_item in model_dict.values():
        idx = model_item['idx']
        if idx not in model_by_idx:
            model_by_idx[idx] = []
        model_by_idx[idx].append(model_item)
    
    for fn_item in false_negatives:
        idx = fn_item['idx']
        gt_text = fn_item['text']
        
        # 检查该句子是否有任何模型输出
        if idx in model_by_idx:
            # 有输出，检查是否有文本匹配
            text_matched = False
            matched_model = None
            
            for model_item in model_by_idx[idx]:
                model_text = model_item['text']
                if (gt_text in model_text) or (model_text in gt_text):
                    text_matched = True
                    matched_model = model_item
                    break
            
            if text_matched:
                # 文本匹配上了但还是被记为false negative，说明是位置没达标
                # 这里只是记录，不判断位置
                analysis['text_mismatch'].append({
                    'groundtruth': fn_item,
                    'model': matched_model,
                    'issue': '文本匹配但位置条件未满足'
                })
            else:
                # 完全没有文本匹配
                analysis['completely_missed'].append({
                    'groundtruth': fn_item,
                    'model_outputs_in_sentence': [
                        {
                            'text': m['text'],
                            'start': m['start'],
                            'end': m['end']
                        } for m in model_by_idx[idx]
                    ],
                    'issue': '句子中有模型输出但文本不匹配'
                })
        else:
            # 完全没有输出
            analysis['completely_missed'].append({
                'groundtruth': fn_item,
                'issue': '整句无任何模型输出'
            })
    
    # 统计数量
    analysis['completely_missed_count'] = len(analysis['completely_missed'])
    analysis['text_mismatch_count'] = len(analysis['text_mismatch'])
    
    return analysis

def analyze_false_positives(false_positives: List[Dict], gt_dict: Dict) -> Dict:
    """
    分析多余的条目：检查是识别了非隐私信息，还是文本不匹配
    
    Args:
        false_positives: 错误识别的模型输出列表
        gt_dict: groundtruth字典
    """
    analysis = {
        'total': len(false_positives),
        'non_private_info': [],  # 句子中完全没有groundtruth
        'text_mismatch': []  # 文本不匹配
    }
    
    # 将groundtruth按句子索引分组
    gt_by_idx = {}
    for gt_item in gt_dict.values():
        idx = gt_item['idx']
        if idx not in gt_by_idx:
            gt_by_idx[idx] = []
        gt_by_idx[idx].append(gt_item)
    
    for fp_item in false_positives:
        idx = fp_item['idx']
        fp_text = fp_item['text']
        
        # 检查该句子是否有groundtruth
        if idx in gt_by_idx:
            # 有groundtruth，检查是否有文本匹配
            text_matched = False
            matched_gt = None
            
            for gt_item in gt_by_idx[idx]:
                gt_text = gt_item['text']
                if (fp_text in gt_text) or (gt_text in fp_text):
                    text_matched = True
                    matched_gt = gt_item
                    break
            
            if text_matched:
                # 文本匹配上了但还是被记为false positive，说明是位置没达标
                analysis['text_mismatch'].append({
                    'model': fp_item,
                    'closest_gt': matched_gt,
                    'issue': '文本匹配但位置条件未满足'
                })
            else:
                # 完全没有文本匹配
                analysis['text_mismatch'].append({
                    'model': fp_item,
                    'groundtruth_in_sentence': [
                        {
                            'text': gt['text'],
                            'start': gt['start'],
                            'end': gt['end']
                        } for gt in gt_by_idx[idx]
                    ],
                    'issue': '文本不匹配'
                })
        else:
            # 该句子没有groundtruth，模型误识别
            analysis['non_private_info'].append({
                'model': fp_item,
                'issue': '该句子没有标注任何隐私信息'
            })
    
    # 统计数量
    analysis['non_private_info_count'] = len(analysis['non_private_info'])
    analysis['text_mismatch_count'] = len(analysis['text_mismatch'])
    
    return analysis

def print_detailed_report(results: Dict):
    """
    打印详细报告（简化版本）
    """
    print("=" * 80)
    print("隐私信息识别准确率评估报告")
    print("=" * 80)
    
    print(f"\n基本统计:")
    print(f"  Groundtruth总数: {results['total_groundtruth']}")
    print(f"  模型输出总数: {results['total_model_output']}")
    print(f"  正确识别数: {results['true_positives']}")
    print(f"  重复正确识别数: {results['repeated_correct_items']}")
    print(f"  错误识别数(False Positives): {results['false_positives']}")
    print(f"  漏识别数(False Negatives): {results['false_negatives']}")
    
    print(f"\n核心指标:")
    print(f"  准确率 (Precision): {results['precision']}%")
    print(f"  召回率 (Recall): {results['recall']}%")
    print(f"  F1分数: {results['f1_score']}%")
    
    print(f"\n漏识别分析 (False Negatives):")
    fn_analysis = results['false_negatives_analysis']
    print(f"  总数: {fn_analysis['total']}")
    print(f"  完全漏掉(整句无输出或文本不匹配): {fn_analysis['completely_missed_count']}")
    print(f"  文本匹配但位置未达标: {fn_analysis['text_mismatch_count']}")
    
    # 显示示例
    if fn_analysis.get('completely_missed', [])[:2]:
        print("\n  完全漏掉示例:")
        for item in fn_analysis['completely_missed'][:2]:
            print(f"    GT: idx={item['groundtruth']['idx']}, text='{item['groundtruth']['text']}'")
            if 'model_outputs_in_sentence' in item:
                print(f"    句子中的模型输出: {[m['text'] for m in item['model_outputs_in_sentence']]}")
            print(f"    问题: {item['issue']}")
    
    if fn_analysis.get('text_mismatch', [])[:2]:
        print("\n  文本匹配但位置未达标示例:")
        for item in fn_analysis['text_mismatch'][:2]:
            print(f"    GT: idx={item['groundtruth']['idx']}, text='{item['groundtruth']['text']}'")
            print(f"    模型: idx={item['model']['idx']}, text='{item['model']['text']}'")
            print(f"    问题: {item['issue']}")
    
    print(f"\n错误识别分析 (False Positives):")
    fp_analysis = results['false_positives_analysis']
    print(f"  总数: {fp_analysis['total']}")
    print(f"  识别了非隐私信息: {fp_analysis['non_private_info_count']}")
    print(f"  文本匹配但位置未达标: {fp_analysis['text_mismatch_count']}")
    
    # 显示错误识别示例
    if fp_analysis.get('non_private_info', [])[:2]:
        print("\n  非隐私信息示例:")
        for item in fp_analysis['non_private_info'][:2]:
            print(f"    模型输出: idx={item['model']['idx']}, text='{item['model']['text']}'")
            print(f"    问题: {item['issue']}")
    
    if fp_analysis.get('text_mismatch', [])[:2]:
        print("\n  文本匹配但位置未达标示例:")
        for item in fp_analysis['text_mismatch'][:2]:
            print(f"    模型输出: idx={item['model']['idx']}, text='{item['model']['text']}'")
            if 'closest_gt' in item:
                print(f"    匹配的GT: idx={item['closest_gt']['idx']}, text='{item['closest_gt']['text']}'")
            elif 'groundtruth_in_sentence' in item:
                print(f"    句子中的GT: {[gt['text'] for gt in item['groundtruth_in_sentence']]}")
            print(f"    问题: {item['issue']}")
    
    print("\n" + "=" * 80)

def calculate_time_based_metrics(gt_dict: dict, model_dict: dict) -> dict:
    """
    基于时间长度的指标计算
    
    对于每个句子(idx)，计算：
    - TP: 模型和GT重叠的时间长度
    - FN: GT有但模型没有的时间长度
    - FP: 模型有但GT没有的时间长度
    
    Args:
        gt_dict: groundtruth字典，key为(idx, start, end)，value为条目
        model_dict: 模型输出字典，key为(idx, start, end)，value为条目
    """
    # 按idx分组
    gt_by_idx = {}
    for key, item in gt_dict.items():
        idx = item['idx']
        if idx not in gt_by_idx:
            gt_by_idx[idx] = []
        gt_by_idx[idx].append((item['start'], item['end']))
    
    model_by_idx = {}
    for key, item in model_dict.items():
        if 'idx_g' in item:
            continue
        idx = item['idx']
        if idx not in model_by_idx:
            model_by_idx[idx] = []
        model_by_idx[idx].append((item['start'], item['end']))
    
    total_tp = 0
    total_fn = 0
    total_fp = 0
    
    all_idxs = set(gt_by_idx.keys()) | set(model_by_idx.keys())
    
    for idx in all_idxs:
        gt_intervals = gt_by_idx.get(idx, [])
        model_intervals = model_by_idx.get(idx, [])
        
        # 合并重叠的区间
        gt_merged = merge_intervals(gt_intervals)
        model_merged = merge_intervals(model_intervals)
        
        # 计算TP: 重叠部分
        tp = calculate_total_overlap(gt_merged, model_merged)
        
        # 计算FN: GT有但模型没有的部分
        fn = calculate_total_unique(gt_merged, model_merged, is_gt=True)
        
        # 计算FP: 模型有但GT没有的部分
        fp = calculate_total_unique(gt_merged, model_merged, is_gt=False)
        
        total_tp += tp
        total_fn += fn
        total_fp += fp
    
    # 计算指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp_duration': total_tp,
        'fn_duration': total_fn,
        'fp_duration': total_fp,
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
    }


def merge_intervals(intervals: list) -> list:
    """合并重叠的区间"""
    if not intervals:
        return []
    
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged


def calculate_total_overlap(gt_intervals: list, model_intervals: list) -> int:
    """计算GT和模型重叠的总时长"""
    total = 0
    for gt_start, gt_end in gt_intervals:
        for model_start, model_end in model_intervals:
            overlap_start = max(gt_start, model_start)
            overlap_end = min(gt_end, model_end)
            total += max(0, overlap_end - overlap_start)
    return total


def calculate_total_unique(gt_intervals: list, model_intervals: list, is_gt: bool) -> int:
    """
    计算独有的时长
    is_gt=True: 计算GT有但模型没有的(FN)
    is_gt=False: 计算模型有但GT没有的(FP)
    """
    if is_gt:
        # GT独有的部分
        total = 0
        for gt_start, gt_end in gt_intervals:
            covered = 0
            for model_start, model_end in model_intervals:
                overlap_start = max(gt_start, model_start)
                overlap_end = min(gt_end, model_end)
                covered += max(0, overlap_end - overlap_start)
            total += (gt_end - gt_start) - covered
        return total
    else:
        # 模型独有的部分
        total = 0
        for model_start, model_end in model_intervals:
            covered = 0
            for gt_start, gt_end in gt_intervals:
                overlap_start = max(model_start, gt_start)
                overlap_end = min(model_end, gt_end)
                covered += max(0, overlap_end - overlap_start)
            total += (model_end - model_start) - covered
        return total

if __name__ == "__main__":
    # # 数据文件路径
    groudtruth_dir = "/path/to/data_backup/crisis/mask_accuracy/mask_gt"
    mask_dir = f"/path/to/others/baseline/mask/crisis/Secure_ner"
    accuracy_output_dir = f"/path/to/others/baseline/mask/crisis/"

    logger, _ = setup_logger(accuracy_output_dir, "accuracy", "DEBUG")
    df, summary_str = compute_directory_accuracy(groudtruth_dir, mask_dir, accuracy_output_dir, logger)
    print(f"\准确率评估汇总:{summary_str}\n详细结果已保存到: {accuracy_output_dir}目录下")

    # models = [
    #     "gpt-5-nano-2025-08-07"
    #          ]
    # for model in models:
    #     groudtruth_dir = "/path/to/data/crisis/mask_accuracy/biyuan/mask_clear_gt"
    #     mask_dir = f"/path/to/data/crisis/mask_accuracy/biyuan/{model}/mask"
    #     accuracy_output_dir = f"/path/to/data/crisis/mask_accuracy/biyuan/{model}"

    #     logger, _ = setup_logger(accuracy_output_dir, "accuracy", "DEBUG")
    #     df, summary_str = compute_directory_accuracy(groudtruth_dir, mask_dir, accuracy_output_dir, logger)
    #     print(f"\n模型: {model} 的准确率评估汇总:{summary_str}\n详细结果已保存到: {accuracy_output_dir}目录下")

