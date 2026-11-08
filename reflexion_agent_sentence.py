"""Memory-enabled Agent with optional sample-level cascade over existing outputs."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from agent import _has_audio, SUPPORTED_FORMATS
from utils.logger_example import setup_logger
from utils.vc_names import canonical_method
from utils.agent_selection import final_vc_from as _final_vc_from
from reflexion_agent_downstream import ReflexionAgent, VC_METHODS

MIN_SCORES_FOR_FAIL_DETECT = 4   # 少于该数量不做失败判定（分位数无意义）


def _compute_fail_stats(scores, iqr_k: float) -> dict:
    """分数分布统计与失败阈值。纯函数。（照搬 2026-09-21 fail_repair 版本）

    threshold = mean + iqr_k * IQR（IQR = q75 - q25，分位数线性插值）。
    scores 为空返回 {}。
    """
    if not scores:
        return {}
    arr = sorted(float(s) for s in scores)
    n = len(arr)
    mean = sum(arr) / n
    var = sum((x - mean) ** 2 for x in arr) / n

    def _q(p):
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac

    q25, q75 = _q(0.25), _q(0.75)
    iqr = q75 - q25
    return {"n": n, "mean": mean, "std": var ** 0.5, "min": arr[0], "max": arr[-1],
            "q25": q25, "q75": q75, "iqr": iqr,
            "threshold": mean + iqr_k * iqr}


def _is_fail_score(score: float, threshold: float, floor) -> bool:
    """fail = score > threshold 且（floor 为 None 或 score > floor）。

    floor 保证"分布整体很好时不硬挑"：只超 IQR 阈值但绝对相似度不高不算失败。
    """
    return score > threshold and (floor is None or score > floor)


def _keyed_audio_files(directory) -> dict:
    """{rel_stem: Path}。rel_stem = 相对 directory 去后缀（与 evaluate CSV name 列同约定）。"""
    d = Path(directory)
    if not d.exists():
        return {}
    keyed = {}
    for p in sorted(d.rglob('*')):
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS:
            keyed[str(p.relative_to(d).with_suffix(''))] = p
    return keyed


def _cascade_assign(round_scores: list, methods: list, threshold, floor,
                    detection_skipped: bool) -> dict:
    """级联分配（纯函数）：决定每个文件最终用哪个方法的输出。

    Args:
        round_scores: 与 methods 对齐，每项 {key: score}，是该轮实际打上分的文件
                      （第 0 轮为全部可配对文件；后续轮只含上一轮失败/缺输出的文件）。
        threshold / floor: 判定参数（detection_skipped=True 时忽略）。
        detection_skipped: 样本太少不做失败判定，全部留在首个方法。

    规则:
        首个通过（不满足 fail 条件）的方法 → status="pass"；
        所有轮都失败 → 取 tried 中 score 最低的方法 → status="best"；
        从未打上分（某方法目录缺该文件且后续也没配上）→ 不在返回的 assignment 里。
    """
    tried = {}      # key -> {method: score}
    assigned = {}   # key -> {"method","score","status","tried"}
    for method, scores in zip(methods, round_scores):
        for key, s in scores.items():
            tried.setdefault(key, {})[method] = s
            if key in assigned:
                continue  # 已通过锁定，正常不会再次出现
            if detection_skipped or not _is_fail_score(s, threshold, floor):
                assigned[key] = {"method": method, "score": s, "status": "pass"}
    for key, tr in tried.items():
        if key in assigned:
            assigned[key]["tried"] = tr
            continue
        best_m = min(tr, key=tr.get)
        assigned[key] = {"method": best_m, "score": tr[best_m],
                         "status": "best", "tried": tr}
    return {"assignment": assigned}


def _score_against_method(orig_embs: dict, anon_files: dict, keys: list,
                          embed_fn, feature_extractor, model, logger) -> dict:
    """对 keys 逐条计算 原始音频 vs output/<method>/ 的说话人嵌入余弦相似度。

    任一侧缺失（原始嵌入提取失败 / 方法目录无该文件）则不打分，
    该文件交由级联逻辑继续升级到下一个方法。
    """
    from sklearn.metrics.pairwise import cosine_similarity
    scores = {}
    for key in keys:
        o = orig_embs.get(key)
        a_path = anon_files.get(key)
        if o is None or a_path is None:
            continue
        a = embed_fn(a_path, feature_extractor, model, logger)
        if a is None:
            logger.warning(f"[Reflex] 匿名侧嵌入提取失败: {a_path}")
            continue
        scores[key] = float(cosine_similarity([o], [a])[0][0])
    return scores


def _rank_methods_for_cascade(output_dir, logger):
    """自动级联排序（Stage 1→2 衔接）：A = agent_result.json 的 final_vc_method，
    其余方法按下游 bacc 降序（数据源 output/<m>/downstream_output/final_result.json；
    无缓存的方法无下游依据，垫后并 warning）。返回 (A, 完整有序列表)。
    """
    result_path = os.path.join(output_dir, 'agent_result.json')
    if not os.path.exists(result_path):
        raise RuntimeError(f"未找到 {result_path}（先跑 Stage-1 agent，或改用 --reflex_methods 手动指定）")
    with open(result_path, 'r', encoding='utf-8') as f:
        a = json.load(f).get('final_vc_method')
    if a not in VC_METHODS:
        raise RuntimeError(f"agent_result.json 的 final_vc_method={a!r} 不在可用方法 {VC_METHODS} 中")

    bacc = {}
    for m in VC_METHODS:
        if m == a:
            continue
        p = os.path.join(output_dir, m, 'downstream_output', 'final_result.json')
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    v = json.load(f).get('balanced_accuracy', {}).get('mean')
                if v is not None:
                    bacc[m] = float(v)
                else:
                    logger.warning(f"[Reflex] 下游结果缺 balanced_accuracy.mean: {p}，排最后")
            except Exception as e:
                logger.warning(f"[Reflex] 读取下游结果失败 {p}: {e}，排最后")
        else:
            logger.warning(f"[Reflex] 无下游缓存，排最后: {m}")
    ranked = (sorted(bacc, key=lambda m: -bacc[m])
              + [m for m in VC_METHODS if m != a and m not in bacc])
    logger.info(f"[Reflex] auto 排序: A={a}，级联={[a] + ranked}，"
                f"bacc={ {k: round(v, 4) for k, v in bacc.items()} }")
    return a, [a] + ranked


def run_reflex_cascade(input_dir, output_dir, methods, logger,
                       iqr_k=2.0, score_floor=0.5, threshold_override=None,
                       dir_name=None):
    """句级 reflex 级联主流程：打分 → 分配 → cp 合并。不跑 VC、不调 LLM。

    Args:
        input_dir: 原始音频目录。
        output_dir: 匿名输出根目录，需已有 output_dir/<method>/ 各方法结果。
        methods: 级联顺序的方法列表（首元素 = 方法A）。
        logger: setup_logger 产出的 logger（嵌入提取内部也用它）。
        iqr_k / score_floor: IQR 口径参数（threshold_override 为 None 时生效）。
        threshold_override: 固定阈值 θ。给定时 fail = score > θ（floor 不参与，
            n<4 仍可判定——固定阈值不依赖分数分布），阈值不再随方法A分布自适应。
        dir_name: 产物目录名。None 时默认 'reflex_<A>_<B>_…'；给定则直接用作
            目录名（如 '0.90_formant_spec_NeuralVC'，便于区分不同 θ 的同方法链）。
    """
    from utils.metrics.eer_calc import load_model, extract_speaker_embedding

    threshold_mode = 'fixed' if threshold_override is not None else 'iqr'
    if threshold_override is not None:
        threshold_override = float(threshold_override)
        score_floor = None

    orig_files = _keyed_audio_files(input_dir)
    all_keys = sorted(orig_files)
    if not all_keys:
        raise RuntimeError(f"输入目录无音频: {input_dir}")

    # 各方法输出目录检查：缺失/无音频的方法从级联中剔除
    method_dirs = {}
    used_methods = []
    for m in methods:
        d = os.path.join(output_dir, m)
        if _has_audio(d):
            method_dirs[m] = d
            used_methods.append(m)
        else:
            logger.warning(f"[Reflex] 方法输出目录不存在或无音频，剔除: {d}")
    if not method_dirs:
        raise RuntimeError(f"没有任何方法的匿名输出可用 (output_dir={output_dir})")

    logger.info(f"[Reflex] input={input_dir} ({len(all_keys)} 条) | "
                f"级联={used_methods} | iqr_k={iqr_k} floor={score_floor}")
    print(f"输入: {input_dir} ({len(all_keys)} 条)")
    print(f"级联方法: {' → '.join(used_methods)}")

    feature_extractor, model = load_model()
    logger.info("[Reflex] 嵌入模型加载完成")

    # 原始侧嵌入只提一次，级联各轮复用
    orig_embs = {}
    for key in all_keys:
        emb = extract_speaker_embedding(orig_files[key], feature_extractor, model, logger)
        if emb is not None:
            orig_embs[key] = emb
    logger.info(f"[Reflex] 原始侧嵌入完成: {len(orig_embs)}/{len(all_keys)}")

    # 级联打分：第 0 轮全量定阈值，后续轮只打上一轮失败/缺输出的文件
    round_scores = []
    stats = {}
    threshold = None
    detection_skipped = False
    remaining = list(all_keys)
    for i, m in enumerate(used_methods):
        anon_files = _keyed_audio_files(method_dirs[m])
        scores = _score_against_method(
            orig_embs, anon_files, remaining,
            extract_speaker_embedding, feature_extractor, model, logger)
        round_scores.append(scores)

        if i == 0:
            stats = _compute_fail_stats(list(scores.values()), iqr_k)
            if threshold_override is not None:
                threshold = threshold_override
                if not stats:
                    detection_skipped = True
                    logger.warning(f"[Reflex] 无任何分数，跳过失败判定，全部使用 {m}")
                    print(f"[Reflex] 无任何分数，跳过失败判定，全部使用 {m}")
                    break
                logger.info(f"[Reflex] 阈值={threshold:.4f} (固定) | "
                            f"A分布 mean={stats['mean']:.4f}")
                print(f"阈值: {threshold:.4f} (固定; 方法A分布 mean={stats['mean']:.4f})")
            else:
                if not stats or stats["n"] < MIN_SCORES_FOR_FAIL_DETECT:
                    detection_skipped = True
                    reason = "no_scores" if not stats else "n_too_small"
                    logger.warning(f"[Reflex] 跳过失败判定 ({reason}, n={stats.get('n', 0)})，"
                                   f"全部使用 {m}")
                    print(f"[Reflex] 样本太少，跳过失败判定 ({reason})，全部使用 {m}")
                    break
                threshold = stats["threshold"]
                logger.info(f"[Reflex] 阈值={threshold:.4f} "
                            f"(mean={stats['mean']:.4f}, IQR={stats['iqr']:.4f}, "
                            f"k={iqr_k}, floor={score_floor})")
                print(f"阈值: {threshold:.4f} (mean={stats['mean']:.4f} + {iqr_k}*{stats['iqr']:.4f})")

        n_fail = sum(1 for s in scores.values() if _is_fail_score(s, threshold, score_floor))
        n_missing = len(remaining) - len(scores)
        remaining = [k for k in remaining
                     if k not in scores or _is_fail_score(scores[k], threshold, score_floor)]
        logger.info(f"[Reflex] 轮{i+1} {m}: 打分 {len(scores)}，失败 {n_fail}，"
                    f"缺输出 {n_missing}，待升级 {len(remaining)}")
        print(f"  [{i+1}/{len(used_methods)}] {m}: 打分 {len(scores)}, "
              f"失败 {n_fail}, 缺输出 {n_missing}, 待换方法重试 {len(remaining)}")
        for k in scores:
            if _is_fail_score(scores[k], threshold, score_floor):
                logger.warning(f"[Reflex] 匿名失败: {k} ({m}, score={scores[k]:.4f})")
        if not remaining:
            break

    invoked_methods = used_methods[:len(round_scores)]
    assigned = _cascade_assign(round_scores, invoked_methods, threshold,
                               score_floor, detection_skipped)["assignment"]
    unassigned = [k for k in all_keys if k not in assigned]

    # cp 合并（目录名可自定义，默认 reflex_<A>_<B>_…）
    reflex_dir = os.path.join(output_dir, dir_name or ('reflex_' + '_'.join(invoked_methods)))
    os.makedirs(reflex_dir, exist_ok=True)
    anon_files_cache = {m: _keyed_audio_files(method_dirs[m]) for m in invoked_methods}
    per_method = {}
    n_best = 0
    for key, info in sorted(assigned.items()):
        src = anon_files_cache[info["method"]].get(key)
        if src is None:
            unassigned.append(key)
            logger.warning(f"[Reflex] 分配到 {info['method']} 但文件缺失: {key}")
            continue
        dst = os.path.join(reflex_dir, key + src.suffix)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        per_method[info["method"]] = per_method.get(info["method"], 0) + 1
        if info["status"] == "best":
            n_best += 1

    report = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "methods_requested": methods,
        "methods_used": invoked_methods,
        "iqr_k": iqr_k,
        "score_floor": score_floor,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "stats": stats,
        "detection_skipped": detection_skipped,
        "n_total": len(all_keys),
        "n_copied": sum(per_method.values()),
        "n_unassigned": len(unassigned),
        "unassigned": unassigned,
        "per_method_counts": per_method,
        "n_leftover_best": n_best,
        "reflex_dir": reflex_dir,
        "per_file": {k: assigned[k] for k in sorted(assigned)},
    }
    report_path = os.path.join(reflex_dir, "reflex_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[Reflex] 完成: {reflex_dir} | 清单: {report_path}")

    print(f"\nreflex 目录: {reflex_dir}")
    print(f"清单: {report_path}")
    for m in invoked_methods:
        print(f"  {m}: {per_method.get(m, 0)} 条")
    if n_best:
        print(f"  其中 {n_best} 条所有方法均超阈值，取相似度最低的方法")
    if unassigned:
        print(f"[WARNING] {len(unassigned)} 条在任何方法输出中都缺失，未复制: {unassigned[:5]}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Reflexion Agent — 基于评估反馈自动调整 VC 方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python reflexion_agent.py -i /path/to/audio -r "匿名化中文音频，SNR尽量高，不需要内容匿名"
  python reflexion_agent.py -i /path/to/audio -r "匿名化，不要改变韵律" --max_reflections 5
  python reflexion_agent.py -i /path/to/audio -r "..." --llm Qwen2.5-32B-Instruct
        """)
    parser.add_argument('--input_dir', '-i', required=True, help='输入音频目录')
    parser.add_argument('--request', '-r', default=None, help='自然语言任务描述（LLM agent 模式必填）')
    parser.add_argument('--llm', default=None, help='模型名称（默认自动推断）')
    parser.add_argument('--open_source', action=argparse.BooleanOptionalAction, default=True,
                        help='使用本地模型（默认）。--no-open_source 使用 OpenAI 兼容 API（读取环境变量 OPENAI_API_KEY / BASE_URL）')
    parser.add_argument('--output_dir', '-o', default=None, help='输出目录')
    parser.add_argument('--middle_dir', default=None, help='中间结果目录')
    parser.add_argument('--mode', choices=['DEBUG', 'INFO'], default='INFO', help='日志级别')
    parser.add_argument('--max_reflections', type=int, default=7, help='最大反思次数（默认7）')
    parser.add_argument('--temperature', type=float, default=0.01, help='LLM 温度（默认0.01，重试时自动降为0）')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='LLM 最大生成 token 数（默认2048）')
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help='LLM 重复惩罚（默认1.05）')
    parser.add_argument('--downstream_data_json', type=str, default=None,
                        help='下游任务原始 data.json 路径（启用下游反思路径）')
    parser.add_argument('--downstream_script', type=str, default=None,
                        help='下游任务 bash 脚本路径（与 --downstream_data_json 配对使用）')
    parser.add_argument('--memory_exclude_dataset', type=str, default=None,
                        help='recall 时排除该数据集的经验/运行记录（leave-one-out 消融实验用）；'
                             '注意 record_run 仍会正常写回（写到 --memory_path 指定的目录，'
                             '实验时建议指向副本目录以免污染主 memory）')
    parser.add_argument('--memory_path', type=str, default=None,
                        help='跨运行双层 memory 根目录（默认 ./memory；含 experiences.json 与 runs/）')
    parser.add_argument('--no_memory', action='store_true',
                        help='禁用跨运行 memory（行为与无 memory 完全一致）')
    # ── 句级 reflex 级联模式（无 LLM、不跑 VC）──
    parser.add_argument('--reflex_methods', type=str, default=None,
                        help='句级 reflex 级联模式：逗号分隔方法列表，顺序=级联顺序，'
                             '如 formant_lpc,NeuralVC,Resyn。要求各方法匿名输出已存在于 '
                             'output_dir/<method>/。给出此参数时不走 LLM agent 流程')
    parser.add_argument('--reflex_auto', action='store_true',
                        help='自动级联：方法A 读 output_dir/agent_result.json 的 '
                             'final_vc_method，其余方法按下游 bacc 降序排列。'
                             '与 --reflex_methods 互斥')
    parser.add_argument('--fail_threshold', type=float, default=0.75,
                        help='固定失败阈值：匿名前后说话人相似度 > 该值判失败（默认 0.75）。'
                             '--no_fail_threshold 禁用后回落 IQR 口径')
    parser.add_argument('--reflex_dir_name', type=str, default=None,
                        help='产物目录名（默认 reflex_<A>_<B>_…）。区分同方法链不同 θ '
                             '时用，如 0.90_formant_spec_NeuralVC')
    parser.add_argument('--no_fail_threshold', action='store_true',
                        help='禁用固定阈值，回落 IQR 口径（阈值 = mean+k·IQR 且 > floor）')
    parser.add_argument('--fail_iqr_k', type=float, default=2.0,
                        help='IQR 口径：阈值 = mean + k*IQR 的 k（默认 2.0，阈值取自方法A分数分布）')
    parser.add_argument('--fail_score_floor', type=float, default=0.5,
                        help='IQR 口径：绝对相似度下限，fail 需同时高于它（默认 0.5；'
                             '--no_fail_score_floor 禁用）')
    parser.add_argument('--no_fail_score_floor', action='store_true',
                        help='IQR 口径：禁用绝对相似度下限（floor=None）')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # ── 句级 reflex 级联模式：打分 → 清单 → cp 合并，不创建 ReflexionAgent ──
    if args.reflex_methods or args.reflex_auto:
        if args.reflex_methods and args.reflex_auto:
            print("错误: --reflex_methods 与 --reflex_auto 互斥")
            sys.exit(1)
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.input_dir)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        floor = None if args.no_fail_score_floor else args.fail_score_floor
        th_override = None if args.no_fail_threshold else args.fail_threshold
        logger, logger_file = setup_logger(output_dir, 'reflex_cascade', args.mode)
        print(f"日志: {logger_file}")
        logger.info(f"[Reflex] 级联模式: input={args.input_dir} output={output_dir} "
                    f"reflex_methods={args.reflex_methods} reflex_auto={args.reflex_auto} "
                    f"threshold={th_override} iqr_k={args.fail_iqr_k} floor={floor}")
        if args.reflex_auto:
            a_method, methods = _rank_methods_for_cascade(output_dir, logger)
            print(f"Agent 选定方法A: {a_method}")
            print(f"级联方法: {' → '.join(methods)}")
        else:
            methods = [m.strip() for m in args.reflex_methods.split(',') if m.strip()]
            unknown = [m for m in methods if m not in VC_METHODS]
            if unknown:
                print(f"错误: 未知 VC 方法: {unknown}（可选: {VC_METHODS}）")
                sys.exit(1)
            if len(set(methods)) != len(methods):
                print("错误: --reflex_methods 含重复方法")
                sys.exit(1)
        run_reflex_cascade(args.input_dir, output_dir, methods, logger,
                           iqr_k=args.fail_iqr_k, score_floor=floor,
                           threshold_override=th_override, dir_name=args.reflex_dir_name)
        sys.exit(0)

    if not args.request:
        parser.error("LLM agent 模式需要 -r/--request（或改用 --reflex_methods 走级联模式）")

    if (args.downstream_data_json or args.downstream_script) and not (
        args.downstream_data_json and args.downstream_script
    ):
        print("错误: --downstream_data_json 和 --downstream_script 必须同时提供")
        sys.exit(1)

    if args.downstream_data_json and not os.path.exists(args.downstream_data_json):
        print(f"错误: data.json 不存在: {args.downstream_data_json}")
        sys.exit(1)

    if args.downstream_script and not os.path.exists(args.downstream_script):
        print(f"错误: 脚本不存在: {args.downstream_script}")
        sys.exit(1)

    agent = ReflexionAgent(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        middle_dir=args.middle_dir,
        llm_model=args.llm,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        open_source=args.open_source,
        memory_path=args.memory_path,
        use_memory=not args.no_memory,
        memory_exclude_dataset=args.memory_exclude_dataset,
    )
    agent.downstream_data_json = args.downstream_data_json
    agent.downstream_script = args.downstream_script

    if agent.downstream_data_json:
        print(f"下游 data.json: {agent.downstream_data_json}")
        print(f"下游脚本: {agent.downstream_script}")
    result = agent.run(args.request, max_reflections=args.max_reflections)

    if result and result.get("success"):
        history = result.get("history", [])
        final_vc = result.get("final_vc_method")
        if history:
            print(f"\n最终选择: {final_vc}")
        else:
            print("\n执行完成（无需反思）。")
    else:
        print("\n执行失败，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
