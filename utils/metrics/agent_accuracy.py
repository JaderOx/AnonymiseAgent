"""
批量跑 reflexion agent，收集日志并自动回填 choice/state/log。
"""

import argparse, json, os, re, sys, subprocess, shutil
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
PATHS = {
    "Androids-I": {
        "input_dir":  "/path/to/data_backup/Androids-Corpus/audio",
        "output_dir": "/path/to/data/agent_accuracy/Androids-I/output",
        "middle_dir": "/path/to/data/agent_accuracy/Androids-I/middle"
    },
    "Androids-R": {
        "input_dir":  "/path/to/data_backup/Androids-Corpus/audio",
        "output_dir": "/path/to/data/agent_accuracy/Androids-R/output",
        "middle_dir": "/path/to/data/agent_accuracy/Androids-R/middle"
    },
    "ENNI":                 {"input_dir":"/path/to/data_backup/ENNI/audio","output_dir":"/path/to/data/agent_accuracy/ENNI/output","middle_dir":"/path/to/data/agent_accuracy/ENNI/middle"},
    "NCMMSC-AD":            {"input_dir":"/path/to/data_backup/NCMMSC_AD/input","output_dir":"/path/to/data/agent_accuracy/NCMMSC-AD/output","middle_dir":"/path/to/data/agent_accuracy/NCMMSC-AD/middle"},
    "Saar-HD":              {"input_dir":"/path/to/data_backup/SVD/Hyperfunktionelle_Dysphonie_binary_intersection_v2/audio/sentence","output_dir":"/path/to/data/agent_accuracy/Saar-HD/output","middle_dir":"/path/to/data/agent_accuracy/Saar-HD/middle"},
    "Saar-Laryngitis":      {"input_dir":"/path/to/data_backup/SVD/Laryngitis_binary_intersection/audio/sentences","output_dir":"/path/to/data/agent_accuracy/Saar-Laryngitis/output","middle_dir":"/path/to/data/agent_accuracy/Saar-Laryngitis/middle"},
    "Saar-Rekurrensparese": {"input_dir":"/path/to/data_backup/SVD/Rekurrensparese_binary_intersection/audio/sentences","output_dir":"/path/to/data/agent_accuracy/Saar-Rekurrensparese/output","middle_dir":"/path/to/data/agent_accuracy/Saar-Rekurrensparese/middle"},
    "NeuroVoz":             {"input_dir":"/path/to/data_backup/NeuroVoz/audios","output_dir":"/path/to/data/agent_accuracy/NeuroVoz/output","middle_dir":"/path/to/data/agent_accuracy/NeuroVoz/middle"},
}

SOURCE_JSON = "/path/to/data/agent_accuracy/reflexion_data_wo_bacc.json"
AGENT_DIR = "/path/to/Anonymise_Agent"


# ═══════════════════════════════════════════════════════════════
# 方法名标准化
# ═══════════════════════════════════════════════════════════════
METHOD_MAP = {
    "mcadams": "McAdams", "formant": "Formant", "pitch": "Pitch",
    "combined": "Combined", "seedvc": "SeedVC",
    "fishaudio_tts": "ASR-TTS"
}

def normalize_method(name: str) -> str:
    return METHOD_MAP.get(name.lower().strip(), name.strip())


# ═══════════════════════════════════════════════════════════════
# 从日志提取最终选择
# ═══════════════════════════════════════════════════════════════
def extract_choice_from_log(log_path: str) -> str | None:
    """
    从 agent 日志提取最终选择的 VC 方法。
    三种模式（按优先级）：
      1. [最终选择] xxx
      2. [反思] 回退到 xxx
      3. [反思结果] {"done": true, "summary": "xxx ..."}
    """
    if not os.path.exists(log_path):
        return None

    lines = open(log_path, "r", encoding="utf-8").readlines()

    last_fallback = None
    last_done_summary = None

    for line in lines:
        # 模式 1: [最终选择] — 最终结论，直接返回
        if "[最终选择]" in line:
            m = re.search(r'\[最终选择\]\s*(.+)', line)
            if m:
                return m.group(1).strip()

        # 模式 2: [反思] 回退到 xxx
        if "回退到" in line:
            m = re.search(r'回退到\s*([^，,\s]+)', line)
            if m:
                last_fallback = m.group(1).strip().rstrip("，,")

        # 模式 3: done:true → 从 summary 提取方法名
        if "[反思结果]" in line and '"done": true' in line:
            m = re.search(r'"summary"\s*:\s*"([^"]+)"', line)
            if m:
                summary_val = m.group(1).strip()
                # summary 可能是 "seedvc 满足 CER<20%"、"seedvc,EER=50.0%"、
                # "mcadams满足情感相似度>0.93的要求"、"选择seedvc 满足..." 或纯 "combined"
                known = '|'.join(re.escape(m) for m in METHOD_MAP)
                m2 = re.search(rf'({known})', summary_val, re.IGNORECASE)
                last_done_summary = m2.group(1) if m2 else summary_val

    # 回退到回退结果
    if last_fallback:
        return last_fallback
    if last_done_summary:
        return last_done_summary

    return None


def check_state(choice_raw: str, target_list: list) -> bool:
    """检查 choice 是否在 target 列表中（大小写不敏感）。"""
    if not choice_raw:
        return False
    choice_lower = choice_raw.lower()
    for t in target_list:
        if t.lower() == choice_lower:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════
def backfill(entries, indices=None):
    """从已有日志回填 choice/state/log。"""
    filled = 0
    for i, entry in enumerate(entries):
        if indices is not None and i not in indices:
            continue
        if entry.get("choice"):
            continue

        ds = entry["dataset"]
        paths = PATHS.get(ds)
        if not paths or not paths["output_dir"]:
            continue

        log_dir = os.path.join(paths["output_dir"], "logs")
        if not os.path.isdir(log_dir):
            continue

        # 找最新日志
        logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".log")], reverse=True)
        if not logs:
            continue

        # 遍历日志，找到匹配 prompt 的
        for log_name in logs:
            log_path = os.path.join(log_dir, log_name)
            content = open(log_path, "r", encoding="utf-8").read()
            # 用 prompt 前 30 字符匹配
            if entry["prompt"][:30] not in content:
                continue

            choice_raw = extract_choice_from_log(log_path)
            if not choice_raw:
                continue

            choice = normalize_method(choice_raw)
            state = check_state(choice, entry.get("target", []))
            entry["choice"] = choice
            entry["state"] = state
            entry["log"] = log_path
            filled += 1
            print(f"[{i}] {ds}: {choice} (state={state}) ← {log_name}")
            break

    return filled


def run_and_fill(entries, llm, open_source, mode, cuda, json_path, indices=None):
    """运行 agent 并回填。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda

    done = skip = fail = 0

    for i, entry in enumerate(entries):
        if indices is not None and i not in indices:
            continue
        if entry.get("choice"):
            skip += 1
            continue

        ds = entry["dataset"]
        paths = PATHS.get(ds)
        if not paths or not paths["input_dir"]:
            print(f"[{i}] {ds} — 路径未配置，跳过")
            skip += 1
            continue

        # 构造命令
        cmd = [
            "python", "reflexion_agent_downstream.py",
            "-i", paths["input_dir"],
            "-o", paths["output_dir"],
            "--middle_dir", paths["middle_dir"],
            "--mode", mode,
            "--llm", llm,
            "--open_source" if open_source else "--no-open_source",
            "-r", entry["prompt"],
        ]

        # 日志路径
        log_dir = os.path.join(paths["output_dir"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(log_dir, f"reflexion_{timestamp}.log")

        print(f"\n[{i}/{len(entries)-1}] {ds}")
        print(f"  prompt: {entry['prompt']}")
        print(f"  log:    {log_path}")

        # 执行，stdout 丢弃，日志由 agent 写文件
        env_run = env.copy()
        env_run["ANONYMISE_LOG_FILE"] = log_path
        with open(os.devnull, "w") as devnull:
            proc = subprocess.run(cmd, stdout=devnull, stderr=subprocess.STDOUT,
                                  env=env_run, cwd=AGENT_DIR)

        if proc.returncode != 0:
            print(f"  ❌ 退出码 {proc.returncode}")
            fail += 1
            continue

        # 从日志提取结果并回填
        choice_raw = extract_choice_from_log(log_path)
        if choice_raw:
            choice = normalize_method(choice_raw)
            state = check_state(choice, entry.get("target", []))
            entry["choice"] = choice
            entry["state"] = state
            entry["log"] = log_path
            print(f"  ✅ choice={choice}, state={state}")
            done += 1
        else:
            print(f"  ⚠️  无法提取结果，请手动检查日志")
            fail += 1

        # 实时保存
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    return done, skip, fail


def main():
    parser = argparse.ArgumentParser(
        description="批量跑 reflexion agent，收集日志并自动回填",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--llm", required=True, help="模型名称，如 Qwen3.5-9B、gpt-4o")
    parser.add_argument("--open_source", type=lambda x: x.lower() == "true", default=True,
                        help="True=本地模型（默认），False=OpenAI 兼容 API")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="DEBUG", help="日志级别")
    parser.add_argument("--cuda", default="0,1,2", help="CUDA_VISIBLE_DEVICES（默认 0,1,2）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--backfill", action="store_true", help="不跑 agent，只从已有日志回填")
    parser.add_argument("--index", nargs=2, type=int, metavar=("START", "END"),
                        help="只跑第 START~END 条（0-indexed）")
    args = parser.parse_args()

    # 每个模型独立 JSON，不存在则从源文件复制
    json_path = os.path.join(os.path.dirname(SOURCE_JSON), f"{args.llm}.json")
    if not os.path.exists(json_path):
        shutil.copy2(SOURCE_JSON, json_path)
        print(f"已复制 {SOURCE_JSON} → {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # 构建索引集合
    indices = None
    if args.index:
        indices = set(range(args.index[0], args.index[1] + 1))

    if args.backfill:
        filled = backfill(entries, indices)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"\n回填 {filled} 条")
        return

    if args.dry_run:
        for i, entry in enumerate(entries):
            if indices is not None and i not in indices:
                continue
            if entry.get("choice"):
                continue
            ds = entry["dataset"]
            paths = PATHS.get(ds)
            if not paths or not paths["input_dir"]:
                print(f"[{i}] {ds} — 路径未配置")
                continue
            print(f"[{i}] {ds}: {entry['prompt'][:60]}")
        return

    done, skip, fail = run_and_fill(entries, args.llm, args.open_source, args.mode, args.cuda, json_path, indices)

    # 最终统计
    total_done = sum(1 for e in entries if e.get("choice"))
    total_correct = sum(1 for e in entries if e.get("state") is True)
    print(f"\n{'='*55}")
    print(f"模型: {args.llm} | source: {'local' if args.open_source else 'api'}")
    print(f"本次: 完成={done} 跳过={skip} 失败={fail}")
    print(f"总计: {total_done}/{len(entries)} 已填 | 正确: {total_correct}/{total_done}" if total_done else "")


if __name__ == "__main__":
    main()
