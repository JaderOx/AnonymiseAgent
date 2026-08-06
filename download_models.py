#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py — 一键下载 Anonymise Agent 所需模型到 .env 指定的路径。

模型来源：
  - funasr(ModelScope): ASR(paraformer/FunASR*)——用 AutoModel 预缓存，路径与运行时一致
  - ModelScope        : LLM(Qwen2.5/gemma)、emotion2vec、CosyVoice
  - HuggingFace       : 年龄性别(w2v2)、wavlm、whisperx、pyannote 说话人分离、Seed-VC
  - torch.hub         : MOS(UTMOS)

用法（在项目根目录运行）：
  python download_models.py                       # 下载默认集合（中文场景最小可跑集）
  python download_models.py --targets asr llm eval
  python download_models.py --targets all
  python download_models.py --list                # 只列出会下载哪些模型，不下载
  python download_models.py --no-llm              # 默认集合里不含 32B 大模型（省空间/省时）
  python download_models.py --gemma               # LLM 额外下 gemma-3-27b-it

可选类别（--targets）：
  asr        paraformer(+vad+punc+campplus) + FunASRNano(+vad) + FunASRMLTNano(+vad)
  llm        Qwen2.5-32B-Instruct（默认）；加 --gemma 额外下 gemma-3-27b-it
  age-gender w2v2-L-robust-6-age-gender（fishaudio/seedvc 目标说话人匹配需要）
  eval       MOS(UTMOS) + emotion2vec + wavlm-base-plus-sv
  whisperx   faster-whisper large-v3 + pyannote 说话人分离（英文/多语言 ASR）
  vc         Seed-VC + CosyVoice（fish-speech 因版本/来源特殊，仅打印获取说明）
  all        以上全部

注意：
  - 路径全部取自 .env（LLM_MODEL_DIR / TORCH_HOME / MODELSCOPE_CACHE），与代码运行时读取一致。
  - ASR 由 funasr AutoModel 预缓存到 TORCH_HOME/<org>/<model>（funasr 的实际缓存位置）。
  - pyannote 说话人分离是 gated 模型，需要 HF_TOKEN 且在 HF 上同意协议（见 .env.example）。
  - fish-speech(s2-pro) 请按脚本提示手动放置到 TORCH_HOME/fish_speech_models/s2-pro 与 TORCH_HOME/fish-speech。
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────
# 1. 读取 .env，设置环境变量（与 agent.py / reflexion_agent 行为一致）
# ──────────────────────────────────────────────────────────────
def _resolve(p: str) -> str:
    """相对路径相对于项目根目录解析为绝对路径。"""
    if not p:
        return p
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (ROOT / pp)
    return str(pp)


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass  # 没有 python-dotenv 或 .env 都没关系，用默认值

    os.environ["MODELSCOPE_CACHE"] = _resolve(os.environ.get("MODELSCOPE_CACHE", "."))
    os.environ["TORCH_HOME"]       = _resolve(os.environ.get("TORCH_HOME", "./models"))
    os.environ["LLM_MODEL_DIR"]    = _resolve(os.environ.get("LLM_MODEL_DIR", "./LLMs"))
    if os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.environ["HF_ENDPOINT"]
    # HF 缓存默认放到 TORCH_HOME/hub，与 MOS 的 torch.hub 一致
    os.environ.setdefault("HF_HOME", _resolve(os.environ.get("TORCH_HOME", "./models") + "/hub"))

    return {
        "MODELSCOPE_CACHE": os.environ["MODELSCOPE_CACHE"],
        "TORCH_HOME":       os.environ["TORCH_HOME"],
        "LLM_MODEL_DIR":    os.environ["LLM_MODEL_DIR"],
        "HF_TOKEN":         os.environ.get("HF_TOKEN"),
    }


# ──────────────────────────────────────────────────────────────
# 2. 下载原语
# ──────────────────────────────────────────────────────────────
def _ms():
    try:
        from modelscope import snapshot_download
    except Exception:
        sys.exit("[ERROR] 未安装 modelscope，请先 pip install modelscope")
    return snapshot_download


def _hf():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        sys.exit("[ERROR] 未安装 huggingface_hub，请先 pip install huggingface_hub")
    return snapshot_download


def ms_dl(model_id, cache_dir=None, local_dir=None):
    kw = {}
    if cache_dir:
        kw["cache_dir"] = cache_dir
    if local_dir:
        kw["local_dir"] = local_dir
    return _ms()(model_id, **kw)


def hf_dl(repo_id, local_dir=None, token=None):
    kw = {}
    if local_dir:
        kw["local_dir"] = local_dir
    if token:
        kw["token"] = token
    return _hf()(repo_id, **kw)


def funasr_precache(model_id, **extra):
    """用 funasr AutoModel 触发下载——与运行时完全一致的缓存路径（funasr 缓存到 TORCH_HOME/<org>/<model>）。"""
    import gc
    from funasr import AutoModel
    m = AutoModel(model=model_id, disable_update=True, ngpu=0, **extra)
    del m
    gc.collect()


# ──────────────────────────────────────────────────────────────
# 3. 模型清单（category -> list of (名字, 调用)）
#    snapshot_download / AutoModel 自带“已存在则跳过”
# ──────────────────────────────────────────────────────────────
def build_targets(env, include_gemma=False):
    MS = env["MODELSCOPE_CACHE"]
    TH = env["TORCH_HOME"]
    LL = env["LLM_MODEL_DIR"]
    TOK = env["HF_TOKEN"]

    T = {}

    # ── ASR（funasr AutoModel 预缓存 → TORCH_HOME/<org>/<model>，与运行时一致）──
    _VAD = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    T["asr"] = [
        ("paraformer+vad+punc+campplus(中文)", lambda: funasr_precache(
            "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model=_VAD,
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="iic/speech_campplus_sv-zh-cn_16k-common")),
        ("FunASRNano+vad(方言/日文)", lambda: funasr_precache(
            "FunAudioLLM/Fun-ASR-Nano-2512", vad_model=_VAD)),
        ("FunASRMLTNano+vad(欧洲)", lambda: funasr_precache(
            "FunAudioLLM/Fun-ASR-MLT-Nano-2512", vad_model=_VAD)),
    ]

    # ── LLM ──
    llm_jobs = [
        ("Qwen2.5-32B-Instruct(中/英/日)",
         lambda: ms_dl("qwen/Qwen2.5-32B-Instruct", local_dir=os.path.join(LL, "Qwen2.5-32B-Instruct"))),
    ]
    if include_gemma:
        llm_jobs.append(
            ("gemma-3-27b-it(默认/其他语言)",
             lambda: ms_dl("LLM-Research/gemma-3-27b-it", local_dir=os.path.join(LL, "gemma-3-27b-it")))
        )
    T["llm"] = llm_jobs

    # ── 年龄/性别 ──
    T["age-gender"] = [
        ("w2v2-L-robust-6-age-gender",
         lambda: hf_dl("audeering/w2v2-L-robust-6-age-gender",
                       local_dir=os.path.join(TH, "w2v2-L-robust-6-age-gender"), token=TOK)),
    ]

    # ── 评估指标 ──
    def _mos():
        import torch
        os.environ["TORCH_HOME"] = TH
        return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    T["eval"] = [
        ("MOS(UTMOS, torch.hub)", _mos),
        ("emotion2vec",
         lambda: ms_dl("iic/emotion2vec_plus_seed",
                       local_dir=os.path.join(MS, "models", "iic", "emotion2vec_plus_seed"))),
        ("wavlm-base-plus-sv(EER)",
         lambda: hf_dl("microsoft/wavlm-base-plus-sv",
                       local_dir=os.path.join(TH, "wavlm-base-plus-sv"), token=TOK)),
    ]

    # ── whisperx（英文/多语言 ASR + 说话人分离）──
    T["whisperx"] = [
        ("faster-whisper large-v3",
         lambda: hf_dl("Systran/faster-whisper-large-v3", local_dir=os.path.join(TH, "whisperx"), token=TOK)),
        ("pyannote speaker-diarization-3.1 (gated, 需 HF_TOKEN)",
         lambda: hf_dl("pyannote/speaker-diarization-3.1", token=TOK)),
    ]

    # ── 声线转换 ──
    def _fishaudio_note():
        print("    [manual] fish-speech 需手动获取并放置：")
        print(f"      - 模型权重 → {os.path.join(TH, 'fish_speech_models', 's2-pro')}")
        print(f"      - 推理代码 → {os.path.join(TH, 'fish-speech')}")
        print("      参考: https://github.com/fishaudio/fish-speech （s2-pro 版本）")
        return "manual"
    T["vc"] = [
        ("Seed-VC(默认神经VC)", lambda: hf_dl("Plachta/Seed-VC", token=TOK)),
        ("CosyVoice2-0.5B(可选)",
         lambda: ms_dl("FunAudioLLM/CosyVoice2-0.5B", local_dir=os.path.join(TH, "cosyvoice2-0.5B"))),
        ("fish-speech(手动)", _fishaudio_note),
    ]

    return T


# ──────────────────────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="一键下载 Anonymise Agent 所需模型")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="类别: asr llm age-gender eval whisperx vc all（可多选；省略=默认集）")
    ap.add_argument("--gemma", action="store_true", help="LLM 额外下载 gemma-3-27b-it")
    ap.add_argument("--no-llm", action="store_true", help="默认集里不含 32B LLM（省空间/省时）")
    ap.add_argument("--list", action="store_true", help="只列出会下载的模型，不下载")
    args = ap.parse_args()

    env = load_env()
    print("== 路径（来自 .env）==")
    print(f"  MODELSCOPE_CACHE = {env['MODELSCOPE_CACHE']}")
    print(f"  TORCH_HOME       = {env['TORCH_HOME']}")
    print(f"  LLM_MODEL_DIR    = {env['LLM_MODEL_DIR']}")
    print(f"  HF_TOKEN         = {'已设置' if env['HF_TOKEN'] else '未设置（gated 模型如 pyannote 会失败）'}")
    print()

    all_T = build_targets(env, include_gemma=args.gemma)

    # 选目标
    if args.targets:
        wanted = []
        for t in args.targets:
            if t == "all":
                wanted = list(all_T.keys()); break
            if t not in all_T:
                sys.exit(f"[ERROR] 未知类别: {t}（可选: {' '.join(all_T.keys())} all）")
            wanted.append(t)
    else:
        wanted = ["asr", "llm", "age-gender", "eval", "vc"]
        if args.no_llm and "llm" in wanted:
            wanted.remove("llm")

    # 收集任务
    jobs = []
    for cat in wanted:
        for name, fn in all_T[cat]:
            jobs.append((cat, name, fn))

    print(f"== 将处理 {len(jobs)} 个模型（类别: {', '.join(wanted)}）==\n")
    if args.list:
        for cat, name, _ in jobs:
            print(f"  [{cat}] {name}")
        return

    ok, fail = [], []
    for cat, name, fn in jobs:
        print(f"▶ [{cat}] {name}")
        try:
            fn()
            print(f"  ✓ 完成\n")
            ok.append(name)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ✗ 失败: {e}\n")
            fail.append((name, str(e)))

    print("== 汇总 ==")
    print(f"  成功 {len(ok)}: {', '.join(ok) if ok else '无'}")
    if fail:
        print(f"  失败 {len(fail)}:")
        for n, e in fail:
            print(f"    - {n}: {e[:160]}")
    print("\n提示: fish-speech 需手动放置（见上方 [manual]）。模型就绪后即可运行 agent.py / reflexion_agent。")


if __name__ == "__main__":
    main()
