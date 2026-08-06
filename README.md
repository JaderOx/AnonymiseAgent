# Anonymise Agent

**简体中文** | [English](README_en.md)

> 面向多场景的语音匿名化工具包：一条可配置的**匿名化流水线** + 一个会**反思**的 LLM Agent。

语音匿名化（去身份 + 去隐私词）一键完成：**ASR 转录 → 隐私词识别（LLM/NER）→ 音频掩蔽 → 声线转换 → 指标评估**。
外层 `ReflexionAgent` 接收自然语言需求，自动规划步骤、执行、读取评估/下游结果并反思，必要时换声线转换方法重试，直到满足目标。

- 🔁 **Pipeline**：`agent.py` 串联四阶段，子进程隔离防 OOM，配置驱动（语言→ASR/LLM/prompt 全在 `configs/*.yaml`）。
- 🤖 **Reflexion Agent**：`reflexion_agent_downstream.py`（含下游任务基线对比）；`reflexion_agent.py` 为不含下游任务的精简版。
- 🎛️ **多种声线转换**：`mcadams / formant / pitch / seedvc / asr-tts `（默认 `seedvc`；可用 `--vc_method` 指定其它，如 `fishaudio_tts`）。
- 📊 **完整指标**：SNR、MOS、CER/WER、情感相似度、基频、说话人 EER、隐私词屏蔽准确率。
- 🖥️ **Gradio Demo**：`demo/demo.py` 可视化「规划-执行-反思」全过程。
- 🌍 **多语言**：中文、中文方言、英文、日文及欧洲多语言（见 [配置文件](#配置文件)）。

---

## 环境准备

需要 **3 个 conda 环境**（pipeline 会按模型自动切换 conda 环境，见 `agent.py` 的 `_ASR_CONDA_ENV` / `_VC_CONDA_ENV`）：

```bash
# 1) 主环境 Agent（ASR / 掩蔽 / DSP-VC / 评估 / Agent 推理）
conda create -n Agent -y python=3.10 && conda activate Agent
# 你自己的torch版本
pip install torch==.. torchvision==.. torchaudio==.. --index-url ..

pip install -r requirements.txt --ignore-installed torch torchvision torchaudio

# （可选）FunASR 补丁：仅当用 FunASRNano / FunASRMLTNano 做 ASR 时才需要，
#     修复 FunASRNano 分句与 VAD KeyError 两处 bug。用 paraformer/whisperx 时可跳过。
# cp utils/auto_model.py $(python -c "import funasr,os;print(os.path.dirname(funasr.__file__))")/auto/auto_model.py

# 2) whisperx 环境（英文/其他语言走 whisperx ASR 时需要）
conda create -n whisperx -y python=3.10 && conda activate whisperx
pip install faster-whisper pyannote.audio whisperx zhconv audonnx

# 3) asr-tts 环境（接入您自己的tts模型，环境按官方文档安装，然后再_VC_CONDA_ENV中配置环境名）
conda create -n _name -y python=3.10
```

复制并填写配置：
```bash
cp .env.example .env   # 填入 HF_TOKEN、模型目录等
```

---

## 准备模型

> 模型体积大，**不在仓库内**。首次运行时 ASR/MOS/emotion2vec/wavlm 会自动下载；以下模型需自行准备并放到对应目录：

| 用途 | 位置 | 模型 |
|---|---|---|
| 隐私词识别 / Agent 推理 | `$LLM_MODEL_DIR/`（默认 `./LLMs`） | `Qwen2.5-32B-Instruct`（中/英/日）、`gemma-3-27b-it`（默认/其他语言）/ `Qwen3.5-9B` 默认reflexion模型 |
| 年龄/性别（asr的meta标注、tts 目标说话人匹配） | `$TORCH_HOME/`（默认 `./models`） | `w2v2-L-robust-6-age-gender` |
| ASR / 评估模型 | 自动下载 | paraformer / FunASR* / whisperx / UTMOS / emotion2vec / wavlm-base-plus-sv |

> 硬件：建议 ≥3 张 NVIDIA RTX 4090；3 张时 `--group_num` 建议 < 30。

---

## 一键下载模型

`download_models.py` 可把所需模型一键下载到 `.env` 指定的路径（funasr / ModelScope / HuggingFace / torch.hub 各走对应通道，已存在自动跳过）：

```bash
python download_models.py --list            # 预览会下哪些
python download_models.py                   # 默认集：asr + llm + age-gender + eval + vc
python download_models.py --targets all     # 全部
python download_models.py --no-llm          # 先不下 32B 大模型，快速跑通 ASR/VC/eval
```

类别：`asr` / `llm` / `age-gender` / `eval` / `whisperx` / `vc` / `all`。

> - LLM 默认下 `Qwen2.5-32B-Instruct`；加 `--gemma` 额外下 `gemma-3-27b-it`。
> - pyannote 说话人分离是 gated 模型，需 `HF_TOKEN`。
> - HuggingFace 不通时，在 `.env` 设 `HF_ENDPOINT=https://hf-mirror.com`。
> - tts模型需自己接入

---

## 快速开始

可见scripts里的example

**A. 直接跑流水线**（`agent.py`）
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python agent.py \
  -i /path/to/input -o /path/to/output --middle_dir /path/to/middle \
  --trans --language 中文 --hotwords "$(grep -v '^#' hotwords.txt | tr '\n' ',')" \
  --mask --method noise --group_num 30 \
  --vc --vc_method seedvc\
  --eval --gt_dir /path/to/gt --subject_map_path /path/to/map.json
```

**B. 让 Reflexion Agent 自己规划 + 反思**（`reflexion_agent_downstream.py`）
```bash
CUDA_VISIBLE_DEVICES=0 python reflexion_agent_downstream.py \
  -i /path/to/input \
  -r "中文单说话人，对比 fishaudio_tts 与 pitch 的下游表现" \
  --downstream_data_json /path/to/data.json \
  --downstream_script /path/to/eval.sh \
  --max_reflections 7
```
> `--downstream_data_json` / `--downstream_script` 为可选；不提供则只做匿名化 + 指标评估 + 反思，不跑下游分类。
> 加 `--no-open_source` 可改用 OpenAI 兼容 API（需在 `.env` 配 `BASE_URL` / `API_KEY`）。

**C. Gradio Demo**
```bash
bash demo/demo.sh        # 或：python demo/demo.py --llm Qwen2.5-32B-Instruct --port 7860
```

---

## 目录结构

```
agent.py                       # 流水线总调度（trans / mask / vc / eval 四阶段）
reflexion_agent_downstream.py  # Reflexion Agent（含下游任务）
reflexion_agent.py             # Reflexion Agent（精简版，不含下游任务）
transcribe.py / mask_llm.py / mask_ner.py / apply_mask.py / voice_convert.py / evaluate.py
configs/   # 语言→ASR/LLM/prompt 映射（改配置不改代码）
utils/     # ASR / VC / 指标 / 日志 / 子进程内存隔离
tts_reference/  # fishaudio/cosyvoice 所需的参考音频与文本（样例数据）
demo/      # Gradio Demo
```

---

## 配置文件

- `configs/language_asr_map.yaml`：语言 → ASR 模型（paraformer / FunASRNano / FunASRMLTNano / whisperx）。
- `configs/language_llm_map.yaml`：语言 → LLM 模型 + 隐私词识别 prompt（改 `prompt` 即可定制，无需改代码）。
- `configs/asr_model_config.yaml`：各 ASR 模型加载参数。

| ASR 模型 | 适用语言 | 热词 | 说话人分离 | conda 环境 |
|---|---|---|---|---|
| `paraformer` | 中文 | ✗ | ✓ | Agent |
| `FunASRNano` | 中文方言、日文 | ✓ | ✗ | Agent |
| `FunASRMLTNano` | 欧洲多语言 | ✓ | ✗ | Agent |
| `whisperx` | 英文 / 其余（默认） | ✓ | ✓（需 HF_TOKEN） | whisperx |

未在配置表中的语言会回退到 `default`（ASR: whisperx；LLM: gemma-3-27b-it），并在日志中告警。

---

## 指标说明

| 指标 | 含义 |
|---|---|
| SNR↑ / MOS↑ | 语音质量 / 自然度 |
| CER↓ / WER↓ | 内容可懂度 |
| EMO↑ | 情感相似度 |
| L1↓ / PCC↑ | 基频相似度 |
| EER | 说话人匿名度（≈50% 最佳，≈0 表示匿名无效） |

---


## 引用
TODO
