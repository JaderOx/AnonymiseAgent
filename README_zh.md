# CASA

[English](README.md) | **简体中文**

> 面向多场景的语音匿名化工具包：一条可配置的**匿名化流水线** + 一个会**反思**的 LLM Agent。

语音匿名化（去身份 + 去隐私词）一键完成：**ASR 转录 → 隐私词识别（LLM/NER）→ 音频掩蔽 → 声线转换 → 指标评估**。
CASA 接收自然语言需求，自动规划步骤、执行、读取评估/下游结果并反思，根据用户目标调整声线转换方法或匿名参数。

- 🔁 **Pipeline**：`agent.py` 串联四阶段，子进程隔离防 OOM，配置驱动（语言→ASR/LLM/prompt 全在 `configs/*.yaml`）。
- 🤖 **CASA**：`casa_downstream.py`（含下游任务基线对比）；`casa.py` 为不含下游任务的精简版。
- 🎛️ **多种声线转换**：`formant_lpc / formant_spec / pitch / NeuralVC / Resyn`（默认 `NeuralVC`；可用 `--vc_method` 指定其它，如 `Resyn`）。
- 📊 **完整指标**：SNR、MOS、CER/WER、情感相似度、基频、说话人 EER、隐私词屏蔽准确率。
- 🖥️ **Gradio Demo**：`demo/demo.py` 可视化「规划-执行-反思」全过程。
- 🌍 **多语言**：中文、中文方言、英文、日文及欧洲多语言（见 [配置文件](#配置文件)）。

---

## 环境准备

在仓库根目录执行以下命令。CASA 使用 `Agent` 主环境，WhisperX 转录使用 `whisperx` 环境，Resyn 重合成使用 `fishaudio` 环境。

```bash
conda create -n Agent -y python=3.10 && conda activate Agent
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

conda create -n whisperx -y python=3.10 && conda activate whisperx
python -m pip install faster-whisper pyannote.audio whisperx zhconv audonnx python-dotenv pyyaml

conda create -n fishaudio -c conda-forge -y python=3.10 pyaudio sox ffmpeg
conda activate fishaudio
mkdir -p models
git clone --depth 1 --branch v2.0.0-beta https://github.com/fishaudio/fish-speech.git models/fish-speech
python -m pip install -e 'models/fish-speech[cu128]' python-dotenv pyyaml

conda activate Agent
hf download fishaudio/s2-pro --local-dir models/fish_speech_models/s2-pro
cp .env.example .env
```

在 `.env` 填写模型目录和令牌。以上命令使用默认的 `TORCH_HOME=./models`。如果修改该路径，请将 Fish Speech 源码和 S2-Pro 权重放在新目录下对应的位置。[Fish Speech 安装说明](https://github.com/fishaudio/fish-speech/blob/v2.0.0-beta/docs/en/install.md) · [S2-Pro 权重](https://huggingface.co/fishaudio/s2-pro)

---

## 准备模型

> 模型体积大，**不在仓库内**。首次运行时 ASR/MOS/emotion2vec/wavlm 会自动下载；以下模型需自行准备并放到对应目录：

| 用途 | 位置 | 模型 |
|---|---|---|
| 隐私词识别 / Agent 推理 | `$LLM_MODEL_DIR/`（默认 `./LLMs`） | `Qwen2.5-32B-Instruct`（中/英/日）、`gemma-3-27b-it`（默认/其他语言）/ `Qwen2.5-32B-Instruct`（Agent 默认，可用 `AGENT_LLM_MODEL` 覆盖） |
| 年龄/性别（asr的meta标注、tts 目标说话人匹配） | `$TORCH_HOME/`（默认 `./models`） | `w2v2-L-robust-6-age-gender` |
| ASR / 评估模型 | 自动下载 | Paraformer / WhisperX / UTMOS / emotion2vec / wavlm-base-plus-sv |

NeuralVC 从 `utils/voice_convert/aishell/aishell_target_speaker.json` 选取参考说话人。使用 NeuralVC 前，将其中的 `wav_path` 改为对应 AISHELL 音频的本地路径。也可通过 `NEURALVC_TARGET_POOL` 指定单独的 JSON 文件，NeuralVC 在加载模型前会检查参考音频是否存在。

> 硬件：建议 ≥3 张 NVIDIA RTX 4090；3 张时 `--group_num` 建议 < 30。

---

## 一键下载模型

`download_models.py` 将选定的模型下载到 `.env` 指定的路径，并提示 Resyn 的独立安装步骤（funasr / ModelScope / HuggingFace / torch.hub 各走对应通道，已存在自动跳过）：

```bash
python download_models.py --list            # 预览会下哪些
python download_models.py                   # 默认集：asr + llm + age-gender + eval + vc
python download_models.py --targets all     # 全部
python download_models.py --no-llm          # 先不下载 32B 大模型，快速跑通 ASR/VC/eval
```

类别：`asr` / `llm` / `age-gender` / `eval` / `whisperx` / `vc` / `all`。

> - LLM 默认下 `Qwen2.5-32B-Instruct`；加 `--gemma` 额外下 `gemma-3-27b-it`。
> - pyannote 说话人分离是 gated 模型，需 `HF_TOKEN`。
> - HuggingFace 不通时，在 `.env` 设 `HF_ENDPOINT=https://hf-mirror.com`。
> - Resyn 使用 Fish Speech S2-Pro 和 `tts_reference/` 中的参考音频。环境准备一节给出了源码和权重的安装命令。

---

## 快速开始

可见scripts里的example

**A. 直接跑流水线**（`agent.py`）
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python agent.py \
  -i /path/to/input -o /path/to/output --middle_dir /path/to/middle \
  --trans --language zh \
  --mask --method noise --group_num 30 \
  --vc --vc_method NeuralVC \
  --eval --gt_dir /path/to/gt --subject_map_path /path/to/map.json
```

**B. 让 CASA 自己规划 + 反思**（`casa_downstream.py`）
```bash
CUDA_VISIBLE_DEVICES=0 python casa_downstream.py \
  -i /path/to/input \
  -r "中文单说话人，对比 Resyn 与 pitch 的下游表现" \
  --downstream_data_json /path/to/data.json \
  --downstream_script /path/to/eval.sh \
  --max_reflections 7
```
> `--downstream_data_json` / `--downstream_script` 为可选；不提供则只做匿名化 + 指标评估 + 反思，不跑下游分类。
> 加 `--no-open_source` 可改用 OpenAI 兼容 API（需在 `.env` 配 `BASE_URL` / `OPENAI_API_KEY`）。

CASA 在规划和反思阶段均可设置 `vc_params`。例如 `{"method":"pitch","vc_params":{"n_steps":-2}}` 将变调幅度设为两个半音。提示词给出了各方法默认值，包括 `pitch.n_steps = -3`、`formant_lpc.alpha = 0.8` 和 `formant_spec.scale = 1.1`。最终选定的参数、指标和音频目录写入 `agent_result.json`。

**C. Gradio Demo**
```bash
bash demo/demo.sh --llm Qwen2.5-32B-Instruct --port 7860
```

默认地址为 `http://127.0.0.1:7860`；远程服务器可通过 SSH 转发端口，或用 `--host` 指定监听地址。填写服务器上的音频目录和匿名化需求，界面可切换本地／在线 LLM、开关 Memory 并指定其目录，展示召回经验、计划、反思、最终指标和选定方法的音频。可选的下游数据和脚本需成对填写；详细日志位于输出目录。

在线模式使用 `OPENAI_API_KEY`（兼容旧变量 `API_KEY`）、可选的 `BASE_URL` 和对应 API 模型名；仅替代决策 LLM，ASR、掩蔽、VC 和评估仍需要本地环境与模型。`AGENT_LLM_MODEL` 可覆盖默认 Agent 模型。

---

## 跨运行 Memory

`casa_downstream.py` 和英文 Demo 共用带记忆的运行流程。`casa.py` 是精简入口。

默认加载仓库根目录的 `memory/`，包含Qwen 总结的 14 条任务经验（`experiences.json`）和 16 份历史运行记录（`runs/`）。Agent 从用户请求提取任务信息，在规划和反思时召回任务经验及相关运行明细；运行结束先保存记录，再由 LLM 总结经验，无需额外回填工具。建议在请求中写明语言、说话人数、下游任务和匿名化目标。

Demo 按任务类型或数据集召回，各取最多三条经验和运行记录。无匹配项时根据当前请求规划，仍可保存本次运行。命令行入口继续使用完整召回策略。

- `--memory_path DIR`：指定读写目录；默认使用并更新随仓库提供的 `memory/`。若要保留初始经验，可先执行 `cp -R memory /path/to/my_memory`，再指定该目录。
- `--no_memory`：关闭记忆。
- `--memory_exclude_dataset NAME`：召回时排除某数据集，仍会写回运行记录。

历史记录中的 `/path/to/...` 是原产物路径的占位符；召回读取已保存的文本和指标，不需要访问这些路径。Memory 读写或总结失败会记录警告，主流程可继续，已保存的运行记录仍保留。历史经验应结合新数据的实际评估使用。

## 目录结构

```
agent.py                       # 流水线总调度（trans / mask / vc / eval 四阶段）
casa_downstream.py  # CASA，含跨运行记忆和可选下游评估
casa.py             # CASA（精简版，不含下游任务）
memory/    # 任务经验与历史运行记录
transcribe.py / mask_llm.py / mask_ner.py / apply_mask.py / voice_convert.py / evaluate.py
configs/   # 语言→ASR/LLM/prompt 映射（改配置不改代码）
utils/     # ASR / VC / 指标 / 日志 / 子进程内存隔离
tts_reference/  # Resyn 使用的参考音频与文本
demo/      # Gradio Demo
```

---

## 配置文件

- `configs/language_asr_map.yaml` 为中文（`zh`）选择 Paraformer，其他语言使用 WhisperX。Paraformer 自带说话人分离，WhisperX 配置 `HF_TOKEN` 后可使用 pyannote 分离说话人。
- `configs/asr_model_config.yaml` 指定 Paraformer 的转录、语音活动检测、标点和说话人模型。
- `configs/language_llm_map.yaml` 指定各语言的隐私词检测模型和提示词。中文、英文和日文默认使用 Qwen，其他语言使用默认配置。

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


## 测试

```bash
python -m pytest tests -q
```

覆盖记忆持久化与召回、匿名参数、最终选择、语言处理和 Demo 接入。测试模拟模型及评估调用；完整推理仍需准备上述环境和模型。

## 引用
TODO
