# Anonymise Agent

[English](README.md) | **简体中文**

> 面向多场景的语音匿名化工具包：一条可配置的**匿名化流水线** + 一个会**反思**的 LLM Agent。

语音匿名化（去身份 + 去隐私词）一键完成：**ASR 转录 → 隐私词识别（LLM/NER）→ 音频掩蔽 → 声线转换 → 指标评估**。
外层 `ReflexionAgent` 接收自然语言需求，自动规划步骤、执行、读取评估/下游结果并反思，必要时换声线转换方法重试，直到满足目标。

- 🔁 **Pipeline**：`agent.py` 串联四阶段，子进程隔离防 OOM，配置驱动（语言→ASR/LLM/prompt 全在 `configs/*.yaml`）。
- 🤖 **Reflexion Agent**：`reflexion_agent_downstream.py`（含下游任务基线对比）；`reflexion_agent.py` 为不含下游任务的精简版。
- 🎛️ **多种声线转换**：`formant_lpc / formant_spec / pitch / NeuralVC / Resyn`（默认 `NeuralVC`；可用 `--vc_method` 指定其它，如 `Resyn`）。
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
| 隐私词识别 / Agent 推理 | `$LLM_MODEL_DIR/`（默认 `./LLMs`） | `Qwen2.5-32B-Instruct`（中/英/日）、`gemma-3-27b-it`（默认/其他语言）/ `Qwen2.5-32B-Instruct`（Agent 默认，可用 `AGENT_LLM_MODEL` 覆盖） |
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
  --vc --vc_method NeuralVC\
  --eval --gt_dir /path/to/gt --subject_map_path /path/to/map.json
```

**B. 让 Reflexion Agent 自己规划 + 反思**（`reflexion_agent_downstream.py`）
```bash
CUDA_VISIBLE_DEVICES=0 python reflexion_agent_downstream.py \
  -i /path/to/input \
  -r "中文单说话人，对比 Resyn 与 pitch 的下游表现" \
  --downstream_data_json /path/to/data.json \
  --downstream_script /path/to/eval.sh \
  --max_reflections 7
```
> `--downstream_data_json` / `--downstream_script` 为可选；不提供则只做匿名化 + 指标评估 + 反思，不跑下游分类。
> 加 `--no-open_source` 可改用 OpenAI 兼容 API（需在 `.env` 配 `BASE_URL` / `OPENAI_API_KEY`）。

**C. Gradio Demo**
```bash
bash demo/demo.sh --llm Qwen2.5-32B-Instruct --port 7860
```

默认地址为 `http://127.0.0.1:7860`；远程服务器可通过 SSH 转发端口，或用 `--host` 指定监听地址。填写服务器上的音频目录和匿名化需求，界面可切换本地／在线 LLM、开关 Memory 并指定其目录，展示召回经验、计划、反思、最终指标和选定方法的音频。可选的下游数据和脚本需成对填写；详细日志位于输出目录。

在线模式使用 `OPENAI_API_KEY`（兼容旧变量 `API_KEY`）、可选的 `BASE_URL` 和对应 API 模型名；仅替代决策 LLM，ASR、掩蔽、VC 和评估仍需要本地环境与模型。`AGENT_LLM_MODEL` 可覆盖默认 Agent 模型。

---

## 跨运行 Memory

`reflexion_agent_downstream.py` 和 Demo 共用带记忆的运行流程；`reflexion_agent_sentence.py` 复用该流程，并提供下文的级联模式。`reflexion_agent.py` 是不含下游任务和跨运行记忆的精简示例。

默认加载仓库根目录的 `memory/`，包含最终 GPT 总结的 14 条任务经验（`experiences.json`）和 16 份历史运行记录（`runs/`）。Agent 从用户请求提取任务信息，在规划和反思时召回任务经验及相关运行明细；运行结束先保存记录，再由 LLM 总结经验，无需额外回填工具。建议在请求中写明语言、说话人数、下游任务和匿名化目标。

- `--memory_path DIR`：指定读写目录；默认使用并更新随仓库提供的 `memory/`。若要保留初始经验，可先执行 `cp -R memory /path/to/my_memory`，再指定该目录。
- `--no_memory`：关闭记忆。
- `--memory_exclude_dataset NAME`：召回时排除某数据集，仍会写回运行记录。

历史记录中的 `/path/to/...` 是原产物路径的占位符；召回读取已保存的文本和指标，不需要访问这些路径。Memory 读写或总结失败会记录警告，主流程可继续，已保存的运行记录仍保留。历史经验应结合新数据的实际评估使用。

## 逐音频级联修复

**前提：各方法的匿名结果已存在于 `output_dir/<method>/`，相对文件名与输入对应。** 级联只做说话人相似度检测、替代选择和复制合并，不自动运行 VC 或决策 LLM。检测单位是一个输入音频文件；需要逐句检测时，应先将输入切成句级文件。

```bash
python reflexion_agent_sentence.py \
  -i /path/to/audio -o /path/to/output \
  --reflex_methods formant_spec,NeuralVC,Resyn --fail_threshold 0.75

# 已有 Agent 结果时，读取最终方法为首候选，其余按下游 bACC 排序：
python reflexion_agent_sentence.py -i /path/to/audio -o /path/to/output \
  --reflex_auto --fail_threshold 0.75
```

相似度严格高于阈值时尝试下一方法。所有方法均未通过时，选取可用候选中相似度最低者并标记；无法打分或缺少全部候选的文件列入 `unassigned`。产物为 `reflex_<methods>/` 及其中的 `reflex_report.json`，可用 `--reflex_dir_name` 指定目录名。合并后的 EER 和下游指标需重新评估。

默认固定阈值为 0.75；`--no_fail_threshold` 切换到首方法分布的 `mean + k × IQR` 规则，并结合相似度下限。可用 `--fail_iqr_k`、`--fail_score_floor`、`--no_fail_score_floor` 调整。阈值应按数据验证。

## 方法名兼容与最终选择

| 当前方法名 | 兼容的旧名称 |
|---|---|
| `formant_lpc` | `mcadams` |
| `formant_spec` | `formant` |
| `NeuralVC` | `seedvc` |
| `Resyn` | `fishaudio_tts` |

`pitch`、`combined` 和 `cosyvoice_tts` 名称不变。流水线 CLI、转换注册表和 Agent 计划接受旧名称；新结果使用规范名称。旧结果目录不会自动改名，复用前应整理目录结构。

LLM 结束探索时可通过 `best_vc_method` 选择已尝试的最佳方法，最终指标与该方法保持一致。未尝试的方法会被拒绝或回落；旧响应未提供此字段时，兼容使用最后尝试的方法。

---

## 目录结构

```
agent.py                       # 流水线总调度（trans / mask / vc / eval 四阶段）
reflexion_agent_downstream.py  # Reflexion Agent（含下游任务）
reflexion_agent.py             # Reflexion Agent（精简版，不含下游任务）
reflexion_agent_sentence.py    # 复用 Agent 流程，增加逐音频级联
memory/    # 任务经验与历史运行记录
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


## 测试

```bash
python -m pytest tests -q
```

覆盖 Memory 持久化与召回、最终方法选择、Demo 接入、旧名称兼容和级联分配。测试模拟模型及评估调用；完整推理仍需准备上述环境和模型。

## 引用
TODO
