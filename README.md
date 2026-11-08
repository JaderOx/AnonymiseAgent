# Anonymise Agent

**English** | [简体中文](README_zh.md)

> A multi-scenario speech anonymization toolkit: a configurable **anonymization pipeline** + a **reflective** LLM Agent.

One-click speech anonymization (de-identification + privacy-word removal): **ASR transcription → privacy-word detection (LLM/NER) → audio masking → voice conversion → metric evaluation**.
The outer `ReflexionAgent` takes natural-language requirements, automatically plans the steps, executes them, reads the evaluation/downstream results and reflects — retrying with a different voice-conversion method when needed until the target is met.

- 🔁 **Pipeline**: `agent.py` chains the four stages with subprocess isolation to prevent OOM, fully configuration-driven (language → ASR/LLM/prompt all live in `configs/*.yaml`).
- 🤖 **Reflexion Agent**: `reflexion_agent_downstream.py` (includes downstream-task baseline comparison); `reflexion_agent.py` is a lightweight version without downstream tasks.
- 🎛️ **Multiple voice-conversion methods**: `formant_lpc / formant_spec / pitch / NeuralVC / Resyn` (default `NeuralVC`; use `--vc_method` to pick others, e.g. `Resyn`).
- 📊 **Comprehensive metrics**: SNR, MOS, CER/WER, emotion similarity, fundamental frequency, speaker EER, privacy-word masking accuracy.
- 🖥️ **Gradio Demo**: `demo/demo.py` visualizes the full "plan–execute–reflect" process.
- 🌍 **Multi-language**: Chinese, Chinese dialects, English, Japanese and several European languages (see [Configuration files](#configuration-files)).

---

## Environment Setup

You need **3 conda environments** (the pipeline switches conda environments automatically per model — see `_ASR_CONDA_ENV` / `_VC_CONDA_ENV` in `agent.py`):

```bash
# 1) Main environment "Agent" (ASR / masking / DSP-VC / evaluation / Agent inference)
conda create -n Agent -y python=3.10 && conda activate Agent
# Your own torch build
pip install torch==.. torchvision==.. torchaudio==.. --index-url ..

pip install -r requirements.txt --ignore-installed torch torchvision torchaudio

# (Optional) FunASR patch: only needed when using FunASRNano / FunASRMLTNano for ASR.
#     Fixes two bugs (FunASRNano sentence segmentation and a VAD KeyError).
#     Skip it if you use paraformer / whisperx.
# cp utils/auto_model.py $(python -c "import funasr,os;print(os.path.dirname(funasr.__file__))")/auto/auto_model.py

# 2) whisperx environment (needed when running English / other languages through whisperx ASR)
conda create -n whisperx -y python=3.10 && conda activate whisperx
pip install faster-whisper pyannote.audio whisperx zhconv audonnx

# 3) asr-tts environment (for plugging in your own TTS model; install per the model's official docs,
#    then set the environment name in _VC_CONDA_ENV)
conda create -n _name -y python=3.10
```

Copy and fill in the configuration:
```bash
cp .env.example .env   # Fill in HF_TOKEN, model directories, etc.
```

---

## Prepare Models

> Models are large and **not stored in the repo**. On first run, ASR/MOS/emotion2vec/wavlm are downloaded automatically; the following must be prepared manually and placed in the right directories:

| Purpose | Location | Model |
|---|---|---|
| Privacy-word detection / Agent inference | `$LLM_MODEL_DIR/` (default `./LLMs`) | `Qwen2.5-32B-Instruct` (Chinese/English/Japanese), `gemma-3-27b-it` (default / other languages) / `Qwen2.5-32B-Instruct` (default Agent model; override with `AGENT_LLM_MODEL`) |
| Age/gender (ASR meta annotation, TTS target-speaker matching) | `$TORCH_HOME/` (default `./models`) | `w2v2-L-robust-6-age-gender` |
| ASR / evaluation models | auto-downloaded | paraformer / FunASR* / whisperx / UTMOS / emotion2vec / wavlm-base-plus-sv |

> Hardware: ≥3× NVIDIA RTX 4090 recommended; with 3 GPUs keep `--group_num` < 30.

---

## One-click Model Download

`download_models.py` downloads all required models to the paths set in `.env` (funasr / ModelScope / HuggingFace / torch.hub each via its own channel; existing files are skipped):

```bash
python download_models.py --list            # Preview what will be downloaded
python download_models.py                   # Default set: asr + llm + age-gender + eval + vc
python download_models.py --targets all     # Everything
python download_models.py --no-llm          # Skip the 32B LLM first, to quickly exercise ASR/VC/eval
```

Categories: `asr` / `llm` / `age-gender` / `eval` / `whisperx` / `vc` / `all`.

> - By default downloads `Qwen2.5-32B-Instruct`; add `--gemma` to also download `gemma-3-27b-it`.
> - pyannote speaker diarization is a gated model and requires `HF_TOKEN`.
> - If HuggingFace is unreachable, set `HF_ENDPOINT=https://hf-mirror.com` in `.env`.
> - The TTS model must be integrated by yourself.

---

## Quick Start

See the examples in `scripts/`.

**A. Run the pipeline directly** (`agent.py`)
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python agent.py \
  -i /path/to/input -o /path/to/output --middle_dir /path/to/middle \
  --trans --language 中文 --hotwords "$(grep -v '^#' hotwords.txt | tr '\n' ',')" \
  --mask --method noise --group_num 30 \
  --vc --vc_method NeuralVC \
  --eval --gt_dir /path/to/gt --subject_map_path /path/to/map.json
```

**B. Let the Reflexion Agent plan and reflect on its own** (`reflexion_agent_downstream.py`)
```bash
CUDA_VISIBLE_DEVICES=0 python reflexion_agent_downstream.py \
  -i /path/to/input \
  -r "Chinese, single speaker; compare downstream performance of Resyn vs pitch" \
  --downstream_data_json /path/to/data.json \
  --downstream_script /path/to/eval.sh \
  --max_reflections 7
```
> `--downstream_data_json` / `--downstream_script` are optional; if omitted, only anonymization + metric evaluation + reflection are performed (no downstream classification).
> Add `--no-open_source` to use an OpenAI-compatible API instead (configure `BASE_URL` / `OPENAI_API_KEY` in `.env`).

**C. Gradio Demo**
```bash
bash demo/demo.sh --llm Qwen2.5-32B-Instruct --port 7860
```

The Demo opens at `http://127.0.0.1:7860`; use SSH port forwarding for a remote server or set `--host` explicitly. Enter a server-side audio directory and your requirements. The interface provides local/API LLM selection, a Memory toggle and directory, recalled experiences, planning/reflection events, final metrics and audio from the selected method. Optional downstream data and script must be supplied together. Detailed logs are saved in the output directory.

For API inference, set `OPENAI_API_KEY` (legacy `API_KEY` remains supported), optionally `BASE_URL`, and select the API model name. Online inference replaces only the decision LLM; local ASR, masking, VC and evaluation environments are still required. `AGENT_LLM_MODEL` overrides the default Agent model.

---

## Cross-run Memory

`reflexion_agent_downstream.py` and the Demo share the memory-enabled runtime. `reflexion_agent_sentence.py` reuses that runtime and adds the cascade mode below. `reflexion_agent.py` is the lightweight example without downstream evaluation or cross-run memory.

By default, the Agent loads the repository's `memory/`, containing 14 GPT-generated task experiences (`experiences.json`) and 16 historical runs (`runs/`). It extracts task information from the request, recalls task experiences and relevant run details for planning and reflection, then saves the run before asking the LLM to summarize experience. No backfill tools are needed. Include the language, speaker count, downstream task and anonymization goals in your request.

- `--memory_path DIR`: select the read/write directory; the default updates the bundled `memory/`. To preserve the initial data, first run `cp -R memory /path/to/my_memory` and use that directory.
- `--no_memory`: disable memory.
- `--memory_exclude_dataset NAME`: exclude a dataset during recall; new runs are still recorded.

Historical `/path/to/...` values are placeholders for original artifacts. Recall uses stored text and metrics without accessing those paths. Memory read/write or summarization failures produce warnings and allow the main flow to continue; saved run records are retained. Validate historical recommendations on your own data.

## Sample-level Cascade

**Prerequisite: each method's anonymized output already exists in `output_dir/<method>/`, with relative filenames matching the input.** The cascade scores speaker similarity, selects alternatives and copies the selected audio; it does not run VC or the decision LLM. Each input file is one detection unit; split audio into sentences first for sentence-level detection.

```bash
python reflexion_agent_sentence.py \
  -i /path/to/audio -o /path/to/output \
  --reflex_methods formant_spec,NeuralVC,Resyn --fail_threshold 0.75

# With existing Agent results, start with the selected method,
# then rank the remaining methods by downstream bACC:
python reflexion_agent_sentence.py -i /path/to/audio -o /path/to/output \
  --reflex_auto --fail_threshold 0.75
```

Similarity strictly above the threshold triggers the next method. If every method fails, the lowest-similarity available candidate is selected and marked accordingly; files that cannot be scored or have no usable candidates appear in `unassigned`. Outputs are written to `reflex_<methods>/` with a `reflex_report.json`; use `--reflex_dir_name` to customize the directory. Re-evaluate EER and downstream metrics on the merged output.

The default fixed threshold is 0.75. `--no_fail_threshold` switches to the first method's `mean + k × IQR` rule with a similarity floor. Adjust it with `--fail_iqr_k`, `--fail_score_floor` or `--no_fail_score_floor`. Validate thresholds on your data.

## Method Names and Final Selection

| Current name | Accepted legacy alias |
|---|---|
| `formant_lpc` | `mcadams` |
| `formant_spec` | `formant` |
| `NeuralVC` | `seedvc` |
| `Resyn` | `fishaudio_tts` |

`pitch`, `combined` and `cosyvoice_tts` retain their names. The pipeline CLI, conversion registry and Agent plans accept legacy names; new results use canonical names. Existing result directories are not renamed automatically; organize them before reuse.

When ending exploration, the LLM can return `best_vc_method` to select an already-tried method. Final metrics follow that selection. Untried selections are rejected or fall back; legacy responses without this field retain the last-attempt behavior.

---

## Directory Structure

```
agent.py                       # Pipeline orchestrator (trans / mask / vc / eval — four stages)
reflexion_agent_downstream.py  # Reflexion Agent (with downstream tasks)
reflexion_agent.py             # Reflexion Agent (lightweight, without downstream tasks)
reflexion_agent_sentence.py    # Shared Agent runtime + sample-level cascade
memory/    # Task experiences and historical run records
transcribe.py / mask_llm.py / mask_ner.py / apply_mask.py / voice_convert.py / evaluate.py
configs/   # language → ASR/LLM/prompt mappings (tweak config, not code)
utils/     # ASR / VC / metrics / logging / subprocess memory isolation
tts_reference/  # reference audio + text needed by fishaudio/cosyvoice (sample data)
demo/      # Gradio Demo
```

---

## Configuration Files

- `configs/language_asr_map.yaml`: language → ASR model (paraformer / FunASRNano / FunASRMLTNano / whisperx).
- `configs/language_llm_map.yaml`: language → LLM model + privacy-word detection prompt (edit `prompt` to customize — no code change needed).
- `configs/asr_model_config.yaml`: loading parameters for each ASR model.

| ASR model | Languages | Hotwords | Diarization | conda env |
|---|---|---|---|---|
| `paraformer` | Chinese | ✗ | ✓ | Agent |
| `FunASRNano` | Chinese dialects, Japanese | ✓ | ✗ | Agent |
| `FunASRMLTNano` | European languages | ✓ | ✗ | Agent |
| `whisperx` | English / others (default) | ✓ | ✓ (needs HF_TOKEN) | whisperx |

Languages not listed in the config fall back to `default` (ASR: whisperx; LLM: gemma-3-27b-it) and emit a warning in the logs.

---

## Metrics

| Metric | Meaning |
|---|---|
| SNR↑ / MOS↑ | Speech quality / naturalness |
| CER↓ / WER↓ | Content intelligibility |
| EMO↑ | Emotion similarity |
| L1↓ / PCC↑ | Fundamental-frequency similarity |
| EER | Speaker anonymity (≈50% is ideal; ≈0 means anonymization is ineffective) |

---


## Tests

```bash
python -m pytest tests -q
```

Covers memory persistence/recall, final selection, Demo integration, legacy aliases and cascade assignment. Model and evaluation calls are simulated in tests; full inference requires the environments and models above.

## Citation
TODO
