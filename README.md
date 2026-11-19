# CASA

**English** | [简体中文](README_zh.md)

> A multi-scenario speech anonymization toolkit: a configurable **anonymization pipeline** + a **reflective** LLM Agent.

One-click speech anonymization (de-identification + privacy-word removal): **ASR transcription → privacy-word detection (LLM/NER) → audio masking → voice conversion → metric evaluation**.
CASA takes natural-language requirements, automatically plans the steps, executes them, reads the evaluation/downstream results and reflects — adjusting the voice-conversion method or its parameters against the requested goal.

- 🔁 **Pipeline**: `agent.py` chains the four stages with subprocess isolation to prevent OOM, fully configuration-driven (language → ASR/LLM/prompt all live in `configs/*.yaml`).
- 🤖 **CASA**: `casa_downstream.py` (includes downstream-task baseline comparison); `casa.py` is a lightweight version without downstream tasks.
- 🎛️ **Multiple voice-conversion methods**: `formant_lpc / formant_spec / pitch / NeuralVC / Resyn` (default `NeuralVC`; use `--vc_method` to pick others, e.g. `Resyn`).
- 📊 **Comprehensive metrics**: SNR, MOS, CER/WER, emotion similarity, fundamental frequency, speaker EER, privacy-word masking accuracy.
- 🖥️ **Gradio Demo**: `demo/demo.py` visualizes the full "plan–execute–reflect" process.
- 🌍 **Multi-language**: Chinese, Chinese dialects, English, Japanese and several European languages (see [Configuration files](#configuration-files)).

---

## Environment Setup

Run the following commands from the repository root. CASA runs in `Agent`, uses `whisperx` for WhisperX transcription, and uses `fishaudio` for Resyn synthesis.

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

Fill in the model directories and token in `.env`. The commands use the default `TORCH_HOME=./models`. If you change it, place both the Fish Speech checkout and S2-Pro weights under that root. [Fish Speech installation](https://github.com/fishaudio/fish-speech/blob/v2.0.0-beta/docs/en/install.md) · [S2-Pro weights](https://huggingface.co/fishaudio/s2-pro)

---

## Prepare Models

> Models are large and **not stored in the repo**. On first run, ASR/MOS/emotion2vec/wavlm are downloaded automatically; the following must be prepared manually and placed in the right directories:

| Purpose | Location | Model |
|---|---|---|
| Privacy-word detection / Agent inference | `$LLM_MODEL_DIR/` (default `./LLMs`) | `Qwen2.5-32B-Instruct` (Chinese/English/Japanese), `gemma-3-27b-it` (default / other languages) / `Qwen2.5-32B-Instruct` (default Agent model; override with `AGENT_LLM_MODEL`) |
| Age/gender (ASR meta annotation, TTS target-speaker matching) | `$TORCH_HOME/` (default `./models`) | `w2v2-L-robust-6-age-gender` |
| ASR / evaluation models | auto-downloaded | Paraformer / WhisperX / UTMOS / emotion2vec / wavlm-base-plus-sv |

NeuralVC selects reference speakers from `utils/voice_convert/aishell/aishell_target_speaker.json`. Set its `wav_path` entries to the corresponding local AISHELL files before running NeuralVC.

> Hardware: ≥3× NVIDIA RTX 4090 recommended; with 3 GPUs keep `--group_num` < 30.

---

## One-click Model Download

`download_models.py` downloads the selected models to the paths set in `.env` and prints the separate Resyn setup instructions (funasr / ModelScope / HuggingFace / torch.hub each via its own channel; existing files are skipped):

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
> - Resyn uses Fish Speech S2-Pro and the reference clips in `tts_reference/`. The setup commands above install its code and weights.

---

## Quick Start

See the examples in `scripts/`.

**A. Run the pipeline directly** (`agent.py`)
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python agent.py \
  -i /path/to/input -o /path/to/output --middle_dir /path/to/middle \
  --trans --language zh \
  --mask --method noise --group_num 30 \
  --vc --vc_method NeuralVC \
  --eval --gt_dir /path/to/gt --subject_map_path /path/to/map.json
```

**B. Run CASA with a natural-language request** (`casa_downstream.py`)
```bash
CUDA_VISIBLE_DEVICES=0 python casa_downstream.py \
  -i /path/to/input \
  -r "Chinese, single speaker; compare downstream performance of Resyn vs pitch" \
  --downstream_data_json /path/to/data.json \
  --downstream_script /path/to/eval.sh \
  --max_reflections 7
```
> `--downstream_data_json` / `--downstream_script` are optional; if omitted, only anonymization + metric evaluation + reflection are performed (no downstream classification).
> Add `--no-open_source` to use an OpenAI-compatible API instead (configure `BASE_URL` / `OPENAI_API_KEY` in `.env`).

CASA can set `vc_params` during planning and reflection. For example, `{"method":"pitch","vc_params":{"n_steps":-2}}` requests a two-semitone shift. The prompt lists each method’s defaults, including `pitch.n_steps = -3`, `formant_lpc.alpha = 0.8` and `formant_spec.scale = 1.1`. The selected parameters, metrics and audio directory are recorded in `agent_result.json`.

**C. Gradio Demo**
```bash
bash demo/demo.sh --llm Qwen2.5-32B-Instruct --port 7860
```

The Demo opens at `http://127.0.0.1:7860`; use SSH port forwarding for a remote server or set `--host` explicitly. Enter a server-side audio directory and your requirements. The interface provides local/API LLM selection, a Memory toggle and directory, recalled experiences, planning/reflection events, final metrics and audio from the selected method. Optional downstream data and script must be supplied together. Detailed logs are saved in the output directory.

For API inference, set `OPENAI_API_KEY` (legacy `API_KEY` remains supported), optionally `BASE_URL`, and select the API model name. Online inference replaces only the decision LLM; local ASR, masking, VC and evaluation environments are still required. `AGENT_LLM_MODEL` overrides the default Agent model.

---

## Cross-run Memory

`casa_downstream.py` and the English demo share the memory-enabled runtime. `casa.py` is the lightweight entry point.

By default, the Agent loads the repository's `memory/`, containing 14 Qwen-generated task experiences (`experiences.json`) and 16 historical runs (`runs/`). It extracts task information from the request, recalls task experiences and relevant run details for planning and reflection, then saves the run before asking the LLM to summarize experience. No backfill tools are needed. Include the language, speaker count, downstream task and anonymization goals in your request.

- `--memory_path DIR`: select the read/write directory; the default updates the bundled `memory/`. To preserve the initial data, first run `cp -R memory /path/to/my_memory` and use that directory.
- `--no_memory`: disable memory.
- `--memory_exclude_dataset NAME`: exclude a dataset during recall; new runs are still recorded.

Historical `/path/to/...` values are placeholders for original artifacts. Recall uses stored text and metrics without accessing those paths. Memory read/write or summarization failures produce warnings and allow the main flow to continue; saved run records are retained. Validate historical recommendations on your own data.

## Directory Structure

```
agent.py                       # Pipeline orchestrator (trans / mask / vc / eval — four stages)
casa_downstream.py  # CASA with memory and optional downstream evaluation
casa.py             # Lightweight CASA
memory/    # Task experiences and historical run records
transcribe.py / mask_llm.py / mask_ner.py / apply_mask.py / voice_convert.py / evaluate.py
configs/   # language → ASR/LLM/prompt mappings (tweak config, not code)
utils/     # ASR / VC / metrics / logging / subprocess memory isolation
tts_reference/  # reference audio and transcripts for Resyn
demo/      # Gradio Demo
```

---

## Configuration Files

- `configs/language_asr_map.yaml` selects Paraformer for Chinese (`zh`) and WhisperX for other languages. Paraformer includes speaker diarization. WhisperX uses pyannote for diarization when `HF_TOKEN` is configured.
- `configs/asr_model_config.yaml` lists the Paraformer model, VAD, punctuation and speaker checkpoints.
- `configs/language_llm_map.yaml` maps languages to privacy-word detection models and prompts. Chinese, English and Japanese use Qwen by default. Other languages use the default configuration.

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

Covers memory persistence and recall, VC parameters, final selection, language handling and demo integration. Model and evaluation calls are simulated in tests; full inference requires the environments and models above.

## Citation
TODO
