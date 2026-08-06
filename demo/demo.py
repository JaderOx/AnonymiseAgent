"""
demo.py — 语音匿名化 Agent Demo（Gradio UI）

基于 reflexion_agent_downstream.py 的 ReflexionAgent，通过 Gradio 展示：
  - Agent 计划生成（步骤 + 理由）
  - 实时执行步骤（工具调用 → 成功/失败）
  - 下游任务结果（bacc / f1）
  - 反思循环（换 VC 方法重试）
  - 评估指标和匿名化音频

用法：
  CUDA_VISIBLE_DEVICES=0 python demo/demo.py --llm Qwen2.5-32B-Instruct
  CUDA_VISIBLE_DEVICES=0 python demo/demo.py --llm Qwen3.5-9B --mode DEBUG --port 8080
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
import shutil
import tempfile
from pathlib import Path

import gradio as gr

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

os.environ['MODELSCOPE_CACHE'] = os.environ.get('MODELSCOPE_CACHE', os.path.join(_PROJECT_ROOT, '.'))
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', os.path.join(_PROJECT_ROOT, 'models'))

from reflexion_agent_downstream import (
    ReflexionAgent, VC_METHODS,
    _build_context_info, _validate_plan,
)

# session 持久化（单用户 demo，跨多次运行累加）
_SESSION_HISTORY = []
_STOP_EVENT = threading.Event()


# ═══════════════════════════════════════════════════════════════════════════════
#  步骤名映射（tool → 中文描述）
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_LABEL = {
    "run_transcription":     "ASR 转录",
    "run_mask":              "内容匿名",
    "run_voice_conversion":  "声线转换",
    "run_evaluation":        "评估",
    "check_directory":       "检查目录",
}


def _tool_label(tool_name: str, params: dict = None) -> str:
    label = _TOOL_LABEL.get(tool_name, tool_name)
    if tool_name == "run_voice_conversion" and params:
        return f"声线转换(method={params.get('method','?')})"
    if tool_name == "run_mask" and params:
        return f"内容匿名(detect={params.get('detect_method','llm')})"
    return label


# ═══════════════════════════════════════════════════════════════════════════════
#  UI-aware ReflexionAgent
# ═══════════════════════════════════════════════════════════════════════════════

class UIReflexionAgent(ReflexionAgent):
    """继承 ReflexionAgent，通过回调向 UI 推送状态。"""

    def __init__(self, *args, on_update=None, session_history=None,
                 ui_language=None, ui_max_speakers=None, ui_min_speakers=None,
                 ui_spk_num=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_update = on_update or (lambda *a: None)
        self._session_history = session_history or []
        self._ui_language = ui_language
        self._ui_max_speakers = ui_max_speakers
        self._ui_min_speakers = ui_min_speakers
        self._ui_spk_num = ui_spk_num

    @property
    def stop_requested(self):
        ev = getattr(self, '_stop_event', None)
        return ev is not None and ev.is_set()

    def _emit(self, kind, data):
        self._on_update(kind, data)

    def _inject_extra_params(self, plan: dict):
        """将 UI 传入的额外参数注入计划步骤。"""
        for step in plan.get("steps", []):
            tool = step["tool"]
            params = step.setdefault("params", {})
            # 评估步骤：gt_dir / subject_map
            if tool == "run_evaluation":
                if self.gt_dir and not params.get("gt_dir"):
                    params["gt_dir"] = self.gt_dir
                if self.subject_map_path and not params.get("subject_map_path"):
                    params["subject_map_path"] = self.subject_map_path
            # 语言覆盖：所有相关步骤
            if self._ui_language:
                params["language"] = self._ui_language
            # 说话人覆盖：转录步骤
            if tool == "run_transcription":
                if self._ui_spk_num:
                    params["spk_num"] = self._ui_spk_num
                    params.pop("min_speakers", None)
                    params.pop("max_speakers", None)
                else:
                    if self._ui_max_speakers:
                        params["max_speakers"] = self._ui_max_speakers
                    if self._ui_min_speakers:
                        params["min_speakers"] = self._ui_min_speakers

    # ── 计划生成 ─────────────────────────────────────────────────────────────

    def _generate_initial_plan(self, user_request):
        self._emit("phase", "生成计划")
        self._emit("status", "正在生成执行计划...")
        self._emit("thinking", "分析请求并生成执行计划...")

        context_info = _build_context_info(self.ctx)
        plan_prompt = f"""你是语音匿名处理 Agent。根据用户请求生成 JSON 执行计划。

【重要】所有工具的 language 参数必须用中文书写（"中文"/"英文"/"日文"/"西班牙语"等），不要用英文或拼音。

【可用工具】
- run_transcription: ASR 转录。参数: language(必填，必须用中文，如"中文"/"英文"/"日文"/"西班牙语"), asr_model(可选), hotwords(可选), spk_num(可选，精确说话人数), min_speakers(可选，最少说话人数), max_speakers(可选，最多说话人数)。spk_num 和 min/max_speakers 三选一，都不填则自动检测
- run_mask: 内容匿名（依赖转录）。参数: language(必填，中文), mask_method(noise/mute), detect_method(llm/ner, 默认llm), group_num(每批处理句数，30-60，不要与说话人数混淆)
- run_voice_conversion: 声线转换。参数: method(必填, 可选: {', '.join(VC_METHODS)}), with_mask(bool), language
- run_evaluation: 评估。参数: language(必填), gt_dir(可选), subject_map_path(可选)

【步骤依赖】
- run_mask 依赖 run_transcription
- run_voice_conversion 独立，with_mask=true 时需要 mask 输出
- run_evaluation 依赖 run_transcription，且需要 mask 或 vc 输出

【输出格式】在适当位置输出 JSON，以 {{ 开头。
{{"language":"语言","reasoning":"分析","steps":[{{"tool":"工具名","params":{{}}}}]}}

示例:
用户: "匿名化中文音频，不需要内容匿名"
{{"language":"中文","reasoning":"跳过mask","steps":[{{"tool":"run_transcription","params":{{"language":"中文"}}}},{{"tool":"run_voice_conversion","params":{{"method":"seedvc","language":"中文"}}}},{{"tool":"run_evaluation","params":{{"language":"中文"}}}}]}}"""

        # 注入 session 历史
        hist_note = ""
        if self._session_history:
            lines = []
            for i, run in enumerate(self._session_history):
                req = run.get("request", "")[:60]
                lines.append(f"  第 {i+1} 次: 「{req}」")
                for h in run.get("history", []):
                    vc = h.get("vc_method", "?")
                    m = h.get("metrics", {})
                    brief = " | ".join(f"{k}={v}" for k, v in list(m.items())[:6])
                    lines.append(f"    VC={vc} | {brief}")
            hist_note = "\n".join(lines)

        user_msg = f"【目录信息】\n{context_info}\n"
        if hist_note:
            user_msg += f"【本轮已有结果】\n{hist_note}\n\n如果你已试过某些 VC 方法，请考虑是否还值得再试一次。除非用户明确要求，否则不要重复已试过的方法。\n\n"
        user_msg += f"【用户请求】\n{user_request}\n\n如果用户要求使用多种声线匿名方法，这不是你的任务，任选一种作为计划执行即可。请输出 JSON 执行计划：\n"

        self.logger.info("向 LLM 请求初始计划...")
        plan = self._call_llm_with_retry(plan_prompt, user_msg, validator=_validate_plan)
        self.logger.debug(f"[解析结果] plan = {json.dumps(plan, ensure_ascii=False, indent=2)}")

        err = _validate_plan(plan)
        if err:
            self.logger.error(f"初始计划解析失败: {err}")
            self._emit("thinking", "计划生成失败")
            return None

        steps = [_tool_label(s["tool"], s.get("params")) for s in plan.get("steps", [])]
        self._emit("thinking", f"计划: {' → '.join(steps)}\n理由: {plan.get('reasoning','')}")
        return plan

    # ── 步骤执行 ─────────────────────────────────────────────────────────────

    def _execute_step(self, step: dict):
        if self.stop_requested:
            self._emit("log", "用户终止，跳过执行")
            return False, {"error": "用户终止"}, 0
        tool_name = step.get("tool", "")
        params = step.get("params", {})
        label = _tool_label(tool_name, params)
        self._emit("step", {"status": "running", "name": label})
        ok, result, elapsed = super()._execute_step(step)
        if ok:
            self._emit("step", {"status": "done", "name": label, "elapsed": elapsed})
        else:
            self._emit("step", {"status": "fail", "name": label,
                                "error": result.get("error", "")[:100]})
        return ok, result, elapsed

    def _execute_plan(self, plan):
        self._emit("phase", "执行计划")
        steps = plan.get("steps", [])
        results = []
        eval_metrics = {}

        for i, step in enumerate(steps):
            if self.stop_requested:
                self._emit("log", "用户终止，停止执行计划")
                break
            tool_name = step["tool"]
            if step.get("_skipped"):
                continue
            label = _tool_label(tool_name, step.get("params"))
            self._emit("step", {"status": "running", "index": i + 1,
                                "total": len(steps), "name": label})

            from reflexion_agent_downstream import TOOL_BY_NAME
            tool = TOOL_BY_NAME.get(tool_name)
            if tool is None:
                self._emit("step", {"status": "fail", "index": i + 1,
                                    "total": len(steps), "name": label,
                                    "error": "未知工具"})
                results.append({"ok": False})
                continue

            t0 = time.time()
            try:
                result = tool.handler(self.ctx, **step.get("params", {}))
            except Exception as e:
                result = {"success": False, "error": str(e)}
            elapsed = time.time() - t0
            ok = result.get("success", False)

            if ok:
                self._emit("step", {"status": "done", "index": i + 1,
                                    "total": len(steps), "name": label,
                                    "elapsed": elapsed})
            else:
                err = result.get("error", "")[:100]
                self._emit("step", {"status": "fail", "index": i + 1,
                                    "total": len(steps), "name": label,
                                    "error": err})
            results.append({"ok": ok, "elapsed": elapsed, "result": result})

            if ok and tool_name == "run_evaluation":
                eval_metrics = result.get("mean_results", {})

        ok = all(r.get("ok", False) for r in results)
        return {"success": ok, "steps_results": results, "eval_metrics": eval_metrics}

    # ── 反思 ─────────────────────────────────────────────────────────────────

    def _reflect(self, user_request, current_metrics, history):
        self._emit("phase", "反思")
        self._emit("thinking", "正在分析指标，决定是否更换 VC 方法...")
        decision = super()._reflect(user_request, current_metrics, history)
        if decision.get("done"):
            self._emit("thinking", f"反思: 完成 — {decision.get('reason','')}")
        else:
            new_vc = decision.get("new_vc_method", "?")
            self._emit("thinking", f"反思: 换方法 → {new_vc} — {decision.get('reason','')}")
        return decision

    # ── 下游任务 ─────────────────────────────────────────────────────────────

    def _run_downstream_for_current_vc(self):
        self._emit("phase", "下游任务")
        self._emit("status", "正在跑下游评估...")
        self._emit("thinking", "下游任务执行中，请稍候...")
        metrics = super()._run_downstream_for_current_vc()
        if metrics:
            bacc = metrics.get("downstream_bacc", "N/A")
            f1 = metrics.get("downstream_f1", "N/A")
            self._emit("thinking", f"下游: bacc={bacc}, f1={f1}")
        else:
            self._emit("thinking", "下游任务无结果")
        return metrics

    # ── 主循环 ───────────────────────────────────────────────────────────────

    def run(self, user_request, max_reflections=3):
        self._emit("log", f"输入目录: {self.input_dir}")
        self._emit("log", f"输出目录: {self.output_dir}")
        self._emit("log", f"请求: {user_request}")
        self._emit("log", "")

        # ── 1. 生成初始计划 ─────────────────────────────────────────────────
        self._emit("phase", "生成计划")
        plan = self._generate_initial_plan(user_request)
        if plan is None:
            self._emit("log", "错误: 无法生成初始计划")
            self._emit("done", None)
            return None
        self._inject_extra_params(plan)

        language = self._ui_language or plan.get("language", "中文")
        steps = plan["steps"]
        initial_vc = None
        for s in steps:
            if s["tool"] == "run_voice_conversion":
                initial_vc = s["params"].get("method", "seedvc")
                break

        # ── 2. 执行初始计划 ─────────────────────────────────────────────────
        self._emit("phase", "执行计划")
        if self.stop_requested:
            self._emit("log", "用户终止")
            return {"success": False, "attempts": 1, "history": []}
        exec_result = self._execute_plan(plan)
        current_metrics = exec_result.get("eval_metrics", {})

        if not exec_result["success"]:
            self._emit("log", "初始计划执行失败")
            self._emit("done", None)
            return {"success": False, "attempts": 1, "history": []}

        if not current_metrics:
            self._emit("log", "未获得评估指标，无法进行反思")
            self._emit("metrics", {})
            self._emit("done", {"attempts": 1, "history": [{"vc_method": initial_vc, "metrics": {}}]})
            return {"success": True, "attempts": 1, "history": []}

        # ── 下游任务（初始）──
        if self.downstream_data_json and self.downstream_script:
            ds_metrics = self._run_downstream_for_current_vc()
            if ds_metrics:
                current_metrics.update(ds_metrics)

        # ── 3. Reflexion 循环 ───────────────────────────────────────────────
        history = [{"vc_method": initial_vc, "metrics": current_metrics}]
        self._emit("metrics", current_metrics)

        for reflection_round in range(max_reflections):
            if self.stop_requested:
                self._emit("log", "用户终止")
                break
            self._emit("phase", "反思")
            self._emit("status", f"反思 #{reflection_round + 1}/{max_reflections}")

            decision = self._reflect(user_request, current_metrics, history)
            self._emit("log", f"决策: {json.dumps(decision, ensure_ascii=False)}")

            if decision.get("done", True):
                break

            new_vc = decision.get("new_vc_method")
            if not new_vc or new_vc not in VC_METHODS:
                self._emit("log", f"无效 VC 方法: {new_vc}，终止")
                break

            tried = [h["vc_method"] for h in history]
            if new_vc in tried:
                self._emit("log", f"{new_vc} 已试过，终止")
                break

            # 重试：VC + 评估
            self._emit("phase", f"重试: {new_vc}")
            self._emit("log", f"更换 VC 方法: {history[-1]['vc_method']} → {new_vc}")

            retry_lang = self._ui_language or language
            vc_step = {"tool": "run_voice_conversion", "params": {"method": new_vc, "language": retry_lang}}
            eval_params = {"language": retry_lang}
            if self.gt_dir:
                eval_params["gt_dir"] = self.gt_dir
            if self.subject_map_path:
                eval_params["subject_map_path"] = self.subject_map_path
            eval_step = {"tool": "run_evaluation", "params": eval_params}

            ok_vc, _, _ = self._execute_step(vc_step)
            if ok_vc:
                ok_ev, ev_result, _ = self._execute_step(eval_step)
                if ok_ev:
                    current_metrics = ev_result.get("mean_results", {})
                else:
                    current_metrics = {}
            else:
                current_metrics = {}

            # 下游（重试后）
            if ok_vc and self.downstream_data_json and self.downstream_script:
                ds_metrics = self._run_downstream_for_current_vc()
                if ds_metrics:
                    current_metrics.update(ds_metrics)

            history.append({"vc_method": new_vc, "metrics": current_metrics})
            self._emit("metrics", current_metrics)

        # ── 完成 ────────────────────────────────────────────────────────────
        last = history[-1] if history else {}
        self._emit("metrics", last.get("metrics", {}))
        vc_list = [h["vc_method"] for h in history]
        self._emit("log", f"\n完成 | 共 {len(history)} 次尝试 | VC: {', '.join(vc_list)}")
        self._emit("done", {"history": history, "attempts": len(history)})
        return {"success": True, "history": history, "attempts": len(history)}


# ═══════════════════════════════════════════════════════════════════════════════
#  后台线程
# ═══════════════════════════════════════════════════════════════════════════════

def _run_agent_thread(input_dir, output_dir, middle_dir, llm_model,
                      mode, max_reflections, user_request, q,
                      downstream_data_json=None, downstream_script=None,
                      session_history=None,
                      gt_dir=None, subject_map_path=None,
                      language=None, max_speakers=None,
                      min_speakers=None, spk_num=None,
                      stop_event=None):
    def on_update(kind, data):
        q.put((kind, data))

    try:
        agent = UIReflexionAgent(
            input_dir=input_dir, output_dir=output_dir, middle_dir=middle_dir,
            llm_model=llm_model, mode=mode, on_update=on_update,
            session_history=session_history,
            ui_language=language, ui_max_speakers=max_speakers,
            ui_min_speakers=min_speakers, ui_spk_num=spk_num,
        )
        if downstream_data_json and downstream_script:
            agent.downstream_data_json = downstream_data_json
            agent.downstream_script = downstream_script
        if gt_dir:
            agent.gt_dir = gt_dir
        if subject_map_path:
            agent.subject_map_path = subject_map_path
        agent._stop_event = stop_event

        result = agent.run(user_request, max_reflections=max_reflections)
        if result is None:
            q.put(("done", None))
        else:
            q.put(("result", result))
    except Exception as e:
        import traceback
        q.put(("log", f"异常: {traceback.format_exc()}"))
        q.put(("done", None))


# ═══════════════════════════════════════════════════════════════════════════════
#  查找输出音频
# ═══════════════════════════════════════════════════════════════════════════════

def _find_output_audio(output_dir):
    """找到输出音频并复制到 /tmp 下（Gradio 只能访问 cwd 和 /tmp）。"""
    p = Path(output_dir)
    src = None
    for name in VC_METHODS:
        for prefix in ("text+", ""):
            d = p / f"{prefix}{name}"
            if d.is_dir():
                wavs = sorted(d.rglob("*.wav"))
                if wavs:
                    src = str(wavs[0])
                    break
        if src:
            break
    if not src:
        wavs = sorted(p.rglob("*.wav"))
        src = str(wavs[0]) if wavs else None
    if not src:
        return None
    # 复制到 /tmp 下，Gradio 可直接访问
    dst = os.path.join(tempfile.gettempdir(), f"anon_agent_{os.path.basename(src)}")
    shutil.copy2(src, dst)
    return dst


# ═══════════════════════════════════════════════════════════════════════════════
#  Hugging Face Spaces 部署指南
# ═══════════════════════════════════════════════════════════════════════════════

HF_DEPLOY_GUIDE = """
## 部署到 Hugging Face Spaces

### 前提条件
- 一个 Hugging Face 账号（https://huggingface.co/join）
- 已安装 `huggingface_hub`：`pip install huggingface_hub`
- 已登录：`huggingface-cli login`

### 步骤一：准备文件结构

在本地创建一个目录，例如 `hf_space/`，包含以下文件：

```
hf_space/
├── app.py              # 入口文件（见步骤二）
├── requirements.txt    # 依赖列表（见步骤三）
├── demo/
│   └── demo.py         # 本文件
├── reflexion_agent_downstream.py
├── agent.py
├── configs/
├── utils/
└── ...                 # 其他项目文件
```

### 步骤二：创建 app.py

在 `hf_space/` 根目录创建 `app.py`：

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Hugging Face Spaces 不支持 GPU 推理，需要连接远程 LLM
# 方案1：通过 API 调用（推荐）
os.environ["LLM_API_BASE"] = "http://your-server:8000/v1"
os.environ["LLM_API_KEY"] = "your-key"

# 方案2：使用免费的小模型 API（如 HuggingFace Inference API）
# os.environ["HF_TOKEN"] = "your-hf-token"

from demo.demo import build_ui

app = build_ui(default_llm="your-model-name", mode="INFO")
app.launch()
```

### 步骤三：创建 requirements.txt

```
gradio>=4.0
# 根据项目实际依赖添加，例如：
# torch
# torchaudio
# modelscope
# funasr
# whisperx
```

### 步骤四：在 Hugging Face 上创建 Space

1. 访问 https://huggingface.co/new-space
2. 填写 Space 名称（如 `voice-anonymization-agent`）
3. 选择 **Gradio** 作为 SDK
4. 选择硬件（免费 CPU 即可，LLM 推理通过 API 连接远程服务器）
5. 点击 "Create Space"

### 步骤五：上传文件

**方法 A：Git 推荐（适合大项目）**
```bash
cd hf_space
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial commit"
git push
```

**方法 B：网页上传**
- 进入 Space 页面 → "Files" → "Add file" → 上传文件

### 步骤六：配置 Secrets（敏感信息）

如果需要 API Key 等敏感信息：
1. 进入 Space → "Settings" → "Variables and secrets"
2. 点击 "New secret" 添加，例如：
   - `LLM_API_BASE` = 你的 LLM 服务地址
   - `LLM_API_KEY` = 你的 API Key
3. 在代码中通过 `os.environ["LLM_API_BASE"]` 读取

### 注意事项
- **GPU**：免费 Spaces 不提供 GPU。如果需要本地推理，需升级为 GPU Space（付费）。
- **大文件**：模型权重等大文件建议用 HuggingFace Hub 的模型仓库托管，不要直接放在 Space 里。
- **超时**：免费 Spaces 空闲 48 小时后会休眠，首次访问需要等待冷启动。
- **并发**：免费 Spaces 只有 2 个 vCPU / 16GB RAM，高并发场景需升级。
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

def build_ui(default_llm=None, mode="INFO"):

    with gr.Blocks(title="语音匿名化 Agent Demo") as app:

        gr.Markdown(
            "# 语音匿名化 Agent\n"
            "基于 Reflexion 架构：自动规划 → 执行 → 反思 → 调整 VC 方法 → 下游评估。"
        )

        # ── 输入 ────────────────────────────────────────────────────────────
        with gr.Row():
            input_dir = gr.Textbox(label="输入音频目录", placeholder="/path/to/audio", scale=3)
            llm_model = gr.Textbox(label="LLM 模型", value=default_llm or "", scale=1)

        with gr.Row():
            output_dir = gr.Textbox(label="输出目录（可选）", scale=2)
            middle_dir = gr.Textbox(label="中间目录（可选）", scale=2)

        with gr.Row():
            user_request = gr.Textbox(label="自然语言请求", lines=2, scale=4,
                                      placeholder="例如: 匿名化中文音频，不需要内容匿名，用seedvc方法")
            with gr.Column(scale=1):
                max_ref = gr.Slider(minimum=0, maximum=5, value=2, step=1,
                                    label="最大反思次数")

        with gr.Row():
            run_btn = gr.Button("开始执行", variant="primary", size="lg")
            stop_btn = gr.Button("停止", variant="stop", size="lg")

        with gr.Accordion("高级选项", open=False):
            with gr.Row():
                language = gr.Dropdown(
                    choices=["", "中文", "英文", "日文", "西班牙语", "法语", "德语", "韩语"],
                    value="", label="语言（留空=由LLM决定）", scale=1, allow_custom_value=True,
                )
                max_speakers = gr.Number(value=None, label="最大说话人数", minimum=1, maximum=20,
                                         precision=0, scale=1)
                min_speakers = gr.Number(value=None, label="最小说话人数", minimum=1, maximum=20,
                                         precision=0, scale=1)
                spk_num = gr.Number(value=None, label="单文件说话人数（精确）", minimum=1, maximum=20,
                                    precision=0, scale=1)
            with gr.Row():
                ds_data = gr.Textbox(label="下游 data.json", scale=2,
                                     placeholder="/path/to/all_data.json")
                ds_script = gr.Textbox(label="下游脚本", scale=2,
                                       placeholder="/path/to/run_downstream.sh")
            with gr.Row():
                gt_dir = gr.Textbox(label="GT 目录（EER评估用）", scale=2,
                                    placeholder="/path/to/gt_audio")
                subject_map = gr.Textbox(label="subject_map.json（EER评估用）", scale=2,
                                         placeholder="/path/to/subject_map.json")

        # ── 输出 ────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                phase_box = gr.Textbox(label="当前阶段", interactive=False, lines=1)
                status_box = gr.Textbox(label="状态", interactive=False, lines=2)
                thinking_box = gr.Textbox(label="Agent 思考", lines=8, max_lines=15,
                                          interactive=False, elem_classes=["log-box"])
                log_box = gr.Textbox(label="执行日志", lines=10, max_lines=20,
                                     interactive=False, elem_classes=["log-box"])
            with gr.Column(scale=1):
                metrics_json = gr.JSON(label="本次指标")
                output_audio = gr.Audio(label="匿名化音频", type="filepath")

        # ── Session 历史 ───────────────────────────────────────────────────
        history_display = gr.JSON(label="Session 历史（所有运行）")

        # ── Generator ───────────────────────────────────────────────────────

        def on_run(input_dir, output_dir, middle_dir, llm_model,
                   user_request, max_ref,
                   language, max_speakers, min_speakers, spk_num,
                   ds_data, ds_script, gt_dir, subject_map):
            global _SESSION_HISTORY
            _STOP_EVENT.clear()
            if not input_dir or not os.path.isdir(input_dir):
                yield ("", "输入目录不存在", "", "", None, None, _SESSION_HISTORY)
                return
            if not user_request or not user_request.strip():
                yield ("", "请输入请求", "", "", None, None, _SESSION_HISTORY)
                return
            if not llm_model or not llm_model.strip():
                yield ("", "请指定 LLM 模型", "", "", None, None, _SESSION_HISTORY)
                return

            output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(input_dir)), "output")
            middle_dir = middle_dir or os.path.join(os.path.dirname(os.path.abspath(input_dir)), "middle")

            ds_data_path = ds_data.strip() if ds_data and ds_data.strip() else None
            ds_script_path = ds_script.strip() if ds_script and ds_script.strip() else None

            gt_dir_path = gt_dir.strip() if gt_dir and gt_dir.strip() else None
            subject_map_path = subject_map.strip() if subject_map and subject_map.strip() else None

            language_val = language.strip() if language and language.strip() else None
            max_spk = int(max_speakers) if max_speakers else None
            min_spk = int(min_speakers) if min_speakers else None
            spk_n = int(spk_num) if spk_num else None

            q = queue.Queue()
            t = threading.Thread(
                target=_run_agent_thread,
                args=(os.path.abspath(input_dir), output_dir, middle_dir,
                      llm_model.strip(), mode, int(max_ref),
                      user_request.strip(), q),
                kwargs={"downstream_data_json": ds_data_path,
                        "downstream_script": ds_script_path,
                        "session_history": _SESSION_HISTORY,
                        "gt_dir": gt_dir_path,
                        "subject_map_path": subject_map_path,
                        "language": language_val,
                        "max_speakers": max_spk,
                        "min_speakers": min_spk,
                        "spk_num": spk_n,
                        "stop_event": _STOP_EVENT},
                daemon=True,
            )
            t.start()

            phase = "初始化..."
            status = "执行中..."
            thinking = ""
            log_lines = []
            metrics = None
            audio_path = None
            start_time = time.time()

            while True:
                try:
                    kind, data = q.get(timeout=0.5)
                except queue.Empty:
                    elapsed = time.time() - start_time
                    disp = f"等待LLM进程关闭中... ({elapsed:.0f}s)" if _STOP_EVENT.is_set() else f"{status} ({elapsed:.0f}s)"
                    yield (phase, disp, thinking,
                           "\n".join(log_lines), metrics, audio_path, _SESSION_HISTORY)
                    if not t.is_alive():
                        break
                    continue

                if kind == "phase":
                    phase = str(data)

                elif kind == "status":
                    status = str(data)

                elif kind == "thinking":
                    thinking = str(data)

                elif kind == "log":
                    log_lines.append(str(data))

                elif kind == "step":
                    if isinstance(data, dict):
                        idx = data.get("index")
                        total = data.get("total")
                        name = data.get("name", "?")
                        st = data.get("status", "")
                        prefix = f"[{idx}/{total}] " if idx is not None else ""
                        if st == "running":
                            status = f"{prefix}{name} 执行中..."
                            log_lines.append(f"  {prefix}{name} ...")
                        elif st == "done":
                            el = data.get("elapsed", 0)
                            status = f"{prefix}{name} 完成 ({el:.0f}s)"
                            log_lines[-1] = f"  {prefix}{name} 完成 ({el:.0f}s)"
                        elif st == "fail":
                            err = data.get("error", "")
                            status = f"{prefix}{name} 失败"
                            log_lines[-1] = f"  {prefix}{name} 失败: {err}"

                elif kind == "metrics":
                    metrics = data

                elif kind == "result":
                    # 追加到 session 历史
                    new_record = {
                        "time": time.strftime("%H:%M:%S"),
                        "request": user_request[:80],
                        "history": [],
                    }
                    for h in (data or {}).get("history", []):
                        m = h.get("metrics", {})
                        new_record["history"].append({
                            "vc_method": h.get("vc_method", "?"),
                            "metrics": {k: v for k, v in list(m.items())[:8]},
                        })
                    _SESSION_HISTORY = _SESSION_HISTORY + [new_record]

                    # 更新 UI
                    history = data.get("history", [])

                    if _STOP_EVENT.is_set():
                        phase = "已停止"
                        status = "用户终止"
                    elif history:
                        last = history[-1]
                        vc = last.get("vc_method", "?")
                        m = last.get("metrics", {})
                        status = f"完成 | VC={vc} | 共 {len(history)} 次"
                        if m:
                            status += f" | {', '.join(f'{k}={v}' for k,v in m.items())}"
                    else:
                        status = "完成"
                        phase = "完成"
                    audio_path = _find_output_audio(output_dir)
                    elapsed = time.time() - start_time
                    yield (phase, f"{status} ({elapsed:.0f}s)", thinking,
                           "\n".join(log_lines), metrics, audio_path, _SESSION_HISTORY)
                    return

                elapsed = time.time() - start_time
                yield (phase, f"{status} ({elapsed:.0f}s)", thinking,
                       "\n".join(log_lines), metrics, audio_path, _SESSION_HISTORY)

        run_btn.click(
            fn=on_run,
            inputs=[input_dir, output_dir, middle_dir, llm_model,
                    user_request, max_ref,
                    language, max_speakers, min_speakers, spk_num,
                    ds_data, ds_script, gt_dir, subject_map],
            outputs=[phase_box, status_box, thinking_box, log_box,
                     metrics_json, output_audio, history_display],
        )
        stop_btn.click(fn=lambda: _STOP_EVENT.set(), inputs=[], outputs=[], queue=False)

        # ── 示例 ────────────────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["/path/to/data/example/agent_zh_text_pumch/input", "", "",
                 "Qwen2.5-32B-Instruct",
                 "匿名化中文音频，不需要内容匿名，用seedvc方法", 2,
                 "", None, None, None, "", "", "", ""],
            ],
            inputs=[input_dir, output_dir, middle_dir, llm_model,
                    user_request, max_ref,
                    language, max_speakers, min_speakers, spk_num,
                    ds_data, ds_script, gt_dir, subject_map],
            label="示例",
        )

        # ── 部署指南 ──────────────────────────────────────────────────────────
        with gr.Accordion("如何部署到 Hugging Face Spaces", open=False):
            gr.Markdown(HF_DEPLOY_GUIDE)

    return app


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="语音匿名化 Agent Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--llm", default=None, help="默认 LLM 模型名")
    parser.add_argument("--mode", choices=["DEBUG", "INFO"], default="INFO")
    args = parser.parse_args()

    if args.llm:
        print(f"默认 LLM: {args.llm}")
    else:
        print("提示: 用 --llm 指定默认 LLM，或在 UI 中填写")

    app = build_ui(default_llm=args.llm, mode=args.mode)
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               theme=gr.themes.Soft(),
               css=".log-box textarea { font-family: monospace; font-size: 13px; }")


if __name__ == "__main__":
    main()
