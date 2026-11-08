"""Local Gradio demo using the same memory-enabled runtime as the CLI."""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import gradio as gr
from reflexion_agent_downstream import ReflexionAgent


class UIReflexionAgent(ReflexionAgent):
    """Forward runtime events without maintaining a second execution loop."""
    def __init__(self, *args, on_update=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_update = on_update or (lambda *args: None)

    def _generate_initial_plan(self, request):
        if self.memory:
            self.on_update('memory', self.memory.recall_for_prompt(task_meta=self.task_meta))
        plan = super()._generate_initial_plan(request)
        self.on_update('plan', plan)
        return plan

    def _execute_step(self, step):
        self.on_update('step', step)
        result = super()._execute_step(step)
        self.on_update('step_result', {'success': result[0], 'seconds': result[2]})
        return result

    def _reflect(self, request, metrics, history, baseline_ds=None):
        self.on_update('metrics', metrics)
        decision = super()._reflect(request, metrics, history, baseline_ds)
        self.on_update('reflection', decision)
        return decision


def _find_output_audio(output_dir, method):
    """Preview only the selected method, never an arbitrary earlier attempt."""
    if not method:
        return None
    for name in (method, f'text_{method}', f'text+{method}'):
        directory = Path(output_dir) / name
        files = sorted(p for p in directory.rglob('*')
                       if p.is_file() and p.suffix.lower() in {'.wav', '.flac', '.mp3', '.ogg'})
        if files:
            dest = Path(tempfile.mkdtemp(prefix='anonymise-preview-')) / files[0].name
            shutil.copy2(files[0], dest)
            return str(dest)
    return None


def run_demo(input_dir, request, llm, local_llm, use_memory, memory_path,
             output_dir, middle_dir, max_reflections, downstream_data, downstream_script,
             gt_dir, subject_map):
    input_path = Path(input_dir).expanduser()
    if not input_path.is_dir():
        raise gr.Error('请输入服务器上存在的音频目录。')
    if not request.strip():
        raise gr.Error('请填写匿名化需求。')
    if bool(downstream_data.strip()) != bool(downstream_script.strip()):
        raise gr.Error('下游数据 JSON 和脚本需要一起填写。')
    output_dir = output_dir.strip() or str(input_path.resolve().parent / 'anonymise_demo' / 'output')
    middle_dir = middle_dir.strip() or str(Path(output_dir).parent / 'middle')
    events = queue.Queue()
    lines, recalled = [], ''

    def worker():
        try:
            agent = UIReflexionAgent(
                input_dir=str(input_path.resolve()), output_dir=output_dir, middle_dir=middle_dir,
                llm_model=llm.strip() or None, open_source=local_llm,
                use_memory=use_memory, memory_path=memory_path.strip() or str(ROOT / 'memory'),
                on_update=lambda kind, data: events.put((kind, data)),
            )
            agent.downstream_data_json = downstream_data.strip() or None
            agent.downstream_script = downstream_script.strip() or None
            agent.gt_dir = gt_dir.strip() or None
            agent.subject_map_path = subject_map.strip() or None
            result = agent.run(request, max_reflections=int(max_reflections))
            events.put(('result', result))
        except Exception as exc:
            events.put(('error', str(exc)))

    threading.Thread(target=worker, daemon=True).start()
    yield '正在运行…', recalled, '', {}, None
    while True:
        try:
            kind, data = events.get(timeout=1)
        except queue.Empty:
            continue
        if kind == 'error':
            raise gr.Error(data)
        if kind == 'memory':
            recalled = data or '暂无匹配经验。'
        elif kind == 'result':
            result = data or {'success': False, 'error': '未生成有效计划'}
            method = result.get('final_vc_method')
            status = f'完成：{method or "未选择方法"}' if result.get('success') else '运行失败，请查看日志'
            lines.append('结果已保存到 ' + output_dir)
            yield status, recalled, '\n'.join(lines), result, _find_output_audio(output_dir, method)
            return
        else:
            lines.append(f'{kind}: {json.dumps(data, ensure_ascii=False, indent=2)}')
        yield f'正在运行：{kind}', recalled, '\n'.join(lines), {}, None


def build_ui(default_llm=None, mode='INFO'):
    with gr.Blocks(title='语音匿名化 Agent · Memory') as app:
        gr.Markdown('# 语音匿名化 Agent\n自动规划 → 执行 → 评估 → 反思，并跨运行积累经验。')
        gr.Markdown('输入服务器上的音频目录。需先按 README 准备模型与环境；在线 API 仅替代决策 LLM。')
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label='输入音频目录', placeholder='/path/to/audio')
                request = gr.Textbox(label='匿名化需求', lines=4,
                    value='语言为中文，说话人数量为1。不需要内容匿名，希望匿名后尽量保留韵律。')
                llm = gr.Textbox(label='LLM 模型名', value=default_llm or os.environ.get('AGENT_LLM_MODEL', 'Qwen2.5-32B-Instruct'))
                local = gr.Checkbox(label='使用本地 LLM（取消则使用在线 API）', value=True)
                memory = gr.Checkbox(label='使用并更新跨运行 Memory', value=True)
                memory_path = gr.Textbox(label='Memory 目录', value=str(ROOT / 'memory'))
                max_ref = gr.Slider(0, 10, value=3, step=1, label='最大反思次数')
                with gr.Accordion('输出与评估设置', open=False):
                    output_dir = gr.Textbox(label='输出目录（留空自动创建）')
                    middle_dir = gr.Textbox(label='中间结果目录（留空自动创建）')
                    downstream_data = gr.Textbox(label='下游数据 JSON（可选）')
                    downstream_script = gr.Textbox(label='下游评估脚本（可选）')
                    gt_dir = gr.Textbox(label='真实转录目录（可选）')
                    subject_map = gr.Textbox(label='说话人映射 JSON（可选）')
                run = gr.Button('开始匿名化', variant='primary')
            with gr.Column():
                status = gr.Textbox(label='状态', interactive=False)
                recalled = gr.Textbox(label='本次召回的历史经验', lines=10, interactive=False)
                trace = gr.Textbox(label='计划、执行与反思', lines=12, interactive=False)
                result = gr.JSON(label='最终选择与指标')
                audio = gr.Audio(label='最终选定方法的音频预览')
        run.click(run_demo, inputs=[input_dir, request, llm, local, memory, memory_path,
                  output_dir, middle_dir, max_ref, downstream_data, downstream_script, gt_dir, subject_map],
                  outputs=[status, recalled, trace, result, audio], concurrency_limit=1)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--llm', default=None)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--mode', choices=['DEBUG', 'INFO'], default='INFO')
    args = parser.parse_args()
    build_ui(args.llm, args.mode).queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
