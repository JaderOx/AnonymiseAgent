"""Gradio interface for running CASA on a server-side audio directory."""
import argparse
from functools import partial
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
from casa_downstream import CASA

DEFAULT_REQUEST = """- These are English recordings. Use one speaker per recording.
- The recordings contain sensitive personal information. Detect and mask every spoken identifier in the audio, including names, phone numbers, email addresses, street addresses, dates of birth, and personal or medical record numbers. Preserve the rest of the wording.
- Anonymize the speaker's voice. Preserve intelligibility and prosody as much as possible.
- Evaluate the final recording after both content masking and voice anonymization. Report the available privacy and speech-quality metrics."""

EVENT_LABELS = {
    'memory': 'Recalling experience',
    'plan': 'Execution plan',
    'step': 'Running tool',
    'step_result': 'Tool result',
    'metrics': 'Evaluation metrics',
    'reflection': 'Reflection',
}


class DemoCASA(CASA):
    """Forward events from the same runtime used by the command line."""

    focused_memory = True
    def __init__(self, *args, on_update=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_update = on_update or (lambda *args: None)

    def _generate_initial_plan(self, request):
        if self.memory:
            experiences, runs, _, related = self.memory.recall_records(
                self.task_meta, self._memory_exclude, focused=self.focused_memory)
            recalled = {
                'Task experiences': [
                    {'Dataset': exp.get('dataset'), 'Task': exp.get('task_type'),
                     'Recommended method': exp.get('recommended_method'),
                     'Methods to avoid': exp.get('avoid_methods') or [],
                     'Source lesson (original language)': exp.get('lesson')}
                    for exp in experiences
                ],
                'Related runs' if related else 'Recent runs': [
                    {'Dataset': run.get('dataset'), 'Selected method': run.get('final_vc_method'),
                     'Metrics': run.get('vc_results', {}).get(run.get('final_vc_method'))}
                    for run in runs
                ],
            }
            if not experiences and not runs:
                recalled = {'Status': 'No matching cross-run experience for this request.'}
            self.on_update('memory', recalled)
        plan = super()._generate_initial_plan(request)
        if plan and self.open_source:
            for step in plan['steps']:
                if step['tool'] == 'run_mask':
                    params = step.setdefault('params', {})
                    if not params.get('llm_model'):
                        params['llm_model'] = self.llm_model
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


def _find_output_audio(selected_dir):
    """Copy a preview from the exact conversion selected by CASA."""
    if not selected_dir:
        return None
    files = sorted(p for p in Path(selected_dir).rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.wav', '.flac', '.mp3', '.ogg', '.aiff'})
    if files:
        dest = Path(tempfile.mkdtemp(prefix='casa-preview-')) / files[0].name
        shutil.copy2(files[0], dest)
        return str(dest)
    return None


def run_demo(input_dir, request, llm, local_llm, use_memory, memory_path,
             output_dir, middle_dir, max_reflections, downstream_data, downstream_script,
             gt_dir, subject_map, *, mode="INFO"):
    input_path = Path(input_dir.strip()).expanduser().resolve()
    if not input_dir.strip() or not input_path.is_dir():
        raise gr.Error('Enter an existing audio directory on the server.')
    if not request.strip():
        raise gr.Error('Describe your anonymization requirements.')
    if bool(downstream_data.strip()) != bool(downstream_script.strip()):
        raise gr.Error('Provide both the downstream data JSON and its evaluation script.')
    output_dir = output_dir.strip() or str(input_path.parent / 'casa_demo' / input_path.name / 'output')
    middle_dir = middle_dir.strip() or str(Path(output_dir).parent / 'middle')
    events = queue.Queue()
    lines, recalled = [], {}

    def worker():
        try:
            agent = DemoCASA(
                input_dir=str(input_path), output_dir=output_dir, middle_dir=middle_dir,
                llm_model=llm.strip() or None, open_source=local_llm,
                mode=mode, use_memory=use_memory, memory_path=memory_path.strip() or str(ROOT / 'memory'),
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
    yield 'Running…', recalled, '', {}, None
    while True:
        kind, data = events.get()
        if kind == 'error':
            raise gr.Error(f'CASA could not complete the run. {data}')
        if kind == 'memory':
            recalled = data
        elif kind == 'result':
            result = data or {'success': False, 'error': 'Could not generate an execution plan.'}
            method = result.get('final_vc_method')
            status = f'Completed · {method}' if method else 'Completed'
            if not result.get('success'):
                status = 'Run failed. See the output log for details.'
            lines.append(f'Output directory: {output_dir}')
            preview = _find_output_audio(result.get('final_audio_dir'))
            yield status, recalled, '\n'.join(lines), result, preview
            return
        else:
            lines.append(f'{EVENT_LABELS[kind]}\n{json.dumps(data, ensure_ascii=False, indent=2)}')
        yield EVENT_LABELS[kind], recalled, '\n'.join(lines), {}, None


def build_ui(default_llm=None, mode='INFO'):
    with gr.Blocks(title='CASA · Speech Anonymization') as app:
        gr.Markdown('# CASA · Speech Anonymization')
        gr.Markdown('Enter a server-side audio directory and describe what the anonymized speech should preserve.')
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label='Input audio directory', placeholder='/path/to/audio')
                request = gr.Textbox(label='Anonymization requirements', lines=9,
                    value=DEFAULT_REQUEST)
                llm = gr.Textbox(label='Decision model', value=default_llm or os.environ.get('AGENT_LLM_MODEL', 'Qwen2.5-32B-Instruct'))
                local = gr.Checkbox(label='Use a local LLM (clear to use the API)', value=True)
                memory = gr.Checkbox(label='Recall and update cross-run memory', value=True)
                memory_path = gr.Textbox(label='Memory directory', value=str(ROOT / 'memory'))
                max_ref = gr.Slider(0, 10, value=3, step=1, label='Maximum reflection rounds')
                with gr.Accordion('Output and evaluation settings', open=False):
                    output_dir = gr.Textbox(label='Output directory (optional)')
                    middle_dir = gr.Textbox(label='Intermediate files directory (optional)')
                    downstream_data = gr.Textbox(label='Downstream data JSON (optional)')
                    downstream_script = gr.Textbox(label='Downstream evaluation script (optional)')
                    gt_dir = gr.Textbox(label='Privacy-word annotations directory (optional)')
                    subject_map = gr.Textbox(label='Speaker mapping JSON (optional)')
                run = gr.Button('Run anonymization', variant='primary')
            with gr.Column():
                status = gr.Textbox(label='Status', interactive=False)
                recalled = gr.JSON(label='Recalled experience', open=1)
                trace = gr.Textbox(label='Plan, tool calls and reflections', lines=12, interactive=False)
                result = gr.JSON(label='Selected method, parameters and metrics')
                audio = gr.Audio(label='Selected audio preview')
        run.click(partial(run_demo, mode=mode), inputs=[input_dir, request, llm, local, memory, memory_path,
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
