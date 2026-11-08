"""Runtime contracts without model downloads, GPU work, or API calls."""
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('module_name', ['reflexion_agent', 'reflexion_agent_downstream', 'reflexion_agent_sentence'])
@pytest.mark.parametrize('decision', [
    {'done': True, 'best_vc_method': 'NeuralVC'},
    {'done': False, 'new_vc_method': 'NeuralVC'},
])
def test_run_uses_selected_metrics_and_records_memory(module_name, decision, tmp_path):
    module = importlib.import_module(module_name)
    agent = module.ReflexionAgent(str(tmp_path/'input'), str(tmp_path/'output'), str(tmp_path/'middle'))
    if hasattr(agent, 'memory'):
        agent.memory = Mock()
        agent._classify_request = Mock(return_value={'task_type': 'other'})
    agent._generate_initial_plan = Mock(return_value={
        'language': '中文', 'steps': [{'tool': 'run_voice_conversion', 'params': {'method': 'NeuralVC'}}]})
    agent._execute_plan = Mock(return_value={'success': True, 'eval_metrics': {'eer': 49}})
    agent._reflect = Mock(side_effect=[{'done': False, 'new_vc_method': 'pitch'}, decision])
    agent._execute_step = Mock(side_effect=[(True, {}, 0), (True, {'mean_results': {'eer': 20}}, 0)])
    result = agent.run('中文，单说话人，不需要内容匿名', max_reflections=2)
    assert result['final_vc_method'] == 'NeuralVC'
    assert result['final_metrics'] == {'eer': 49}
    assert agent.ctx.vc_method_used == 'NeuralVC'
    if hasattr(agent, 'memory'):
        args = agent.memory.record_run.call_args.kwargs
        assert args['final_vc_method'] == 'NeuralVC'
        assert args['final_metrics'] == {'eer': 49}
        saved = json.loads((tmp_path/'output'/'agent_result.json').read_text())
        assert saved['final_vc_method'] == 'NeuralVC'


def test_bundled_memory_loads_from_other_working_directory(tmp_path, monkeypatch):
    from reflexion_agent_downstream import ReflexionAgent
    monkeypatch.chdir(tmp_path)
    agent = ReflexionAgent(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'middle'))
    assert len(agent.memory.experiences) == 14
    assert len(agent.memory.runs) == 16
    assert agent.memory.recall_for_prompt({'task_type': 'dementia'})


def test_demo_uses_runtime_memory_and_selected_audio(tmp_path, monkeypatch):
    import demo.demo as demo
    original = demo.UIReflexionAgent.run
    assert original is demo.ReflexionAgent.run
    memory_root = tmp_path/'memory'
    events = []
    agent = demo.UIReflexionAgent(str(tmp_path), str(tmp_path/'out'), str(tmp_path/'middle'),
                                  memory_path=str(memory_root), on_update=lambda *args: events.append(args))
    agent._call_llm_with_retry = Mock(return_value={'language': '中文', 'steps': [
        {'tool': 'run_voice_conversion', 'params': {'method': 'NeuralVC'}}]})
    agent._generate_initial_plan('中文，不需要内容匿名')
    assert any(kind == 'memory' for kind, _ in events)
    for method in ['NeuralVC', 'pitch']:
        path=tmp_path/'out'/method/'sample.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(method.encode())
    preview=demo._find_output_audio(tmp_path/'out', 'pitch')
    assert Path(preview).read_bytes() == b'pitch'
    assert demo._find_output_audio(tmp_path/'out', 'Resyn') is None


def test_method_aliases_work_in_cli_and_plans(tmp_path):
    import subprocess
    import sys
    from reflexion_agent_downstream import _validate_plan
    plan={'language': '中文', 'steps': [{'tool':'run_voice_conversion','params':{'method':'seedvc'}}]}
    assert _validate_plan(plan) is None
    assert plan['steps'][0]['params']['method'] == 'NeuralVC'
    cli=subprocess.run([sys.executable, 'agent.py', '--vc_method', 'seedvc', '--help'], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr


@pytest.mark.parametrize('module_name', ['reflexion_agent_downstream', 'reflexion_agent_sentence'])
def test_mask_plan_never_reuses_unmasked_cache(module_name, tmp_path):
    module=importlib.import_module(module_name)
    agent=module.ReflexionAgent(str(tmp_path/'input'),str(tmp_path/'out'),str(tmp_path/'mid'),use_memory=False)
    (tmp_path/'out'/'evaluate_NeuralVC.csv').write_text('name,eer\n-mean-,45\n')
    agent._execute_step=Mock(return_value=(True, {'mean_results': {'eer': 49}}, 0))
    plan={'steps': [
        {'tool':'run_mask','params':{}},
        {'tool':'run_voice_conversion','params':{'method':'NeuralVC'}},
        {'tool':'run_evaluation','params':{}},
    ]}
    result=agent._execute_plan(plan)
    assert agent._execute_step.call_count==3
    assert result['eval_metrics']=={'eer':49}
    assert plan['steps'][1]['params']['with_mask'] is True
    assert plan['steps'][2]['params']['with_mask'] is True
