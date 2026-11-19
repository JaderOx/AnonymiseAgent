"""Invalid model selections must return concrete options to the planner."""
import importlib
import inspect
import json
from unittest.mock import Mock

import pytest


@pytest.fixture(params=["casa", "casa_downstream"])
def runtime(request, tmp_path):
    module = importlib.import_module(request.param)
    kwargs = {"use_memory": False} if request.param == "casa_downstream" else {}
    agent = module.CASA(str(tmp_path / "input"), str(tmp_path / "out"),
                        str(tmp_path / "middle"), **kwargs)
    return module, agent


@pytest.mark.parametrize("tool,field,bad,good,choices", [
    ("run_transcription", "asr_model", "best_cer_model", "paraformer", ("paraformer", "whisperx")),
    ("run_transcription", "asr_model", ["paraformer"], "whisperx", ("paraformer", "whisperx")),
    ("run_voice_conversion", "method", "combine", "pitch", ("formant_lpc", "formant_spec", "pitch", "NeuralVC", "Resyn")),
    ("run_voice_conversion", "method", "combined", "pitch", ("formant_lpc", "formant_spec", "pitch", "NeuralVC", "Resyn")),
    ("run_voice_conversion", "method", "best_eer_model", "Resyn", ("formant_lpc", "formant_spec", "pitch", "NeuralVC", "Resyn")),
    ("run_mask", "mask_method", "silence", "mute", ("noise", "mute")),
    ("run_mask", "detect_method", "best_detector", "llm", ("llm", "ner")),
    ("check_directory", "file_type", "wav", "audio", ("audio", "json", "any")),
])
def test_initial_plan_reselects_with_available_choices(runtime, tool, field, bad, good, choices):
    module, agent = runtime
    def plan(value):
        return {"language": "中文", "steps": [{"tool": tool, "params": {"language": "中文", field: value}}]}
    agent._call_llm = Mock(side_effect=[json.dumps(plan(bad)), json.dumps(plan(good))])

    result = agent._generate_initial_plan("匿名化中文音频")

    assert result["steps"][0]["params"][field] == good
    assert agent._call_llm.call_count == 2
    retry_prompt = agent._call_llm.call_args_list[1].args[1]
    assert repr(bad) in retry_prompt
    assert "Available choices: " + ", ".join(choices) in retry_prompt
    initial_prompt = agent._call_llm.call_args_list[0].args[0]
    assert "paraformer, whisperx" in initial_prompt
    assert "combined" not in initial_prompt


@pytest.mark.parametrize("params", [{}, {"asr_model": ""}, {"asr_model": "paraformer"}, {"asr_model": "whisperx"}])
def test_valid_asr_selection_needs_no_retry(runtime, params):
    module, agent = runtime
    plan = {"steps": [{"tool": "run_transcription", "params": {"language": "中文", **params}}]}
    agent._call_llm = Mock(return_value=json.dumps(plan))
    assert agent._generate_initial_plan("转录中文音频") == plan
    agent._call_llm.assert_called_once()


@pytest.mark.parametrize("done,key", [(False, "new_vc_method"), (True, "best_vc_method")])
@pytest.mark.parametrize("invalid", ["combined", "best_eer_model", ["pitch"]])
def test_reflection_reselects_invalid_method(runtime, done, key, invalid):
    module, agent = runtime
    agent._call_llm = Mock(side_effect=[
        json.dumps({"done": done, key: invalid}),
        json.dumps({"done": done, key: "pitch"}),
    ])
    history = [{"vc_method": "pitch", "vc_params": {"n_steps": -3.0}, "metrics": {"eer": 40}}]

    result = agent._reflect("提高匿名程度", {"eer": 40}, history)

    assert result[key] == "pitch"
    assert agent._call_llm.call_count == 2
    retry_prompt = agent._call_llm.call_args_list[1].args[1]
    assert repr(invalid) in retry_prompt
    assert "Available choices: " + ", ".join(module.VC_METHODS) in retry_prompt
    assert "combined" not in agent._call_llm.call_args_list[0].args[0]


def test_invalid_selections_stop_at_existing_retry_limit(runtime):
    module, agent = runtime
    bad = {"steps": [{"tool": "run_transcription", "params": {"asr_model": "best_cer_model"}}]}
    agent._call_llm = Mock(return_value=json.dumps(bad))
    assert agent._generate_initial_plan("转录") is None
    assert agent._call_llm.call_count == 3


def test_unknown_tool_gets_available_tool_names(runtime):
    module, agent = runtime
    bad = {"steps": [{"tool": "best_transcriber", "params": {}}]}
    good = {"steps": [{"tool": "run_transcription", "params": {"language": "中文"}}]}
    agent._call_llm = Mock(side_effect=[json.dumps(bad), json.dumps(good)])
    assert agent._generate_initial_plan("转录")["steps"][0]["tool"] == "run_transcription"
    assert "Available choices: " + ", ".join(module.TOOL_BY_NAME) in agent._call_llm.call_args_list[1].args[1]


def test_vc_choices_are_consistent_and_exclude_combined():
    import agent
    import casa
    import casa_downstream
    import voice_convert
    from utils.vc_names import METHODS
    from utils.vc_parameters import VC_DEFAULTS, VC_PARAMETER_GUIDE
    from utils.voice_convert import list_methods

    expected = {"formant_lpc", "formant_spec", "pitch", "NeuralVC", "Resyn"}
    for choices in (METHODS, VC_DEFAULTS, agent._VC_CONDA_ENV, casa.VC_METHODS,
                    casa_downstream.VC_METHODS, list_methods()):
        assert set(choices) == expected
    assert "combined" not in VC_PARAMETER_GUIDE
    assert inspect.signature(voice_convert.convert_audio).parameters["method"].default in expected
    assert inspect.signature(voice_convert.batch_convert).parameters["method"].default in expected
