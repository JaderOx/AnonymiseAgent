"""Tool dispatch filters arguments against the actual handler signature."""
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest


@pytest.mark.parametrize("module_name", ["casa", "casa_downstream"])
@pytest.mark.parametrize("tool_name,params,expected", [
    ("run_evaluation", {"language": "中文", "gt_dir": "truth", "subject_map_path": "subjects.json",
                        "with_mask": True, "extra_argument": 1, "ctx": "untrusted"},
     {"language": "中文", "gt_dir": "truth", "subject_map_path": "subjects.json"}),
    ("run_voice_conversion", {"method": "Resyn", "language": "中文", "with_mask": True,
                              "extra_argument": 1},
     {"method": "Resyn", "language": "中文", "with_mask": True}),
    ("check_directory", {"directory": "audio", "extra_argument": 1}, {"directory": "audio"}),
])
def test_dispatch_drops_only_undeclared_arguments(module_name, tool_name, params, expected, monkeypatch):
    module = importlib.import_module(module_name)
    tool = module.TOOL_BY_NAME[tool_name]
    handler = create_autospec(tool.handler, return_value={"success": True})
    monkeypatch.setattr(tool, "handler", handler)
    agent = module.CASA.__new__(module.CASA)
    agent.ctx = SimpleNamespace()
    agent.logger = Mock()
    original_params = dict(params)

    ok, result, elapsed = agent._execute_step({"tool": tool_name, "params": params})

    if module_name == "casa_downstream" and tool_name == "run_evaluation":
        expected = {**expected, "with_mask": True}
    assert ok is True
    assert result == {"success": True}
    assert elapsed >= 0
    handler.assert_called_once_with(agent.ctx, **expected)
    agent.logger.warning.assert_called_once()
    assert params == original_params


@pytest.mark.parametrize("module_name", ["casa", "casa_downstream"])
def test_dispatch_preserves_valid_arguments_and_defaults(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    tool = module.TOOL_BY_NAME["run_evaluation"]
    handler = create_autospec(tool.handler, return_value={"success": True})
    monkeypatch.setattr(tool, "handler", handler)
    agent = module.CASA.__new__(module.CASA)
    agent.ctx = SimpleNamespace()
    agent.logger = Mock()

    ok, _, _ = agent._execute_step({"tool": "run_evaluation", "params": {"language": "中文"}})

    assert ok is True
    handler.assert_called_once_with(agent.ctx, language="中文")
    agent.logger.warning.assert_not_called()
