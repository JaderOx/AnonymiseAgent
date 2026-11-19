"""tests/test_final_vc.py — 最终方法裁决语义单测：LLM done=true 的 best_vc_method
优先于"最后尝试的方法"（2026-10-20 Saar-HD 运行暴露：探索终点 formant_lpc ≠
裁决最优 NeuralVC，旧逻辑记了前者）。

运行: python -m pytest tests/test_final_vc.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.agent_selection import final_vc_from as _final_vc_from  # noqa: E402

HIST = [
    {"vc_method": "NeuralVC", "metrics": {"eer": 51.0}, "reason": ""},
    {"vc_method": "Resyn", "metrics": {"eer": 51.7}, "reason": ""},
    {"vc_method": "formant_lpc", "metrics": {"eer": 25.8}, "reason": ""},
]


def test_no_decision_falls_back_to_last():
    assert _final_vc_from(None, HIST) == "formant_lpc"


def test_decision_without_best_falls_back_to_last():
    # 旧 prompt 的 done=true 只有 reason/summary，无 best_vc_method
    d = {"done": True, "reason": "完成", "summary": ""}
    assert _final_vc_from(d, HIST) == "formant_lpc"


def test_best_in_history_wins():
    d = {"done": True, "reason": "NeuralVC 最佳平衡", "best_vc_method": "NeuralVC"}
    assert _final_vc_from(d, HIST) == "NeuralVC"


def test_best_not_in_history_falls_back_to_last():
    # LLM 幻觉出未尝试过的方法 → 不能采用（其指标不存在）
    d = {"done": True, "reason": "pitch 最好", "best_vc_method": "pitch"}
    assert _final_vc_from(d, HIST) == "formant_lpc"


def test_rollback_branch_format():
    # 回退分支构造的裁决 {"best_vc_method": new_vc}
    d = {"best_vc_method": "Resyn"}
    assert _final_vc_from(d, HIST) == "Resyn"


def test_empty_history_returns_none():
    assert _final_vc_from(None, []) is None
    assert _final_vc_from({"best_vc_method": "NeuralVC"}, []) is None
