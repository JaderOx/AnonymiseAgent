"""tests/test_reflex_cascade.py — 句级 reflex 级联纯逻辑单测（无 GPU、无音频模型）。

运行: python -m pytest tests/test_reflex_cascade.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reflexion_agent_sentence import (  # noqa: E402
    MIN_SCORES_FOR_FAIL_DETECT,
    _cascade_assign,
    _compute_fail_stats,
    _is_fail_score,
    _keyed_audio_files,
    _rank_methods_for_cascade,
)


# ── _compute_fail_stats ───────────────────────────────────────────────────────

def test_compute_fail_stats_basic():
    # [0.1, 0.2, 0.3, 0.4]: q25=0.175, q75=0.325 (线性插值), mean=0.25, IQR=0.15
    stats = _compute_fail_stats([0.4, 0.1, 0.3, 0.2], iqr_k=2.0)
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(0.25)
    assert stats["q25"] == pytest.approx(0.175)
    assert stats["q75"] == pytest.approx(0.325)
    assert stats["iqr"] == pytest.approx(0.15)
    assert stats["threshold"] == pytest.approx(0.25 + 2.0 * 0.15)


def test_compute_fail_stats_empty():
    assert _compute_fail_stats([], iqr_k=2.0) == {}


def test_compute_fail_stats_iqr_zero():
    # 全部分数相同 → IQR=0，threshold=mean → score>mean 恒假，天然无 fail
    stats = _compute_fail_stats([0.3] * 5, iqr_k=2.0)
    assert stats["iqr"] == 0.0
    assert stats["threshold"] == pytest.approx(0.3)


# ── _is_fail_score ────────────────────────────────────────────────────────────

def test_is_fail_score_threshold_and_floor():
    th, floor = 0.55, 0.5
    assert _is_fail_score(0.6, th, floor)          # 超 IQR 阈值且超 floor → fail
    assert not _is_fail_score(0.52, th, floor)     # 超 IQR 阈值但没超 floor → 不算
    assert not _is_fail_score(0.4, th, floor)      # 没超 IQR 阈值 → 不算


def test_is_fail_score_no_floor():
    assert _is_fail_score(0.56, 0.55, None)        # floor 禁用时只看阈值
    assert not _is_fail_score(0.55, 0.55, None)    # 等于阈值不算（严格大于）


# ── _cascade_assign ───────────────────────────────────────────────────────────

METHODS = ["A", "B", "C"]


def test_cascade_assign_first_pass_stays():
    rounds = [{"f1": 0.2, "f2": 0.8, "f3": 0.9},
              {"f2": 0.3, "f3": 0.7},
              {"f3": 0.6}]
    out = _cascade_assign(rounds, METHODS, threshold=0.55, floor=None,
                          detection_skipped=False)["assignment"]
    assert out["f1"] == {"method": "A", "score": 0.2, "status": "pass",
                         "tried": {"A": 0.2}}
    assert out["f2"]["method"] == "B" and out["f2"]["status"] == "pass"
    assert out["f2"]["tried"] == {"A": 0.8, "B": 0.3}


def test_cascade_assign_leftover_takes_lowest():
    # f3 三轮全失败 → 取 tried 中最低分（C=0.6）
    rounds = [{"f3": 0.9}, {"f3": 0.7}, {"f3": 0.6}]
    out = _cascade_assign(rounds, METHODS, threshold=0.55, floor=None,
                          detection_skipped=False)["assignment"]
    assert out["f3"]["status"] == "best"
    assert out["f3"]["method"] == "C"
    assert out["f3"]["score"] == pytest.approx(0.6)


def test_cascade_assign_missing_output_escalates():
    # f2 在 A 轮没打上分（方法目录缺文件）→ 不分配 A，B 轮通过则归 B
    rounds = [{"f1": 0.2}, {"f2": 0.3}]
    out = _cascade_assign(rounds, METHODS, threshold=0.55, floor=None,
                          detection_skipped=False)["assignment"]
    assert out["f2"]["method"] == "B" and out["f2"]["status"] == "pass"
    assert out["f2"]["tried"] == {"B": 0.3}


def test_cascade_assign_never_scored_not_in_assignment():
    # 从未打上分的文件不出现在 assignment（由调用方归入 unassigned）
    rounds = [{"f1": 0.2}]
    out = _cascade_assign(rounds, METHODS, threshold=0.55, floor=None,
                          detection_skipped=False)["assignment"]
    assert "ghost" not in out


def test_cascade_assign_detection_skipped_all_stay_first():
    # 样本太少跳过判定 → 已打分文件全部留在首个方法
    rounds = [{"f1": 0.9, "f2": 0.95}]
    out = _cascade_assign(rounds, METHODS, threshold=None, floor=0.5,
                          detection_skipped=True)["assignment"]
    assert all(v["method"] == "A" and v["status"] == "pass"
               for v in out.values())


def test_min_scores_constant():
    assert MIN_SCORES_FOR_FAIL_DETECT == 4


# ── _keyed_audio_files ────────────────────────────────────────────────────────

def test_keyed_audio_files_rel_stem_with_subdir(tmp_path):
    (tmp_path / "healthy").mkdir()
    (tmp_path / "Hyperfunktionelle_Dysphonie").mkdir()
    (tmp_path / "healthy" / "1002__1002-phrase.wav").write_bytes(b"x")
    (tmp_path / "Hyperfunktionelle_Dysphonie" / "1037__1037-phrase.wav").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("not audio")
    keyed = _keyed_audio_files(tmp_path)
    assert set(keyed) == {"healthy/1002__1002-phrase",
                          "Hyperfunktionelle_Dysphonie/1037__1037-phrase"}
    assert keyed["healthy/1002__1002-phrase"].suffix == ".wav"


def test_keyed_audio_files_missing_dir(tmp_path):
    assert _keyed_audio_files(tmp_path / "nope") == {}


# ── _rank_methods_for_cascade ─────────────────────────────────────────────────

def test_rank_methods_by_downstream_bacc(tmp_path):
    import json
    import logging
    import reflexion_agent_sentence as ras

    out = tmp_path / "output"
    out.mkdir()
    (out / "agent_result.json").write_text(json.dumps({"final_vc_method": "pitch"}))
    # pitch=A；下游 bacc：NeuralVC 0.65 > formant_spec 0.60 > Resyn 0.49；
    # formant_lpc 无缓存 → 垫后
    for m, bacc in [("NeuralVC", 0.65), ("formant_spec", 0.60), ("Resyn", 0.49)]:
        d = out / m / "downstream_output"
        d.mkdir(parents=True)
        (d / "final_result.json").write_text(
            json.dumps({"balanced_accuracy": {"mean": bacc}}))

    a, order = _rank_methods_for_cascade(str(out), logging.getLogger("t"))
    assert a == "pitch"
    assert order == ["pitch", "NeuralVC", "formant_spec", "Resyn", "formant_lpc"]


def test_rank_methods_requires_agent_result(tmp_path):
    import logging
    import reflexion_agent_sentence as ras
    with pytest.raises(RuntimeError, match="agent_result"):
        _rank_methods_for_cascade(str(tmp_path), logging.getLogger("t"))


# ── run_reflex_cascade 端到端（monkeypatch 掉嵌入模型，无 GPU）────────────────

def _build_cascade_fixture(tmp_path, monkeypatch):
    """5 输入文件 × 3 方法输出。返回 (inp, out)。

    A 分数: f1=0.0, f2=1.0, f3=1.0, f5=0.0（f4 在 A/B/C 都缺输出）。
    B 分数: f2=0.0, f3=0.8；C 分数: f3≈0.9。
    """
    import numpy as np
    import utils.metrics.eer_calc as eer

    inp, out = tmp_path / "audio", tmp_path / "output"
    (inp / "healthy").mkdir(parents=True)
    vectors = {}  # path -> embedding

    def _mk(base, rel, vec):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        vectors[str(p)] = np.array(vec, dtype=float)

    _mk(inp, "healthy/f1.wav", [1, 0])
    _mk(inp, "healthy/f2.wav", [1, 0])
    _mk(inp, "healthy/f3.wav", [1, 0])
    _mk(inp, "healthy/f4.wav", [1, 0])
    _mk(inp, "healthy/f5.wav", [1, 0])
    _mk(out, "A/healthy/f1.wav", [0, 1])   # cos 0.0 → pass
    _mk(out, "A/healthy/f2.wav", [1, 0])   # cos 1.0 → fail
    _mk(out, "A/healthy/f3.wav", [1, 0])   # cos 1.0 → fail
    # A 缺 f4
    _mk(out, "A/healthy/f5.wav", [0, 1])   # cos 0.0 → pass（凑够 n>=4 才做 IQR 判定）
    _mk(out, "B/healthy/f2.wav", [0, 1])   # cos 0.0 → pass
    _mk(out, "B/healthy/f3.wav", [0.8, 0.6])  # cos 0.8
    _mk(out, "C/healthy/f3.wav", [0.9, 0.436])  # cos ≈0.9

    monkeypatch.setattr(eer, "load_model", lambda: (None, None))
    monkeypatch.setattr(
        eer, "extract_speaker_embedding",
        lambda path, fe, model, logger: vectors[str(path)])
    return inp, out


def _assert_cascade_outcome(report, reflex_dir):
    # 分配结果（两种阈值模式下相同）：f1/f5→A pass，f2→B pass，
    # f3 全失败 → tried 最低 B(0.8)（不是最后的 C）；f4 无任何输出 → unassigned
    assert report["reflex_dir"] == str(reflex_dir)
    assert report["per_method_counts"] == {"A": 2, "B": 2}
    assert report["n_leftover_best"] == 1
    assert report["unassigned"] == ["healthy/f4"]
    assert report["detection_skipped"] is False
    assert (reflex_dir / "healthy" / "f1.wav").read_bytes() == b"x"
    assert (reflex_dir / "healthy" / "f2.wav").exists()
    assert (reflex_dir / "healthy" / "f3.wav").exists()
    assert not (reflex_dir / "healthy" / "f4.wav").exists()
    assert (reflex_dir / "reflex_report.json").exists()
    assert report["per_file"]["healthy/f2"]["method"] == "B"
    assert report["per_file"]["healthy/f2"]["status"] == "pass"
    assert report["per_file"]["healthy/f3"]["method"] == "B"
    assert report["per_file"]["healthy/f3"]["status"] == "best"
    assert report["per_file"]["healthy/f3"]["tried"]["C"] == pytest.approx(0.9, abs=1e-2)


def test_run_reflex_cascade_iqr_mode(tmp_path, monkeypatch):
    import logging
    import reflexion_agent_sentence as ras
    inp, out = _build_cascade_fixture(tmp_path, monkeypatch)
    # A 分布 [0,0,1,1] → mean=0.5, IQR=1.0, k=0.2 → th=0.7
    report = ras.run_reflex_cascade(
        str(inp), str(out), ["A", "B", "C"], logging.getLogger("t"),
        iqr_k=0.2, score_floor=0.5)
    assert report["threshold_mode"] == "iqr"
    assert report["threshold"] == pytest.approx(0.7)
    _assert_cascade_outcome(report, tmp_path / "output" / "reflex_A_B_C")


def test_run_reflex_cascade_fixed_threshold(tmp_path, monkeypatch):
    import logging
    import reflexion_agent_sentence as ras
    inp, out = _build_cascade_fixture(tmp_path, monkeypatch)
    # 固定 θ=0.75：f2/f3(1.0) 失败，f3 在 B 轮 0.8 仍失败、C 轮 0.9 失败 → best=B(0.8)
    report = ras.run_reflex_cascade(
        str(inp), str(out), ["A", "B", "C"], logging.getLogger("t"),
        threshold_override=0.75)
    assert report["threshold_mode"] == "fixed"
    assert report["threshold"] == pytest.approx(0.75)
    _assert_cascade_outcome(report, tmp_path / "output" / "reflex_A_B_C")


def test_run_reflex_cascade_fixed_threshold_tiny_n(tmp_path, monkeypatch):
    """固定阈值不依赖分布：n<4 也能判定（IQR 模式下会跳过）。"""
    import logging
    import reflexion_agent_sentence as ras
    inp, out = _build_cascade_fixture(tmp_path, monkeypatch)
    # 只留 2 个可配对文件（删掉 f2/f5 的输出对，保留 f1/f3）
    (out / "A" / "healthy" / "f2.wav").unlink()
    (out / "A" / "healthy" / "f5.wav").unlink()
    (out / "B" / "healthy" / "f2.wav").unlink()
    report = ras.run_reflex_cascade(
        str(inp), str(out), ["A", "B", "C"], logging.getLogger("t"),
        threshold_override=0.85)
    assert report["detection_skipped"] is False
    # f1 pass A；f3 失败 A→B(0.8)→C(0.9)，best=B；f2/f5/f4 unassigned
    assert report["per_method_counts"] == {"A": 1, "B": 1}
    assert sorted(report["unassigned"]) == ["healthy/f2", "healthy/f4", "healthy/f5"]
