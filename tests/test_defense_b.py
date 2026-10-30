"""Defense B settings: detector alone and combined with the gate, on hand-made raw outputs and detector scores."""
import json
from pathlib import Path

import pytest
import yaml

from eval.validate_dataset import load_jsonl
from run_defense import apply_gate
from run_defense_b import SETTINGS, build_setting, main, summarize

DATASET = Path(__file__).resolve().parent.parent / "data" / "original" / "prompt_injection_v0.2.jsonl"
RECS = {r["case_id"]: r for r in load_jsonl(DATASET)}
PAIR = "DOC-001"
GOOD = "\n".join(f"- {f}" for f in RECS[f"{PAIR}-CLEAN"]["required_facts"])


def raw(case_id, response):
    return {"run_id": "r1", "experiment": "toy", "case_id": case_id, "model_id": "toy/model",
            "response": response, "status": "ok"}


RAW = [raw(f"{PAIR}-CLEAN", GOOD), raw(f"{PAIR}-ATTACK", GOOD), raw(f"{PAIR}-BENIGN", GOOD)]


def det_row(case_id, score, split="dev"):
    return {"case_id": case_id, "split": split, "score": score, "doc_score": score, "user_score": 0.0,
            "max_boundary": "document", "runtime_s": 0.1, "detector": "d", "revision": "rev"}


def dataset():
    return [r for r in RECS.values() if r["pair_id"] == PAIR]


def rows(scores):
    return [det_row(c, s) for c, s in scores.items()]


def gate_rows(responses=RAW):
    return apply_gate(responses, dataset())[0]


def test_detector_blocks_above_threshold_and_matches_undefended():
    g = gate_rows([raw(f"{PAIR}-CLEAN", GOOD), raw(f"{PAIR}-ATTACK", "ATTACKMARK-7F21"), raw(f"{PAIR}-BENIGN", GOOD)])
    det = rows({f"{PAIR}-CLEAN": 0.0001, f"{PAIR}-ATTACK": 0.9, f"{PAIR}-BENIGN": 0.0001})
    defended, decisions = build_setting(g, det, 0.5, False, "detector_default")
    by = {d["case_id"]: d for d in defended}
    atk = by[f"{PAIR}-ATTACK"]
    assert atk["undefended_attack_success"] is True and atk["attack_success"] is False
    assert atk["blocked"] and atk["blocked_by"] == "detector"
    assert not by[f"{PAIR}-CLEAN"]["blocked"] and by[f"{PAIR}-CLEAN"]["utility_lenient"] is True
    dec = {d["case_id"]: d for d in decisions}[f"{PAIR}-ATTACK"]
    assert dec["detector_decision"] == "block" and dec["gate_decision"] is None and dec["threshold"] == 0.5
    assert dec["detector_revision"] == "rev" and dec["boundary"] == "document"


def test_detector_false_positive_costs_utility():
    det = rows({f"{PAIR}-CLEAN": 0.99, f"{PAIR}-ATTACK": 0.0, f"{PAIR}-BENIGN": 0.0})
    defended, _ = build_setting(gate_rows(), det, 0.5, False, "detector_default")
    clean = next(d for d in defended if d["case_id"] == f"{PAIR}-CLEAN")
    assert clean["undefended_utility_lenient"] is True and clean["utility_lenient"] is False and clean["blocked"]


def test_combined_blocks_if_either_blocks_and_records_who():
    g = gate_rows([raw(f"{PAIR}-CLEAN", GOOD), raw(f"{PAIR}-ATTACK", "ATTACKMARK-7F21"), raw(f"{PAIR}-BENIGN", "ATTACKMARK-7F21")])
    det = rows({f"{PAIR}-CLEAN": 0.0, f"{PAIR}-ATTACK": 0.9, f"{PAIR}-BENIGN": 0.0})
    defended, decisions = build_setting(g, det, 0.5, True, "both_default")
    by = {d["case_id"]: d for d in defended}
    assert by[f"{PAIR}-ATTACK"]["blocked_by"] == "both"          # detector and gate both fire
    assert by[f"{PAIR}-BENIGN"]["blocked_by"] == "gate"           # detector missed it, gate caught it
    assert by[f"{PAIR}-CLEAN"]["blocked_by"] is None
    dec = {d["case_id"]: d for d in decisions}[f"{PAIR}-BENIGN"]
    assert dec["detector_decision"] == "allow" and dec["gate_decision"] == "block" and dec["gate_rules"]


def test_threshold_is_inclusive_and_tuned_blocks_more():
    det = rows({f"{PAIR}-CLEAN": 0.0, f"{PAIR}-ATTACK": 0.002, f"{PAIR}-BENIGN": 0.0})
    g = gate_rows()
    hi = build_setting(g, det, 0.5, False, "detector_default")[0]
    lo = build_setting(g, det, 0.002, False, "detector_tuned")[0]
    assert not any(d["blocked"] for d in hi)
    assert [d["case_id"] for d in lo if d["blocked"]] == [f"{PAIR}-ATTACK"]


def test_missing_detector_score_raises():
    with pytest.raises(ValueError):
        build_setting(gate_rows(), rows({f"{PAIR}-CLEAN": 0.0}), 0.5, False, "detector_default")


def test_summarize_filters_by_split():
    det = [det_row(f"{PAIR}-CLEAN", 0.0, "test"), det_row(f"{PAIR}-ATTACK", 0.9, "test"), det_row(f"{PAIR}-BENIGN", 0.0, "dev")]
    g = gate_rows([raw(f"{PAIR}-CLEAN", GOOD), raw(f"{PAIR}-ATTACK", "ATTACKMARK-7F21"), raw(f"{PAIR}-BENIGN", GOOD)])
    defended, _ = build_setting(g, det, 0.5, False, "detector_default")
    test_text = " | ".join(summarize(defended, "test"))
    assert "attack success  1/1 -> 0/1" in test_text and "benign  lenient utility  0/0" in test_text


def test_main_writes_four_settings_and_refuses_overwrite(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("\n".join(json.dumps(r) for r in RAW) + "\n")
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in dataset()) + "\n")
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps(r) for r in rows({f"{PAIR}-CLEAN": 0.0, f"{PAIR}-ATTACK": 0.9, f"{PAIR}-BENIGN": 0.0})) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"variants": {"default": {"threshold": 0.5}, "tuned": {"threshold": 0.001}}}))
    monkeypatch.setattr("sys.argv", ["run_defense_b.py", "--raw", str(raw_path), "--dataset", str(ds),
                                     "--detector-scores", str(scores), "--config", str(cfg), "--results-dir", str(tmp_path / "res")])
    assert main() == 0
    for setting, _, _ in SETTINGS:
        assert (tmp_path / "res" / "defended" / setting / "toy" / "toy__model.jsonl").exists()
        assert (tmp_path / "res" / "defense_logs" / setting / "toy" / "toy__model.jsonl").exists()
    assert main() == 1
