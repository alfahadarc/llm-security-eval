"""verify_matching must accept consistent files and reject each kind of tampering."""
import json
from pathlib import Path

import pytest

from eval.validate_dataset import load_jsonl
from run_defense import apply_gate
from score_results import score_records
from verify_matching import verify

DATASET = Path(__file__).resolve().parent.parent / "data" / "original" / "prompt_injection_v0.2.jsonl"
PAIR = "DOC-001"
RECS = [r for r in load_jsonl(DATASET) if r["pair_id"] == PAIR]
GOOD = "\n".join(f"- {f}" for f in RECS[0]["required_facts"])


def raw_rows():
    texts = {f"{PAIR}-CLEAN": GOOD, f"{PAIR}-ATTACK": "ATTACKMARK-7F21", f"{PAIR}-BENIGN": GOOD}
    return [{"run_id": "r1", "experiment": "toy", "case_id": c, "model_id": "toy/m", "seed": 0,
             "response": t, "status": "ok"} for c, t in texts.items()]


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def files(tmp_path):
    raw = raw_rows()
    scored = score_records(raw, RECS)
    defended, decisions = apply_gate(raw, RECS)
    return {"raw": raw, "scored": scored, "defended": defended, "log": decisions, "dir": tmp_path}


def run(f):
    d = f["dir"]
    return verify(write(d / "raw.jsonl", f["raw"]), write(d / "scored.jsonl", f["scored"]),
                  write(d / "defended.jsonl", f["defended"]), write(d / "log.jsonl", f["log"]))


def test_consistent_files_pass(files):
    assert run(files) == []


def test_missing_defended_case_is_reported(files):
    files["defended"] = files["defended"][:2]
    assert any("differ from raw cases" in p for p in run(files))


def test_changed_undefended_label_is_reported(files):
    files["defended"][1]["undefended_label"] = "utility_pass"
    assert any("undefended_label" in p for p in run(files))


def test_changed_undefended_flag_is_reported(files):
    files["defended"][1]["undefended_attack_success"] = False
    assert any("undefended_attack_success" in p for p in run(files))


def test_defense_creating_a_success_is_reported(files):
    row = files["defended"][0]
    row["undefended_attack_success"], row["attack_success"] = False, True
    assert any("created an attack success" in p for p in run(files))


def test_blocked_row_counted_as_success_is_reported(files):
    row = next(d for d in files["defended"] if d["blocked"])
    row["utility_lenient"] = True
    assert any("blocked reply still counted" in p for p in run(files))


def test_wrong_run_id_is_reported(files):
    files["defended"][0]["run_id"] = "other"
    assert any("run_id/model_id differ" in p for p in run(files))


def test_log_decision_mismatch_and_missing_log_are_reported(files):
    flip = files["log"][1]
    flip["decision"] = "allow" if flip["decision"] == "block" else "block"
    assert any("logged decision disagrees" in p for p in run(files))
    files["log"] = files["log"][:2]
    assert any("decision log cases differ" in p for p in run(files))


def test_several_seeds_are_reported(files):
    files["raw"][0]["seed"] = 1
    assert any("several seeds" in p for p in run(files))
