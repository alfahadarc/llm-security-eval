import json
from pathlib import Path

import pytest

import run_eval
from score_results import main as score_main, score_records, scored_path, summarize
from eval.validate_dataset import load_jsonl
from models.base import GenerationResult
from results_io import read_records, results_path, RECORD_KEYS

DATASET = Path(__file__).resolve().parent.parent / "data" / "original" / "prompt_injection_v0.1.jsonl"
CFG = {"model_id": "Fake/Model", "revision": None, "dtype": "float16",
       "max_input_tokens": 100, "max_new_tokens": 10, "do_sample": False}


class FakeModel:
    cfg = CFG

    def __init__(self, text="- a\n- b\n- c", status="ok"):
        self.text, self.status, self.calls = text, status, []

    def generate(self, messages):
        self.calls.append(messages)
        return GenerationResult(text=self.text, runtime_s=0.01, status=self.status,
                                input_tokens=5, output_tokens=3)


def records():
    return load_jsonl(DATASET)


def test_run_cases_writes_one_valid_raw_record_per_case(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    model = FakeModel()
    run_id = run_eval.run_cases(model, records(), "exp", 0, out, log=lambda *_: None)
    raw = read_records(out)
    assert len(raw) == len(model.calls) == 30
    assert [r["case_id"] for r in raw] == [r["case_id"] for r in records()]
    assert all(set(RECORD_KEYS) <= set(r) for r in raw)
    assert {r["run_id"] for r in raw} == {run_id}
    assert raw[0]["messages"][0]["role"] == "system" and raw[0]["messages"][1]["role"] == "user"


def test_run_cases_refuses_existing_output(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_eval.run_cases(FakeModel(), records(), "exp", 0, out, log=lambda *_: None)
    with pytest.raises(FileExistsError):
        run_eval.run_cases(FakeModel(), records(), "exp", 0, out, log=lambda *_: None)


def raw_run(tmp_path, **kw):
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_eval.run_cases(FakeModel(**kw), records(), "exp", 0, out, log=lambda *_: None)
    return out, read_records(out)


def test_score_records_labels_and_columns(tmp_path):
    _, raw = raw_run(tmp_path)
    scored = score_records(raw, records())
    assert len(scored) == 30
    assert {"case_id", "pair_id", "condition", "objective", "technique", "label"} <= set(scored[0])
    assert all(s["label"] == "utility_fail" for s in scored)  # "- a/- b/- c" lacks the facts
    assert summarize(scored)["attack"]["utility_fail"] == 10


def test_score_records_flags_marker(tmp_path):
    _, raw = raw_run(tmp_path, text="ATTACKMARK-7F21")
    scored = {s["case_id"]: s for s in score_records(raw, records())}
    assert scored["DOC-001-ATTACK"]["label"] == "attack_success"
    assert scored["DOC-001-CLEAN"]["label"] == "utility_fail"


def test_error_status_is_labeled_error(tmp_path):
    _, raw = raw_run(tmp_path, text="", status="error")
    assert {s["label"] for s in score_records(raw, records())} == {"error"}


def test_score_records_rejects_bad_joins(tmp_path):
    _, raw = raw_run(tmp_path)
    with pytest.raises(ValueError, match="no raw record"):
        score_records(raw[:-1], records())
    with pytest.raises(ValueError, match="duplicate"):
        score_records(raw + [raw[0]], records())
    bad = [dict(raw[0], case_id="NOPE-1")] + raw[1:]
    with pytest.raises(ValueError, match="not in the dataset"):
        score_records(bad, records())


def test_score_cli_writes_scored_file(tmp_path, capsys):
    out, _ = raw_run(tmp_path)
    assert score_main_argv(out, tmp_path) == 0
    scored = read_records(scored_path(tmp_path, "exp", "Fake/Model"))
    assert len(scored) == 30
    assert "DOC-010-BENIGN" in capsys.readouterr().out


def score_main_argv(raw_path, results_dir):
    import sys
    old = sys.argv
    sys.argv = ["score_results.py", "--raw", str(raw_path), "--dataset", str(DATASET),
                "--results-dir", str(results_dir)]
    try:
        return score_main()
    finally:
        sys.argv = old
