import random

import numpy as np
import pytest
import torch

from models.base import GenerationResult
from results_io import append_record, make_record, model_slug, new_run_id, read_records, results_path
from seed import set_seed

CFG = {
    "model_id": "Qwen/Qwen3-4B", "revision": "abc", "dtype": "float16",
    "max_input_tokens": 8192, "max_new_tokens": 256, "do_sample": False,
}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_same_numbers():
    set_seed(7)
    a = draw()
    set_seed(7)
    assert draw() == a


def test_different_seed_different_numbers():
    set_seed(7)
    a = draw()
    set_seed(8)
    assert draw() != a


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, None])
def test_bad_seed_rejected(bad):
    with pytest.raises(ValueError):
        set_seed(bad)


def test_results_path_layout(tmp_path):
    p = results_path(tmp_path, "smoke_test", "Qwen/Qwen3-4B")
    assert p == tmp_path / "raw" / "smoke_test" / "Qwen__Qwen3-4B.jsonl"
    assert model_slug("a/b") == "a__b"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden"])
def test_bad_experiment_name_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        results_path(tmp_path, bad, "m")


def test_roundtrip_stores_full_prompt_and_unicode(tmp_path):
    res = GenerationResult(text="réponse ✓", runtime_s=1.5, status="ok", input_tokens=10, output_tokens=3)
    rec = make_record("run-1", "exp", "c1", CFG, 0, MSGS, res)
    p = results_path(tmp_path, "exp", CFG["model_id"])
    append_record(p, rec)
    append_record(p, rec)  # append, never overwrite
    got = read_records(p)
    assert len(got) == 2
    assert got[0] == rec
    assert got[0]["messages"] == MSGS
    assert got[0]["revision"] == "abc" and got[0]["seed"] == 0


def test_error_result_is_recorded(tmp_path):
    res = GenerationResult(text="", runtime_s=0.1, status="error", error="ValueError: too long")
    rec = make_record("run-1", "exp", "c1", CFG, 0, MSGS, res)
    p = tmp_path / "e.jsonl"
    append_record(p, rec)
    got = read_records(p)[0]
    assert got["status"] == "error" and got["error"] == "ValueError: too long"
    assert got["input_tokens"] is None


def test_make_record_rejects_invalid_messages():
    res = GenerationResult(text="", runtime_s=0, status="ok")
    with pytest.raises(ValueError):
        make_record("run-1", "exp", "c1", CFG, 0, [], res)


def test_append_rejects_incomplete_record(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        append_record(tmp_path / "x.jsonl", {"experiment": "e"})


def test_read_reports_bad_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        read_records(p)


def test_record_has_run_id_and_generation_settings():
    res = GenerationResult(text="a", runtime_s=0.1, status="ok")
    rec = make_record("run-1", "exp", "c1", CFG, 0, MSGS, res)
    assert rec["run_id"] == "run-1"
    assert rec["generation"] == {"max_input_tokens": 8192, "max_new_tokens": 256, "do_sample": False}


def test_run_ids_are_unique():
    assert new_run_id() != new_run_id()
