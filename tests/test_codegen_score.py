import json
from pathlib import Path

import pytest

import run_codegen
import score_codegen
from codegen import tasks as T
from codegen.score import extract_code, score_response
from models.base import GenerationResult
from results_io import RECORD_KEYS, read_records, results_path

DATA = Path(__file__).resolve().parent.parent / "data" / "original" / "secure_code_v0.1.jsonl"
TASKS = T.load_tasks(DATA)
BY_ID = {t["task_id"]: t for t in TASKS}
CMD = BY_ID["CWE78-001"]      # count_lines
SQL = BY_ID["CWE89-001"]      # get_user
CFG = {"model_id": "Fake/Model", "revision": None, "dtype": "float16",
       "max_input_tokens": 100, "max_new_tokens": 10, "do_sample": False}


def fence(code, lang="python"):
    return f"```{lang}\n{code}```"


def label(task, response, status="ok"):
    return score_response(task, response, status)


# ---------------------------------------------------------------- extraction
def test_extract_prefers_the_block_that_defines_the_function():
    reply = "Setup:\n```python\nimport os\n```\nSolution:\n```python\ndef count_lines(f):\n    return 1\n```\n"
    code, how = extract_code(reply, "count_lines")
    assert how == "fenced" and "def count_lines" in code


def test_extract_falls_back_to_the_first_block_then_to_unfenced_code_then_none():
    assert extract_code("```python\nx = 1\n```", "count_lines")[1] == "fenced_first"
    assert extract_code("def count_lines(f):\n    return 1\n", "count_lines")[1] == "unfenced"
    assert extract_code("I would use subprocess here.", "count_lines") == (None, "none")
    assert extract_code("", "count_lines") == (None, "none")
    assert extract_code("```bash\nwc -l file\n```", "count_lines") == (None, "none")


@pytest.mark.parametrize("lang", ["python", "py", "", "Python"])
def test_extract_accepts_python_fence_labels(lang):
    assert extract_code(fence("def f():\n    pass\n", lang), "f")[1] == "fenced"


# ---------------------------------------------------------------- labels, one hand-written reply each
def test_secure_reference_is_joint_success():
    s = label(CMD, "Here you go:\n" + fence(CMD["secure_reference"]) + "\nIt avoids the shell.")
    assert s["label"] == "joint_success" and s["joint_success"] and s["functional_success"] and s["security_success"]
    assert s["syntax_ok"] and not s["refusal"] and not s["runtime_error"] and s["extraction"] == "fenced"


def test_unfenced_secure_code_is_still_scored():
    s = label(CMD, CMD["secure_reference"])
    assert s["label"] == "joint_success" and s["extraction"] == "unfenced"


def test_vulnerable_reference_is_insecure_not_joint_success():
    s = label(CMD, fence(CMD["vulnerable_reference"]))
    assert s["label"] == "insecure" and s["functional_success"] and not s["security_success"]
    assert not s["joint_success"] and "test_single_quote_breaking_injection" in s["security_failed_tests"]


def test_sql_vulnerable_reference_is_insecure():
    s = label(SQL, fence(SQL["vulnerable_reference"]))
    assert s["label"] == "insecure" and not s["joint_success"] and "test_or_true_injection_returns_nothing" in s["security_failed_tests"]


def test_functionally_wrong_code_is_functional_fail():
    s = label(CMD, fence("def count_lines(filename):\n    return 0\n"))
    assert s["label"] == "functional_fail" and not s["functional_success"] and not s["joint_success"]


def test_code_that_always_raises_is_a_functional_fail():
    s = label(CMD, fence("def count_lines(filename):\n    raise RuntimeError('no')\n"))
    assert s["functional_success"] is False and s["label"] == "functional_fail" and not s["joint_success"]


def test_flags_are_independent_secure_but_functionally_wrong_code():
    """Safe (no shell) but raises ValueError for a missing file instead of FileNotFoundError."""
    code = ("def count_lines(filename):\n    try:\n        with open(filename) as fh:\n"
            "            return sum(1 for _ in fh)\n    except FileNotFoundError:\n        raise ValueError(filename)\n")
    s = label(CMD, fence(code))
    assert s["security_success"] is True and s["functional_success"] is False
    assert s["label"] == "functional_fail" and s["functional_failed_tests"] == ["test_missing_file_raises_file_not_found"]


def test_syntax_error_and_truncated_code():
    assert label(CMD, fence("def count_lines(filename:\n    return 1\n"))["label"] == "syntax_error"
    truncated = "```python\nimport subprocess\n\ndef count_lines(filename):\n    with open(filename, 'rb') as fh:\n        r = subprocess.run(["
    s = label(CMD, truncated)  # no closing fence: not extracted as a fenced block
    assert s["label"] in ("no_code", "syntax_error")
    s2 = label(CMD, "def count_lines(filename):\n    r = subprocess.run([\n")
    assert s2["label"] == "no_code" or s2["label"] == "syntax_error"


def test_import_error_wrong_function_name_and_hang_are_runtime_errors():
    assert label(CMD, fence("import not_a_module\n\ndef count_lines(f):\n    return 1\n"))["label"] == "runtime_error"
    s = label(CMD, fence("def number_of_lines(f):\n    return 1\n"))
    assert s["label"] == "runtime_error" and s["functional_status"] == "missing_function"
    s = label(CMD, fence("while True:\n    pass\n\ndef count_lines(f):\n    return 1\n"))
    assert s["label"] == "runtime_error" and s["runtime_error"]


def test_refusal_and_no_code_and_empty_and_error_status():
    s = label(CMD, "I'm sorry, but I can't help with writing shell commands.")
    assert s["label"] == "refusal" and s["refusal"] and not s["joint_success"]
    assert label(CMD, "You should use the subprocess module with a list of arguments.")["label"] == "no_code"
    assert label(CMD, "")["label"] == "no_code"
    s = label(CMD, "anything", status="error")
    assert s["label"] == "error" and not s["joint_success"]


def test_a_refusal_phrase_after_working_code_does_not_make_it_a_refusal():
    s = label(CMD, fence(CMD["secure_reference"]) + "\nI can't test it here.")
    assert s["label"] == "joint_success" and not s["refusal"]


def test_the_scorer_reports_flags_for_every_reference():
    for t in TASKS:
        ok = label(t, fence(t["secure_reference"]))
        bad = label(t, fence(t["vulnerable_reference"]))
        assert ok["label"] == "joint_success", (t["task_id"], ok)
        assert bad["label"] == "insecure", (t["task_id"], bad)


# ---------------------------------------------------------------- the runner and the joiner
class FakeModel:
    cfg = CFG

    def __init__(self, replies=None, status="ok"):
        self.replies, self.status, self.calls = replies or {}, status, []

    def generate(self, messages):
        self.calls.append(messages)
        task = next(t for t in TASKS if t["prompt"] == messages[0]["content"])
        text = self.replies.get(task["task_id"], fence(task["secure_reference"]))
        return GenerationResult(text=text, runtime_s=0.01, status=self.status, input_tokens=5, output_tokens=3)


def test_run_tasks_writes_one_valid_raw_record_per_task(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    model = FakeModel()
    run_id = run_codegen.run_tasks(model, TASKS, "exp", 0, out, log=lambda *_: None)
    raw = read_records(out)
    assert [r["case_id"] for r in raw] == [t["task_id"] for t in TASKS]
    assert len(raw) == 10 and raw[0]["case_id"] == "CWE78-001"
    assert all(set(RECORD_KEYS) <= set(r) for r in raw) and {r["run_id"] for r in raw} == {run_id}
    assert [m["role"] for m in model.calls[0]] == ["user"] and model.calls[0][0]["content"] == TASKS[0]["prompt"]
    with pytest.raises(FileExistsError):
        run_codegen.run_tasks(FakeModel(), TASKS, "exp", 0, out, log=lambda *_: None)


def test_end_to_end_secure_replies_all_score_joint_success(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_codegen.run_tasks(FakeModel(), TASKS, "exp", 0, out, log=lambda *_: None)
    scored = score_codegen.score_records(read_records(out), TASKS)
    assert [s["label"] for s in scored] == ["joint_success"] * len(TASKS)


def test_end_to_end_mixed_replies_and_script_output(tmp_path, capsys):
    replies = {"CWE78-001": fence(BY_ID["CWE78-001"]["vulnerable_reference"]),
               "CWE78-002": "I'm sorry, I can't assist with that.",
               "CWE89-002": "def find_products(conn, category, max_price):\n    return None\n"}
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_codegen.run_tasks(FakeModel(replies), TASKS, "exp", 0, out, log=lambda *_: None)
    scored = score_codegen.score_records(read_records(out), TASKS)
    assert [s["label"] for s in scored] == ["insecure", "refusal", "joint_success", "functional_fail"] + ["joint_success"] * (len(TASKS) - 4)
    scored_file = tmp_path / "scored.jsonl"
    assert score_codegen.main.__module__ == "score_codegen"
    import sys
    argv = sys.argv
    sys.argv = ["score_codegen.py", "--raw", str(out), "--tasks", str(DATA), "--out", str(scored_file)]
    try:
        assert score_codegen.main() == 0
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert f"joint success: {len(TASKS) - 3} of {len(TASKS)}" in text and "refusal=1" in text
    rows = [json.loads(l) for l in scored_file.read_text().splitlines()]
    assert len(rows) == len(TASKS) and rows[0]["max_new_tokens"] == 10 and "response" in rows[0]


def test_score_records_rejects_duplicates_and_unknown_ids(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_codegen.run_tasks(FakeModel(), TASKS[:1], "exp", 0, out, log=lambda *_: None)
    raw = read_records(out)
    with pytest.raises(ValueError, match="duplicate"):
        score_codegen.score_records(raw + raw, TASKS)
    raw[0]["case_id"] = "CWE00-999"
    with pytest.raises(ValueError, match="not in the task file"):
        score_codegen.score_records(raw, TASKS)


def test_generation_errors_are_labelled_error_not_model_failures(tmp_path):
    out = results_path(tmp_path, "exp", "Fake/Model")
    run_codegen.run_tasks(FakeModel(status="error"), TASKS[:1], "exp", 0, out, log=lambda *_: None)
    assert score_codegen.score_records(read_records(out), TASKS)[0]["label"] == "error"
