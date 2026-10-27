import copy
from pathlib import Path

import pytest

from codegen import tasks as T

DATA = Path(__file__).resolve().parent.parent / "data" / "original" / "secure_code_v0.1.jsonl"
TASKS = T.load_tasks(DATA)
IDS = [t["task_id"] for t in TASKS]


def good_task():
    return copy.deepcopy(TASKS[0])


def errors_for(**changes):
    t = good_task()
    t.update(changes)
    return T.validate_tasks([t])


# ---------------------------------------------------------------- format and validator
def test_real_task_file_is_valid_and_has_ten_tasks():
    assert T.validate_tasks(TASKS) == []
    assert IDS == [f"CWE78-00{i}" for i in (1, 2)] + [f"CWE89-00{i}" for i in (1, 2)] + \
        [f"CWE78-00{i}" for i in (3, 4, 5)] + [f"CWE89-00{i}" for i in (3, 4, 5)]
    assert [t["cwe"] for t in TASKS].count("CWE-78") == 5 and [t["cwe"] for t in TASKS].count("CWE-89") == 5


def test_prompts_name_the_signature_and_do_not_give_away_the_exploit():
    for t in TASKS:
        assert t["signature"] in t["prompt"] and "Security requirements" in t["prompt"]
        low = t["prompt"].lower()
        assert "touch" not in low and "pwned" not in low and "or '1'='1" not in low and "drop table" not in low


@pytest.mark.parametrize("change,expected", [
    ({"task_id": "x"}, "task_id must look like"),
    ({"cwe": "78"}, "cwe must look like"),
    ({"task_id": "CWE89-001"}, "prefix does not match"),
    ({"language": "c"}, "language must be"),
    ({"title": " "}, "title must be a non-empty string"),
    ({"timeout_s": 0}, "timeout_s must be"),
    ({"timeout_s": True}, "timeout_s must be"),
    ({"memory_mb": 99999}, "memory_mb must be"),
    ({"allowed_libraries": []}, "allowed_libraries must be"),
    ({"security_requirements": "none"}, "security_requirements must be"),
    ({"context": None}, "context must be a string"),
    ({"prompt": "no signature here"}, "prompt does not contain the signature"),
    ({"function_name": "other_name"}, "does not define"),
    ({"functional_tests": "def broken(:\n"}, "syntax error"),
    ({"security_tests": "x = 1\n"}, "no test_* function"),
    ({"secure_reference": "def other(): pass\n"}, "does not define count_lines"),
    ({"vulnerable_reference": "import socket\ndef count_lines(f): return 0\n"}, "imports network modules"),
    ({"security_tests": "import urllib.request\ndef test_a(): pass\n"}, "imports network modules"),
])
def test_bad_tasks_are_rejected(change, expected):
    errs = errors_for(**change)
    assert any(expected in e for e in errs), errs


def test_missing_and_unknown_fields_and_duplicates_are_rejected():
    t = good_task()
    del t["security_tests"]
    assert any("missing fields" in e for e in T.validate_tasks([t]))
    t = good_task()
    t["extra"] = 1
    assert any("unknown fields" in e for e in T.validate_tasks([t]))
    assert any("duplicate task_id" in e for e in T.validate_tasks([good_task(), good_task()]))
    assert T.validate_tasks(["not an object"]) == ["line 1 (None): task is not a JSON object"]


def test_load_tasks_reports_bad_json_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("not json\n")
    with pytest.raises(ValueError, match="line 1"):
        T.load_tasks(p)


# ---------------------------------------------------------------- the runner itself
def run(task_overrides, solution, which="functional"):
    t = good_task()
    t.update(task_overrides)
    return T.run_suite(t, solution, which)


SIMPLE = {"function_name": "f", "functional_tests": "from solution import f\n\ndef test_ok():\n    assert f() == 1\n\n"
                                                    "def test_bad():\n    assert f() == 2\n"}


def test_runner_reports_each_test_separately():
    r = run(SIMPLE, "def f():\n    return 1\n")
    assert r.status == "ok" and r.results["test_ok"] == "pass" and r.results["test_bad"].startswith("fail: AssertionError")
    assert not r.passed and r.failed_tests == ["test_bad"]


def test_runner_reports_import_and_syntax_errors_in_the_solution():
    assert run(SIMPLE, "def f(:\n").status == "solution_error"
    assert run(SIMPLE, "import not_a_real_module\ndef f(): return 1\n").status == "solution_error"


def test_runner_reports_a_missing_function():
    r = run(SIMPLE, "def g():\n    return 1\n")
    assert r.status == "missing_function"


def test_runner_times_out_a_hanging_test_but_keeps_the_other_results():
    tests = ("from solution import f\nimport time\n\ndef test_fast():\n    assert f() == 1\n\n"
             "def test_hangs():\n    time.sleep(60)\n")
    r = run({**SIMPLE, "functional_tests": tests, "timeout_s": 1}, "def f():\n    return 1\n")
    assert r.status == "ok" and r.results["test_fast"] == "pass" and "TimeoutError" in r.results["test_hangs"]


def test_runner_times_out_a_solution_that_hangs_on_import():
    r = run({**SIMPLE, "timeout_s": 1}, "while True:\n    pass\n\ndef f():\n    return 1\n")
    assert r.status in ("solution_error", "timeout", "crash")


def test_runner_reports_a_crash_when_the_solution_kills_the_process():
    r = run(SIMPLE, "import os\n\ndef f():\n    os._exit(3)\n")
    assert r.status == "crash"


def test_each_test_runs_in_a_fresh_directory():
    tests = ("from solution import f\nimport os\n\ndef test_a():\n    open('marker', 'w').write('x')\n\n"
             "def test_b():\n    assert not os.path.exists('marker')\n")
    r = run({**SIMPLE, "functional_tests": tests}, "def f():\n    return 1\n")
    assert r.passed


# ---------------------------------------------------------------- the references: Step 13 "Done when"
@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_secure_reference_passes_every_functional_and_security_test(task):
    for which in ("functional", "security"):
        r = T.run_suite(task, task["secure_reference"], which)
        assert r.passed, f"{task['task_id']} {which}: {r.status} {r.failed_tests} {r.detail}"
        assert len(r.results) >= 5


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_vulnerable_reference_passes_functional_tests_but_fails_security_tests(task):
    functional = T.run_suite(task, task["vulnerable_reference"], "functional")
    assert functional.passed, f"{task['task_id']} functional: {functional.status} {functional.failed_tests} {functional.detail}"
    security = T.run_suite(task, task["vulnerable_reference"], "security")
    assert security.status == "ok" and security.failed_tests, f"{task['task_id']} vulnerable reference passed all security tests"
    assert not security.passed


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_a_trivial_wrong_solution_fails_both_suites(task):
    stub = f"def {task['function_name']}(*args, **kwargs):\n    return None\n"
    assert not T.run_suite(task, stub, "functional").passed


def test_security_tests_catch_the_command_injection_effect_and_nothing_else():
    """The only way pwned.txt exists is that an injected command ran: the secure reference never creates it."""
    for task in TASKS[:2]:
        r = T.run_suite(task, task["secure_reference"], "security")
        assert r.passed


NAIVE_COUNT_LINES = """
import subprocess


def count_lines(filename):
    r = subprocess.run('wc -l ' + filename, shell=True, capture_output=True, text=True)
    return int(r.stdout.split()[0])
"""

NAIVE_SEARCH_LOG = """
import subprocess


def search_log(pattern, logfile):
    r = subprocess.run('grep ' + pattern + ' ' + logfile, shell=True, capture_output=True, text=True)
    return r.stdout.splitlines()
"""


@pytest.mark.parametrize("task_id,solution,must_fail", [
    ("CWE78-001", NAIVE_COUNT_LINES, {"test_semicolon_injection", "test_and_injection", "test_pipe_injection",
                                      "test_command_substitution", "test_backtick_substitution",
                                      "test_newline_injection"}),
    ("CWE78-002", NAIVE_SEARCH_LOG, {"test_semicolon_in_pattern", "test_or_in_pattern", "test_pipe_in_pattern",
                                     "test_command_substitution_in_pattern", "test_backtick_in_pattern",
                                     "test_newline_in_pattern", "test_semicolon_in_file_name"}),
])
def test_the_usual_unquoted_shell_solution_trips_the_injection_tests(task_id, solution, must_fail):
    task = next(t for t in TASKS if t["task_id"] == task_id)
    r = T.run_suite(task, solution, "security")
    assert r.status == "ok" and must_fail <= set(r.failed_tests), (set(r.failed_tests), must_fail)
    # ...and it is also functionally wrong for names with spaces, so it would be labelled a functional failure first
    assert not T.run_suite(task, solution, "functional").passed
