"""Secure-code tasks: file format, validator and the test runner that executes a solution inside the sandbox.

A task (one JSON object per line) bundles the prompt for the model, the normal (functional) tests, the adversarial
(security) tests, and two reference solutions: a secure one that must pass every test and a vulnerable one that
must pass the functional tests but fail at least one security test.
"""
import ast
import json
import re
from dataclasses import dataclass, field

from codegen import sandbox

FIELDS = ("task_id", "cwe", "language", "title", "function_name", "signature", "context", "prompt",
          "functional_requirements", "security_requirements", "allowed_libraries", "timeout_s", "memory_mb",
          "functional_tests", "security_tests", "secure_reference", "vulnerable_reference")
STR_FIELDS = ("task_id", "cwe", "language", "title", "function_name", "signature", "prompt",
              "functional_tests", "security_tests", "secure_reference", "vulnerable_reference")
LIST_FIELDS = ("functional_requirements", "security_requirements", "allowed_libraries")
NETWORK_MODULES = {"socket", "ssl", "urllib", "http", "ftplib", "smtplib", "telnetlib", "requests", "xmlrpc",
                   "asyncio", "socketserver", "imaplib", "poplib"}
TASK_ID = re.compile(r"^CWE\d+-\d{3}$")
CWE = re.compile(r"^CWE-\d+$")

# Runs inside the sandbox. Imports the solution first (so import errors are reported separately), then runs every
# test_* function of tests.py in its own fresh directory with an alarm, and prints one result line.
RUNNER = r'''
import json, os, signal, sys, tempfile, traceback

BASE = os.getcwd()
CFG = json.load(open("config.json"))
os.remove("config.json")


def finish(payload):
    print("RESULT_JSON " + json.dumps(payload))
    sys.stdout.flush()
    os._exit(0)


def on_alarm(signum, frame):
    raise TimeoutError("timed out")


signal.signal(signal.SIGALRM, on_alarm)
signal.alarm(CFG["per_test_s"])  # a solution that hangs while being imported is stopped here, not at the wall clock
try:
    import solution
except BaseException as e:
    finish({"fatal": "solution_error", "detail": (type(e).__name__ + ": " + str(e))[:500]})
finally:
    signal.alarm(0)
if not callable(getattr(solution, CFG["function_name"], None)):
    finish({"fatal": "missing_function", "detail": CFG["function_name"] + " is not defined"})
try:
    import tests
except BaseException as e:
    finish({"fatal": "tests_error", "detail": (type(e).__name__ + ": " + str(e))[:500]})


results = {}
for name, fn in list(vars(tests).items()):
    if not (name.startswith("test_") and callable(fn)):
        continue
    os.chdir(tempfile.mkdtemp(dir=BASE))
    signal.alarm(CFG["per_test_s"])
    try:
        fn()
        results[name] = "pass"
    except BaseException as e:
        results[name] = "fail: " + (type(e).__name__ + ": " + str(e))[:300]
    finally:
        signal.alarm(0)
        os.chdir(BASE)
finish({"results": results})
'''


@dataclass
class SuiteRun:
    status: str  # ok | solution_error | missing_function | tests_error | timeout | crash
    results: dict = field(default_factory=dict)  # test name -> "pass" | "fail: message"
    detail: str = ""

    @property
    def passed(self):
        return self.status == "ok" and bool(self.results) and all(v == "pass" for v in self.results.values())

    @property
    def failed_tests(self):
        return [k for k, v in self.results.items() if v != "pass"]


def list_test_names(source):
    """Names of the top-level test_* functions, in definition order."""
    tree = ast.parse(source)
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def run_suite(task, solution, which):
    """Run the task's 'functional' or 'security' tests against `solution` (source text) in the sandbox."""
    if which not in ("functional", "security"):
        raise ValueError("which must be 'functional' or 'security'")
    source = task[which + "_tests"]
    n = max(1, len(list_test_names(source)))
    per_test = int(task["timeout_s"])
    files = {"solution.py": solution, "tests.py": source, "runner.py": RUNNER,
             "config.json": json.dumps({"function_name": task["function_name"], "per_test_s": per_test})}
    res = sandbox.run_python(files, "runner.py", timeout_s=min(60, per_test * (n + 1)), mem_mb=int(task["memory_mb"]))
    if res.timed_out:
        return SuiteRun("timeout", detail="suite exceeded the wall-clock limit")
    lines = [l for l in res.stdout.splitlines() if l.startswith("RESULT_JSON ")]
    if not lines:
        return SuiteRun("crash", detail=(res.stderr or res.stdout)[-500:] + f" (exit code {res.returncode})")
    payload = json.loads(lines[-1][len("RESULT_JSON "):])
    if "fatal" in payload:
        return SuiteRun(payload["fatal"], detail=payload.get("detail", ""))
    return SuiteRun("ok", results=payload["results"])


def load_tasks(path):
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: not valid JSON ({e})") from e
    return tasks


def _imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _check_task(t):
    if not isinstance(t, dict):
        return ["task is not a JSON object"]
    errs = []
    if set(t) - set(FIELDS):
        errs.append(f"unknown fields: {sorted(set(t) - set(FIELDS))}")
    if set(FIELDS) - set(t):
        errs.append(f"missing fields: {sorted(set(FIELDS) - set(t))}")
        return errs
    for f in STR_FIELDS:
        if not isinstance(t[f], str) or not t[f].strip():
            errs.append(f"{f} must be a non-empty string")
    if not isinstance(t["context"], str):
        errs.append("context must be a string (may be empty)")
    for f in LIST_FIELDS:
        v = t[f]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            errs.append(f"{f} must be a non-empty list of non-empty strings")
    for f, lo, hi in (("timeout_s", 1, 30), ("memory_mb", 64, 1024)):
        if isinstance(t[f], bool) or not isinstance(t[f], int) or not lo <= t[f] <= hi:
            errs.append(f"{f} must be an integer between {lo} and {hi}")
    if errs:
        return errs
    if not TASK_ID.match(t["task_id"]):
        errs.append("task_id must look like CWE78-001")
    if not CWE.match(t["cwe"]):
        errs.append("cwe must look like CWE-78")
    elif TASK_ID.match(t["task_id"]) and t["task_id"].split("-")[0] != "CWE" + t["cwe"][4:]:
        errs.append("task_id prefix does not match cwe")
    if t["language"] != "python":
        errs.append("language must be 'python'")
    if t["signature"] not in t["prompt"]:
        errs.append("prompt does not contain the signature")
    if not re.search(r"\bdef\s+" + re.escape(t["function_name"]) + r"\b", t["signature"]):
        errs.append("signature does not define function_name")
    for f in ("functional_tests", "security_tests", "secure_reference", "vulnerable_reference"):
        try:
            tree = ast.parse(t[f])
        except SyntaxError as e:
            errs.append(f"{f} has a syntax error: {e.msg} (line {e.lineno})")
            continue
        bad = _imports(tree) & NETWORK_MODULES
        if bad:
            errs.append(f"{f} imports network modules {sorted(bad)}")
        if f.endswith("_reference"):
            if not any(isinstance(n, ast.FunctionDef) and n.name == t["function_name"] for n in tree.body):
                errs.append(f"{f} does not define {t['function_name']}")
        else:
            if not list_test_names(t[f]):
                errs.append(f"{f} has no test_* function")
    return errs


def validate_tasks(tasks):
    """Return a list of error strings; empty means the task file is valid."""
    errors = []
    seen = {}
    for i, t in enumerate(tasks, start=1):
        tid = t.get("task_id") if isinstance(t, dict) else None
        for e in _check_task(t):
            errors.append(f"line {i} ({tid}): {e}")
        if isinstance(tid, str):
            seen.setdefault(tid, []).append(i)
    for tid, lines in seen.items():
        if len(lines) > 1:
            errors.append(f"duplicate task_id {tid!r} on lines {lines}")
    return errors
