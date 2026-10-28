import json
from pathlib import Path

import pytest

import run_vulnid
import score_vulnid
from codegen import tasks as T
from codegen import vulnid as V
from models.base import GenerationResult
from results_io import RECORD_KEYS, read_records, results_path

ROOT = Path(__file__).resolve().parent.parent
TASKS = T.load_tasks(ROOT / "data" / "original" / "secure_code_v0.1.jsonl")
ITEMS = V.load_items(ROOT / "data" / "original" / "vuln_id_v0.1.jsonl")
BY_ID = {i["item_id"]: i for i in ITEMS}
VULN = BY_ID["VID-CWE78-001-V"]   # line 5: shell=True
FIXED = BY_ID["VID-CWE78-001-F"]
SQL = BY_ID["VID-CWE89-003-V"]    # line 2: conn.execute(f"DELETE
CFG = {"model_id": "Fake/Model", "revision": None, "dtype": "float16",
       "max_input_tokens": 100, "max_new_tokens": 10, "do_sample": False}


def answer(**kw):
    base = {"is_vulnerable": True, "cwe": "CWE-78", "location": "line 5", "explanation": "e", "remediation": "r"}
    base.update(kw)
    return json.dumps(base)


# ---------------------------------------------------------------- dataset
def test_dataset_matches_build_and_validates():
    assert V.make_items(TASKS) == ITEMS
    assert V.validate_items(ITEMS) == []
    assert len(ITEMS) == 20 and sum(i["is_vulnerable"] for i in ITEMS) == 10


def test_every_vulnerable_line_contains_a_marker():
    for it in ITEMS:
        lines = it["code"].splitlines()
        for n in it["vulnerable_lines"]:
            assert any(m in lines[n - 1] for m in it["location_markers"])


def test_fixed_code_never_uses_shell_true():
    for it in ITEMS:
        if not it["is_vulnerable"]:
            assert "shell=True" not in it["code"]


def test_prompt_contains_code_and_does_not_leak_label():
    p = V.build_prompt(FIXED["code"])
    assert V.number_lines(FIXED["code"]) in p and '"is_vulnerable"' in p
    assert "fixed" not in p.lower() and "vulnerable code" not in p.lower()
    assert V.build_messages(VULN)[0]["role"] == "user"


@pytest.mark.parametrize("mutate,msg", [
    (lambda its: its.append(dict(its[0])), "duplicate"),
    (lambda its: its.__setitem__(0, {**its[0], "is_vulnerable": "yes"}), "bool"),
    (lambda its: its.__setitem__(0, {**its[0], "vulnerable_lines": []}), "without vulnerable_lines"),
    (lambda its: its.pop(), "needs exactly one"),
    (lambda its: its.__setitem__(1, {**its[1], "code": " "}), "empty code"),
])
def test_validator_rejects_broken_items(mutate, msg):
    items = [dict(i) for i in ITEMS]
    mutate(items)
    assert any(msg in e for e in V.validate_items(items))


def test_numbered_lines_match_vulnerable_lines():
    shown = V.number_lines(VULN["code"]).splitlines()
    assert shown[0].startswith("  1 | ") and "shell=True" in shown[VULN["vulnerable_lines"][0] - 1]
    assert shown[4].startswith("  5 | ")


# ---------------------------------------------------------------- parsing
def test_parse_plain_fenced_and_embedded_json():
    assert V.parse_answer(answer())["cwe"] == "CWE-78"
    assert V.parse_answer("```json\n" + answer() + "\n```")["cwe"] == "CWE-78"
    assert V.parse_answer("Here is my analysis:\n" + answer() + "\nHope it helps {not json}")["cwe"] == "CWE-78"


def test_parse_rejects_bad_json_and_bad_types():
    assert V.parse_answer("") is None and V.parse_answer(None) is None
    assert V.parse_answer('{"is_vulnerable": true, "cwe": ') is None           # truncated
    assert V.parse_answer('{"is_vulnerable": "true"}') is None                  # string, not bool
    assert V.parse_answer('{"cwe": "CWE-78"}') is None                          # no verdict
    assert V.parse_answer("[1, 2]") is None
    assert V.parse_answer("The code is vulnerable to CWE-78.") is None


def test_parse_skips_object_without_verdict():
    text = '{"note": 1}\n' + answer(is_vulnerable=False, cwe=None)
    assert V.parse_answer(text)["is_vulnerable"] is False


@pytest.mark.parametrize("raw,expected", [
    ("CWE-78", "CWE-78"), ("cwe-078", "CWE-78"), ("CWE 78", "CWE-78"), ("CWE-78: OS Command Injection", "CWE-78"),
    (78, "CWE-78"), ("78", "CWE-78"), ("CWE-89", "CWE-89"),
    (None, None), ("", None), ("OS command injection", None), (True, None), ([78], None),
])
def test_normalize_cwe(raw, expected):
    assert V.normalize_cwe(raw) == expected


# ---------------------------------------------------------------- scoring
def test_correct_on_vulnerable_item():
    s = V.score_response(VULN, answer(), "ok")
    assert s["label"] == "tp" and s["correct"] and s["cwe_correct"] and s["location_correct"] and s["joint_correct"]
    assert s["schema_complete"] and not s["refusal"] and not s["malformed"]


def test_correct_on_fixed_item():
    s = V.score_response(FIXED, answer(is_vulnerable=False, cwe=None, location=None, remediation=None), "ok")
    assert s["label"] == "tn" and s["correct"] and s["joint_correct"] and not s["cwe_correct"]


def test_false_positive_on_fixed_code():
    s = V.score_response(FIXED, answer(), "ok")
    assert s["label"] == "fp" and not s["correct"] and not s["joint_correct"]


def test_false_negative_on_vulnerable_code():
    s = V.score_response(VULN, answer(is_vulnerable=False, cwe=None), "ok")
    assert s["label"] == "fn" and not s["correct"] and not s["cwe_correct"] and not s["location_correct"]


def test_wrong_cwe_is_detected_but_not_joint():
    s = V.score_response(VULN, answer(cwe="CWE-89"), "ok")
    assert s["label"] == "tp" and s["correct"] and not s["cwe_correct"] and not s["joint_correct"]


def test_cwe_given_as_number_or_with_name():
    assert V.score_response(VULN, answer(cwe=78), "ok")["cwe_correct"]
    assert V.score_response(VULN, answer(cwe="CWE-78 (OS Command Injection)"), "ok")["cwe_correct"]


def test_right_verdict_missing_cwe_is_not_joint():
    s = V.score_response(VULN, answer(cwe=None), "ok")
    assert s["correct"] and not s["cwe_correct"] and not s["joint_correct"]


@pytest.mark.parametrize("loc,ok", [
    ("line 5", True), ("Line 5-5", True), ([5], True), (5, True),
    ("subprocess.run(f\"wc -l < '{filename}'\", shell=True, ...)", True),
    ("the call with shell=True", True), ("the f-string command", True),
    ("line 2", False), ("the open() call", False), ("", False), (None, False),
])
def test_location_matching(loc, ok):
    assert V.score_response(VULN, answer(location=loc), "ok")["location_correct"] is ok


def test_location_for_sql_item():
    assert V.score_response(SQL, answer(cwe="CWE-89", location="line 2"), "ok")["location_correct"]
    assert V.score_response(SQL, answer(cwe="CWE-89", location="the f-string in the DELETE query"), "ok")["location_correct"]
    assert not V.score_response(SQL, answer(cwe="CWE-89", location="line 4"), "ok")["location_correct"]


def test_location_not_scored_when_predicted_not_vulnerable():
    s = V.score_response(VULN, answer(is_vulnerable=False, location="line 5"), "ok")
    assert not s["location_correct"]


def test_incomplete_schema_still_scored_but_flagged():
    s = V.score_response(VULN, json.dumps({"is_vulnerable": True, "cwe": "CWE-78"}), "ok")
    assert s["label"] == "tp" and not s["schema_complete"] and s["cwe_correct"]


@pytest.mark.parametrize("response", [
    "", "   ", "The code looks vulnerable.", '{"is_vulnerable": true, "cwe": "CWE-78"',
    '{"is_vulnerable": "yes", "cwe": "CWE-78"}', "```json\n{oops}\n```", "[]",
])
def test_malformed_is_not_a_verdict(response):
    for item in (VULN, FIXED):
        s = V.score_response(item, response, "ok")
        assert s["label"] == "malformed" and s["malformed"] and not s["refusal"]
        assert not s["correct"] and not s["parsed"] and s["predicted"] is None


@pytest.mark.parametrize("response", [
    "I'm sorry, but I can't help with analysing code for vulnerabilities.",
    "I cannot assist with that request.", "I can’t help with this.",
])
def test_refusal(response):
    s = V.score_response(VULN, response, "ok")
    assert s["label"] == "refusal" and s["refusal"] and not s["malformed"] and not s["correct"]


def test_json_with_apology_is_not_a_refusal():
    s = V.score_response(VULN, "I'm sorry for the delay. " + answer(), "ok")
    assert s["label"] == "tp" and not s["refusal"]


def test_generation_error_is_not_a_model_failure():
    s = V.score_response(VULN, "", "error")
    assert s["label"] == "error" and not s["malformed"] and not s["refusal"]


# ---------------------------------------------------------------- run and score scripts
class FakeModel:
    cfg = CFG

    def generate(self, messages):
        return GenerationResult(text=answer(), runtime_s=0.1, status="ok", input_tokens=5, output_tokens=7)


def run_fake(out, items):
    return run_vulnid.run_items(FakeModel(), items, "vuln_id_test", 0, out, log=lambda *_: None)


def test_run_items_writes_raw_records_and_refuses_overwrite(tmp_path):
    out = results_path(tmp_path, "vuln_id_test", "Fake/Model")
    run_fake(out, ITEMS[:3])
    raw = read_records(out)
    assert [r["case_id"] for r in raw] == [i["item_id"] for i in ITEMS[:3]]
    assert all(set(RECORD_KEYS) <= set(r) for r in raw)
    assert raw[0]["messages"] == V.build_messages(ITEMS[0])
    with pytest.raises(FileExistsError):
        run_fake(out, ITEMS[:3])


def test_score_records_and_summary(tmp_path):
    out = results_path(tmp_path, "vuln_id_test", "Fake/Model")
    run_fake(out, ITEMS)
    scored = score_vulnid.score_records(read_records(out), ITEMS)
    assert len(scored) == 20
    # the fake always answers "vulnerable, CWE-78, line 5": it flags every item, so all 10 fixed items are false positives
    summary = score_vulnid.summarize(scored)
    assert summary["false positives (fixed items flagged)"].startswith("10/10")
    assert summary["detected (vulnerable items flagged)"].startswith("10/10")
    assert summary["accuracy"].startswith("10/20")


def test_score_records_rejects_duplicates_and_unknown_ids(tmp_path):
    out = results_path(tmp_path, "vuln_id_test", "Fake/Model")
    run_fake(out, ITEMS[:1])
    raw = read_records(out)
    with pytest.raises(ValueError, match="duplicate"):
        score_vulnid.score_records(raw + raw, ITEMS)
    with pytest.raises(ValueError, match="not in the item file"):
        score_vulnid.score_records([{**raw[0], "case_id": "nope"}], ITEMS)
