"""Score a raw secure-code run. Runs the generated code in the sandbox (CPU only), safe to re-run.

Usage: python src/score_codegen.py --raw results/raw/<experiment>/<model>.jsonl
Writes results/scored/<experiment>/<model>.jsonl and prints one line per task plus label counts.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from codegen.score import score_response
from codegen.tasks import load_tasks
from results_io import model_slug, read_records

DEFAULT_TASKS = Path("data/original/secure_code_v0.1.jsonl")
FLAGS = ("syntax_ok", "functional_success", "security_success", "joint_success", "refusal", "runtime_error")


def score_records(raw_records, tasks):
    """Join raw records to tasks by task_id (= raw case_id) and score each one."""
    by_id = {t["task_id"]: t for t in tasks}
    seen, scored = set(), []
    for raw in raw_records:
        tid = raw["case_id"]
        if tid in seen:
            raise ValueError(f"duplicate raw record for task {tid!r}")
        seen.add(tid)
        if tid not in by_id:
            raise ValueError(f"raw task id {tid!r} is not in the task file")
        s = score_response(by_id[tid], raw["response"], raw["status"])
        scored.append({"run_id": raw["run_id"], "model_id": raw["model_id"], **s,
                       "output_tokens": raw.get("output_tokens"),
                       "max_new_tokens": (raw.get("generation") or {}).get("max_new_tokens"),
                       "response": raw["response"]})
    return scored


def main():
    parser = argparse.ArgumentParser(description="Score a raw secure-code run")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = read_records(args.raw)
    if not raw:
        print(f"ERROR: no records in {args.raw}")
        return 1
    try:
        scored = score_records(raw, load_tasks(args.tasks))
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    out = args.out or (args.results_dir / "scored" / raw[0]["experiment"] / f"{model_slug(raw[0]['model_id'])}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    for s in scored:
        flags = ",".join(f for f in FLAGS if s[f]) or "-"
        cut = " (hit the token limit)" if s["output_tokens"] and s["output_tokens"] == s["max_new_tokens"] else ""
        print(f"{s['task_id']:<11} {s['label']:<16} {flags:<46}{cut}")
    print()
    print(", ".join(f"{k}={v}" for k, v in sorted(Counter(s["label"] for s in scored).items())))
    print(f"joint success: {sum(s['joint_success'] for s in scored)} of {len(scored)}")
    print(f"scored records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
