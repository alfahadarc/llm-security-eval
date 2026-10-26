"""Score a raw run against the dataset. CPU only, safe to re-run after tuning the scorer.

Usage: python src/score_results.py --raw results/raw/<experiment>/<model>.jsonl
Writes results/scored/<experiment>/<model>.jsonl and prints a per-case table and label counts.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from eval.prompt_injection import score
from eval.validate_dataset import load_jsonl, select_pairs
from results_io import model_slug, read_records

DEFAULT_DATASET = Path("data/original/prompt_injection_v0.1.jsonl")
COPIED = ("pair_id", "condition", "objective", "technique")
FLAGS = ("utility_success", "attack_success", "format_success", "refusal",
         "canary_leak", "marker_present", "action_claim")


def score_records(raw_records, dataset_records):
    """Join raw records to dataset records by case_id and score each one.

    Raises ValueError for a duplicate raw case_id, a raw case_id missing from the dataset,
    or a dataset case with no raw record.
    """
    by_id = {r["case_id"]: r for r in dataset_records}
    seen = set()
    scored = []
    for raw in raw_records:
        cid = raw["case_id"]
        if cid in seen:
            raise ValueError(f"duplicate raw record for case_id {cid!r}")
        seen.add(cid)
        if cid not in by_id:
            raise ValueError(f"raw case_id {cid!r} is not in the dataset")
        rec = by_id[cid]
        s = score(rec, raw["response"], raw["status"])
        scored.append({
            "run_id": raw["run_id"], "case_id": cid, "model_id": raw["model_id"],
            **{k: rec[k] for k in COPIED},
            "label": s["label"],
            **{k: s[k] for k in FLAGS},
            "parse_failure": s["parse_failure"],
            "response": raw["response"],
        })
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"dataset cases with no raw record: {missing}")
    return scored


def scored_path(root, experiment, model_id):
    return Path(root) / "scored" / experiment / f"{model_slug(model_id)}.jsonl"


def summarize(scored):
    """Label counts per condition."""
    counts = {}
    for s in scored:
        counts.setdefault(s["condition"], Counter())[s["label"]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Score a raw run against the dataset")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pairs", default=None,
                        help="score only these pairs (use the same value as in run_eval.py)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = read_records(args.raw)
    if not raw:
        print(f"ERROR: no records in {args.raw}")
        return 1
    try:
        dataset = load_jsonl(args.dataset)
        if args.pairs:
            dataset = select_pairs(dataset, args.pairs)
        scored = score_records(raw, dataset)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    out = args.out or scored_path(args.results_dir, raw[0]["experiment"], raw[0]["model_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:  # rescoring overwrites the scored file by design
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    for s in scored:
        flags = ",".join(k for k in FLAGS if s[k]) or "-"
        head = s["response"].strip().replace("\n", " | ")[:80]
        print(f"{s['case_id']:<16} {s['label']:<15} {flags:<28} {head}")
    print()
    for cond, counter in sorted(summarize(scored).items()):
        print(f"{cond:<7} " + ", ".join(f"{label}={n}" for label, n in sorted(counter.items())))
    print(f"scored records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
