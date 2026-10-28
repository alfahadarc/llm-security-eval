"""Score a raw vulnerability-identification run (CPU only, safe to re-run).

Usage: python src/score_vulnid.py --raw results/raw/<experiment>/<model>.jsonl
Writes results/scored/<experiment>/<model>.jsonl, prints one line per item and the summary rates.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from codegen.vulnid import load_items, score_response
from results_io import model_slug, read_records

DEFAULT_ITEMS = Path("data/original/vuln_id_v0.1.jsonl")


def score_records(raw_records, items):
    """Join raw records to items by item_id (= raw case_id) and score each one."""
    by_id = {t["item_id"]: t for t in items}
    seen, scored = set(), []
    for raw in raw_records:
        iid = raw["case_id"]
        if iid in seen:
            raise ValueError(f"duplicate raw record for item {iid!r}")
        seen.add(iid)
        if iid not in by_id:
            raise ValueError(f"raw item id {iid!r} is not in the item file")
        s = score_response(by_id[iid], raw["response"], raw["status"])
        scored.append({"run_id": raw["run_id"], "model_id": raw["model_id"], **s,
                       "output_tokens": raw.get("output_tokens"),
                       "max_new_tokens": (raw.get("generation") or {}).get("max_new_tokens"),
                       "response": raw["response"]})
    return scored


def summarize(scored):
    """Rates over all items (denominator = every scored item, so refusals and malformed replies count as misses)."""
    n = len(scored)
    vuln = [s for s in scored if s["is_vulnerable"]]
    fixed = [s for s in scored if not s["is_vulnerable"]]

    def rate(k, d):
        return f"{k}/{d}" + (f" = {k / d:.0%}" if d else "")

    return {
        "items": n,
        "parsed": rate(sum(s["parsed"] for s in scored), n),
        "schema_complete": rate(sum(s["schema_complete"] for s in scored), n),
        "refusal": rate(sum(s["refusal"] for s in scored), n),
        "malformed": rate(sum(s["malformed"] for s in scored), n),
        "accuracy": rate(sum(s["correct"] for s in scored), n),
        "detected (vulnerable items flagged)": rate(sum(s["label"] == "tp" for s in vuln), len(vuln)),
        "false positives (fixed items flagged)": rate(sum(s["label"] == "fp" for s in fixed), len(fixed)),
        "CWE correct (vulnerable items)": rate(sum(s["cwe_correct"] for s in vuln), len(vuln)),
        "location correct (vulnerable items)": rate(sum(s["location_correct"] for s in vuln), len(vuln)),
        "joint correct (all items)": rate(sum(s["joint_correct"] for s in scored), n),
    }


def main():
    parser = argparse.ArgumentParser(description="Score a raw vulnerability-identification run")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = read_records(args.raw)
    if not raw:
        print(f"ERROR: no records in {args.raw}")
        return 1
    try:
        scored = score_records(raw, load_items(args.items))
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    out = args.out or (args.results_dir / "scored" / raw[0]["experiment"] / f"{model_slug(raw[0]['model_id'])}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    for s in scored:
        cut = " (hit the token limit)" if s["output_tokens"] and s["output_tokens"] == s["max_new_tokens"] else ""
        print(f"{s['item_id']:<17} {s['label']:<10} cwe={s['predicted_cwe'] or '-':<8} "
              f"cwe_ok={int(s['cwe_correct'])} loc_ok={int(s['location_correct'])}{cut}")
    print()
    print(", ".join(f"{k}={v}" for k, v in sorted(Counter(s["label"] for s in scored).items())))
    for k, v in summarize(scored).items():
        print(f"{k}: {v}")
    print(f"scored records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
