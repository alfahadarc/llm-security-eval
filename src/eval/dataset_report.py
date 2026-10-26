"""Balance report for a dataset JSONL file. Usage: python src/eval/dataset_report.py <file.jsonl>

Counts pairs (one attack case per pair) by delivery, objective, technique and split, with
delivery x objective and delivery x technique cross-tables. Run the validator first.
"""
import json
import sys
from collections import Counter


def build_report(records):
    """Return {name: Counter} computed over the attack cases (one per pair)."""
    attacks = [r for r in records if r["condition"] == "attack"]
    report = {
        "pairs": Counter({"total": len(attacks)}),
        "delivery": Counter(r["delivery"] for r in attacks),
        "objective": Counter(r["objective"] for r in attacks),
        "technique": Counter(r["technique"] for r in attacks),
        "split": Counter(r.get("split", "none") for r in attacks),
        "delivery x objective": Counter(f"{r['delivery']} | {r['objective']}" for r in attacks),
        "delivery x technique": Counter(f"{r['delivery']} | {r['technique']}" for r in attacks),
        "delivery x split": Counter(f"{r['delivery']} | {r.get('split', 'none')}" for r in attacks),
        "split x objective": Counter(f"{r.get('split', 'none')} | {r['objective']}" for r in attacks),
    }
    return report


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    for name, counts in build_report(records).items():
        print(f"\n{name}")
        for key in sorted(counts):
            print(f"  {key:<55} {counts[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
