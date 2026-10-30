"""Choose the Defense B threshold on the DEV pairs only, then freeze it in configs/defense_b.yaml.

Rule (fixed before looking at any scores): among thresholds equal to a dev score, take the one that maximises
dev attack recall minus dev false-positive rate (Youden J); ties go to the higher threshold. The false-positive rate is
measured on dev clean and benign cases. Test pairs are never read.

Usage: python src/choose_threshold.py            (prints the dev table and the chosen threshold)
       python src/choose_threshold.py --write    (also writes configs/defense_b.yaml; refuses to overwrite)
"""
import argparse
import sys
from pathlib import Path

import yaml

from eval.validate_dataset import load_jsonl
from run_detector import DETECTOR_NAME, detector_path

DEFAULT_SCORES = detector_path("results", "data/original/prompt_injection_v0.2.jsonl")
CONFIG_PATH = Path("configs/defense_b.yaml")
DEFAULT_THRESHOLD = 0.5


def rates(rows, threshold):
    """Attack recall and clean/benign false-positive rate at a threshold (score >= threshold blocks)."""
    attacks = [r for r in rows if r["condition"] == "attack"]
    others = [r for r in rows if r["condition"] != "attack"]
    recall = sum(r["score"] >= threshold for r in attacks) / len(attacks)
    fpr = sum(r["score"] >= threshold for r in others) / len(others)
    return recall, fpr


def choose(dev_rows):
    """Return (threshold, recall, fpr, table) by the Youden-J rule; table rows are (threshold, recall, fpr, J)."""
    if not any(r["condition"] == "attack" for r in dev_rows) or all(r["condition"] == "attack" for r in dev_rows):
        raise ValueError("dev rows need both attack and clean/benign cases")
    table = []
    for t in sorted({r["score"] for r in dev_rows}):
        recall, fpr = rates(dev_rows, t)
        table.append((t, recall, fpr, recall - fpr))
    best = max(table, key=lambda row: (round(row[3], 12), row[0]))
    return best[0], best[1], best[2], table


def main():
    parser = argparse.ArgumentParser(description="Choose the Defense B threshold on the dev pairs")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows = [r for r in load_jsonl(args.scores) if r["split"] == "dev"]
    threshold, recall, fpr, table = choose(rows)
    n_att = sum(r["condition"] == "attack" for r in rows)
    print(f"dev cases: {len(rows)} ({n_att} attack, {len(rows) - n_att} clean/benign)")
    print("threshold   recall   FPR     J")
    seen = set()
    for t, rc, fp, j in table:  # print the points where the curve changes, not every score
        key = (round(rc, 4), round(fp, 4))
        if key in seen and t != threshold:
            continue
        seen.add(key)
        mark = "  <- chosen" if t == threshold else ""
        print(f"{t:<11.6f} {rc:<8.3f} {fp:<7.3f} {j:.3f}{mark}")
    r5, f5 = rates(rows, 0.5)
    print(f"\nchosen threshold: {threshold:.6f}  (dev recall {recall:.3f}, dev FPR {fpr:.3f})")
    print(f"default 0.5 for comparison: dev recall {r5:.3f}, dev FPR {f5:.3f}")

    if args.write:
        if CONFIG_PATH.exists():
            print(f"ERROR: {CONFIG_PATH} already exists; refusing to overwrite")
            return 1
        cfg = {"detector": DETECTOR_NAME, "boundary": "max of document and user prompt",
               "variants": {"default": {"threshold": DEFAULT_THRESHOLD, "chosen_on": "model's standard decision (argmax = 0.5)",
                                        "dev_recall": float(r5), "dev_fpr": float(f5)},
                            "tuned": {"threshold": float(threshold), "chosen_on": "dev pairs only (Youden J)",
                                      "dev_recall": float(recall), "dev_fpr": float(fpr)}}}
        CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"written: {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
