"""Score every dataset case with the Prompt Guard 2 detector. No generation, CPU by default.

Usage: python src/run_detector.py --dataset data/original/prompt_injection_v0.2.jsonl
Writes results/detector/prompt_guard_2_86m/<dataset name>.jsonl: one row per case with the score of each boundary
(the untrusted document and the user prompt checked separately) and their maximum. The threshold is NOT applied here;
it is chosen on the dev pairs (src/choose_threshold.py) and applied in src/run_defense_b.py.
The detector does not depend on the model under test, so one score file serves every model. Refuses to overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

from defenses.prompt_guard import MODEL_ID, REVISION, PromptGuard
from eval.validate_dataset import load_jsonl

DETECTOR_NAME = "prompt_guard_2_86m"
DEFAULT_DATASET = Path("data/original/prompt_injection_v0.2.jsonl")


def detector_path(results_dir, dataset_path):
    return Path(results_dir) / "detector" / DETECTOR_NAME / f"{Path(dataset_path).stem}.jsonl"


def score_case(detector, record):
    """Score the untrusted document and the user prompt separately; the case score is the larger one."""
    doc, doc_segs, doc_s = detector.score_text(record["untrusted_content"])
    usr, usr_segs, usr_s = detector.score_text(record["user_prompt"])
    return {
        "case_id": record["case_id"], "pair_id": record["pair_id"], "condition": record["condition"],
        "split": record["split"], "delivery": record.get("delivery"), "objective": record.get("objective"),
        "technique": record.get("technique"),
        "doc_score": doc, "user_score": usr, "score": max(doc, usr),
        "max_boundary": "document" if doc >= usr else "user_prompt",
        "doc_segments": doc_segs, "user_segments": usr_segs, "runtime_s": doc_s + usr_s,
        "detector": MODEL_ID, "revision": REVISION, "device": detector.device,
    }


def main():
    parser = argparse.ArgumentParser(description="Score dataset cases with Prompt Guard 2")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = detector_path(args.results_dir, args.dataset)
    if out.exists():
        print(f"ERROR: {out} already exists; refusing to overwrite")
        return 1
    records = load_jsonl(args.dataset)
    detector = PromptGuard(device=args.device)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for n, rec in enumerate(records, start=1):
            row = score_case(detector, rec)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{n}/{len(records)}] {row['case_id']} score={row['score']:.4f} ({row['max_boundary']}) {row['runtime_s']:.2f}s")
    print(f"detector scores written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
