"""Apply Defense B (Prompt Guard 2 detector) alone and combined with Defense A (capability gate) to a finished raw run.

Usage: python src/run_defense_b.py --raw results/raw/<experiment>/<model>.jsonl
CPU only. Reads the detector scores written by src/run_detector.py and the thresholds frozen in configs/defense_b.yaml.
Builds four settings per model (the other two, no defense and gate only, already exist as the scored run and
results/defended/capability_gate):
  detector_default   detector at the standard 0.5 threshold
  detector_tuned     detector at the dev-tuned threshold
  both_default       detector (0.5) or gate blocks
  both_tuned         detector (dev-tuned) or gate blocks
Writes results/defended/<setting>/<experiment>/<model>.jsonl and results/defense_logs/<setting>/... (every decision
with detector name, revision, boundary, document and user-prompt scores, threshold, decision, runtime).
A request blocked by the detector never reaches the model, so the matched undefended result is the existing scored run.
Refuses to overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from defenses.prompt_guard import decide
from eval.validate_dataset import load_jsonl
from results_io import model_slug, read_records
from run_defense import apply_gate
from run_detector import DEFAULT_DATASET, detector_path

SETTINGS = (("detector_default", "default", False), ("detector_tuned", "tuned", False),
            ("both_default", "default", True), ("both_tuned", "tuned", True))
CONFIG_PATH = Path("configs/defense_b.yaml")


def build_setting(gate_rows, det_rows, threshold, use_gate, setting):
    """Combine the gate rows (with undefended fields) and the detector rows into defended and decision rows."""
    det = {r["case_id"]: r for r in det_rows}
    defended, decisions = [], []
    for g in gate_rows:
        d = det.get(g["case_id"])
        if d is None:
            raise ValueError(f"no detector score for case_id {g['case_id']!r}")
        det_block = decide(d["score"], threshold) == "block"
        gate_block = use_gate and g["blocked"]
        blocked = det_block or gate_block
        by = "both" if det_block and gate_block else "detector" if det_block else "gate" if gate_block else None
        row = {k: v for k, v in g.items() if k not in ("defense", "blocked", "rules", "attack_success",
                                                       "utility_success", "utility_lenient")}
        row.update({
            "defense": setting, "split": d["split"], "blocked": blocked, "blocked_by": by,
            "rules": g["rules"] if gate_block else [],
            "detector_score": d["score"], "detector_threshold": threshold,
            "attack_success": g["undefended_attack_success"] and not blocked,
            "utility_success": g["undefended_utility_success"] and not blocked,
            "utility_lenient": g["undefended_utility_lenient"] and not blocked,
        })
        defended.append(row)
        decisions.append({
            "run_id": g["run_id"], "case_id": g["case_id"], "model_id": g["model_id"], "defense": setting,
            "detector": d["detector"], "detector_revision": d["revision"], "boundary": d["max_boundary"],
            "doc_score": d["doc_score"], "user_score": d["user_score"], "score": d["score"],
            "threshold": threshold, "detector_decision": "block" if det_block else "allow",
            "detector_runtime_s": d["runtime_s"],
            "gate_decision": ("block" if g["blocked"] else "allow") if use_gate else None,
            "gate_rules": g["rules"] if use_gate else None,
            "decision": "block" if blocked else "allow", "blocked_by": by,
        })
    return defended, decisions


def summarize(defended, split=None):
    rows = [r for r in defended if split is None or r["split"] == split]
    def n(cond, key):
        sel = [r for r in rows if r["condition"] == cond]
        return sum(1 for r in sel if r[key]), len(sel)
    a0, an = n("attack", "undefended_attack_success")
    out = [f"attack success  {a0}/{an} -> {n('attack', 'attack_success')[0]}/{an}"]
    for cond in ("clean", "benign"):
        u0, un = n(cond, "undefended_utility_lenient")
        out.append(f"{cond:<7} lenient utility  {u0}/{un} -> {n(cond, 'utility_lenient')[0]}/{un}"
                   f"   blocked {sum(r['blocked'] for r in rows if r['condition'] == cond)}/{un}")
    return out


def out_paths(root, setting, experiment, model_id):
    name = f"{model_slug(model_id)}.jsonl"
    return (Path(root) / "defended" / setting / experiment / name,
            Path(root) / "defense_logs" / setting / experiment / name)


def main():
    parser = argparse.ArgumentParser(description="Apply the detector, alone and with the gate, to a raw run")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--detector-scores", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    raw = read_records(args.raw)
    if not raw:
        print(f"ERROR: no records in {args.raw}")
        return 1
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scores_path = args.detector_scores or detector_path(args.results_dir, args.dataset)
    experiment, model_id = raw[0]["experiment"], raw[0]["model_id"]
    paths = {s: out_paths(args.results_dir, s, experiment, model_id) for s, _, _ in SETTINGS}
    existing = [p for pair in paths.values() for p in pair if p.exists()]
    if existing:
        print(f"ERROR: {existing[0]} already exists; refusing to overwrite")
        return 1
    try:
        gate_rows, _ = apply_gate(raw, load_jsonl(args.dataset))
        det_rows = load_jsonl(scores_path)
        thresholds = {s: cfg["variants"][v]["threshold"] for s, v, _ in SETTINGS}
        results = {s: build_setting(gate_rows, det_rows, thresholds[s], gate, s) for s, _, gate in SETTINGS}
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1

    for setting, (defended, decisions) in results.items():
        for path, rows in zip(paths[setting], (defended, decisions)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{model_id} {setting} (threshold {thresholds[setting]:.6f})")
        for split in ("test", None):
            print(f"  [{split or 'all'}] " + " | ".join(summarize(defended, split)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
