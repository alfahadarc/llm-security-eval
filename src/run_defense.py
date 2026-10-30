"""Apply Defense A (capability gate) to a finished raw run. CPU only, no model is loaded.

Usage: python src/run_defense.py --raw results/raw/<experiment>/<model>.jsonl --dataset <dataset.jsonl>
Writes:
  results/defended/capability_gate/<experiment>/<model>.jsonl     (defended result matched to the undefended one by case_id)
  results/defense_logs/capability_gate/<experiment>/<model>.jsonl  (every allow/block decision, rules fired, policy)
Blocking never changes what the model generates, so the matching undefended result is the existing scored run.
Refuses to overwrite existing output.
"""
import argparse
import json
import sys
from pathlib import Path

from defenses.capability_gate import gate_response, policy_for_case
from eval.validate_dataset import load_jsonl
from results_io import model_slug, read_records
from score_results import FLAGS, score_records

DEFENSE = "capability_gate"
STRICT_DEFENSE = "capability_gate_strict"  # same gate, strict "- " bullets (sensitivity variant)
DEFAULT_DATASET = Path("data/original/prompt_injection_v0.2.jsonl")


def apply_gate(raw_records, dataset_records, strict=False):
    """Return (defended_rows, decision_rows), one of each per raw record."""
    scored = {s["case_id"]: s for s in score_records(raw_records, dataset_records)}
    by_id = {r["case_id"]: r for r in dataset_records}
    name = STRICT_DEFENSE if strict else DEFENSE
    defended, decisions = [], []
    for raw in raw_records:
        cid = raw["case_id"]
        rec, base = by_id[cid], scored[cid]
        verdict = gate_response(rec, raw["response"], lenient_bullets=not strict)
        blocked = verdict["decision"] == "block"
        decisions.append({
            "run_id": raw["run_id"], "case_id": cid, "model_id": raw["model_id"], "defense": name,
            "decision": verdict["decision"], "rules": verdict["rules"], "policy": policy_for_case(rec, lenient_bullets=not strict),
        })
        defended.append({
            "run_id": raw["run_id"], "case_id": cid, "model_id": raw["model_id"], "defense": name,
            "pair_id": base["pair_id"], "condition": base["condition"],
            "objective": base["objective"], "technique": base["technique"],
            "blocked": blocked, "rules": verdict["rules"],
            "undefended_label": base["label"],
            **{f"undefended_{k}": base[k] for k in FLAGS},
            "attack_success": base["attack_success"] and not blocked,   # a blocked reply never reaches the user
            "utility_success": base["utility_success"] and not blocked,
            "utility_lenient": base["utility_lenient"] and not blocked,
        })
    return defended, decisions


def summarize(defended):
    """Attack success and clean/benign utility, before and after the gate."""
    def count(rows, cond, key):
        sel = [r for r in rows if r["condition"] == cond]
        return sum(1 for r in sel if r[key]), len(sel)
    lines = []
    n = count(defended, "attack", "undefended_attack_success")[1]
    lines.append(f"attack success   before {count(defended, 'attack', 'undefended_attack_success')[0]}/{n}"
                 f"  after {count(defended, 'attack', 'attack_success')[0]}/{n}")
    for cond in ("clean", "benign"):
        n = count(defended, cond, "undefended_utility_success")[1]
        lines.append(f"{cond:<7} utility  before {count(defended, cond, 'undefended_utility_success')[0]}/{n}"
                     f"  after {count(defended, cond, 'utility_success')[0]}/{n}"
                     f"   | lenient before {count(defended, cond, 'undefended_utility_lenient')[0]}/{n}"
                     f"  after {count(defended, cond, 'utility_lenient')[0]}/{n}")
    lines.append(f"blocked replies  {sum(r['blocked'] for r in defended)}/{len(defended)}")
    return lines


def out_paths(root, experiment, model_id, strict=False):
    name = f"{model_slug(model_id)}.jsonl"
    defense = STRICT_DEFENSE if strict else DEFENSE
    return (Path(root) / "defended" / defense / experiment / name,
            Path(root) / "defense_logs" / defense / experiment / name)


def main():
    parser = argparse.ArgumentParser(description="Apply the capability gate to a raw run")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--strict", action="store_true",
                        help="sensitivity variant: strict '- ' bullets only (output folder capability_gate_strict); "
                             "the default is the lenient bullet style")
    args = parser.parse_args()

    raw = read_records(args.raw)
    if not raw:
        print(f"ERROR: no records in {args.raw}")
        return 1
    defended_path, log_path = out_paths(args.results_dir, raw[0]["experiment"], raw[0]["model_id"], args.strict)
    for p in (defended_path, log_path):
        if p.exists():
            print(f"ERROR: {p} already exists; refusing to overwrite")
            return 1
    try:
        defended, decisions = apply_gate(raw, load_jsonl(args.dataset), args.strict)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    for path, rows in ((defended_path, defended), (log_path, decisions)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{raw[0]['model_id']} ({raw[0]['experiment']})")
    print("\n".join(summarize(defended)))
    print(f"defended results: {defended_path}\ndecision log:     {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
