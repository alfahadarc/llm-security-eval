"""Check Step 16's "Done when": every defended result has a matching undefended result for the same case, model and seed,
and every defended case has a logged decision. CPU only, reads files, changes nothing.

Usage: python src/verify_matching.py            (all defended settings for the PI experiment, all models)
"""
import argparse
import json
import sys
from pathlib import Path

from results_io import read_records
from score_results import FLAGS

EXPERIMENT = "prompt_injection_v0.2_full"


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def verify(raw_path, scored_path, defended_path, log_path):
    """Return a list of problems (empty = every defended row matches its undefended row and has a logged decision)."""
    problems = []
    raw = {r["case_id"]: r for r in read_records(raw_path)}
    scored = {r["case_id"]: r for r in read_jsonl(scored_path)}
    defended = read_jsonl(defended_path)
    log = {r["case_id"]: r for r in read_jsonl(log_path)}
    seeds = {r["seed"] for r in raw.values()}
    if len(seeds) != 1:
        problems.append(f"raw run has several seeds: {sorted(seeds)}")
    ids = [d["case_id"] for d in defended]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case_id in defended rows")
    if set(ids) != set(raw):
        problems.append(f"defended cases differ from raw cases (missing {len(set(raw) - set(ids))}, extra {len(set(ids) - set(raw))})")
    if set(log) != set(ids):
        problems.append("decision log cases differ from defended cases")
    for d in defended:
        cid = d["case_id"]
        r, s, entry = raw.get(cid), scored.get(cid), log.get(cid)
        if r is None or s is None:
            problems.append(f"{cid}: no raw/scored record")
            continue
        if d["run_id"] != r["run_id"] or d["model_id"] != r["model_id"]:
            problems.append(f"{cid}: run_id/model_id differ from the raw record")
        if d["undefended_label"] != s["label"]:
            problems.append(f"{cid}: undefended_label {d['undefended_label']} != scored {s['label']}")
        for k in FLAGS:
            if d.get(f"undefended_{k}") != s[k]:
                problems.append(f"{cid}: undefended_{k} differs from the scored record")
        if d["attack_success"] and not d["undefended_attack_success"]:
            problems.append(f"{cid}: defense created an attack success")
        if d["blocked"] and (d["attack_success"] or d["utility_success"] or d["utility_lenient"]):
            problems.append(f"{cid}: blocked reply still counted as success")
        if entry is not None:
            if entry.get("decision", "block" if d["blocked"] else "allow") != ("block" if d["blocked"] else "allow"):
                problems.append(f"{cid}: logged decision disagrees with the defended row")
            if entry["run_id"] != r["run_id"] or entry["model_id"] != r["model_id"]:
                problems.append(f"{cid}: log run_id/model_id differ from the raw record")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Verify defended results against undefended results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args()

    root = args.results_dir
    settings = sorted(p.name for p in (root / "defended").iterdir() if (p / args.experiment).is_dir())
    total_files, failed = 0, 0
    for setting in settings:
        for defended_path in sorted((root / "defended" / setting / args.experiment).glob("*.jsonl")):
            name = defended_path.name
            problems = verify(root / "raw" / args.experiment / name, root / "scored" / args.experiment / name,
                              defended_path, root / "defense_logs" / setting / args.experiment / name)
            total_files += 1
            status = "OK" if not problems else f"{len(problems)} PROBLEMS"
            print(f"{setting:<22} {name:<42} {status}")
            for p in problems[:5]:
                print(f"    {p}")
            failed += bool(problems)
    print(f"\n{total_files} defended files checked, {failed} with problems")
    return 1 if failed or not total_files else 0


if __name__ == "__main__":
    sys.exit(main())
