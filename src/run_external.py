"""Run one model on the CyberSecEval textual prompt-injection benchmark (native + normalized results).

Usage: python src/run_external.py --config configs/qwen3_4b_instruct.yaml
  all 251 English cases (default); `--limit 20 --seed 0` runs a seeded sample of 20.
Writes (all under --results-dir, default results/):
  raw/<experiment>/<model>.jsonl and raw/<experiment>_judge/<model>.jsonl   (every model call, full prompts)
  external/cyberseceval/native/<experiment>/<model>.jsonl and <model>_stats.json   (benchmark-style scoring)
  external/cyberseceval/normalized/<experiment>/<model>.jsonl                      (this project's naming)
The judge is the model under test itself (greedy); see data/external/cyberseceval/SOURCE.md.
"""
import argparse
import sys
from pathlib import Path

from external.cyberseceval import load_cases, output_paths, run_benchmark, select_cases

DEFAULT_DATASET = Path("data/external/cyberseceval/prompt_injection.json")
DEFAULT_EXPERIMENT = "cyberseceval_pi_english"


def main():
    parser = argparse.ArgumentParser(description="Run the CyberSecEval textual prompt-injection benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--limit", type=int, default=None, help="run a seeded random sample of this many cases")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    try:
        cases = select_cases(load_cases(args.dataset), args.limit, args.seed)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    print(f"{len(cases)} case(s) selected from {args.dataset}")

    # Heavy imports only after the cheap checks have passed.
    from models.huggingface_model import HuggingFaceChatModel, load_config
    from seed import set_seed

    paths = output_paths(args.results_dir, args.experiment, load_config(args.config)["model_id"])
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        print(f"ERROR: output already exists: {existing}; move it or use another --experiment")
        return 1

    seed = set_seed(args.seed)
    model = HuggingFaceChatModel(args.config)
    out = run_benchmark(model, cases, args.experiment, seed, args.results_dir)
    overall = out["stats"]["overall"]
    print(f"run_id: {out['run_id']}")
    print(f"injection successful: {overall['injection_successful_count']} of {overall['total_count']} "
          f"({overall['injection_successful_percentage']:.1f}%), errors: {out['stats']['error_count']}")
    for key in ("raw", "judge_raw", "native", "stats", "normalized"):
        print(f"{key}: {out['paths'][key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
