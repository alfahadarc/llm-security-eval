"""Generate model responses for every dataset case and append raw records. No scoring here.

Usage: python src/run_eval.py --config configs/qwen3_4b_instruct.yaml
Scoring is a separate step: python src/score_results.py --raw <raw jsonl>
"""
import argparse
import sys
from pathlib import Path

from eval.prompt_injection import build_messages
from eval.validate_dataset import load_jsonl, validate_records
from results_io import append_record, make_record, new_run_id, results_path

DEFAULT_DATASET = Path("data/original/prompt_injection_v0.1.jsonl")
DEFAULT_EXPERIMENT = "prompt_injection_v0.1"


def run_cases(model, records, experiment, seed, out_path, run_id=None, log=print):
    """Generate for each record and append one raw record per case. Returns the run_id.

    Refuses to write into a file that already exists, so runs never mix silently.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; move it or use another --experiment")
    run_id = run_id or new_run_id()
    for i, rec in enumerate(records, start=1):
        messages = build_messages(rec)
        result = model.generate(messages)
        append_record(out_path, make_record(run_id, experiment, rec["case_id"], model.cfg, seed, messages, result))
        log(f"[{i}/{len(records)}] {rec['case_id']} status={result.status} "
            f"tokens={result.input_tokens}/{result.output_tokens} {result.runtime_s:.1f}s")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run the prompt-injection dataset on one model (raw output only)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    records = load_jsonl(args.dataset)
    errors = validate_records(records)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("Dataset is invalid; nothing was run.")
        return 1

    # Heavy imports only after the cheap checks have passed.
    from models.huggingface_model import HuggingFaceChatModel, load_config
    from seed import set_seed

    # Check the output path before the slow model load.
    out = results_path(args.results_dir, args.experiment, load_config(args.config)["model_id"])
    if out.exists():
        print(f"ERROR: {out} already exists; move it or use another --experiment")
        return 1

    seed = set_seed(args.seed)
    model = HuggingFaceChatModel(args.config)
    run_id = run_cases(model, records, args.experiment, seed, out)
    print(f"run_id: {run_id}")
    print(f"raw records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
