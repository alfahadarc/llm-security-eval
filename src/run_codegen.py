"""Generate model replies for the secure-code tasks and append raw records. No scoring here.

Usage: python src/run_codegen.py --config configs/qwen3_4b_instruct.yaml [--ids CWE78-001,CWE89-002]
Scoring is a separate step: python src/score_codegen.py --raw <raw jsonl>
Each task prompt is sent as one user message. --max-new-tokens overrides the config value (and is recorded in the
raw records), because the default may cut generated code off.
"""
import argparse
import sys
from pathlib import Path

from codegen.tasks import load_tasks, validate_tasks
from results_io import append_record, make_record, new_run_id, results_path

DEFAULT_TASKS = Path("data/original/secure_code_v0.1.jsonl")
DEFAULT_EXPERIMENT = "secure_code_v0.1"


def build_messages(task):
    return [{"role": "user", "content": task["prompt"]}]


def run_tasks(model, tasks, experiment, seed, out_path, run_id=None, log=print):
    """Generate for each task and append one raw record per task. Returns the run_id; refuses an existing file."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; move it or use another --experiment")
    run_id = run_id or new_run_id()
    for i, task in enumerate(tasks, start=1):
        messages = build_messages(task)
        result = model.generate(messages)
        append_record(out_path, make_record(run_id, experiment, task["task_id"], model.cfg, seed, messages, result))
        log(f"[{i}/{len(tasks)}] {task['task_id']} status={result.status} "
            f"tokens={result.input_tokens}/{result.output_tokens} {result.runtime_s:.1f}s")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run the secure-code tasks on one model (raw output only)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--ids", default=None, help="comma-separated task ids to run (default: all)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    errors = validate_tasks(tasks)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("Task file is invalid; nothing was run.")
        return 1
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = sorted(set(wanted) - {t["task_id"] for t in tasks})
        if unknown or not wanted:
            print(f"ERROR: unknown or empty --ids {unknown or args.ids!r}")
            return 1
        tasks = [t for t in tasks if t["task_id"] in wanted]

    from models.huggingface_model import HuggingFaceChatModel, load_config
    from seed import set_seed

    out = results_path(args.results_dir, args.experiment, load_config(args.config)["model_id"])
    if out.exists():
        print(f"ERROR: {out} already exists; move it or use another --experiment")
        return 1
    seed = set_seed(args.seed)
    model = HuggingFaceChatModel(args.config)
    if args.max_new_tokens:
        model.cfg["max_new_tokens"] = args.max_new_tokens  # recorded in every raw record's `generation`
    run_id = run_tasks(model, tasks, args.experiment, seed, out)
    print(f"run_id: {run_id}")
    print(f"raw records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
