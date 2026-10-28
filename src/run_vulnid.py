"""Generate model replies for the vulnerability-identification items and append raw records. No scoring here.

Usage: python src/run_vulnid.py --config configs/qwen3_4b_instruct.yaml [--ids VID-CWE78-001-V,...]
Scoring is a separate step: python src/score_vulnid.py --raw <raw jsonl>
--max-new-tokens overrides the config value (recorded in the raw records); the JSON answer needs more than 256.
"""
import argparse
import sys
from pathlib import Path

from codegen.vulnid import build_messages, load_items, validate_items
from results_io import append_record, make_record, new_run_id, results_path

DEFAULT_ITEMS = Path("data/original/vuln_id_v0.1.jsonl")
DEFAULT_EXPERIMENT = "vuln_id_v0.1"
DEFAULT_MAX_NEW_TOKENS = 512


def run_items(model, items, experiment, seed, out_path, run_id=None, log=print):
    """Generate for each item and append one raw record per item. Returns the run_id; refuses an existing file."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; move it or use another --experiment")
    run_id = run_id or new_run_id()
    for i, item in enumerate(items, start=1):
        messages = build_messages(item)
        result = model.generate(messages)
        append_record(out_path, make_record(run_id, experiment, item["item_id"], model.cfg, seed, messages, result))
        log(f"[{i}/{len(items)}] {item['item_id']} status={result.status} "
            f"tokens={result.input_tokens}/{result.output_tokens} {result.runtime_s:.1f}s")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run the vulnerability-identification items on one model (raw only)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--ids", default=None, help="comma-separated item ids to run (default: all)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    items = load_items(args.items)
    errors = validate_items(items)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("Item file is invalid; nothing was run.")
        return 1
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = sorted(set(wanted) - {t["item_id"] for t in items})
        if unknown or not wanted:
            print(f"ERROR: unknown or empty --ids {unknown or args.ids!r}")
            return 1
        items = [t for t in items if t["item_id"] in wanted]

    from models.huggingface_model import HuggingFaceChatModel, load_config
    from seed import set_seed

    out = results_path(args.results_dir, args.experiment, load_config(args.config)["model_id"])
    if out.exists():
        print(f"ERROR: {out} already exists; move it or use another --experiment")
        return 1
    seed = set_seed(args.seed)
    model = HuggingFaceChatModel(args.config)
    model.cfg["max_new_tokens"] = args.max_new_tokens
    run_id = run_items(model, items, args.experiment, seed, out)
    print(f"run_id: {run_id}")
    print(f"raw records written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
