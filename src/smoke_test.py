import argparse
from pathlib import Path

from models.huggingface_model import HuggingFaceChatModel
from results_io import append_record, make_record, new_run_id, results_path
from seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Send one system+user prompt to the configured model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    seed = set_seed(args.seed)
    model = HuggingFaceChatModel(args.config)
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "In one sentence, what is a prompt injection attack?"},
    ]
    result = model.generate(messages)

    out = results_path(args.results_dir, "smoke_test", model.cfg["model_id"])
    run_id = new_run_id()
    append_record(out, make_record(run_id, "smoke_test", "smoke-001", model.cfg, seed, messages, result))

    print(f"status: {result.status}")
    print(f"error: {result.error}")
    print(f"runtime_s: {result.runtime_s:.2f}")
    print(f"tokens in/out: {result.input_tokens}/{result.output_tokens}")
    print(f"seed: {seed}")
    print(f"run_id: {run_id}")
    print(f"record appended to: {out}")
    print("--- response ---")
    print(result.text)


if __name__ == "__main__":
    main()
