import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models.base import validate_messages

RECORD_KEYS = (
    "run_id", "experiment", "case_id", "model_id", "revision", "dtype", "generation", "seed",
    "messages", "response", "status", "error",
    "input_tokens", "output_tokens", "runtime_s", "timestamp_utc",
)
GENERATION_KEYS = ("max_input_tokens", "max_new_tokens", "do_sample")


def new_run_id() -> str:
    """UTC timestamp + short random suffix; one per run, shared by all records of that run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def model_slug(model_id: str) -> str:
    """Filesystem-safe model name, e.g. 'Qwen/Qwen3-4B' -> 'Qwen__Qwen3-4B'."""
    return model_id.replace("/", "__")


def results_path(root, experiment: str, model_id: str) -> Path:
    """results/raw/<experiment>/<model>.jsonl under root."""
    if not experiment or "/" in experiment or experiment.startswith("."):
        raise ValueError(f"invalid experiment name {experiment!r}")
    return Path(root) / "raw" / experiment / f"{model_slug(model_id)}.jsonl"


def make_record(run_id, experiment, case_id, cfg, seed, messages, result) -> dict:
    """One raw record: the full prompt (messages) and everything needed to reproduce the call."""
    validate_messages(messages)
    return {
        "run_id": run_id,
        "experiment": experiment,
        "case_id": case_id,
        "model_id": cfg["model_id"],
        "revision": cfg.get("revision"),
        "dtype": cfg["dtype"],
        "generation": {k: cfg[k] for k in GENERATION_KEYS},
        "seed": seed,
        "messages": messages,
        "response": result.text,
        "status": result.status,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "runtime_s": result.runtime_s,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_record(path, record: dict) -> None:
    """Append one JSON line and flush, so a crash never loses finished records."""
    missing = set(RECORD_KEYS) - set(record)
    if missing:
        raise ValueError(f"record missing keys: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def read_records(path) -> list:
    """Read all records; a malformed line raises with its line number."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {n} is not valid JSON ({e})") from e
    return records
