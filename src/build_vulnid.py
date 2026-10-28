"""Build the vulnerability-identification items from the secure-code tasks (deterministic, CPU only).

Usage: python src/build_vulnid.py [--tasks data/original/secure_code_v0.1.jsonl] [--out data/original/vuln_id_v0.1.jsonl]
Refuses to overwrite an existing output file. Prints the SHA-256 of the written file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from codegen.tasks import load_tasks
from codegen.vulnid import make_items, validate_items


def main():
    parser = argparse.ArgumentParser(description="Build vulnerability-identification items")
    parser.add_argument("--tasks", type=Path, default=Path("data/original/secure_code_v0.1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/original/vuln_id_v0.1.jsonl"))
    args = parser.parse_args()
    if args.out.exists():
        print(f"ERROR: {args.out} already exists; move it first")
        return 1
    items = make_items(load_tasks(args.tasks))
    errors = validate_items(items)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items written to {args.out}")
    print(f"sha256: {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
