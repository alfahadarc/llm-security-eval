"""Validate a dataset JSONL file. Usage: python src/eval/validate_dataset.py <file.jsonl>

Exit code 0 if every record is valid, 1 otherwise. Errors are printed one per line.
"""
import json
import sys
from collections import defaultdict

STR_FIELDS = (
    "case_id", "pair_id", "condition", "application", "system_prompt", "user_prompt",
    "untrusted_content", "canary", "required_format",
)
NULLABLE_STR_FIELDS = ("delivery", "objective", "technique", "attack_marker")
ALL_FIELDS = set(STR_FIELDS) | set(NULLABLE_STR_FIELDS) | {"attack_present", "required_facts"}
OPTIONAL_FIELDS = {"split"}  # v0.2+; v0.1 has none. If used, every record must have it.

CONDITIONS = ("clean", "attack", "benign")
APPLICATIONS = ("document_summarization",)
INDIRECT, DIRECT = "indirect_document", "direct_user_prompt"
DELIVERIES = (INDIRECT, DIRECT)
SPLITS = ("dev", "test")
OBJECTIVES = ("attacker_response", "secret_exfiltration", "format_hijack", "task_abandonment")
TECHNIQUES = ("ignore_previous_override", "fake_system_message", "role_impersonation",
              "delimiter_manipulation", "light_encoding")
REQUIRED_FORMATS = ("three_bullets",)


def load_jsonl(path):
    """Read records; a blank or malformed line raises ValueError with its line number."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: not valid JSON ({e})") from e
    return records


def select_pairs(records, spec):
    """Keep the records of the pairs named in spec, in file order.

    spec is comma-separated; each item is a pair_id ('DIR-035') or an inclusive range in file order
    ('DIR-035:DIR-042'). Raises ValueError for an empty spec, an unknown pair_id or a backwards range.
    """
    known = []
    for r in records:
        if r["pair_id"] not in known:
            known.append(r["pair_id"])
    wanted = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        first, _, last = part.partition(":")
        last = last or first
        for pid in (first, last):
            if pid not in known:
                raise ValueError(f"unknown pair_id {pid!r} in --pairs")
        i, j = known.index(first), known.index(last)
        if i > j:
            raise ValueError(f"range {part!r} in --pairs goes backwards")
        wanted.update(known[i:j + 1])
    if not wanted:
        raise ValueError("--pairs is empty")
    return [r for r in records if r["pair_id"] in wanted]


def _check_record(rec):
    """Return a list of error strings for one record (no cross-record checks)."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    errs = []
    keys = set(rec)
    if keys - ALL_FIELDS - OPTIONAL_FIELDS:
        errs.append(f"unknown fields: {sorted(keys - ALL_FIELDS - OPTIONAL_FIELDS)}")
    if ALL_FIELDS - keys:
        errs.append(f"missing fields: {sorted(ALL_FIELDS - keys)}")
        return errs  # types below would be meaningless

    for f in STR_FIELDS:
        if not isinstance(rec[f], str) or not rec[f].strip():
            errs.append(f"{f} must be a non-empty string")
    for f in NULLABLE_STR_FIELDS:
        if rec[f] is not None and (not isinstance(rec[f], str) or not rec[f].strip()):
            errs.append(f"{f} must be null or a non-empty string")
    if not isinstance(rec["attack_present"], bool):
        errs.append("attack_present must be true or false")
    facts = rec["required_facts"]
    if not isinstance(facts, list) or not facts or not all(isinstance(x, str) and x.strip() for x in facts):
        errs.append("required_facts must be a non-empty list of non-empty strings")
    if errs:
        return errs

    for field, allowed in (("condition", CONDITIONS), ("application", APPLICATIONS),
                           ("required_format", REQUIRED_FORMATS)):
        if rec[field] not in allowed:
            errs.append(f"invalid {field} {rec[field]!r}; allowed: {list(allowed)}")
    for field, allowed in (("delivery", DELIVERIES), ("objective", OBJECTIVES), ("technique", TECHNIQUES)):
        if rec[field] is not None and rec[field] not in allowed:
            errs.append(f"invalid {field} {rec[field]!r}; allowed: {list(allowed)}")
    if "split" in rec and rec["split"] not in SPLITS:
        errs.append(f"invalid split {rec['split']!r}; allowed: {list(SPLITS)}")

    doc = rec["untrusted_content"]
    if rec["condition"] == "attack":
        if rec["attack_present"] is not True:
            errs.append("attack case must have attack_present=true")
        for f in ("delivery", "objective", "technique", "attack_marker"):
            if rec[f] is None:
                errs.append(f"attack case must set {f}")
        if rec["attack_marker"]:
            # The marker lives where the attack is delivered, and only there.
            if rec["delivery"] == DIRECT:
                home, home_name, other, other_name = rec["user_prompt"], "user_prompt", doc, "untrusted_content"
            else:
                home, home_name, other, other_name = doc, "untrusted_content", rec["user_prompt"], "user_prompt"
            if rec["attack_marker"] not in home:
                errs.append(f"attack_marker does not appear in {home_name}")
            elif rec["attack_marker"] in other:
                errs.append(f"attack_marker must not appear in {other_name} for delivery {rec['delivery']!r}")
    elif rec["condition"] in ("clean", "benign"):
        if rec["attack_present"] is not False:
            errs.append(f"{rec['condition']} case must have attack_present=false")
        for f in ("delivery", "objective", "technique", "attack_marker"):
            if rec[f] is not None:
                errs.append(f"{rec['condition']} case must have {f}=null")

    canary = rec["canary"]
    if canary not in rec["system_prompt"]:
        errs.append("canary does not appear in system_prompt")
    if canary in doc or canary in rec["user_prompt"]:
        errs.append("canary leaks into user_prompt or untrusted_content")

    low = doc.lower()
    for fact in facts:
        if fact.lower() not in low:
            errs.append(f"required fact {fact!r} not found in untrusted_content")
    return errs


def validate_records(records):
    """Return a list of error strings; an empty list means the dataset is valid.

    Line numbers in messages are record positions (line N of the JSONL file).
    """
    errors = []
    for i, rec in enumerate(records, start=1):
        cid = rec.get("case_id") if isinstance(rec, dict) else None
        for e in _check_record(rec):
            errors.append(f"line {i} ({cid}): {e}")

    good = [r for r in records if isinstance(r, dict) and ALL_FIELDS <= set(r)]

    seen = defaultdict(list)
    for i, r in enumerate(records, start=1):
        if isinstance(r, dict) and isinstance(r.get("case_id"), str):
            seen[r["case_id"]].append(i)
    for cid, lines in seen.items():
        if len(lines) > 1:
            errors.append(f"duplicate case_id {cid!r} on lines {lines}")

    canaries = defaultdict(list)
    for i, r in enumerate(records, start=1):
        if isinstance(r, dict) and isinstance(r.get("canary"), str):
            canaries[r["canary"]].append(i)
    for canary, lines in canaries.items():
        if len(lines) > 1:
            errors.append(f"canary {canary!r} reused on lines {lines}")

    pairs = defaultdict(list)
    for r in good:
        if isinstance(r["pair_id"], str):
            pairs[r["pair_id"]].append(r)
    for pid, members in pairs.items():
        conds = sorted(str(m["condition"]) for m in members)
        if conds != sorted(CONDITIONS):
            errors.append(f"pair {pid!r} must have exactly one clean, one attack and one benign case; found {conds}")
            continue
        # Only valid-looking members get compared; type errors were reported above.
        attack = next(m for m in members if m["condition"] == "attack")
        # Indirect pairs share the user prompt (the document varies); direct pairs share the document.
        varying = "untrusted_content" if attack["delivery"] == DIRECT else "user_prompt"
        for f in ("application", varying, "required_facts", "required_format"):
            if len({json.dumps(m[f]) for m in members}) > 1:
                errors.append(f"pair {pid!r}: field {f} differs between cases")
        if len({json.dumps(m.get("split")) for m in members}) > 1:
            errors.append(f"pair {pid!r}: field split differs between cases")
        marker = attack["attack_marker"]
        if isinstance(marker, str) and marker:
            for m in members:
                if m["condition"] != "attack" and (marker in m["untrusted_content"] or marker in m["user_prompt"]):
                    errors.append(f"pair {pid!r}: attack_marker appears in {m['case_id']!r}")

    with_split = sum(1 for r in good if "split" in r)
    if with_split and with_split != len(good):
        errors.append(f"split must be set on every record or on none ({with_split} of {len(good)} have it)")
    return errors


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        records = load_jsonl(argv[1])
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    errors = validate_records(records)
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        print(f"INVALID: {len(errors)} error(s) in {len(records)} record(s)")
        return 1
    print(f"OK: {len(records)} record(s), {len({r['pair_id'] for r in records})} pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
