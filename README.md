# LLM Security Evaluation

A starter scaffold for evaluating prompt-injection attacks and defenses against large language models.

## Project structure

- `configs/` for experiment YAML configs
- `data/` for prompts, attack sets, and task definitions
- `src/models/` for model loading
- `src/attacks/` for attack implementations
- `src/defenses/` for mitigation baselines
- `src/eval/` for metrics and reporting
- `results/` for figures and tables
- `report/` for the technical report

## Quick start

1. Create and activate the conda environment:
   - `conda env create -f environment.yml`
   - `conda activate llm-sec`
2. Validate the dataset:
   - `python src/eval/validate_dataset.py data/original/prompt_injection_v0.1.jsonl`
3. Generate responses (raw output only, writes `results/raw/prompt_injection_v0.1/<model>.jsonl`):
   - `python src/run_eval.py --config configs/qwen3_4b_instruct.yaml`
4. Score the raw run (CPU only, safe to re-run):
   - `python src/score_results.py --raw results/raw/prompt_injection_v0.1/Qwen__Qwen3-4B-Instruct-2507.jsonl`
5. Run the tests:
   - `pytest -q`

## Datasets

### `dataset-v0.1.0`: prompt injection, document summarization

- File: `data/original/prompt_injection_v0.1.jsonl`
- Contents: 30 records = 10 pairs (`DOC-001` to `DOC-010`) x clean / attack / benign. Indirect injection through the untrusted document; the scenario and scoring rules are in `docs/POLICY.md`.
- Coverage (attack cases): objectives attacker_response 3, secret_exfiltration 3, format_hijack 2, task_abandonment 2; techniques ignore_previous_override, fake_system_message, role_impersonation, delimiter_manipulation, light_encoding, 2 each.
- SHA-256: `687beca4caa23c7cc68f288a5d62721281eede68065f48c893bd8293107cbb96`
- Verify: `sha256sum data/original/prompt_injection_v0.1.jsonl` must print the hash above, and `python src/eval/validate_dataset.py data/original/prompt_injection_v0.1.jsonl` must print `OK: 30 record(s), 10 pair(s)`.
- First run (Qwen3-4B-Instruct-2507, fp16, greedy, seed 0): attack 8 of 10 `attack_success`, 2 `utility_pass`; clean 10 of 10 and benign 10 of 10 `utility_pass`. After this run, `required_facts` in pairs DOC-008 and DOC-009 were changed (peripheral facts replaced by central ones); prompts did not change.

## Notes / decisions log

| Date | Decision |
|---|---|
| 2026-10-21 | Environment: conda env `llm-sec` from `environment.yml`; PyTorch pinned to `2.8.0+cu128` (server driver supports CUDA up to 12.8) |
| 2026-10-21 | First model: `Qwen/Qwen3-4B-Instruct-2507`, config in `configs/qwen3_4b_instruct.yaml` (revision pinned: `cdbee75f17c01a7cc42f958dc650907174af0554`) |
| 2026-10-21 | Weights cache: `HF_HOME=/research2/abdfahad/hf_cache` set via `conda env config vars` for `llm-sec` (`/research` disk is full) |
| 2026-10-22 | Dataset v0.1.0 frozen (30 cases, SHA-256 in the Datasets section); tag `dataset-v0.1.0` |
| 2026-10-21 | Primary runs use fp16 + greedy. Note: bf16 (Qwen's training dtype) gives different greedy text than fp16; fp16 is run-to-run deterministic |
