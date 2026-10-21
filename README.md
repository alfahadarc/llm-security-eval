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
2. Check the config loads:
   - `python src/run.py --config configs/qwen3_4b_instruct.yaml`

## Notes / decisions log

| Date | Decision |
|---|---|
| 2026-10-21 | Environment: conda env `llm-sec` from `environment.yml`; PyTorch pinned to `2.8.0+cu128` (server driver supports CUDA up to 12.8) |
| 2026-10-21 | First model: `Qwen/Qwen3-4B-Instruct-2507`, config in `configs/qwen3_4b_instruct.yaml` (revision pinned: `cdbee75f17c01a7cc42f958dc650907174af0554`) |
| 2026-10-21 | Weights cache: `HF_HOME=/research2/abdfahad/hf_cache` set via `conda env config vars` for `llm-sec` (`/research` disk is full) |
| 2026-10-21 | Primary runs use fp16 + greedy. Note: bf16 (Qwen's training dtype) gives different greedy text than fp16; fp16 is run-to-run deterministic |
