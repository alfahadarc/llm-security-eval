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
