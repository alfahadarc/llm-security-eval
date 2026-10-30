# Models

All models: Hugging Face `transformers` 5.20.2, greedy decoding, `max_new_tokens: 256`, one Tesla V100 16 GB, system+user messages through each model's own chat template. Revisions are pinned in `configs/*.yaml` (checked 2026-10-28; each passed the same smoke test, `src/smoke_test.py`).

| Config | Model ID | Role | License | Access | Revision | dtype | Weights | Peak VRAM at 4,096 in + 256 out |
|---|---|---|---|---|---|---|---|---|
| `qwen3_4b_instruct.yaml` | Qwen/Qwen3-4B-Instruct-2507 | core, first model | Apache-2.0 | open | `cdbee75f…0554` | float16 | 8.0 GB | not measured here |
| `phi4_mini_instruct.yaml` | microsoft/Phi-4-mini-instruct | core | MIT | open | `cfbefacb…3083` | float16 | 7.7 GB | 12.3 GB |
| `granite4_micro.yaml` | ibm-granite/granite-4.0-micro | core | Apache-2.0 | open | `56111ae1…e979` | float16 | 6.8 GB | 13.5 GB |
| `gemma3_4b_it.yaml` | google/gemma-3-4b-it | core | Gemma terms of use | gated (manual approval, token) | `093f9f38…9767` | **bfloat16** | 8.6 GB | 15.2 GB at 8,095 in |
| `mistral7b_instruct_v03.yaml` | mistralai/Mistral-7B-Instruct-v0.3 | established anchor | Apache-2.0 | open | `c170c708…bab71` | float16 | 14.5 GB | does not fit (see below) |

Llama-3.1-8B-Instruct was rejected: 8.03B parameters = 16.1 GB in FP16, more than the GPU. Llama-3.2-3B-Instruct (gated, access works) is available as a freshness/extra model if wanted.

## Chat templates (system role checked with marker strings)

| Model | System prompt handling |
|---|---|
| Qwen3 | own system turn |
| Phi-4-mini | own `<\|system\|>` turn |
| Granite 4.0 micro | own `system` role turn (template is in a separate `chat_template.jinja`) |
| Gemma 3 | no system role: the system text is prepended to the first user turn |
| Mistral v0.3 | no system role: the system text is prepended to the user `[INST]` turn |

Gemma and Mistral therefore see the system prompt as part of the user turn. This matters for prompt-injection results (the system/user boundary is weaker for them) and must be stated in the report.

## Model-specific exceptions

1. **Gemma 3 runs in bfloat16, not float16.** In float16 the first logits are NaN and the model emits only pad tokens (empty output, 256 tokens). Eager attention in float16 did not help. bfloat16 works on the V100 (emulated, no speed problem: 1.8 s for the smoke prompt). Greedy decoding is deterministic. Report as: "Gemma 3 uses bf16 because fp16 overflows."
2. **Mistral-7B uses `max_input_tokens: 2048` (others 8192).** 14.5 GB of weights leave about 1.3 GB for attention; inputs of 2,304 tokens run out of memory (2,049 works). The longest real prompt in the benchmark under Mistral's tokenizer is 978 tokens (CyberSecEval; PI v0.2: 533, secure code: 260, vuln-ID: 345), so the effective limit is the same for all models. The CyberSecEval judge prompt (response up to 256 tokens + question) still has to be checked in Step 15b. The adapter raises an error instead of truncating, so an over-long input would show as an error row, never as silent truncation.
3. **Phi-4-mini and Granite run out of memory above about 4,096 to 6,144 input tokens**, again attention memory on the V100 (no flash attention). Irrelevant for this benchmark (max 978 tokens) but the 8192 setting is a ceiling, not something tested.

## Smoke-test results (one prompt, seed 0)

| Model | status | tokens in/out | repeat run identical |
|---|---|---|---|
| Phi-4-mini | ok | 22/25 | yes |
| Granite 4.0 micro | ok | 30/46 | yes |
| Gemma 3 4B (bf16) | ok | 27/29 | yes |
| Mistral 7B v0.3 | ok | 23/43 | yes |
