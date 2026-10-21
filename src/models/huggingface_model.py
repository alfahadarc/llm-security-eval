import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import ChatModel, GenerationResult, validate_messages

REQUIRED_KEYS = {"model_id", "dtype", "device", "max_input_tokens", "max_new_tokens", "do_sample"}
OPTIONAL_KEYS = {"revision", "trust_remote_code"}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_config(path):
    """Load a model YAML and reject missing or unknown keys (no silent setting changes)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    keys = set(cfg)
    missing = REQUIRED_KEYS - keys
    unknown = keys - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"config has unknown keys: {sorted(unknown)}")
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"unsupported dtype {cfg['dtype']!r}; use one of {sorted(DTYPES)}")
    return cfg


class HuggingFaceChatModel(ChatModel):
    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        c = self.cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            c["model_id"], revision=c.get("revision"),
            trust_remote_code=c.get("trust_remote_code", False),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            c["model_id"], revision=c.get("revision"), dtype=DTYPES[c["dtype"]],
            trust_remote_code=c.get("trust_remote_code", False),
        ).to(c["device"])
        self.model.eval()

    def generate(self, messages) -> GenerationResult:
        validate_messages(messages)
        c = self.cfg
        start = time.perf_counter()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            n_in = inputs["input_ids"].shape[1]
            if n_in > c["max_input_tokens"]:
                # never truncate silently: report it as an error instead
                raise ValueError(f"input has {n_in} tokens, over max_input_tokens={c['max_input_tokens']}")
            inputs = inputs.to(c["device"])
            gen_kwargs = dict(max_new_tokens=c["max_new_tokens"], do_sample=c["do_sample"])
            if not c["do_sample"]:
                # greedy: clear sampling defaults that the model's generation_config may carry
                gen_kwargs.update(temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = out[0, n_in:]  # slice off the prompt: generated text only
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return GenerationResult(
                text=text, runtime_s=time.perf_counter() - start, status="ok",
                input_tokens=n_in, output_tokens=int(new_tokens.shape[0]),
            )
        except Exception as e:  # record the failure instead of crashing a whole run
            return GenerationResult(
                text="", runtime_s=time.perf_counter() - start, status="error",
                error=f"{type(e).__name__}: {e}",
            )
