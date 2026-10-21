"""generate() with a fake tokenizer and model: no GPU, no weights."""
import pytest
import torch

from models.huggingface_model import HuggingFaceChatModel

PROMPT_IDS = [11, 12, 13, 14]
NEW_IDS = [21, 22]


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, prompt_ids=PROMPT_IDS):
        self.prompt_ids = prompt_ids
        self.decoded = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "PROMPT"

    def __call__(self, text, return_tensors, add_special_tokens):
        assert add_special_tokens is False  # template already added them
        return FakeBatch(input_ids=torch.tensor([self.prompt_ids]))

    def decode(self, ids, skip_special_tokens):
        self.decoded = ids.tolist()
        return "decoded"


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.tensor([NEW_IDS])], dim=1)


def make_model(max_input_tokens=100, do_sample=False, tokenizer=None):
    m = HuggingFaceChatModel.__new__(HuggingFaceChatModel)  # skip loading weights
    m.cfg = {"device": "cpu", "max_input_tokens": max_input_tokens, "max_new_tokens": 5, "do_sample": do_sample}
    m.tokenizer = tokenizer or FakeTokenizer()
    m.model = FakeModel()
    return m


MSGS = [{"role": "user", "content": "hi"}]


def test_only_new_tokens_are_decoded():
    m = make_model()
    r = m.generate(MSGS)
    assert r.status == "ok" and r.text == "decoded"
    assert m.tokenizer.decoded == NEW_IDS  # prompt tokens sliced off
    assert (r.input_tokens, r.output_tokens) == (len(PROMPT_IDS), len(NEW_IDS))


def test_greedy_clears_sampling_params():
    m = make_model(do_sample=False)
    m.generate(MSGS)
    kw = m.model.kwargs
    assert kw["do_sample"] is False
    assert kw["temperature"] is None and kw["top_p"] is None and kw["top_k"] is None


def test_over_limit_input_is_error_not_truncation():
    m = make_model(max_input_tokens=3)
    r = m.generate(MSGS)
    assert r.status == "error" and "max_input_tokens" in r.error
    assert r.text == ""
    assert m.model.kwargs is None  # model never called


def test_exception_in_model_is_captured():
    m = make_model()
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    m.model.generate = boom
    r = m.generate(MSGS)
    assert r.status == "error" and "RuntimeError" in r.error and "out of memory" in r.error


def test_invalid_messages_raise_before_generation():
    m = make_model()
    with pytest.raises(ValueError):
        m.generate([])
