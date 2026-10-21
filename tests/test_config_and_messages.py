from pathlib import Path

import pytest
import yaml

from models.base import validate_messages
from models.huggingface_model import load_config

REPO = Path(__file__).resolve().parent.parent
GOOD = {
    "model_id": "x/y", "dtype": "float16", "device": "cpu",
    "max_input_tokens": 100, "max_new_tokens": 10, "do_sample": False,
}


def write_cfg(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_real_qwen_config_loads_and_is_pinned():
    cfg = load_config(REPO / "configs" / "qwen3_4b_instruct.yaml")
    assert cfg["dtype"] == "float16"
    assert cfg["do_sample"] is False
    assert cfg["revision"], "revision must be pinned"


def test_good_config_loads(tmp_path):
    assert load_config(write_cfg(tmp_path, GOOD))["model_id"] == "x/y"


def test_missing_key_rejected(tmp_path):
    cfg = {k: v for k, v in GOOD.items() if k != "dtype"}
    with pytest.raises(ValueError, match="missing"):
        load_config(write_cfg(tmp_path, cfg))


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        load_config(write_cfg(tmp_path, {**GOOD, "temprature": 0.7}))


def test_bad_dtype_rejected(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        load_config(write_cfg(tmp_path, {**GOOD, "dtype": "int4"}))


def test_valid_messages_pass():
    validate_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])


@pytest.mark.parametrize("bad", [
    [],
    "hello",
    [{"role": "user"}],
    [{"role": "user", "content": "x", "extra": 1}],
    [{"role": "tool", "content": "x"}],
    [{"role": "user", "content": 5}],
    [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
])
def test_invalid_messages_rejected(bad):
    with pytest.raises(ValueError):
        validate_messages(bad)
