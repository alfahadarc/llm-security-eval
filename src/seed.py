import random

import numpy as np
import torch


def set_seed(seed: int) -> int:
    """Seed Python, NumPy and torch (CPU + CUDA). Returns the seed so it can be logged."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    return seed
