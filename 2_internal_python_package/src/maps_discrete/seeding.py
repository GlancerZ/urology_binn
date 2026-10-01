from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Set training RNGs without changing the frozen cohort split."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    audit: dict[str, Any] = {"python": seed, "numpy": seed, "torch": None}
    try:
        import torch
    except ImportError:
        return audit
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    audit["torch"] = seed
    audit["cuda_available"] = bool(torch.cuda.is_available())
    return audit

