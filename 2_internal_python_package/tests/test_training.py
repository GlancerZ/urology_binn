import unittest

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from maps_discrete.training import TrainingSettings, _optimizer


class TinyTeacher(nn.Module):
    """Only the attributes _optimizer reads."""

    def __init__(self):
        super().__init__()
        self.binn = nn.Linear(3, 2)
        self.h6_bn = nn.BatchNorm1d(2)
        self.prs_bn = nn.BatchNorm1d(1)
        self.fusion = nn.Linear(3, 1)
        self.baseline_logits = nn.Parameter(torch.zeros(4))


class OptimizerTests(unittest.TestCase):
    def test_frozen_recipe_keeps_adam_with_l2(self):
        optimizer = _optimizer(TinyTeacher(), TrainingSettings())
        self.assertIs(type(optimizer), Adam)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [1e-2, 0.0])

    def test_decoupled_weight_decay_switches_to_adamw(self):
        settings = TrainingSettings(decoupled_weight_decay=True, binn_weight_decay=0.1)
        optimizer = _optimizer(TinyTeacher(), settings)
        self.assertIs(type(optimizer), AdamW)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.1, 0.0])


if __name__ == "__main__":
    unittest.main()
