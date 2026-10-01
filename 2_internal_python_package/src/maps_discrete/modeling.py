from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn


class InterpretableRiskFusion(nn.Module):
    """Attention-weighted H6 and PRS fusion returning one scalar log-risk."""

    def __init__(self, n_tokens: int, embed_dim: int = 32):
        super().__init__()
        self.value_proj = nn.Linear(1, embed_dim, bias=False)
        self.token_embed = nn.Parameter(torch.zeros(n_tokens, embed_dim))
        nn.init.normal_(self.token_embed, std=0.02)
        self.score_vec = nn.Linear(embed_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, values: torch.Tensor):
        embedded = torch.tanh(self.value_proj(values.unsqueeze(-1)) + self.token_embed.unsqueeze(0))
        attention = torch.softmax(self.score_vec(embedded).squeeze(-1), dim=1)
        contribution = self.gamma * attention * values
        return self.bias + contribution.sum(dim=1), attention, contribution


class DiscreteTimeSurvivalBINN(nn.Module):
    """Six-layer Reactome-informed network with proportional discrete hazards."""

    def __init__(self, binn: nn.Module, h6_dim: int, n_intervals: int, attention_dim: int = 32):
        super().__init__()
        self.binn = binn
        self.h6_bn = nn.BatchNorm1d(h6_dim, eps=1e-5, affine=True)
        self.prs_bn = nn.BatchNorm1d(1, eps=1e-5, affine=True)
        self.fusion = InterpretableRiskFusion(h6_dim + 1, attention_dim)
        self.baseline_logits = nn.Parameter(torch.full((n_intervals,), -5.0))

    def forward(self, omics: torch.Tensor, prs: torch.Tensor) -> dict[str, torch.Tensor]:
        h6 = self.binn.layers.forward_features(omics)
        tokens = torch.cat([self.h6_bn(h6), self.prs_bn(prs)], dim=1)
        log_risk, attention, contribution = self.fusion(tokens)
        return {
            "hazard_logits": self.baseline_logits.unsqueeze(0) + log_risk.unsqueeze(1),
            "log_risk": log_risk,
            "attention": attention,
            "contribution": contribution,
            "h6": h6,
        }


def _binn_class(binn_root: str | Path):
    root = str(Path(binn_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from binn.model.binn import BINN

    return BINN


def build_model(
    data_matrix: pd.DataFrame,
    mapping: pd.DataFrame,
    pathways: pd.DataFrame,
    entity_col: str,
    binn_root: str | Path,
    device: torch.device,
    n_layers: int = 6,
    n_intervals: int = 10,
    attention_dim: int = 32,
    residual_width: int = 1,
) -> DiscreteTimeSurvivalBINN:
    BINN = _binn_class(binn_root)
    binn = BINN(
        data_matrix=data_matrix,
        mapping=mapping,
        pathways=pathways,
        entity_col=entity_col,
        n_layers=n_layers,
        n_outputs=2,
        heads_ensemble=True,
        activation="tanh",
        device="cpu",
        routing_mode="simple_dual",
        residual_width=residual_width,
    )
    h6_dim = int(binn.layers.blocks[-1][0].out_features)
    return DiscreteTimeSurvivalBINN(binn, h6_dim, n_intervals, attention_dim).to(device)


def pathway_layer_sizes(model: DiscreteTimeSurvivalBINN) -> list[int]:
    return [int(block[0].out_features) for block in model.binn.layers.blocks]


def routing_frame(model: DiscreteTimeSurvivalBINN) -> pd.DataFrame:
    rows = []
    for feature in model.binn.inputs:
        layers = model.binn.input_connected_layers[feature]
        rows.append({
            "input": feature,
            "connected_layers": "|".join(f"H{x}" for x in layers),
            "entry_layer": "Other" if not layers else f"H{layers[0]}",
            "uses_h1_other": bool(model.binn.uses_h1_residual[feature]),
            "uses_direct_high_layer": bool(model.binn.uses_direct_high_layer[feature]),
        })
    return pd.DataFrame(rows)
