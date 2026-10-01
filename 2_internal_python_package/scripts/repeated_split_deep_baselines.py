"""Generic deep-learning baselines (no Reactome structure) on the Teacher's repeated splits.

Both models share the Teacher's inputs (3,016 omics + PRS), discrete-time survival head
(hazard logit = yearly baseline + one risk score), optimiser settings, batching and val10
early stopping; only the network between the inputs and the risk score differs:
  mlp            3,017 -> 256 -> 64 -> risk
  mlp_attention  MLP tokeniser (3,016 -> 256 -> 16 latent tokens) plus a PRS token,
                 2 transformer blocks (self-attention + MLP), attention pooling -> risk
Evaluated on test10 of the frozen split and of the 20 random splits used by
repeated_split_validation.py.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.prediction import predict
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.seeding import set_global_seed
from maps_discrete.splits import stratified_split
from maps_discrete.training import initialize_baseline, masked_bce, validation_nll
from repeated_split_validation import calibration, discrimination

STUDY = Path(__file__).resolve().parents[2]
MODELS = ("mlp", "mlp_attention")


class MLPRisk(nn.Module):
    def __init__(self, n_inputs: int, n_intervals: int, hidden: tuple[int, ...] = (256, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs + 1
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.GELU(), nn.Dropout(dropout)]
            width = size
        self.encoder = nn.Sequential(*layers)
        self.risk = nn.Linear(width, 1)
        nn.init.zeros_(self.risk.weight)
        nn.init.zeros_(self.risk.bias)
        self.baseline_logits = nn.Parameter(torch.full((n_intervals,), -5.0))

    def forward(self, omics: torch.Tensor, prs: torch.Tensor) -> dict[str, torch.Tensor]:
        log_risk = self.risk(self.encoder(torch.cat([omics, prs], dim=1))).squeeze(1)
        return {
            "hazard_logits": self.baseline_logits.unsqueeze(0) + log_risk.unsqueeze(1),
            "log_risk": log_risk,
            "attention": log_risk.new_zeros(len(log_risk), 1),
        }


class MLPAttentionRisk(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_intervals: int,
        n_tokens: int = 16,
        dim: int = 32,
        n_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.tokenizer = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_tokens * dim),
        )
        self.prs_projection = nn.Linear(1, dim)
        self.token_embed = nn.Parameter(torch.randn(n_tokens + 1, dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=2 * dim, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.pool_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.risk = nn.Linear(dim, 1)
        nn.init.zeros_(self.risk.weight)
        nn.init.zeros_(self.risk.bias)
        self.baseline_logits = nn.Parameter(torch.full((n_intervals,), -5.0))

    def forward(self, omics: torch.Tensor, prs: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(omics).view(-1, self.n_tokens, self.dim)
        tokens = torch.cat([tokens, self.prs_projection(prs).unsqueeze(1)], dim=1) + self.token_embed
        tokens = self.norm(self.blocks(tokens))
        attention = torch.softmax(tokens @ self.pool_query / self.dim ** 0.5, dim=1)
        log_risk = self.risk((attention.unsqueeze(-1) * tokens).sum(dim=1)).squeeze(1)
        return {
            "hazard_logits": self.baseline_logits.unsqueeze(0) + log_risk.unsqueeze(1),
            "log_risk": log_risk,
            "attention": attention,
        }


def build(name: str, n_inputs: int, n_intervals: int, dropout: float) -> nn.Module:
    if name == "mlp":
        return MLPRisk(n_inputs, n_intervals, dropout=dropout)
    return MLPAttentionRisk(n_inputs, n_intervals, dropout=dropout)


def fit(model, x, prs, data, train, val, device, seed, settings) -> tuple[int, float]:
    """Same loop as training.fit_model; weight decay on every weight except the yearly baseline."""
    initialize_baseline(model, data.targets, data.interval_mask, train)
    weights = [p for name, p in model.named_parameters() if name != "baseline_logits"]
    optimizer = Adam([
        {"params": weights, "lr": settings.learning_rate, "weight_decay": settings.binn_weight_decay},
        {"params": [model.baseline_logits], "lr": settings.learning_rate, "weight_decay": 0.0},
    ])
    dataset = TensorDataset(*[torch.from_numpy(a[train]) for a in (x, prs, data.targets, data.interval_mask)])
    best, best_epoch, waiting, best_state = float("inf"), 0, 0, None
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(int(seed) + epoch))
        for xb, pb, yb, mb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), pb.to(device))["hazard_logits"]
            loss = masked_bce(logits, yb.to(device), mb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
        loss_val = validation_nll(model, x, prs, data.targets, data.interval_mask, val, device)
        if loss_val < best - settings.min_delta:
            best, best_epoch, waiting = loss_val, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            waiting += 1
            if waiting >= settings.patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "deep_baselines"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=None,
                        help="default: the frozen Teacher learning rate")
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.2])
    parser.add_argument("--frozen-only", action="store_true", help="only the frozen split (for tuning on val10)")
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    learning_rates = args.learning_rates or [settings.learning_rate]
    data = load_internal_cohort(config, include_reactome=False)
    omics = list(dict.fromkeys([*data.nmr_features, *data.protein_features]))
    splits = [("frozen", data.idx_train, data.idx_val, data.idx_test)]
    if not args.frozen_only:
        splits += [(f"random_{seed:02d}", *stratified_split(data.event, seed)) for seed in args.split_seeds]

    rows: list[dict] = []
    for split, train, val, test in splits:
        fitted = fit_preprocessor(replace(data, idx_train=train, idx_val=val, idx_test=test), omics)
        x, prs = fitted.matrix(omics), fitted.prs()
        duration, event = data.duration_years[test], data.event[test]
        for name, learning_rate, dropout in [
            (n, lr, d) for n in args.models for lr in learning_rates for d in args.dropouts
        ]:
            set_global_seed(args.training_seed)
            model = build(name, x.shape[1], config.n_intervals, dropout).to(device)
            run_settings = replace(settings, learning_rate=learning_rate)
            best_epoch, best_nll = fit(model, x, prs, data, train, val, device, args.training_seed, run_settings)
            cumulative = predict(model, x, prs, device).cumulative_incidence[:, 9].astype(float)
            risk = cumulative[test]
            val_known = data.duration_years[val] >= 10
            val_cases = (data.event[val] == 1) & (data.duration_years[val] <= 10)
            val_mask = val_known | val_cases
            row = {"split": split, "model": name, "learning_rate": learning_rate, "dropout": dropout,
                   "best_epoch": best_epoch, "val_nll": best_nll,
                   "val_auc_10y": float(roc_auc_score(val_cases[val_mask], cumulative[val][val_mask])),
                   "parameters": sum(p.numel() for p in model.parameters())}
            row.update({k.replace(f"{name}_", ""): v for k, v in discrimination(name, duration, event, risk).items()})
            row.update({k.replace(f"{name}_", ""): v for k, v in calibration(name, duration, event, risk).items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "deep_test10_metrics_by_split.csv", index=False)
            print(f"[{(time.time() - started) / 60:.1f} min] {split} {name} lr={learning_rate:g} dropout={dropout:g} "
                  f"best_epoch={best_epoch} val_nll={best_nll:.5f} valAUC={row['val_auc_10y']:.4f} "
                  f"AUC10={row['auc_10y']:.4f} PR10={row['pr_auc_10y']:.4f} Brier10={row['brier_10y']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    random = frame.loc[frame["split"].ne("frozen")] if not args.frozen_only else frame
    metrics = ["auc_10y", "pr_auc_10y", "c_index", "brier_10y", "observed_expected_10y", "auc_3y", "auc_5y", "best_epoch"]
    summary = random.groupby(["model", "learning_rate", "dropout"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(output / "deep_summary_random_splits.csv")
    (output / "deep_manifest.json").write_text(json.dumps({
        "purpose": "generic deep baselines without Reactome structure on the Teacher's repeated splits",
        "models": args.models, "split_seeds": [] if args.frozen_only else args.split_seeds,
        "training_seed": args.training_seed, "learning_rates": learning_rates, "dropouts": args.dropouts,
        "settings": asdict(settings), "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    print("\nRandom splits, test10:")
    print(summary.T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
