"""BINN-based improvements for 10-year discrimination, on the Teacher's repeated splits.

Every variant starts from the frozen BINN initialisation of its training seed:
  frozen                  the frozen Teacher (discrete-time survival likelihood)
  direct_l1_<lam>         + a sparse linear path from all 3,016 omics inputs straight to the
                          risk score (bypassing pathway layers and attention), L1 penalty lam
  binary                  the same network trained on 10-year onset status (BCE) instead of
                          the discrete-time survival likelihood
  binary_direct_l1_<lam>  both
  binary_relaxed          10-year objective plus an unpenalised direct path restricted to the
                          features a stage-1 L1 run kept (|w| > 0.01) on the same split
                          (--relaxed-from); selection uses only train80 and val10
Model selection must use the validation log-loss of the 10-year risk (val10 of each split);
test10 metrics are reported for every variant. 10-year val/test risks are saved per run so
seed ensembles can be evaluated afterwards.
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
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.modeling import build_model
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.prediction import predict
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.seeding import set_global_seed
from maps_discrete.splits import stratified_split
from maps_discrete.training import initialize_baseline, masked_bce, validation_nll
from repeated_split_validation import calibration, discrimination

STUDY = Path(__file__).resolve().parents[2]


def parse_variant(name: str) -> tuple[bool, float | None, bool]:
    """Return (binary objective, L1 strength of the direct path or None, relaxed direct path)."""
    binary = name.startswith("binary")
    if "direct_l1_" in name:
        return binary, float(name.split("direct_l1_")[1]), False
    if name in {"relaxed", "binary_relaxed"}:
        return binary, None, True
    if name not in {"frozen", "binary"}:
        raise ValueError(f"Unknown variant {name!r}")
    return binary, None, False


def stage1_selection(path: Path, threshold: float = 1e-2) -> dict[str, list[str]]:
    """Per-split features whose stage-1 direct weight exceeded the threshold."""
    frame = pd.read_csv(path)
    selection = {}
    for _, row in frame.iterrows():
        pairs = [item.split(":") for item in str(row["direct_top5"]).split(";")]
        selection[row["split"]] = [name for name, weight in pairs if abs(float(weight)) > threshold]
    return selection


class ImprovedBINN(nn.Module):
    """Frozen Teacher plus an optional sparse direct path and a 10-year logit."""

    def __init__(self, base: nn.Module, n_inputs: int, direct: bool, direct_columns: list[int] | None = None):
        super().__init__()
        self.base = base
        self.register_buffer("direct_columns", torch.tensor(
            direct_columns if direct_columns is not None else range(n_inputs), dtype=torch.long))
        self.direct = nn.Linear(len(self.direct_columns), 1, bias=False) if direct else None
        if self.direct is not None:
            nn.init.zeros_(self.direct.weight)
        self.intercept_10y = nn.Parameter(torch.zeros(1))

    def forward(self, omics: torch.Tensor, prs: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.base(omics, prs)
        log_risk = output["log_risk"]
        if self.direct is not None:
            log_risk = log_risk + self.direct(omics[:, self.direct_columns]).squeeze(1)
        return {
            "hazard_logits": self.base.baseline_logits.unsqueeze(0) + log_risk.unsqueeze(1),
            "logit_10y": self.intercept_10y + log_risk,
            "log_risk": log_risk,
            "attention": output["attention"],
        }


def make_optimizer(model: ImprovedBINN, settings) -> Adam:
    """Frozen parameter groups (Adam, L2 on BINN weights only); direct path unpenalised by L2."""
    base = model.base
    heads = [*base.h6_bn.parameters(), *base.prs_bn.parameters(), *base.fusion.parameters(),
             base.baseline_logits, model.intercept_10y]
    groups = [
        {"params": base.binn.parameters(), "lr": settings.learning_rate, "weight_decay": settings.binn_weight_decay},
        {"params": heads, "lr": settings.learning_rate, "weight_decay": 0.0},
    ]
    if model.direct is not None:
        groups.append({"params": model.direct.parameters(), "lr": settings.learning_rate, "weight_decay": 0.0})
    return Adam(groups)


def binary_loss(model, x, prs, y10, index, device) -> float:
    model.eval()
    total = 0.0
    loader = DataLoader(TensorDataset(torch.from_numpy(x[index]), torch.from_numpy(prs[index]),
                                      torch.from_numpy(y10[index])), batch_size=256, shuffle=False)
    with torch.no_grad():
        for xb, pb, yb in loader:
            logit = model(xb.to(device), pb.to(device))["logit_10y"]
            total += float(F.binary_cross_entropy_with_logits(logit, yb.to(device), reduction="sum"))
    return total / len(index)


def fit(model: ImprovedBINN, x, prs, data, y10, train, val, device, seed, settings, binary, l1,
        soft_logits: np.ndarray | None = None):
    """training.fit_model's loop; binary objective and L1 on the direct path are optional.

    With soft_logits (a Teacher's 10-year logits, binary objective only) the loss is
    (1 - alpha) * hard BCE + alpha * T^2 * BCE(logit / T, sigmoid(soft / T)) with the frozen
    distillation alpha and temperature; early stopping still uses the hard validation loss.
    """
    if soft_logits is not None and not binary:
        raise ValueError("soft_logits distillation is implemented for the binary objective only")
    initialize_baseline(model.base, data.targets, data.interval_mask, train)
    rate = (y10[train].sum() + 0.5) / (len(train) + 1.0)
    with torch.no_grad():
        model.intercept_10y.fill_(float(np.log(rate / (1.0 - rate))))
    optimizer = make_optimizer(model, settings)
    min_delta = settings.min_delta
    if binary:  # per-participant loss: rescale the per person-interval threshold
        min_delta *= float(data.interval_mask[val].sum(axis=1).mean())
    soft = soft_logits if soft_logits is not None else np.zeros(len(y10), dtype=np.float32)
    dataset = TensorDataset(*[torch.from_numpy(np.asarray(a[train], dtype=np.float32))
                              for a in (x, prs, data.targets, data.interval_mask, y10, soft)])
    alpha, temperature = settings.distillation_alpha, settings.distillation_temperature
    best, best_epoch, waiting, best_state = float("inf"), 0, 0, None
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(int(seed) + epoch))
        for xb, pb, yb, mb, y10b, softb in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(xb.to(device), pb.to(device))
            if binary:
                loss = F.binary_cross_entropy_with_logits(output["logit_10y"], y10b.to(device))
                if soft_logits is not None:
                    target = torch.sigmoid(softb.to(device) / temperature)
                    distill = F.binary_cross_entropy_with_logits(output["logit_10y"] / temperature, target)
                    loss = (1.0 - alpha) * loss + alpha * temperature ** 2 * distill
            else:
                loss = masked_bce(output["hazard_logits"], yb.to(device), mb.to(device))
            if l1 is not None:
                loss = loss + l1 * model.direct.weight.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
        if binary:
            loss_val = binary_loss(model, x, prs, y10, val, device)
        else:
            loss_val = validation_nll(model, x, prs, data.targets, data.interval_mask, val, device)
        if loss_val < best - min_delta:
            best, best_epoch, waiting = loss_val, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            waiting += 1
            if waiting >= settings.patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, best


def risk_10y(model: ImprovedBINN, x, prs, device, binary: bool) -> np.ndarray:
    if not binary:
        return predict(model, x, prs, device).cumulative_incidence[:, 9].astype(float)
    model.eval()
    parts = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(prs)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for xb, pb in loader:
            parts.append(torch.sigmoid(model(xb.to(device), pb.to(device))["logit_10y"]).cpu().numpy())
    return np.concatenate(parts).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "binn_improvements"))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--relaxed-from", default=str(
        STUDY / "14_repeated_split_validation" / "binn_improvements" / "binary_direct_l1_1e-2_seed7_by_split.csv"))
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    (output / "risks").mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    data = load_internal_cohort(config)
    y10 = data.event.astype(np.float32)  # every non-case is followed through 10 years
    splits = [("frozen", data.idx_train, data.idx_val, data.idx_test)]
    splits += [(f"random_{seed:02d}", *stratified_split(data.event, seed)) for seed in args.split_seeds]
    teacher_matrix = data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy()
    base_kwargs = dict(
        mapping=data.mapping, pathways=data.pathways, entity_col=data.entity_col,
        binn_root=config.raw["data"]["binn_root"], device=device,
        n_layers=int(config.raw["training"]["n_layers"]), n_intervals=config.n_intervals,
        attention_dim=int(config.raw["training"]["attention_dim"]),
    )

    for variant in args.variants:
        binary, l1, relaxed = parse_variant(variant)
        selection = stage1_selection(Path(args.relaxed_from)) if relaxed else {}
        for seed in args.training_seeds:
            set_global_seed(seed)
            base = build_model(data_matrix=teacher_matrix, **base_kwargs)
            features = list(base.binn.inputs)
            base_initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
            rows: list[dict] = []
            for split, train, val, test in splits:
                fitted = fit_preprocessor(replace(data, idx_train=train, idx_val=val, idx_test=test), features)
                x, prs = fitted.matrix(features), fitted.prs()
                base.load_state_dict(base_initial)
                columns = [features.index(name) for name in selection[split]] if relaxed else None
                model = ImprovedBINN(base, len(features), direct=l1 is not None or relaxed,
                                     direct_columns=columns).to(device)
                best_epoch, best_loss = fit(model, x, prs, data, y10, train, val, device, seed, settings, binary, l1)
                risk = risk_10y(model, x, prs, device, binary)
                np.savez(output / "risks" / f"{variant}_{split}_seed{seed}.npz", val=risk[val], test=risk[test])
                duration, event = data.duration_years[test], data.event[test]
                row = {
                    "variant": variant, "split": split, "training_seed": seed, "best_epoch": best_epoch,
                    "early_stopping_loss": best_loss,
                    "val_logloss_10y": float(log_loss(y10[val], np.clip(risk[val], 1e-7, 1 - 1e-7))),
                    "val_auc_10y": float(roc_auc_score(y10[val], risk[val])),
                }
                if l1 is not None or relaxed:
                    weights = model.direct.weight.detach().cpu().numpy().ravel()
                    names = [features[i] for i in model.direct_columns.tolist()]
                    order = np.argsort(-np.abs(weights))[:5]
                    row["direct_nonzero"] = int((np.abs(weights) > 1e-2).sum())
                    row["direct_top5"] = ";".join(f"{names[i]}:{weights[i]:+.3f}" for i in order)
                row.update({k.replace("model_", ""): v for k, v in discrimination("model", duration, event, risk[test]).items()})
                row.update({k.replace("model_", ""): v for k, v in calibration("model", duration, event, risk[test]).items()})
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / f"{variant}_seed{seed}_by_split.csv", index=False)
                print(f"[{(time.time() - started) / 60:.1f} min] {variant} seed={seed} {split} best_epoch={best_epoch} "
                      f"valLL={row['val_logloss_10y']:.5f} AUC10={row['auc_10y']:.4f} PR10={row['pr_auc_10y']:.4f}"
                      + (f" direct={row['direct_top5']}" if relaxed else "")
                      + (f" nonzero={row['direct_nonzero']}" if l1 is not None else ""), flush=True)
    (output / f"manifest_{'_'.join(args.variants)}_seeds{'_'.join(map(str, args.training_seeds))}.json").write_text(
        json.dumps({"variants": args.variants, "training_seeds": args.training_seeds, "split_seeds": args.split_seeds,
                    "settings": asdict(settings), "elapsed_minutes": (time.time() - started) / 60.0}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
