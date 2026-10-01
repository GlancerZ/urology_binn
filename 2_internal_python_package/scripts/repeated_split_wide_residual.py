"""Wider Residual channel for the Teacher BINN on the Teacher's repeated splits.

residual_width K replaces the single Residual node per layer with K nodes: inputs
without an H1 pathway feed all K nodes at H1, Residual nodes connect all-to-all
between layers and never mix with pathway nodes, and H6 contributes K Residual tokens
to the attention fusion. Everything else is the frozen Teacher recipe, so K=1
reproduces the current Teacher. --optimizer adamw swaps Adam with an L2 term in the
gradient (frozen recipe) for AdamW's decoupled weight decay. Choose configurations on
val_nll (val10 of each split); test10 metrics are reported for every configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.mixture_pipeline import train_teacher
from maps_discrete.modeling import build_model
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.seeding import set_global_seed
from maps_discrete.splits import stratified_split
from repeated_split_validation import calibration, discrimination

STUDY = Path(__file__).resolve().parents[2]


def token_usage(model, x: np.ndarray, prs: np.ndarray, index: np.ndarray, device) -> tuple[float, int]:
    """Residual tokens' share of log-risk variance and the number of tokens with >2% mean attention."""
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x[index]).to(device), torch.from_numpy(prs[index]).to(device))
    contribution = output["contribution"].cpu().numpy()
    risk = output["log_risk"].cpu().numpy()
    columns = [i for i, name in enumerate(model.binn.layer_names[-1]) if str(name).startswith("Residual")]
    covariance = np.cov(contribution[:, columns].sum(axis=1), risk)
    active = int((output["attention"].cpu().numpy().mean(axis=0) > 0.02).sum())
    return float(covariance[0, 1] / covariance[1, 1]), active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "wide_residual"))
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=None, help="default: the frozen binn_weight_decay")
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    frozen_decay = settings.binn_weight_decay
    decay = frozen_decay if args.weight_decay is None else args.weight_decay
    settings = replace(settings, decoupled_weight_decay=args.optimizer == "adamw", binn_weight_decay=decay)
    tag = "" if (args.optimizer == "adam" and decay == frozen_decay) else f"_{args.optimizer}_wd{decay:g}"
    data = load_internal_cohort(config)
    splits = [("frozen", data.idx_train, data.idx_val, data.idx_test)]
    splits += [(f"random_{seed:02d}", *stratified_split(data.event, seed)) for seed in args.split_seeds]

    for width in args.widths:
        set_global_seed(args.training_seed)
        model = build_model(
            data_matrix=data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy(),
            mapping=data.mapping, pathways=data.pathways, entity_col=data.entity_col,
            binn_root=config.raw["data"]["binn_root"], device=device,
            n_layers=int(config.raw["training"]["n_layers"]), n_intervals=config.n_intervals,
            attention_dim=int(config.raw["training"]["attention_dim"]), residual_width=width,
        )
        features = list(model.binn.inputs)
        initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        rows: list[dict] = []
        for split, train, val, test in splits:
            split_data = replace(data, idx_train=train, idx_val=val, idx_test=test)
            fitted = fit_preprocessor(split_data, features)
            x, prs = fitted.matrix(features), fitted.prs()
            model.load_state_dict(initial_state)
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                fit, cumulative, _ = train_teacher(
                    "ph", model, x, prs, split_data, train, val, device, checkpoint_dir,
                    args.training_seed, settings,
                )
            duration, event = data.duration_years[test], data.event[test]
            risk = cumulative[test, 9]
            share, active = token_usage(model, x, prs, test, device)
            row = {
                "split": split, "residual_width": width, "optimizer": args.optimizer, "weight_decay": decay,
                "h6_tokens": len(model.binn.layer_names[-1]), "best_epoch": fit.best_epoch,
                "val_nll": fit.best_validation_nll, "residual_token_share": share, "active_tokens": active,
            }
            row.update({k.replace("model_", ""): v for k, v in discrimination("model", duration, event, risk).items()})
            row.update({k.replace("model_", ""): v for k, v in calibration("model", duration, event, risk).items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / f"width_{width:03d}{tag}_by_split.csv", index=False)
            print(f"[{(time.time() - started) / 60:.1f} min] K={width}{tag} {split} best_epoch={fit.best_epoch} "
                  f"val_nll={fit.best_validation_nll:.6f} AUC10={row['auc_10y']:.4f} PR10={row['pr_auc_10y']:.4f} "
                  f"residual_share={row['residual_token_share']:.3f} active_tokens={active}", flush=True)
    (output / f"manifest_widths_{'_'.join(map(str, args.widths))}{tag}.json").write_text(json.dumps({
        "purpose": "wider Residual channel on the Teacher's repeated splits",
        "widths": args.widths, "split_seeds": args.split_seeds, "training_seed": args.training_seed,
        "settings": asdict(settings), "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
