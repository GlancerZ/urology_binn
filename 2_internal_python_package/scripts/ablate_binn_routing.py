"""Ablation of the Teacher BINN's Residual channel and direct deep-layer input entries.

Variants start from the frozen seed-specific initialisation and are then modified:
  full         frozen simple_dual routing (reference)
  no_residual  the H1 Residual node receives no inputs, so unmapped features leave the model
  no_direct    no raw-input entries at H2-H5, so those features reach the model only via Residual
  neither      both removed: only H1-mapped features and PRS remain
Same development protocol as compare_mixture_cure_development.py: fit on 90% of train80,
early-stop on the other 10%, evaluate on val10. test10 is never evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.mixture_cure import case_interval_distribution
from maps_discrete.mixture_pipeline import (
    DEVELOPMENT_SPLIT_SEED,
    build_teacher,
    development_split,
    evaluate_cumulative,
    train_teacher,
)
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.preprocessing import fit_preprocessor

STUDY = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_residual", "no_direct", "neither")
KEY_METRICS = [
    "auc_10y", "pr_auc_10y", "brier_10y", "c_index", "auc_1y", "auc_3y", "auc_5y",
    "observed_expected_10y", "residual_token_share",
]


def remove_residual_inputs(binn) -> int:
    """Disconnect every raw input from the H1 Residual node."""
    column = list(binn.connectivity_matrices[0].columns).index("Residual")
    first = binn.layers.blocks[0][0]
    removed = int(first.weight_mask[column].sum())
    with torch.no_grad():
        first.weight_mask[column] = 0
        first.weight_orig[column] = 0
    return removed


def remove_direct_entries(binn) -> int:
    """Drop the raw-input entries at H2-Hn; those inputs keep only their Residual route."""
    removed = sum(
        int(binn.layers.direct_inputs[i].weight_mask.sum())
        for i, used in enumerate(binn.layers.has_direct_input) if used
    )
    binn.layers.has_direct_input = [False] * len(binn.layers.has_direct_input)
    return removed


def residual_token_share(model, x: np.ndarray, prs: np.ndarray, index: np.ndarray, device) -> float:
    """Share of log-risk variance carried by the H6 Residual token: cov(contribution, risk) / var(risk)."""
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x[index]).to(device), torch.from_numpy(prs[index]).to(device))
    contribution = output["contribution"].cpu().numpy()
    risk = output["log_risk"].cpu().numpy()
    column = list(model.binn.layer_names[-1]).index("Residual")
    covariance = np.cov(contribution[:, column], risk)
    return float(covariance[0, 1] / covariance[1, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "13_binn_routing_ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    data = load_internal_cohort(config)
    fit_index, stop_index = development_split(data)
    val = data.idx_val
    reference = case_interval_distribution(data.targets, fit_index)
    horizons = [int(h) for h in config.raw["endpoint"]["horizons_years"]]

    features = x = prs = None
    rows: list[dict] = []
    for variant in args.variants:
        for seed in args.seeds:
            model = build_teacher(config, data, "ph", seed, device)
            if features is None:
                features = list(model.binn.inputs)
                fitted = fit_preprocessor(data, features)
                x, prs = fitted.matrix(features), fitted.prs()
            removed = {"residual_links_removed": 0, "direct_links_removed": 0}
            if variant in {"no_residual", "neither"}:
                removed["residual_links_removed"] = remove_residual_inputs(model.binn)
            if variant in {"no_direct", "neither"}:
                removed["direct_links_removed"] = remove_direct_entries(model.binn)
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                fit, cumulative, _ = train_teacher(
                    "ph", model, x, prs, data, fit_index, stop_index, device, checkpoint_dir, seed, settings,
                )
            metrics = evaluate_cumulative(data, cumulative, val, reference, horizons)
            metrics.update(removed)
            metrics.update({
                "variant": variant, "seed": seed, "best_epoch": fit.best_epoch,
                "early_stopping_nll": fit.best_validation_nll,
                "residual_token_share": residual_token_share(model, x, prs, val, device),
            })
            rows.append(metrics)
            pd.DataFrame(rows).to_csv(output / "val10_metrics_by_seed.csv", index=False)
            print(
                f"[{(time.time() - started) / 60:.1f} min] {variant} seed={seed} best_epoch={fit.best_epoch} "
                f"AUC10={metrics['auc_10y']:.4f} PR10={metrics['pr_auc_10y']:.4f} "
                f"Brier10={metrics['brier_10y']:.4f} residual_share={metrics['residual_token_share']:.4f}",
                flush=True,
            )

    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby("variant", sort=False)[KEY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "val10_summary_mean_sd.csv")
    if "full" in args.variants:
        full = by_seed.loc[by_seed["variant"].eq("full")].set_index("seed")
        deltas = []
        for variant in args.variants:
            if variant == "full":
                continue
            other = by_seed.loc[by_seed["variant"].eq(variant)].set_index("seed")
            delta = (other[KEY_METRICS[:-1]] - full[KEY_METRICS[:-1]]).dropna()
            deltas.append(delta.agg(["mean", "std", "min", "max"]).assign(variant=variant))
        pd.concat(deltas).to_csv(output / "val10_paired_differences_vs_full.csv")
    (output / "ablation_manifest.json").write_text(json.dumps({
        "purpose": "BINN routing ablation; test10 not evaluated",
        "variants": args.variants,
        "seeds": args.seeds,
        "early_stopping": f"10% of train80 (stratified, seed {DEVELOPMENT_SPLIT_SEED}), n={len(stop_index)}",
        "evaluation": f"val10, n={len(val)}, events={int(data.event[val].sum())}",
        "settings": asdict(settings),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\nMean (SD) over seeds, val10:")
    mean = by_seed.groupby("variant", sort=False)[KEY_METRICS].mean()
    sd = by_seed.groupby("variant", sort=False)[KEY_METRICS].std()
    print((mean.map(lambda v: f"{v:.4f}") + " (" + sd.map(lambda v: f"{v:.4f}") + ")").T.to_string())


if __name__ == "__main__":
    main()
