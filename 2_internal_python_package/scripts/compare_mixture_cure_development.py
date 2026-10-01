"""Development comparison of Teacher variants on val10; test10 is never evaluated.

Variants: "ph" (frozen proportional-hazards Teacher), "ordinal" and "softmax" mixture
cure Teachers. Every variant is fitted on 90% of train80, early-stopped on the other
10% and evaluated on val10, per seed and as a seed ensemble.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.mixture_cure import case_interval_distribution
from maps_discrete.mixture_pipeline import (
    DEVELOPMENT_SPLIT_SEED,
    VARIANTS,
    build_teacher,
    development_split,
    evaluate_cumulative,
    prediction_frame,
    train_teacher,
)
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.preprocessing import fit_preprocessor

STUDY = Path(__file__).resolve().parents[2]
KEY_METRICS = [
    "auc_10y", "pr_auc_10y", "brier_10y", "c_index", "auc_1y", "auc_3y", "auc_5y",
    "observed_expected_10y", "timing_log_score_gain", "timing_concordance", "early_vs_late_auc",
    "median_year_mae", "reference_median_year_mae", "interval80_coverage", "interval80_mean_width",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "12_mixture_cure_development"))
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
    cumulative_by_variant: dict[str, list[np.ndarray]] = {variant: [] for variant in args.variants}
    for variant in args.variants:
        for seed in args.seeds:
            model = build_teacher(config, data, variant, seed, device)
            if features is None:
                features = list(model.binn.inputs)
                fitted = fit_preprocessor(data, features)
                x, prs = fitted.matrix(features), fitted.prs()
            if list(model.binn.inputs) != features:
                raise RuntimeError("Teacher input order changed between runs")
            run_dir = output / variant / f"seed_{seed:03d}"
            fit, cumulative, extras = train_teacher(
                variant, model, x, prs, data, fit_index, stop_index, device, run_dir, seed, settings,
            )
            prediction_frame(data, cumulative, val, extras.get("timing_score")).to_csv(
                run_dir / "val10_predictions.csv", index=False
            )
            metrics = evaluate_cumulative(data, cumulative, val, reference, horizons)
            metrics.update({
                "variant": variant, "seed": seed, "best_epoch": fit.best_epoch,
                "early_stopping_nll": fit.best_validation_nll,
            })
            rows.append(metrics)
            cumulative_by_variant[variant].append(cumulative)
            pd.DataFrame(rows).to_csv(output / "val10_metrics_by_seed.csv", index=False)
            print(
                f"[{(time.time() - started) / 60:.1f} min] {variant} seed={seed} best_epoch={fit.best_epoch} "
                f"AUC10={metrics['auc_10y']:.4f} PR10={metrics['pr_auc_10y']:.4f} "
                f"timing_gain={metrics['timing_log_score_gain']:+.4f} "
                f"timing_C={metrics['timing_concordance']:.3f} early_vs_late={metrics['early_vs_late_auc']:.3f}",
                flush=True,
            )

    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby("variant", sort=False)[KEY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "val10_summary_mean_sd.csv")
    ensembles = []
    for variant, arrays in cumulative_by_variant.items():
        metrics = evaluate_cumulative(data, np.mean(arrays, axis=0), val, reference, horizons)
        metrics["variant"] = f"{variant}_ensemble{len(arrays)}"
        ensembles.append(metrics)
    ensemble = pd.DataFrame(ensembles).set_index("variant")
    ensemble.to_csv(output / "val10_seed_ensembles.csv")

    manifest = {
        "purpose": "development comparison; test10 not evaluated",
        "variants": args.variants,
        "seeds": args.seeds,
        "fit_n": int(len(fit_index)),
        "early_stopping": f"10% of train80 (stratified, seed {DEVELOPMENT_SPLIT_SEED}), n={len(stop_index)}",
        "evaluation": f"val10, n={len(val)}, events={int(data.event[val].sum())}",
        "settings": asdict(settings),
        "config_path": str(config.path),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output / "development_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    mean_table = by_seed.groupby("variant", sort=False)[KEY_METRICS].mean().T
    print("\nMean over seeds (val10):\n" + mean_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nSeed ensembles (val10):\n" + ensemble[KEY_METRICS].T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
