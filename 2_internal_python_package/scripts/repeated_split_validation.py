"""Repeated stratified 80/10/10 splits of the internal cohort for the frozen Teacher recipe.

For every split, preprocessing is fitted on train80 and the Teacher (training seed 7,
frozen settings) is early-stopped on val10 and evaluated on test10. Two references are
evaluated on the same test10: KLK3 (P07288) alone and a logistic model on KLK3 + PRS
fitted on train80. Split "frozen" is the protocol split; the others are random.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.endpoint import fixed_horizon_status
from maps_discrete.metrics import harrell_c_index
from maps_discrete.mixture_pipeline import build_teacher, train_teacher
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.splits import stratified_split

STUDY = Path(__file__).resolve().parents[2]
HORIZONS = (1, 3, 5, 10)
MODELS = ("teacher", "klk3", "klk3_prs_logistic")


def discrimination(prefix: str, duration: np.ndarray, event: np.ndarray, score: np.ndarray) -> dict[str, float]:
    result = {f"{prefix}_c_index": harrell_c_index(duration, event, score)}
    for horizon in HORIZONS:
        known, outcome = fixed_horizon_status(duration, event, horizon)
        result[f"{prefix}_auc_{horizon}y"] = float(roc_auc_score(outcome[known], score[known]))
        result[f"{prefix}_pr_auc_{horizon}y"] = float(average_precision_score(outcome[known], score[known]))
    return result


def calibration(prefix: str, duration: np.ndarray, event: np.ndarray, risk_10y: np.ndarray) -> dict[str, float]:
    known, outcome = fixed_horizon_status(duration, event, 10)
    return {
        f"{prefix}_brier_10y": float(brier_score_loss(outcome[known], risk_10y[known])),
        f"{prefix}_observed_expected_10y": float(outcome[known].sum() / risk_10y[known].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    data = load_internal_cohort(config)
    model = build_teacher(config, data, "ph", args.training_seed, device)
    features = list(model.binn.inputs)
    klk3_column = features.index("P07288")
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    splits = [("frozen", data.idx_train, data.idx_val, data.idx_test)]
    splits += [(f"random_{seed:02d}", *stratified_split(data.event, seed)) for seed in args.split_seeds]
    rows: list[dict] = []
    for name, train, val, test in splits:
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
        teacher_risk = cumulative[test, 9]
        klk3 = x[test, klk3_column].astype(float)
        known_train, outcome_train = fixed_horizon_status(data.duration_years[train], data.event[train], 10)
        two = np.column_stack([x[:, klk3_column], prs[:, 0]]).astype(float)
        logistic = LogisticRegression(C=1.0, max_iter=5000).fit(two[train][known_train], outcome_train[known_train])
        logistic_risk = logistic.predict_proba(two[test])[:, 1]

        row = {
            "split": name, "train_n": len(train), "val_n": len(val), "test_n": len(test),
            "test_events": int(event.sum()), "best_epoch": fit.best_epoch,
        }
        row.update(discrimination("teacher", duration, event, teacher_risk))
        row.update(calibration("teacher", duration, event, teacher_risk))
        row.update(discrimination("klk3", duration, event, klk3))
        row.update(discrimination("klk3_prs_logistic", duration, event, logistic_risk))
        row.update(calibration("klk3_prs_logistic", duration, event, logistic_risk))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "test10_metrics_by_split.csv", index=False)
        print(
            f"[{(time.time() - started) / 60:.1f} min] {name} best_epoch={fit.best_epoch} "
            f"Teacher AUC10={row['teacher_auc_10y']:.4f} PR10={row['teacher_pr_auc_10y']:.4f} | "
            f"KLK3 AUC10={row['klk3_auc_10y']:.4f} | KLK3+PRS AUC10={row['klk3_prs_logistic_auc_10y']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    random = frame.loc[frame["split"].ne("frozen")]
    metrics = [column for column in frame.columns if column.split("_")[0] in {"teacher", "klk3"}]
    summary = random[metrics].agg(["mean", "std", "min", "max"]).T
    summary["frozen_split"] = frame.loc[frame["split"].eq("frozen"), metrics].iloc[0]
    summary.to_csv(output / "test10_summary_random_splits.csv")
    paired = {}
    for reference in ("klk3", "klk3_prs_logistic"):
        for metric in ("auc_10y", "pr_auc_10y", "c_index"):
            delta = random[f"teacher_{metric}"] - random[f"{reference}_{metric}"]
            paired[f"teacher_minus_{reference}_{metric}"] = {
                "mean": float(delta.mean()), "sd": float(delta.std()),
                "p2.5": float(delta.quantile(0.025)), "p97.5": float(delta.quantile(0.975)),
                "share_teacher_better": float((delta > 0).mean()),
            }
    pd.DataFrame(paired).T.to_csv(output / "test10_paired_differences_random_splits.csv")
    (output / "repeated_split_manifest.json").write_text(json.dumps({
        "purpose": "robustness of the frozen Teacher recipe across event-stratified 80/10/10 splits",
        "split_seeds": args.split_seeds,
        "training_seed": args.training_seed,
        "early_stopping": "val10 of each split (frozen protocol)",
        "evaluation": "test10 of each split",
        "settings": asdict(settings),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    key = [f"{m}_{k}" for m in MODELS for k in ("auc_10y", "pr_auc_10y", "c_index")] + [
        "teacher_brier_10y", "teacher_observed_expected_10y", "teacher_auc_1y", "teacher_auc_3y", "teacher_auc_5y",
    ]
    print("\nRandom splits (n=%d):" % len(random))
    print(summary.loc[key].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nPaired differences:")
    print(pd.DataFrame(paired).T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
