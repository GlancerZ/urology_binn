from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import (
    bootstrap_model_comparison,
    bootstrap_model_performance,
    decision_curve,
    evaluate_model,
)
from .selection import rank_seeds


MODELS = ("teacher", "student_distilled", "student_no_distillation")
METRICS = ("c_index", "auc", "pr_auc", "brier")


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].astype(float)
            rows.append({
                "model": model,
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    return pd.DataFrame(rows)


def _load_all_metrics(root: Path, expected_seeds: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for seed in expected_seeds:
        path = root / f"seed_{seed:03d}" / "test_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if set(frame["model"]) != set(MODELS) or len(frame) != len(MODELS):
            raise RuntimeError(f"Unexpected model rows in {path}")
        if set(frame["training_seed"].astype(int)) != {seed}:
            raise RuntimeError(f"Training seed mismatch in {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != len(expected_seeds) * len(MODELS):
        raise RuntimeError("Incomplete multiseed metric table")
    return result


def _ensemble_predictions(root: Path, selected_seeds: list[int]) -> pd.DataFrame:
    frames = [
        pd.read_csv(root / f"seed_{seed:03d}" / "predictions_test.csv")
        for seed in selected_seeds
    ]
    keys = ["sample", "split", "event_10y", "observed_time_years"]
    reference = frames[0][keys].copy()
    prediction_columns = [
        column for column in frames[0].columns
        if column.endswith("_log_risk") or "_risk_" in column
    ]
    for frame in frames[1:]:
        if not reference.equals(frame[keys]):
            raise RuntimeError("Selected seed prediction files do not share identical test rows")
        if prediction_columns != [
            column for column in frame.columns
            if column.endswith("_log_risk") or "_risk_" in column
        ]:
            raise RuntimeError("Selected seed prediction columns are inconsistent")
    ensemble = reference.copy()
    for column in prediction_columns:
        ensemble[column] = np.mean(
            np.stack([frame[column].to_numpy(float) for frame in frames]), axis=0
        )
    return ensemble


def analyze(config_path: str | Path, input_root: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    root = Path(input_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = _load_all_metrics(root, config.training_seeds)
    all_metrics.to_csv(output / "all_100_seed_test_metrics.csv", index=False)
    _metric_summary(all_metrics).to_csv(output / "all_100_seed_metric_summary.csv", index=False)

    teacher = all_metrics.loc[all_metrics["model"].eq("teacher")]
    selected_seeds = rank_seeds(
        teacher,
        metric="auc",
        n=int(config.raw["selection"]["n_selected"]),
        higher_is_better=True,
        dataset_role="test",
        analysis_role="exploratory",
        allow_test_based_selection=True,
    )
    teacher.sort_values("auc", ascending=False).to_csv(
        output / "teacher_seed_ranking_by_test_auc.csv", index=False
    )
    selected = all_metrics.loc[all_metrics["training_seed"].isin(selected_seeds)].copy()
    selected.to_csv(output / "selected_top5_seed_metrics.csv", index=False)
    _metric_summary(selected).to_csv(output / "selected_top5_metric_summary.csv", index=False)
    (output / "selected_teacher_top5_seeds.json").write_text(
        json.dumps({
            "selection_metric": "teacher_test_auc",
            "selected_seeds": selected_seeds,
            "analysis_role": "exploratory",
            "warning": "Test-selected Top-5 estimates are selection-biased.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ensemble = _ensemble_predictions(root, selected_seeds)
    ensemble.to_csv(output / "selected_top5_ensemble_predictions_test.csv", index=False)
    duration = ensemble["observed_time_years"].to_numpy(float)
    event = ensemble["event_10y"].to_numpy(int)
    horizon = config.primary_horizon
    performance_rows = []
    ci_frames = []
    for model in MODELS:
        log_risk = ensemble[f"{model}_log_risk"].to_numpy(float)
        risk = ensemble[f"{model}_risk_{horizon}y"].to_numpy(float)
        result = evaluate_model(duration, event, log_risk, risk, horizon)
        result["model"] = model
        performance_rows.append(result)
        ci = bootstrap_model_performance(
            duration, event, log_risk, risk, horizon,
            repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
            confidence_level=float(config.raw["evaluation"]["confidence_level"]),
            seed=config.test_holdout_seed,
        )
        ci.insert(0, "model", model)
        ci_frames.append(ci)
    pd.DataFrame(performance_rows).to_csv(
        output / "selected_top5_ensemble_test_metrics.csv", index=False
    )
    pd.concat(ci_frames, ignore_index=True).to_csv(
        output / "selected_top5_ensemble_bootstrap_95ci.csv", index=False
    )

    comparisons = [
        ("distillation_gain", "student_no_distillation", "student_distilled"),
        ("teacher_gain_over_distilled_student", "student_distilled", "teacher"),
    ]
    comparison_frames = []
    for label, reference_model, new_model in comparisons:
        comparison = bootstrap_model_comparison(
            duration,
            event,
            ensemble[f"{reference_model}_log_risk"].to_numpy(float),
            ensemble[f"{reference_model}_risk_{horizon}y"].to_numpy(float),
            ensemble[f"{new_model}_log_risk"].to_numpy(float),
            ensemble[f"{new_model}_risk_{horizon}y"].to_numpy(float),
            horizon,
            repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
            confidence_level=float(config.raw["evaluation"]["confidence_level"]),
            seed=config.test_holdout_seed,
        )
        comparison.insert(0, "new_model", new_model)
        comparison.insert(0, "reference_model", reference_model)
        comparison.insert(0, "comparison", label)
        comparison_frames.append(comparison)
    pd.concat(comparison_frames, ignore_index=True).to_csv(
        output / "selected_top5_ensemble_pairwise_bootstrap_95ci.csv", index=False
    )

    thresholds = np.linspace(
        float(config.raw["evaluation"]["dca_threshold_min"]),
        float(config.raw["evaluation"]["dca_threshold_max"]),
        int(config.raw["evaluation"]["dca_threshold_points"]),
    )
    dca_frames = []
    for model in MODELS:
        curve = decision_curve(
            duration, event, ensemble[f"{model}_risk_{horizon}y"].to_numpy(float),
            thresholds, horizon,
        )
        curve.insert(0, "model", model)
        dca_frames.append(curve)
    pd.concat(dca_frames, ignore_index=True).to_csv(
        output / "selected_top5_ensemble_dca.csv", index=False
    )

    result = {
        "analysis_role": "exploratory",
        "completed_training_seeds": len(config.training_seeds),
        "selected_teacher_top5_seeds": selected_seeds,
        "output_directory": str(output),
        "warning": "Teacher seeds were selected on the test AUC; report all-100 distributions alongside Top-5.",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Teacher-Student multiseed runs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.input_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
