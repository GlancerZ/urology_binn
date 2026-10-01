from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import bootstrap_model_performance, decision_curve, evaluate_model
from .selection import rank_seeds


METRICS = ("c_index", "auc", "pr_auc", "brier")


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=True):
        for metric in METRICS:
            values = group[metric].astype(float)
            rows.append({
                "model": model,
                "modality": group["modality"].iloc[0],
                "algorithm": group["algorithm"].iloc[0],
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    return pd.DataFrame(rows)


def load_metrics(root: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    expected_models: set[str] | None = None
    for seed in seeds:
        path = root / f"seed_{seed:03d}" / "ml_test_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        models = set(frame["model"].astype(str))
        if expected_models is None:
            expected_models = models
        elif models != expected_models:
            raise RuntimeError(f"Model set mismatch in {path}")
        if len(frame) != len(models) or set(frame["training_seed"].astype(int)) != {seed}:
            raise RuntimeError(f"Malformed seed metric file: {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["training_seed"].nunique() != len(seeds):
        raise RuntimeError("Incomplete ML seed grid")
    return result


def analyze(config_path: str | Path, input_root: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    root = Path(input_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(root, config.training_seeds)
    metrics.to_csv(output / "all_100_seed_ml_test_metrics.csv", index=False)
    metric_summary(metrics).to_csv(output / "all_100_seed_ml_metric_summary.csv", index=False)

    selected_by_model: dict[str, list[int]] = {}
    ranking_frames = []
    selected_frames = []
    n_selected = int(config.raw["selection"]["n_selected"])
    for model, group in metrics.groupby("model", sort=True):
        selected = rank_seeds(
            group,
            metric="auc",
            n=n_selected,
            higher_is_better=False,
            dataset_role="test",
            analysis_role="exploratory",
            allow_test_based_selection=True,
        )
        selected_by_model[str(model)] = selected
        ranking = group.sort_values("auc", ascending=True).copy()
        ranking.insert(0, "rank_worst_to_best", np.arange(1, len(ranking) + 1))
        ranking_frames.append(ranking)
        selected_frames.append(group.loc[group["training_seed"].isin(selected)].copy())
    pd.concat(ranking_frames, ignore_index=True).to_csv(
        output / "ml_seed_rankings_by_test_auc.csv", index=False
    )
    selected_metrics = pd.concat(selected_frames, ignore_index=True)
    selected_metrics.to_csv(output / "selected_bottom5_seed_metrics.csv", index=False)
    metric_summary(selected_metrics).to_csv(output / "selected_bottom5_metric_summary.csv", index=False)
    (output / "selected_ml_bottom5_seeds_by_model.json").write_text(
        json.dumps({
            "selection_metric": "test_auc",
            "selection_direction": "lowest_5_per_model",
            "analysis_role": "exploratory",
            "selected_seeds_by_model": selected_by_model,
            "warning": "Test-selected Bottom-5 estimates are deliberately downward-biased.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    first_prediction = pd.read_csv(root / "seed_001" / "ml_predictions_test.csv")
    keys = ["sample", "split", "event_10y", "observed_time_years"]
    ensemble = first_prediction[keys].copy()
    duration = ensemble["observed_time_years"].to_numpy(float)
    event = ensemble["event_10y"].to_numpy(int)
    horizon = config.primary_horizon
    performance_rows = []
    ci_frames = []
    dca_frames = []
    thresholds = np.linspace(
        float(config.raw["evaluation"]["dca_threshold_min"]),
        float(config.raw["evaluation"]["dca_threshold_max"]),
        int(config.raw["evaluation"]["dca_threshold_points"]),
    )
    for position, (model, selected_seeds) in enumerate(selected_by_model.items(), start=1):
        column = f"{model}_risk_{horizon}y"
        risks = []
        for seed in selected_seeds:
            prediction = pd.read_csv(root / f"seed_{seed:03d}" / "ml_predictions_test.csv")
            if not ensemble[keys].equals(prediction[keys]):
                raise RuntimeError(f"Test rows differ for seed {seed}")
            risks.append(prediction[column].to_numpy(float))
        risk = np.mean(np.stack(risks), axis=0)
        ensemble[column] = risk
        result = evaluate_model(duration, event, risk, risk, horizon)
        result.update({"model": model, "selected_seeds": "|".join(map(str, selected_seeds))})
        performance_rows.append(result)
        ci = bootstrap_model_performance(
            duration, event, risk, risk, horizon,
            repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
            confidence_level=float(config.raw["evaluation"]["confidence_level"]),
            seed=config.test_holdout_seed,
        )
        ci.insert(0, "model", model)
        ci_frames.append(ci)
        dca = decision_curve(duration, event, risk, thresholds, horizon)
        dca.insert(0, "model", model)
        dca_frames.append(dca)
        print(f"ML ensemble bootstrap {position}/{len(selected_by_model)} model={model}", flush=True)
    ensemble.to_csv(output / "selected_bottom5_ensemble_predictions_test.csv", index=False)
    pd.DataFrame(performance_rows).to_csv(
        output / "selected_bottom5_ensemble_test_metrics.csv", index=False
    )
    pd.concat(ci_frames, ignore_index=True).to_csv(
        output / "selected_bottom5_ensemble_bootstrap_95ci.csv", index=False
    )
    pd.concat(dca_frames, ignore_index=True).to_csv(
        output / "selected_bottom5_ensemble_dca.csv", index=False
    )
    manifest = {
        "analysis_role": "exploratory",
        "completed_training_seeds": len(config.training_seeds),
        "n_models_per_seed": int(metrics["model"].nunique()),
        "n_fitted_model_results": int(len(metrics)),
        "selection": "lowest 5 test AUC seeds separately for each model",
        "output_directory": str(output),
        "warning": "Bottom-5 test selection is intentionally pessimistic and not an unbiased comparison.",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze traditional ML 100-seed results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.input_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
