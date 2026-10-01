from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .endpoint import fixed_horizon_status
from .metrics import continuous_nri_idi, decision_curve, evaluate_model


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    label: str
    risk: np.ndarray
    log_risk: np.ndarray | None = None
    color: str | None = None

    def ranking_score(self) -> np.ndarray:
        return np.asarray(self.risk if self.log_risk is None else self.log_risk, dtype=float)


def _quantile_interval(values: Iterable[float], confidence_level: float = 0.95) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), 0
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper), int(len(array))


def calibration_statistics(event: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    y = np.asarray(event, dtype=int)
    p = np.clip(np.asarray(risk, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        fitted = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit, y)
        intercept = float(fitted.intercept_[0])
        slope = float(fitted.coef_[0, 0])
    except Exception:
        intercept = float("nan")
        slope = float("nan")
    observed = float(np.mean(y))
    mean_predicted = float(np.mean(p))
    oe_ratio = observed / mean_predicted if mean_predicted > 0 else float("nan")
    bins = calibration_bins(y, p, n_bins=10)
    ece = float(np.sum(bins["n"] / bins["n"].sum() * np.abs(bins["observed"] - bins["predicted"])))
    return {
        "observed_event_rate": observed,
        "mean_predicted_risk": mean_predicted,
        "observed_expected_ratio": oe_ratio,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
    }


def calibration_bins(event: np.ndarray, risk: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"event": np.asarray(event, dtype=int), "risk": np.asarray(risk, dtype=float)})
    try:
        frame["bin"] = pd.qcut(frame["risk"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    return (
        frame.groupby("bin", observed=True)
        .agg(n=("event", "size"), observed=("event", "mean"), predicted=("risk", "mean"))
        .reset_index(drop=True)
    )


def _point_performance(
    duration: np.ndarray,
    event: np.ndarray,
    prediction: ModelPrediction,
    horizon: float,
) -> dict[str, float]:
    result = evaluate_model(
        duration,
        event,
        prediction.ranking_score(),
        np.asarray(prediction.risk, dtype=float),
        horizon,
    )
    return {key: float(result[key]) for key in ("c_index", "auc", "pr_auc", "brier")}


def _comparison(
    duration: np.ndarray,
    event: np.ndarray,
    reference: ModelPrediction,
    new: ModelPrediction,
    horizon: float,
) -> dict[str, float]:
    ref = _point_performance(duration, event, reference, horizon)
    nxt = _point_performance(duration, event, new, horizon)
    reclassification = continuous_nri_idi(duration, event, reference.risk, new.risk, horizon)
    return {
        "delta_c_index": nxt["c_index"] - ref["c_index"],
        "delta_auc": nxt["auc"] - ref["auc"],
        "delta_pr_auc": nxt["pr_auc"] - ref["pr_auc"],
        "delta_brier": nxt["brier"] - ref["brier"],
        "nri_event": float(reclassification["nri_event"]),
        "nri_nonevent": float(reclassification["nri_nonevent"]),
        "nri": float(reclassification["nri"]),
        "idi": float(reclassification["idi"]),
    }


def _recovery(
    metric: str,
    student: float,
    teacher: float,
    reference: float | None = None,
) -> float:
    """Return direction-aware Student performance retained relative to Teacher.

    With no reference, discrimination metrics use Student / Teacher. Because a
    lower Brier score is better, Brier uses Teacher / Student so that 1.0 still
    means complete retention and values below 1.0 mean worse Student performance.
    A reference can still be supplied for legacy incremental-recovery analyses.
    """
    if reference is None:
        numerator = teacher if metric == "brier" else student
        denominator = student if metric == "brier" else teacher
        if denominator <= 0 or not np.isfinite(denominator):
            return float("nan")
        return float(numerator / denominator)
    if metric == "brier":
        numerator = reference - student
        denominator = reference - teacher
    else:
        numerator = student - reference
        denominator = teacher - reference
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def evaluate_models(
    *,
    samples: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    predictions: list[ModelPrediction],
    output_dir: str | Path,
    reference_model_id: str,
    comparison_reference_by_model: dict[str, str] | None = None,
    recovery_pairs: list[dict[str, str]] | None = None,
    horizon: float = 10.0,
    repetitions: int = 2000,
    bootstrap_seed: int = 2026,
    confidence_level: float = 0.95,
    dca_threshold_min: float = 0.005,
    dca_threshold_max: float = 0.15,
    dca_threshold_points: int = 100,
    title_prefix: str = "",
) -> dict:
    """Create uniform tables and figures for one fixed evaluation cohort."""
    output = Path(output_dir)
    tables = output / "1_tables"
    figures = output / "2_figures"
    plot_data = output / "3_plot_data"
    for directory in (output, tables, figures, plot_data):
        directory.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples).astype(str)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    by_id = {prediction.model_id: prediction for prediction in predictions}
    if reference_model_id not in by_id:
        raise KeyError(f"Reference model {reference_model_id!r} is unavailable")
    if not (len(samples) == len(duration) == len(event)):
        raise ValueError("Samples, duration and event arrays must align")
    for prediction in predictions:
        if len(prediction.risk) != len(event):
            raise ValueError(f"Prediction length mismatch for {prediction.model_id}")

    known, y_all = fixed_horizon_status(duration, event, horizon)
    if not known.all():
        evaluation_index = np.flatnonzero(known)
    else:
        evaluation_index = np.arange(len(event))
    y = y_all[evaluation_index]

    point = {
        prediction.model_id: _point_performance(duration, event, prediction, horizon)
        for prediction in predictions
    }
    performance_draws = {
        prediction.model_id: {metric: [] for metric in ("c_index", "auc", "pr_auc", "brier")}
        for prediction in predictions
    }

    reference_map = {
        prediction.model_id: reference_model_id for prediction in predictions
        if prediction.model_id != reference_model_id
    }
    if comparison_reference_by_model:
        reference_map.update(comparison_reference_by_model)
    comparison_point = {
        model_id: _comparison(duration, event, by_id[reference_id], by_id[model_id], horizon)
        for model_id, reference_id in reference_map.items()
        if model_id != reference_id
    }
    comparison_draws = {
        model_id: {metric: [] for metric in metrics}
        for model_id, metrics in comparison_point.items()
    }

    recovery_pairs = recovery_pairs or []
    recovery_point: dict[str, dict[str, float]] = {}
    recovery_draws: dict[str, dict[str, list[float]]] = {}
    for pair in recovery_pairs:
        name = pair["name"]
        teacher = point[pair["teacher"]]
        student = point[pair["student"]]
        reference_id = pair.get("reference")
        reference = point[reference_id] if reference_id else None
        recovery_point[name] = {
            metric: _recovery(
                metric,
                student[metric],
                teacher[metric],
                reference[metric] if reference is not None else None,
            )
            for metric in ("c_index", "auc", "pr_auc", "brier")
        }
        recovery_draws[name] = {metric: [] for metric in recovery_point[name]}

    rng = np.random.default_rng(bootstrap_seed)
    valid_repetitions = 0
    for _ in range(repetitions):
        index = rng.integers(0, len(event), size=len(event))
        try:
            draw_point = {
                prediction.model_id: _point_performance(
                    duration[index],
                    event[index],
                    ModelPrediction(
                        prediction.model_id,
                        prediction.label,
                        np.asarray(prediction.risk)[index],
                        prediction.ranking_score()[index],
                        prediction.color,
                    ),
                    horizon,
                )
                for prediction in predictions
            }
        except (ValueError, ZeroDivisionError):
            continue
        valid_repetitions += 1
        for model_id, metrics in draw_point.items():
            for metric, value in metrics.items():
                performance_draws[model_id][metric].append(value)
        for model_id, reference_id in reference_map.items():
            if model_id == reference_id:
                continue
            try:
                comparison = _comparison(
                    duration[index],
                    event[index],
                    ModelPrediction(
                        reference_id,
                        by_id[reference_id].label,
                        np.asarray(by_id[reference_id].risk)[index],
                        by_id[reference_id].ranking_score()[index],
                    ),
                    ModelPrediction(
                        model_id,
                        by_id[model_id].label,
                        np.asarray(by_id[model_id].risk)[index],
                        by_id[model_id].ranking_score()[index],
                    ),
                    horizon,
                )
            except (ValueError, ZeroDivisionError):
                continue
            for metric, value in comparison.items():
                comparison_draws[model_id][metric].append(value)
        for pair in recovery_pairs:
            name = pair["name"]
            reference_id = pair.get("reference")
            for metric in recovery_point[name]:
                recovery_draws[name][metric].append(_recovery(
                    metric,
                    draw_point[pair["student"]][metric],
                    draw_point[pair["teacher"]][metric],
                    draw_point[reference_id][metric] if reference_id else None,
                ))

    performance_rows = []
    for prediction in predictions:
        row: dict[str, float | str | int] = {
            "model_id": prediction.model_id,
            "model": prediction.label,
            "n": len(event),
            "events": int(event.sum()),
        }
        for metric, estimate in point[prediction.model_id].items():
            lower, upper, valid = _quantile_interval(
                performance_draws[prediction.model_id][metric], confidence_level
            )
            row[metric] = estimate
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
            row[f"{metric}_valid_bootstrap"] = valid
        performance_rows.append(row)
    performance_table = pd.DataFrame(performance_rows)
    performance_table.to_csv(tables / "Table_1_model_performance_95ci.csv", index=False)

    comparison_rows = []
    for model_id, estimates in comparison_point.items():
        reference_id = reference_map[model_id]
        row = {
            "model_id": model_id,
            "model": by_id[model_id].label,
            "reference_model_id": reference_id,
            "reference_model": by_id[reference_id].label,
        }
        for metric, estimate in estimates.items():
            lower, upper, valid = _quantile_interval(
                comparison_draws[model_id][metric], confidence_level
            )
            row[metric] = estimate
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
            row[f"{metric}_valid_bootstrap"] = valid
        comparison_rows.append(row)
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(tables / "Table_2_model_differences_nri_idi_95ci.csv", index=False)

    calibration_rows = []
    calibration_curve_rows = []
    for prediction in predictions:
        stats = calibration_statistics(y, np.asarray(prediction.risk)[evaluation_index])
        calibration_rows.append({"model_id": prediction.model_id, "model": prediction.label, **stats})
        bins = calibration_bins(y, np.asarray(prediction.risk)[evaluation_index], n_bins=10)
        bins.insert(0, "model", prediction.label)
        bins.insert(0, "model_id", prediction.model_id)
        calibration_curve_rows.append(bins)
    calibration_table = pd.DataFrame(calibration_rows)
    calibration_table.to_csv(tables / "Table_3_calibration_metrics.csv", index=False)
    calibration_curve_table = pd.concat(calibration_curve_rows, ignore_index=True)
    calibration_curve_table.to_csv(plot_data / "calibration_curve_data.csv", index=False)

    recovery_rows = []
    for pair in recovery_pairs:
        name = pair["name"]
        reference_id = pair.get("reference")
        comparison_basis = (
            "reference_adjusted_incremental_recovery"
            if reference_id
            else "direct_student_teacher_ratio"
        )
        for metric, estimate in recovery_point[name].items():
            lower, upper, valid = _quantile_interval(
                recovery_draws[name][metric], confidence_level
            )
            recovery_rows.append({
                "recovery_definition": name,
                "comparison_basis": comparison_basis,
                "teacher_model": by_id[pair["teacher"]].label,
                "student_model": by_id[pair["student"]].label,
                "reference_model": by_id[reference_id].label if reference_id else "",
                "metric": metric,
                "ratio_formula": (
                    "teacher_brier/student_brier"
                    if metric == "brier" and not reference_id
                    else "student_metric/teacher_metric"
                    if not reference_id
                    else "reference_adjusted"
                ),
                "recovery_proportion": estimate,
                "recovery_percent": estimate * 100.0,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_lower_percent": lower * 100.0,
                "ci_upper_percent": upper * 100.0,
                "valid_bootstrap_repetitions": valid,
            })
    recovery_table = pd.DataFrame(recovery_rows)
    if len(recovery_table):
        recovery_table.to_csv(tables / "Table_4_student_performance_recovery_95ci.csv", index=False)

    prediction_frame = pd.DataFrame({
        "sample": samples,
        "observed_time_years": duration,
        "event_10y": event,
    })
    for prediction in predictions:
        prediction_frame[f"{prediction.model_id}_risk_10y"] = prediction.risk
        prediction_frame[f"{prediction.model_id}_ranking_score"] = prediction.ranking_score()
    prediction_frame.to_csv(plot_data / "aligned_predictions.csv", index=False)

    thresholds = np.linspace(dca_threshold_min, dca_threshold_max, dca_threshold_points)
    dca_frames = []
    for prediction in predictions:
        frame = decision_curve(duration, event, prediction.risk, thresholds, horizon)
        frame.insert(0, "model", prediction.label)
        frame.insert(0, "model_id", prediction.model_id)
        dca_frames.append(frame)
    dca_table = pd.concat(dca_frames, ignore_index=True)
    dca_table.to_csv(plot_data / "dca_curve_data.csv", index=False)

    _plot_roc_pr(figures, title_prefix, predictions, y, evaluation_index, performance_table)
    _plot_dca(figures, title_prefix, predictions, dca_table)
    _plot_difference_forest(figures, title_prefix, comparison_table)
    _plot_auc_pr_forest(figures, title_prefix, performance_table)
    _plot_calibration(figures, title_prefix, predictions, calibration_curve_table)

    manifest = {
        "title_prefix": title_prefix,
        "n": int(len(event)),
        "events": int(event.sum()),
        "known_10y_n": int(known.sum()),
        "horizon_years": horizon,
        "bootstrap_repetitions_requested": repetitions,
        "bootstrap_repetitions_valid_for_performance": valid_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "dca_threshold_min": dca_threshold_min,
        "dca_threshold_max": dca_threshold_max,
        "reference_model_id": reference_model_id,
        "student_performance_recovery": [
            {
                "name": pair["name"],
                "student": pair["student"],
                "teacher": pair["teacher"],
                "reference": pair.get("reference"),
                "basis": (
                    "reference_adjusted_incremental_recovery"
                    if pair.get("reference")
                    else "direct_student_teacher_ratio"
                ),
            }
            for pair in recovery_pairs
        ],
        "model_ids": [prediction.model_id for prediction in predictions],
        "selection_note": "Internal selected-seed results are exploratory when test10 was used for seed selection.",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "performance": performance_table,
        "comparison": comparison_table,
        "calibration": calibration_table,
        "recovery": recovery_table,
        "manifest": manifest,
    }


def _colors(predictions: list[ModelPrediction]) -> dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {
        prediction.model_id: prediction.color or palette(index % 10)
        for index, prediction in enumerate(predictions)
    }


def _plot_roc_pr(
    figures: Path,
    title_prefix: str,
    predictions: list[ModelPrediction],
    y: np.ndarray,
    evaluation_index: np.ndarray,
    performance: pd.DataFrame,
) -> None:
    colors = _colors(predictions)
    metrics = performance.set_index("model_id")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    for prediction in predictions:
        score = np.asarray(prediction.risk)[evaluation_index]
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, lw=2, color=colors[prediction.model_id], label=f"{prediction.label} ({metrics.loc[prediction.model_id, 'auc']:.3f})")
        axes[1].plot(recall, precision, lw=2, color=colors[prediction.model_id], label=f"{prediction.label} ({metrics.loc[prediction.model_id, 'pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    axes[1].axhline(float(np.mean(y)), ls="--", color="#777777", lw=1, label=f"Prevalence ({np.mean(y):.3f})")
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="10-year ROC curves", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel="Recall (Sensitivity)", ylabel="Precision", title="10-year precision-recall curves", xlim=(0, 1), ylim=(0, 1.01))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
            ncol=1,
            borderaxespad=0,
        )
    fig.suptitle(f"{title_prefix}: discrimination curves".strip(": "), fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.28, 1, 0.96))
    fig.savefig(figures / "Figure_1_ROC_PR_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "Figure_1_ROC_PR_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_dca(figures: Path, title_prefix: str, predictions: list[ModelPrediction], dca: pd.DataFrame) -> None:
    colors = _colors(predictions)
    fig, axis = plt.subplots(figsize=(13, 6.5))
    for prediction in predictions:
        frame = dca.loc[dca["model_id"].eq(prediction.model_id)]
        axis.plot(frame["threshold"] * 100, frame["model_net_benefit"], lw=2, color=colors[prediction.model_id], label=prediction.label)
    first = dca.loc[dca["model_id"].eq(predictions[0].model_id)]
    axis.plot(first["threshold"] * 100, first["treat_all_net_benefit"], "--", color="#555555", label="Screen all")
    axis.axhline(0, color="#111111", lw=1, label="Screen none")
    axis.set(xlabel="10-year risk threshold (%)", ylabel="Net benefit", title=f"{title_prefix}: decision curve analysis".strip(": "))
    axis.set_xlim(float(first["threshold"].min() * 100), float(first["threshold"].max() * 100))
    axis.grid(alpha=0.25)
    axis.legend(
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        ncol=1,
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0, 0, 0.68, 1))
    fig.savefig(figures / "Figure_2_DCA.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "Figure_2_DCA.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_difference_forest(figures: Path, title_prefix: str, table: pd.DataFrame) -> None:
    metrics = [
        ("delta_c_index", "Δ C-index"),
        ("delta_auc", "Δ 10-year AUC"),
        ("nri", "Continuous NRI"),
        ("idi", "IDI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, max(8, 0.55 * len(table) + 5)), sharey=True)
    y = np.arange(len(table))
    labels = [f"{row.model}\nvs {row.reference_model}" for row in table.itertuples()]
    for plot_index, (axis, (metric, title)) in enumerate(zip(axes.flat, metrics)):
        estimate = table[metric].to_numpy(float)
        lower = table[f"{metric}_ci_lower"].to_numpy(float)
        upper = table[f"{metric}_ci_upper"].to_numpy(float)
        axis.errorbar(estimate, y, xerr=[estimate - lower, upper - estimate], fmt="o", color="#173F5F", ecolor="#5E6B75", capsize=3)
        axis.axvline(0, color="#C43C39", ls="--", lw=1)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(y)
        if plot_index % 2 == 0:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
    fig.suptitle(f"{title_prefix}: paired model differences (95% CI)".strip(": "), fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "Figure_3_model_difference_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "Figure_3_model_difference_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_auc_pr_forest(figures: Path, title_prefix: str, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.6 * len(table) + 2)), sharey=True)
    y = np.arange(len(table))
    for plot_index, (axis, metric, title) in enumerate(
        zip(axes, ("auc", "pr_auc"), ("10-year AUC", "PR-AUC"))
    ):
        estimate = table[metric].to_numpy(float)
        lower = table[f"{metric}_ci_lower"].to_numpy(float)
        upper = table[f"{metric}_ci_upper"].to_numpy(float)
        axis.errorbar(estimate, y, xerr=[estimate - lower, upper - estimate], fmt="o", color="#173F5F", ecolor="#5E6B75", capsize=3)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(y)
        if plot_index == 0:
            axis.set_yticklabels(table["model"])
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
    fig.suptitle(f"{title_prefix}: AUC / PR-AUC forest plot (95% CI)".strip(": "), fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "Figure_4_AUC_PR_AUC_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "Figure_4_AUC_PR_AUC_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(figures: Path, title_prefix: str, predictions: list[ModelPrediction], table: pd.DataFrame) -> None:
    colors = _colors(predictions)
    fig, axis = plt.subplots(figsize=(11.5, 6.5))
    for prediction in predictions:
        frame = table.loc[table["model_id"].eq(prediction.model_id)]
        axis.plot(frame["predicted"], frame["observed"], marker="o", lw=1.8, color=colors[prediction.model_id], label=prediction.label)
    limit = max(0.16, float(table[["predicted", "observed"]].max().max()) * 1.08)
    axis.plot([0, limit], [0, limit], "--", color="#555555", lw=1, label="Ideal")
    axis.set(xlabel="Mean predicted 10-year risk", ylabel="Observed 10-year risk", title=f"{title_prefix}: calibration by risk decile".strip(": "), xlim=(0, limit), ylim=(0, limit))
    axis.grid(alpha=0.25)
    axis.legend(
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        ncol=1,
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0, 0, 0.68, 1))
    fig.savefig(figures / "Figure_5_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "Figure_5_calibration.pdf", bbox_inches="tight")
    plt.close(fig)
