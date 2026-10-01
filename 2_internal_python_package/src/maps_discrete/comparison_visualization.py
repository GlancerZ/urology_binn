from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# Matplotlib must have a writable configuration directory on managed Windows hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "maps_mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_SPECS = OrderedDict(
    [
        (
            "teacher",
            {
                "label": "Teacher MAPS",
                "column": "teacher_risk_10y",
                "source": "teacher_student",
                "color": "#173F5F",
                "linestyle": "-",
            },
        ),
        (
            "student_distilled",
            {
                "label": "Student MAPS (distilled)",
                "column": "student_distilled_risk_10y",
                "source": "teacher_student",
                "color": "#E07A1F",
                "linestyle": "-",
            },
        ),
        (
            "student_no_distillation",
            {
                "label": "Student BINN (no distillation)",
                "column": "student_no_distillation_risk_10y",
                "source": "teacher_student",
                "color": "#7A5195",
                "linestyle": "--",
            },
        ),
        (
            "nmr_only__elastic_net",
            {
                "label": "NMR-only Elastic Net",
                "column": "nmr_only__elastic_net_risk_10y",
                "source": "ml",
                "color": "#2A9D8F",
                "linestyle": "-.",
            },
        ),
        (
            "prs_only__logistic",
            {
                "label": "PRS-only Logistic",
                "column": "prs_only__logistic_risk_10y",
                "source": "ml",
                "color": "#9C755F",
                "linestyle": ":",
            },
        ),
        (
            "nmr_prs__logistic",
            {
                "label": "NMR+PRS Logistic",
                "column": "nmr_prs__logistic_risk_10y",
                "source": "ml",
                "color": "#4E79A7",
                "linestyle": "--",
            },
        ),
        (
            "nmr_olink_prs__xgboost",
            {
                "label": "NMR+Olink+PRS XGBoost",
                "column": "nmr_olink_prs__xgboost_risk_10y",
                "source": "ml",
                "color": "#C43C39",
                "linestyle": "-",
            },
        ),
    ]
)

MODALITY_LABELS = OrderedDict(
    [
        ("nmr_only", "NMR-only"),
        ("prs_only", "PRS-only"),
        ("nmr_prs", "NMR+PRS"),
        ("nmr_olink_prs", "NMR+Olink+PRS"),
    ]
)

ALGORITHM_LABELS = OrderedDict(
    [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("random_forest", "Random Forest"),
        ("logistic", "Logistic"),
        ("elastic_net", "Elastic Net"),
        ("svm_rbf", "SVM (RBF)"),
    ]
)

ALGORITHM_COLORS = {
    "xgboost": "#C43C39",
    "lightgbm": "#59A14F",
    "random_forest": "#F28E2B",
    "logistic": "#4E79A7",
    "elastic_net": "#B07AA1",
    "svm_rbf": "#76B7B2",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#AEB7C2",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DDE3E8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _seed_directories(parent: Path) -> list[Path]:
    paths = sorted(parent.glob("seed_[0-9][0-9][0-9]"))
    if len(paths) != 100:
        raise RuntimeError(f"Expected 100 seed directories in {parent}, found {len(paths)}")
    return paths


def _average_seed_predictions(
    parent: Path,
    filename: str,
    prediction_columns: list[str],
) -> pd.DataFrame:
    metadata_columns = ["sample", "split", "event_10y", "observed_time_years"]
    total: np.ndarray | None = None
    reference: pd.DataFrame | None = None
    for seed_dir in _seed_directories(parent):
        path = seed_dir / filename
        frame = pd.read_csv(path, usecols=metadata_columns + prediction_columns)
        if reference is None:
            reference = frame[metadata_columns].copy()
            total = np.zeros((len(frame), len(prediction_columns)), dtype=np.float64)
        else:
            if not np.array_equal(reference["sample"].to_numpy(), frame["sample"].to_numpy()):
                raise RuntimeError(f"Sample order differs in {path}")
            if not np.array_equal(reference["event_10y"].to_numpy(), frame["event_10y"].to_numpy()):
                raise RuntimeError(f"Endpoint differs in {path}")
        assert total is not None
        total += frame[prediction_columns].to_numpy(dtype=np.float64)
    assert reference is not None and total is not None
    reference[prediction_columns] = total / 100.0
    return reference


def build_symmetric_all100_predictions(study_root: Path) -> pd.DataFrame:
    ts_parent = study_root / "5_teacher_student_multiseed"
    ml_parent = study_root / "6_traditional_ml_multiseed"
    ts_columns = [spec["column"] for spec in MODEL_SPECS.values() if spec["source"] == "teacher_student"]
    ml_columns = [spec["column"] for spec in MODEL_SPECS.values() if spec["source"] == "ml"]
    ts = _average_seed_predictions(ts_parent, "predictions_test.csv", ts_columns)
    ml = _average_seed_predictions(ml_parent, "ml_predictions_test.csv", ml_columns)
    merged = ts.merge(
        ml[["sample"] + ml_columns],
        on="sample",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(ts):
        raise RuntimeError("Teacher-student and ML test cohorts do not align")
    return merged


def load_exploratory_selected_predictions(study_root: Path) -> pd.DataFrame:
    ts_path = (
        study_root
        / "5_teacher_student_multiseed"
        / "101_multiseed_analysis"
        / "selected_top5_ensemble_predictions_test.csv"
    )
    ml_path = (
        study_root
        / "6_traditional_ml_multiseed"
        / "101_multiseed_analysis"
        / "selected_bottom5_ensemble_predictions_test.csv"
    )
    ts = pd.read_csv(ts_path)
    ml = pd.read_csv(ml_path)
    ml_columns = [spec["column"] for spec in MODEL_SPECS.values() if spec["source"] == "ml"]
    merged = ts.merge(ml[["sample"] + ml_columns], on="sample", how="inner", validate="one_to_one")
    if len(merged) != len(ts):
        raise RuntimeError("Selected teacher-student and ML test cohorts do not align")
    return merged


def _bootstrap_auc_pr(
    y: np.ndarray,
    predictions: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = 20260919,
) -> pd.DataFrame:
    scores = {model: predictions[spec["column"]].to_numpy(dtype=float) for model, spec in MODEL_SPECS.items()}
    estimates = {
        model: {
            "auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
        }
        for model, score in scores.items()
    }
    draws = {
        model: {"auc": [], "pr_auc": []}
        for model in MODEL_SPECS
    }
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = rng.integers(0, len(y), size=len(y))
        y_boot = y[index]
        if len(np.unique(y_boot)) < 2:
            continue
        for model, score in scores.items():
            boot_score = score[index]
            draws[model]["auc"].append(float(roc_auc_score(y_boot, boot_score)))
            draws[model]["pr_auc"].append(float(average_precision_score(y_boot, boot_score)))
    rows: list[dict[str, object]] = []
    for model, spec in MODEL_SPECS.items():
        for metric in ("auc", "pr_auc"):
            values = np.asarray(draws[model][metric], dtype=float)
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "label": spec["label"],
                    "metric": metric,
                    "estimate": estimates[model][metric],
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "valid_bootstrap_repetitions": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _curve_tables(predictions: pd.DataFrame, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = predictions["event_10y"].to_numpy(dtype=int)
    prevalence = float(y.mean())
    curve_rows: list[dict[str, object]] = []
    dca_rows: list[dict[str, object]] = []
    thresholds = np.linspace(0.005, 0.15, 120)
    for model, spec in MODEL_SPECS.items():
        score = predictions[spec["column"]].to_numpy(dtype=float)
        auc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        curve_rows.extend(
            {
                "analysis": analysis,
                "curve": "roc",
                "model": model,
                "label": spec["label"],
                "x": float(x),
                "y": float(value),
                "metric_value": auc,
            }
            for x, value in zip(fpr, tpr)
        )
        curve_rows.extend(
            {
                "analysis": analysis,
                "curve": "pr",
                "model": model,
                "label": spec["label"],
                "x": float(x),
                "y": float(value),
                "metric_value": ap,
            }
            for x, value in zip(recall, precision)
        )
        for threshold in thresholds:
            positive = score >= threshold
            tp = int(np.sum(positive & (y == 1)))
            fp = int(np.sum(positive & (y == 0)))
            odds = threshold / (1.0 - threshold)
            dca_rows.append(
                {
                    "analysis": analysis,
                    "model": model,
                    "label": spec["label"],
                    "threshold": float(threshold),
                    "net_benefit": float(tp / len(y) - fp / len(y) * odds),
                    "treat_all": float(prevalence - (1.0 - prevalence) * odds),
                    "treat_none": 0.0,
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(dca_rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18, metadata={"Creator": "MAPS discrete research framework"})
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#263238",
        va="top",
    )


def plot_main_comparison(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
    curve_table: pd.DataFrame,
    dca_table: pd.DataFrame,
    title: str,
    subtitle: str,
    footer: str,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.5))
    ax_roc, ax_pr, ax_dca, ax_forest = axes.flat
    prevalence = float(predictions["event_10y"].mean())

    for model, spec in MODEL_SPECS.items():
        roc_data = curve_table[(curve_table["curve"] == "roc") & (curve_table["model"] == model)]
        auc = float(roc_data["metric_value"].iloc[0])
        ax_roc.plot(
            roc_data["x"],
            roc_data["y"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.2,
            label=f"{spec['label']}  ({auc:.3f})",
        )
        pr_data = curve_table[(curve_table["curve"] == "pr") & (curve_table["model"] == model)]
        ap = float(pr_data["metric_value"].iloc[0])
        ax_pr.plot(
            pr_data["x"],
            pr_data["y"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.2,
            label=f"{spec['label']}  ({ap:.3f})",
        )
        dca_data = dca_table[dca_table["model"] == model]
        ax_dca.plot(
            dca_data["threshold"] * 100,
            dca_data["net_benefit"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.0,
            label=spec["label"],
        )

    ax_roc.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", linewidth=1.2, label="Chance")
    ax_roc.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title("10-year ROC curves")
    ax_roc.legend(loc="lower right", fontsize=8.0, title="Model (AUC)", title_fontsize=8.5)
    _panel_label(ax_roc, "A")

    ax_pr.axhline(prevalence, color="#7F8C8D", linestyle="--", linewidth=1.2, label=f"Prevalence ({prevalence:.3f})")
    ax_pr.set(xlabel="Recall (Sensitivity)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set_title("10-year precision-recall curves")
    ax_pr.legend(loc="upper right", fontsize=8.0, title="Model (PR-AUC)", title_fontsize=8.5)
    _panel_label(ax_pr, "B")

    reference = dca_table[dca_table["model"] == next(iter(MODEL_SPECS))]
    ax_dca.plot(reference["threshold"] * 100, reference["treat_all"], color="#5E6A71", linestyle="--", linewidth=1.5, label="Screen all")
    ax_dca.axhline(0, color="#111827", linewidth=1.3, label="Screen none")
    ax_dca.set(
        xlabel="10-year risk threshold (%)",
        ylabel="Net benefit",
        xlim=(0.5, 15),
        ylim=(-0.01, max(0.055, float(dca_table["net_benefit"].max()) * 1.08)),
    )
    ax_dca.set_title("Decision curve analysis")
    ax_dca.legend(loc="upper right", fontsize=8.0, ncol=2)
    ax_dca.text(
        0.01,
        0.02,
        "Display range is clipped below -0.01 for readability.",
        transform=ax_dca.transAxes,
        fontsize=8,
        color="#5E6A71",
        va="bottom",
    )
    _panel_label(ax_dca, "C")

    labels = [spec["label"] for spec in MODEL_SPECS.values()]
    y_positions = np.arange(len(labels))[::-1]
    for row_index, (model, spec) in enumerate(MODEL_SPECS.items()):
        y_pos = y_positions[row_index]
        if row_index % 2 == 0:
            ax_forest.axhspan(y_pos - 0.45, y_pos + 0.45, color="#F5F7F9", zorder=0)
        for metric, marker, offset, color in (
            ("auc", "o", 0.13, "#173F5F"),
            ("pr_auc", "s", -0.13, "#E07A1F"),
        ):
            row = intervals[(intervals["model"] == model) & (intervals["metric"] == metric)].iloc[0]
            estimate = float(row["estimate"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
            ax_forest.errorbar(
                estimate,
                y_pos + offset,
                xerr=np.array([[estimate - lower], [upper - estimate]]),
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=3,
                markersize=6,
                zorder=3,
            )
    ax_forest.set_yticks(y_positions, labels)
    ax_forest.set_xlim(0, 1.01)
    ax_forest.set_xlabel("Estimate (95% participant-bootstrap CI)")
    ax_forest.set_title("AUC and PR-AUC forest plot")
    ax_forest.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#173F5F", linestyle="", label="AUC"),
            Line2D([0], [0], marker="s", color="#E07A1F", linestyle="", label="PR-AUC"),
        ],
        loc="lower right",
        fontsize=9,
    )
    _panel_label(ax_forest, "D")

    fig.suptitle(title, fontsize=20, fontweight="bold", color="#162A3A", y=0.985)
    fig.text(0.5, 0.954, subtitle, ha="center", va="top", fontsize=10.5, color="#52616B")
    fig.text(0.012, 0.012, footer, ha="left", va="bottom", fontsize=8.5, color="#5E6A71")
    fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.925), h_pad=3.2, w_pad=2.8)
    _save_figure(fig, output_stem)


def _combined_seed_metrics(study_root: Path) -> pd.DataFrame:
    ts = pd.read_csv(
        study_root
        / "5_teacher_student_multiseed"
        / "101_multiseed_analysis"
        / "all_100_seed_test_metrics.csv"
    )
    ml = pd.read_csv(
        study_root
        / "6_traditional_ml_multiseed"
        / "101_multiseed_analysis"
        / "all_100_seed_ml_test_metrics.csv"
    )
    frames: list[pd.DataFrame] = []
    for model, spec in MODEL_SPECS.items():
        source = ts if spec["source"] == "teacher_student" else ml
        part = source[source["model"] == model].copy()
        part["label"] = spec["label"]
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True)
    expected = 100 * len(MODEL_SPECS)
    if len(combined) != expected:
        raise RuntimeError(f"Expected {expected} seed-level metric rows, found {len(combined)}")
    return combined


def plot_seed_metric_distributions(metric_data: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.5), sharey=True)
    labels = [spec["label"] for spec in MODEL_SPECS.values()]
    colors = [spec["color"] for spec in MODEL_SPECS.values()]
    positions = np.arange(len(labels))[::-1] + 1
    for ax, metric, title in zip(axes, ("auc", "pr_auc"), ("ROC-AUC across 100 seeds", "PR-AUC across 100 seeds")):
        values = [metric_data.loc[metric_data["model"] == model, metric].to_numpy() for model in MODEL_SPECS]
        boxes = ax.boxplot(
            values,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#172B3A", "markersize": 4.5},
            medianprops={"color": "#172B3A", "linewidth": 1.4},
            whiskerprops={"color": "#6B7780"},
            capprops={"color": "#6B7780"},
            flierprops={"marker": ".", "markersize": 2.5, "markerfacecolor": "#6B7780", "markeredgecolor": "#6B7780", "alpha": 0.45},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor(color)
        ax.set_yticks(positions, labels)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(title.split()[0])
        ax.set_title(title)
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    fig.suptitle("Model performance variability across all training seeds", fontsize=20, fontweight="bold", color="#162A3A", y=0.98)
    fig.text(
        0.5,
        0.938,
        "Each box summarizes 100 independently trained models evaluated on the same fixed test cohort.",
        ha="center",
        color="#52616B",
        fontsize=10.5,
    )
    fig.text(0.015, 0.018, "Diamond = mean; center line = median; boxes = interquartile range.", fontsize=8.5, color="#5E6A71")
    fig.tight_layout(rect=(0.03, 0.045, 0.99, 0.91), w_pad=4.0)
    _save_figure(fig, output_stem)


def _all_ml_curves(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["event_10y"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for modality in MODALITY_LABELS:
        for algorithm in ALGORITHM_LABELS:
            model = f"{modality}__{algorithm}"
            score = predictions[f"{model}_risk_10y"].to_numpy(dtype=float)
            auc = float(roc_auc_score(y, score))
            ap = float(average_precision_score(y, score))
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            rows.extend(
                {"modality": modality, "algorithm": algorithm, "model": model, "curve": "roc", "x": float(x), "y": float(value), "metric_value": auc}
                for x, value in zip(fpr, tpr)
            )
            rows.extend(
                {"modality": modality, "algorithm": algorithm, "model": model, "curve": "pr", "x": float(x), "y": float(value), "metric_value": ap}
                for x, value in zip(recall, precision)
            )
    return pd.DataFrame(rows)


def plot_all_ml_curves(predictions: pd.DataFrame, curve_data: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16.5, 20), sharex="col", sharey="col")
    prevalence = float(predictions["event_10y"].mean())
    for row_index, (modality, modality_label) in enumerate(MODALITY_LABELS.items()):
        ax_roc, ax_pr = axes[row_index]
        for algorithm, algorithm_label in ALGORITHM_LABELS.items():
            color = ALGORITHM_COLORS[algorithm]
            roc_data = curve_data[(curve_data["modality"] == modality) & (curve_data["algorithm"] == algorithm) & (curve_data["curve"] == "roc")]
            pr_data = curve_data[(curve_data["modality"] == modality) & (curve_data["algorithm"] == algorithm) & (curve_data["curve"] == "pr")]
            ax_roc.plot(roc_data["x"], roc_data["y"], color=color, linewidth=2.0, label=f"{algorithm_label} ({float(roc_data['metric_value'].iloc[0]):.3f})")
            ax_pr.plot(pr_data["x"], pr_data["y"], color=color, linewidth=2.0, label=f"{algorithm_label} ({float(pr_data['metric_value'].iloc[0]):.3f})")
        ax_roc.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", linewidth=1.1)
        ax_pr.axhline(prevalence, color="#7F8C8D", linestyle="--", linewidth=1.1)
        ax_roc.set_title(f"{modality_label}: ROC")
        ax_pr.set_title(f"{modality_label}: precision-recall")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.01)
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.01)
        ax_roc.set_ylabel("Sensitivity")
        ax_pr.set_ylabel("Precision")
        ax_roc.legend(loc="lower right", fontsize=8, title="Algorithm (AUC)", title_fontsize=8.5)
        ax_pr.legend(loc="upper right", fontsize=8, title="Algorithm (PR-AUC)", title_fontsize=8.5)
    axes[-1, 0].set_xlabel("1 - Specificity")
    axes[-1, 1].set_xlabel("Recall (Sensitivity)")
    fig.suptitle("Traditional ML performance by modality", fontsize=20, fontweight="bold", color="#162A3A", y=0.995)
    fig.text(
        0.5,
        0.978,
        "Exploratory Bottom-5 ensemble for each modality-algorithm pair; fixed test cohort, n=1,726 (82 cases).",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#52616B",
    )
    fig.text(
        0.015,
        0.008,
        "Bottom-5 seeds were selected separately by each model's test AUC; estimates are intentionally downward-selected and not confirmatory.",
        fontsize=8.5,
        color="#5E6A71",
    )
    fig.tight_layout(rect=(0.02, 0.025, 0.99, 0.96), h_pad=3.0, w_pad=2.5)
    _save_figure(fig, output_stem)


def plot_all_ml_forest(intervals: pd.DataFrame, output_stem: Path) -> None:
    ordered_models = [f"{modality}__{algorithm}" for modality in MODALITY_LABELS for algorithm in ALGORITHM_LABELS]
    labels = [
        f"{MODALITY_LABELS[model.split('__')[0]]} / {ALGORITHM_LABELS[model.split('__')[1]]}"
        for model in ordered_models
    ]
    y_positions = np.arange(len(ordered_models))[::-1]
    fig, ax = plt.subplots(figsize=(14.5, 16.5))
    for index, model in enumerate(ordered_models):
        y_pos = y_positions[index]
        modality_index = index // len(ALGORITHM_LABELS)
        if modality_index % 2 == 0:
            ax.axhspan(y_pos - 0.5, y_pos + 0.5, color="#F4F7F8", zorder=0)
        for metric, marker, offset, color in (
            ("auc", "o", 0.14, "#173F5F"),
            ("pr_auc", "s", -0.14, "#E07A1F"),
        ):
            row = intervals[(intervals["model"] == model) & (intervals["metric"] == metric)].iloc[0]
            estimate = float(row["estimate"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
            ax.errorbar(
                estimate,
                y_pos + offset,
                xerr=np.array([[estimate - lower], [upper - estimate]]),
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=2.5,
                markersize=5.5,
                zorder=3,
            )
    for boundary in (5.5, 11.5, 17.5):
        ax.axhline(boundary, color="#AEB7C2", linewidth=1.1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 1.01)
    ax.set_xlabel("Estimate (95% participant-bootstrap CI)")
    ax.set_title("AUC and PR-AUC for all traditional ML models", fontsize=18, pad=28)
    ax.text(
        0.5,
        1.012,
        "Exploratory Bottom-5 ensemble selected separately for each modality-algorithm pair",
        transform=ax.transAxes,
        ha="center",
        color="#52616B",
        fontsize=10.5,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#173F5F", linestyle="", label="AUC"),
            Line2D([0], [0], marker="s", color="#E07A1F", linestyle="", label="PR-AUC"),
        ],
        loc="lower right",
    )
    fig.text(
        0.015,
        0.012,
        "Bottom-5 test-AUC selection produces downward-biased exploratory estimates; 95% CIs use 2,000 participant-level bootstrap repetitions.",
        fontsize=8.5,
        color="#5E6A71",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.965))
    _save_figure(fig, output_stem)


def _write_readme(output_root: Path, case_count: int, total_count: int) -> None:
    content = f"""# Teacher-Student-Machine Learning visualization package

## Primary interpretation

- `1_main_figures/Figure_1_symmetric_all100_comparison` is the recommended primary visualization. It averages patient-level predictions across all 100 training seeds for every displayed model, so all model families use a symmetric aggregation rule.
- `1_main_figures/Figure_2_exploratory_top5_vs_bottom5` reproduces the prespecified exploratory selection scheme: Teacher/Students use the five Teacher seeds with the highest test AUC, while every ML model uses its own five seeds with the lowest test AUC. This contrast is intentionally biased in opposite directions and must not be used as confirmatory evidence of superiority.
- `1_main_figures/Figure_3_all100_seed_metric_distributions` shows seed-to-seed variability on the fixed test cohort.
- `2_supplementary_ml_by_modality/Figure_S1_bottom5_roc_pr_by_modality` shows all 24 ML modality-algorithm combinations.
- `2_supplementary_ml_by_modality/Figure_S2_bottom5_auc_prauc_forest` shows their AUC and PR-AUC with participant-bootstrap 95% CIs.

## Cohort and statistics

- Fixed test cohort: n={total_count:,}; 10-year cases={case_count}; non-cases={total_count - case_count:,}.
- ROC-AUC and PR-AUC are calculated from participant-level 10-year predicted risks.
- Forest-plot confidence intervals use 2,000 participant-level percentile-bootstrap repetitions.
- DCA uses risk thresholds from 0.5% to 15%; the display is clipped below net benefit -0.01 to keep clinically relevant differences legible.
- The PR horizontal reference is the observed 10-year event prevalence.

## Representative ML models in the compact comparison

- NMR-only: Elastic Net.
- PRS-only: Logistic regression.
- NMR+PRS: Logistic regression.
- NMR+Olink+PRS: XGBoost.

PNG files are 300 dpi for preview and insertion into common office documents. PDF files retain vector graphics for publication workflows. Plot-ready CSV files and the generation manifest are stored in `3_plot_data`.
"""
    (output_root / "README_VISUALIZATION.md").write_text(content, encoding="utf-8")


def generate_visualizations(study_root: Path, bootstrap_repetitions: int = 2000) -> Path:
    _set_style()
    output_root = study_root / "10_tables_and_figures" / "1_teacher_student_ml_comparison"
    main_dir = output_root / "1_main_figures"
    supplementary_dir = output_root / "2_supplementary_ml_by_modality"
    data_dir = output_root / "3_plot_data"
    for directory in (main_dir, supplementary_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all100_predictions = build_symmetric_all100_predictions(study_root)
    selected_predictions = load_exploratory_selected_predictions(study_root)
    all100_predictions.to_csv(data_dir / "symmetric_all100_ensemble_predictions_test.csv", index=False)
    selected_predictions.to_csv(data_dir / "exploratory_selected_ensemble_predictions_test.csv", index=False)

    all100_intervals = _bootstrap_auc_pr(
        all100_predictions["event_10y"].to_numpy(dtype=int),
        all100_predictions,
        repetitions=bootstrap_repetitions,
        seed=20260919,
    )
    selected_intervals = _bootstrap_auc_pr(
        selected_predictions["event_10y"].to_numpy(dtype=int),
        selected_predictions,
        repetitions=bootstrap_repetitions,
        seed=20260920,
    )
    all100_intervals.to_csv(data_dir / "symmetric_all100_auc_prauc_bootstrap_95ci.csv", index=False)
    selected_intervals.to_csv(data_dir / "exploratory_selected_auc_prauc_bootstrap_95ci.csv", index=False)

    all100_curves, all100_dca = _curve_tables(all100_predictions, "symmetric_all100")
    selected_curves, selected_dca = _curve_tables(selected_predictions, "exploratory_top5_vs_bottom5")
    pd.concat([all100_curves, selected_curves], ignore_index=True).to_csv(data_dir / "main_roc_pr_plot_data.csv", index=False)
    pd.concat([all100_dca, selected_dca], ignore_index=True).to_csv(data_dir / "main_dca_plot_data.csv", index=False)

    n = len(all100_predictions)
    cases = int(all100_predictions["event_10y"].sum())
    cohort_text = f"Fixed 10-year test cohort: n={n:,}; cases={cases}; non-cases={n - cases}."
    plot_main_comparison(
        all100_predictions,
        all100_intervals,
        all100_curves,
        all100_dca,
        "Teacher, Student and representative ML models",
        "Primary symmetric comparison: patient-level predictions averaged across all 100 training seeds",
        cohort_text + " AUC/PR-AUC CIs: 2,000 participant-bootstrap repetitions.",
        main_dir / "Figure_1_symmetric_all100_comparison",
    )
    plot_main_comparison(
        selected_predictions,
        selected_intervals,
        selected_curves,
        selected_dca,
        "Teacher, Student and representative ML models",
        "Exploratory comparison: Teacher Top-5 seeds versus model-specific ML Bottom-5 seeds",
        cohort_text + " Opposite test-set selection directions make this visualization exploratory and selection-biased.",
        main_dir / "Figure_2_exploratory_top5_vs_bottom5",
    )

    seed_metrics = _combined_seed_metrics(study_root)
    seed_metrics.to_csv(data_dir / "all100_seed_metrics_compact_models.csv", index=False)
    plot_seed_metric_distributions(seed_metrics, main_dir / "Figure_3_all100_seed_metric_distributions")

    ml_analysis_dir = study_root / "6_traditional_ml_multiseed" / "101_multiseed_analysis"
    selected_ml_predictions = pd.read_csv(ml_analysis_dir / "selected_bottom5_ensemble_predictions_test.csv")
    all_ml_curves = _all_ml_curves(selected_ml_predictions)
    all_ml_curves.to_csv(data_dir / "all24_ml_bottom5_roc_pr_plot_data.csv", index=False)
    plot_all_ml_curves(
        selected_ml_predictions,
        all_ml_curves,
        supplementary_dir / "Figure_S1_bottom5_roc_pr_by_modality",
    )
    ml_intervals = pd.read_csv(ml_analysis_dir / "selected_bottom5_ensemble_bootstrap_95ci.csv")
    plot_all_ml_forest(
        ml_intervals,
        supplementary_dir / "Figure_S2_bottom5_auc_prauc_forest",
    )

    specs = [
        {
            "model": model,
            "label": spec["label"],
            "prediction_column": spec["column"],
            "source": spec["source"],
        }
        for model, spec in MODEL_SPECS.items()
    ]
    pd.DataFrame(specs).to_csv(data_dir / "compact_model_specifications.csv", index=False)
    manifest = {
        "study_root": str(study_root),
        "test_n": n,
        "test_cases": cases,
        "bootstrap_repetitions": bootstrap_repetitions,
        "primary_analysis": "symmetric patient-level average across all 100 seeds",
        "exploratory_analysis": "Teacher Top-5 test-AUC seeds versus each ML model's own Bottom-5 test-AUC seeds",
        "pdf_outputs": [
            str(main_dir / "Figure_1_symmetric_all100_comparison.pdf"),
            str(main_dir / "Figure_2_exploratory_top5_vs_bottom5.pdf"),
            str(main_dir / "Figure_3_all100_seed_metric_distributions.pdf"),
            str(supplementary_dir / "Figure_S1_bottom5_roc_pr_by_modality.pdf"),
            str(supplementary_dir / "Figure_S2_bottom5_auc_prauc_forest.pdf"),
        ],
    }
    (data_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(output_root, cases, n)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Teacher-Student-ML comparison figures")
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = generate_visualizations(args.study_root.resolve(), args.bootstrap_repetitions)
    print(output)


if __name__ == "__main__":
    main()
