from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


MODEL_LABELS = {
    "teacher_maps": "Teacher-MAPS",
    "student_distilled": "Student-MAPS",
    "student_no_distillation": "Non-distilled Student",
    "nmr_only_lightgbm": "NMR-only LightGBM",
    "prs_only_logistic": "PRS-only logistic",
    "nmr_prs_logistic": "NMR+PRS logistic",
    "full_omics_xgboost": "NMR+Olink+PRS XGBoost",
    "base": "Base",
    "lifestyle": "Lifestyle",
    "lifestyle_klk3": "Lifestyle + BMI + KLK3",
    "lifestyle_klk3_prs": "Lifestyle + BMI + KLK3 + PRS",
    "prs_only": "PRS-only",
}

MODEL_COLORS = {
    "teacher_maps": "#0072B2",
    "student_distilled": "#D55E00",
    "student_no_distillation": "#CC79A7",
    "nmr_only_lightgbm": "#009E73",
    "prs_only_logistic": "#A6761D",
    "nmr_prs_logistic": "#56B4E9",
    "full_omics_xgboost": "#E69F00",
    "base": "#4C78A8",
    "lifestyle": "#009E73",
    "lifestyle_klk3": "#E69F00",
    "lifestyle_klk3_prs": "#7A5195",
    "prs_only": "#C44E52",
}

MODEL_LINESTYLES = {
    "teacher_maps": "-",
    "student_distilled": "-",
    "student_no_distillation": "--",
    "nmr_only_lightgbm": "-.",
    "prs_only_logistic": ":",
    "nmr_prs_logistic": "--",
    "full_omics_xgboost": "-.",
    "base": "-",
    "lifestyle": "--",
    "lifestyle_klk3": "-.",
    "lifestyle_klk3_prs": "-",
    "prs_only": ":",
}

RISK_COLUMNS = {
    "teacher_maps": "teacher_maps_risk_10y",
    "student_distilled": "student_distilled_risk_10y",
    "student_no_distillation": "student_no_distillation_risk_10y",
    "nmr_only_lightgbm": "nmr_only_lightgbm_risk_10y",
    "prs_only_logistic": "prs_only_logistic_risk_10y",
    "nmr_prs_logistic": "nmr_prs_logistic_risk_10y",
    "full_omics_xgboost": "full_omics_xgboost_risk_10y",
    "base": "base_risk_10y",
    "lifestyle": "lifestyle_risk_10y",
    "lifestyle_klk3": "lifestyle_klk3_risk_10y",
    "lifestyle_klk3_prs": "lifestyle_klk3_prs_risk_10y",
    "prs_only": "prs_only_risk_10y",
}


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.set_facecolor("white")
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.tick_params(colors="#333333")


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.08,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )


def save_square(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(8.0, 8.0, forward=True)
    fig.savefig(figures / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def ordered_model_ids(performance: pd.DataFrame) -> list[str]:
    return [str(value) for value in performance["model_id"].tolist()]


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id.replace("_", " ").title())


def model_color(model_id: str, index: int = 0) -> str:
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    return MODEL_COLORS.get(model_id, fallback[index % len(fallback)])


def _plot_roc_pr(
    output: Path,
    title: str,
    predictions: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    y = predictions["event_10y"].to_numpy(int)
    ids = ordered_model_ids(performance)
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for index, model_id in enumerate(ids):
        column = RISK_COLUMNS[model_id]
        score = predictions[column].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        style = MODEL_LINESTYLES.get(model_id, "-")
        color = model_color(model_id, index)
        line = axes[0].plot(fpr, tpr, lw=1.9, color=color, ls=style)[0]
        axes[1].plot(recall, precision, lw=1.9, color=color, ls=style)
        handles.append(line)
        labels.append(model_label(model_id))

    axes[0].plot([0, 1], [0, 1], ls="--", color="#777777", lw=1.0)
    axes[1].axhline(float(y.mean()), ls="--", color="#777777", lw=1.0)
    axes[0].set(
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        title="10-year ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[1].set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
        title="10-year precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    for axis, label in zip(axes, ("A", "B")):
        clean_axis(axis)
        axis.set_aspect("equal", adjustable="box")
        panel_label(axis, label)
    fig.suptitle(title, y=0.955, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        frameon=False,
        ncol=2,
        handlelength=3.0,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.30, wspace=0.32)
    save_square(fig, output, "Figure_1_ROC_PR_curves")


def _plot_dca(output: Path, title: str, performance: pd.DataFrame, dca: pd.DataFrame) -> None:
    ids = ordered_model_ids(performance)
    fig, axis = plt.subplots(figsize=(8, 8))
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for index, model_id in enumerate(ids):
        frame = dca.loc[dca["model_id"].eq(model_id)]
        line = axis.plot(
            frame["threshold"] * 100,
            frame["model_net_benefit"],
            lw=1.9,
            color=model_color(model_id, index),
            ls=MODEL_LINESTYLES.get(model_id, "-"),
        )[0]
        handles.append(line)
        labels.append(model_label(model_id))
    first = dca.loc[dca["model_id"].eq(ids[0])]
    handles.append(
        axis.plot(
            first["threshold"] * 100,
            first["treat_all_net_benefit"],
            ls="--",
            color="#666666",
            lw=1.2,
        )[0]
    )
    labels.append("Screen all")
    handles.append(axis.axhline(0, color="#111111", lw=1.0))
    labels.append("Screen none")
    axis.set(
        xlabel="10-year risk threshold (%)",
        ylabel="Net benefit",
        title=title,
        xlim=(float(first["threshold"].min() * 100), float(first["threshold"].max() * 100)),
    )
    clean_axis(axis)
    axis.set_box_aspect(1)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        ncol=3,
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.26)
    save_square(fig, output, "Figure_2_DCA")


def _plot_difference_forest(output: Path, title: str, table: pd.DataFrame) -> None:
    metrics = [
        ("delta_c_index", "Difference in C-index"),
        ("delta_auc", "Difference in 10-year AUC"),
        ("nri", "Continuous NRI"),
        ("idi", "IDI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    y = np.arange(len(table))
    labels = [
        f"{model_label(str(row.model_id))} vs\n{model_label(str(row.reference_model_id))}"
        for row in table.itertuples()
    ]
    for index, (axis, (metric, heading)) in enumerate(zip(axes.flat, metrics)):
        estimate = table[metric].to_numpy(float)
        lower = table[f"{metric}_ci_lower"].to_numpy(float)
        upper = table[f"{metric}_ci_upper"].to_numpy(float)
        axis.errorbar(
            estimate,
            y,
            xerr=np.vstack((estimate - lower, upper - estimate)),
            fmt="o",
            markersize=4.8,
            color="#0072B2",
            ecolor="#555555",
            elinewidth=1.1,
            capsize=2.5,
        )
        axis.axvline(0, color="#B2182B", ls="--", lw=1.0)
        axis.set_title(heading)
        axis.set_yticks(y)
        if index % 2 == 0:
            axis.set_yticklabels(labels, fontsize=7.2)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        clean_axis(axis)
        panel_label(axis, chr(ord("A") + index))
    fig.suptitle(title, y=0.975, fontweight="bold")
    fig.supxlabel("Estimate (95% participant-bootstrap CI)", y=0.035, fontsize=9.5)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.90, bottom=0.10, hspace=0.28, wspace=0.22)
    save_square(fig, output, "Figure_3_model_difference_forest")


def _plot_comparison2_primary_forest(
    output: Path, performance: pd.DataFrame, comparisons: pd.DataFrame
) -> None:
    """Absolute C-index/AUC; only NRI/IDI have model references."""
    panels = [
        (performance, "c_index", "C-index", False),
        (performance, "auc", "10-year AUC", False),
        (comparisons, "nri", "Continuous NRI", True),
        (comparisons, "idi", "IDI", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=False)
    for index, (axis, (table, metric, heading, paired)) in enumerate(zip(axes.flat, panels)):
        estimate = table[metric].to_numpy(float)
        lower = table[f"{metric}_ci_lower"].to_numpy(float)
        upper = table[f"{metric}_ci_upper"].to_numpy(float)
        y = np.arange(len(table))
        axis.errorbar(
            estimate, y, xerr=np.vstack((estimate - lower, upper - estimate)),
            fmt="o", markersize=4.8, color="#0072B2", ecolor="#555555",
            elinewidth=1.1, capsize=2.5,
        )
        if paired:
            axis.axvline(0, color="#B2182B", ls="--", lw=1.0)
        axis.set_title(heading)
        axis.set_yticks(y)
        if index % 2 == 0:
            if paired:
                labels = [
                    f"{model_label(str(row.model_id))} vs\n{model_label(str(row.reference_model_id))}"
                    for row in table.itertuples()
                ]
            else:
                labels = [model_label(str(mid)) for mid in table["model_id"]]
            axis.set_yticklabels(labels, fontsize=6.5 if paired else 7.7)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        clean_axis(axis)
        panel_label(axis, chr(ord("A") + index))
    fig.suptitle("Model comparison 2: performance and reclassification", y=0.975, fontweight="bold")
    fig.supxlabel(
        "Absolute performance (top); NRI/IDI vs Base (bottom); 95% participant-bootstrap CI",
        y=0.025, fontsize=8.1,
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.90, bottom=0.10, hspace=0.29, wspace=0.24)
    save_square(fig, output, "Figure_3_model_difference_forest")


def _plot_auc_pr_forest(output: Path, title: str, performance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    y = np.arange(len(performance))
    labels = [model_label(str(value)) for value in performance["model_id"]]
    for index, (axis, metric, heading) in enumerate(
        zip(axes, ("auc", "pr_auc"), ("10-year AUC", "PR-AUC"))
    ):
        estimate = performance[metric].to_numpy(float)
        lower = performance[f"{metric}_ci_lower"].to_numpy(float)
        upper = performance[f"{metric}_ci_upper"].to_numpy(float)
        colors = [model_color(str(mid), i) for i, mid in enumerate(performance["model_id"])]
        for row, est, lo, hi, color in zip(y, estimate, lower, upper, colors):
            axis.errorbar(
                est,
                row,
                xerr=np.array([[est - lo], [hi - est]]),
                fmt="o",
                markersize=5.2,
                color=color,
                ecolor="#555555",
                elinewidth=1.1,
                capsize=2.5,
            )
        axis.set_title(heading)
        axis.set_yticks(y)
        if index == 0:
            axis.set_yticklabels(labels, fontsize=8.0)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        clean_axis(axis)
        panel_label(axis, chr(ord("A") + index))
    fig.suptitle(title, y=0.955, fontweight="bold")
    fig.supxlabel("Estimate (95% participant-bootstrap CI)", y=0.055, fontsize=9.5)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.87, bottom=0.13, wspace=0.20)
    save_square(fig, output, "Figure_4_AUC_PR_AUC_forest")


def _plot_calibration(
    output: Path,
    title: str,
    performance: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    ids = ordered_model_ids(performance)
    fig, axis = plt.subplots(figsize=(8, 8))
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for index, model_id in enumerate(ids):
        frame = calibration.loc[calibration["model_id"].eq(model_id)]
        line = axis.plot(
            frame["predicted"],
            frame["observed"],
            marker="o",
            markersize=3.6,
            lw=1.5,
            color=model_color(model_id, index),
            ls=MODEL_LINESTYLES.get(model_id, "-"),
        )[0]
        handles.append(line)
        labels.append(model_label(model_id))
    limit = max(0.16, float(calibration[["predicted", "observed"]].max().max()) * 1.06)
    handles.append(axis.plot([0, limit], [0, limit], ls="--", color="#666666", lw=1.1)[0])
    labels.append("Ideal")
    axis.set(
        xlabel="Mean predicted 10-year risk",
        ylabel="Observed 10-year risk",
        title=title,
        xlim=(0, limit),
        ylim=(0, limit),
    )
    clean_axis(axis)
    axis.set_aspect("equal", adjustable="box")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        ncol=2,
        handlelength=2.7,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.26)
    save_square(fig, output, "Figure_5_calibration")


def rebuild_internal_analysis(directory: Path, title_prefix: str) -> None:
    figures = directory / "2_figures"
    tables = directory / "1_tables"
    plot_data = directory / "3_plot_data"
    performance = pd.read_csv(tables / "Table_1_model_performance_95ci.csv")
    differences = pd.read_csv(tables / "Table_2_model_differences_nri_idi_95ci.csv")
    predictions = pd.read_csv(plot_data / "aligned_predictions.csv")
    dca = pd.read_csv(plot_data / "dca_curve_data.csv")
    calibration = pd.read_csv(plot_data / "calibration_curve_data.csv")
    _plot_roc_pr(figures, f"{title_prefix}: discrimination", predictions, performance)
    _plot_dca(figures, f"{title_prefix}: decision curve analysis", performance, dca)
    if directory.name == "8_model_comparison_2_clinical":
        _plot_comparison2_primary_forest(figures, performance, differences)
    else:
        _plot_difference_forest(figures, f"{title_prefix}: paired model differences", differences)
    _plot_auc_pr_forest(figures, f"{title_prefix}: discrimination estimates", performance)
    _plot_calibration(figures, f"{title_prefix}: calibration by risk decile", performance, calibration)
    manifest = {
        "style": "journal-ready square figures",
        "canvas_inches": [8.0, 8.0],
        "png_dpi": 600,
        "pdf": "vector",
        "gridlines": False,
        "legend_policy": "outside the plotting axes",
        "statistics_recomputed": False,
        "source_tables": [
            "Table_1_model_performance_95ci.csv",
            "Table_2_model_differences_nri_idi_95ci.csv",
        ],
    }
    if directory.name == "8_model_comparison_2_clinical":
        manifest["figure_3_metric_policy"] = (
            "C-index and 10-year AUC are absolute; all non-Base NRI and IDI estimates use Base as reference"
        )
    (directory / "publication_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _single_external_roc(
    output: Path,
    prefix: str,
    title: str,
    y: np.ndarray,
    risk: np.ndarray,
    row: pd.Series,
    color: str,
) -> None:
    fpr, tpr, _ = roc_curve(y, risk)
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.plot(fpr, tpr, color=color, lw=2.2)
    axis.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1.0)
    axis.set(
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.text(
        0.98,
        0.04,
        f"AUC {row['auc']:.3f} (95% CI {row['auc_ci_lower']:.3f}-{row['auc_ci_upper']:.3f})",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    clean_axis(axis)
    axis.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.13)
    save_square(fig, output, f"Figure_{prefix}1_ROC_curve")


def _single_external_pr(
    output: Path,
    prefix: str,
    title: str,
    y: np.ndarray,
    risk: np.ndarray,
    row: pd.Series,
    color: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y, risk)
    prevalence = float(y.mean())
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.plot(recall, precision, color=color, lw=2.2)
    axis.axhline(prevalence, ls="--", color="#777777", lw=1.0)
    axis.set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.text(
        0.98,
        0.96,
        f"PR-AUC {row['pr_auc']:.3f} (95% CI {row['pr_auc_ci_lower']:.3f}-{row['pr_auc_ci_upper']:.3f})\n"
        f"Event prevalence {prevalence:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    clean_axis(axis)
    axis.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.13)
    save_square(fig, output, f"Figure_{prefix}2_PR_curve")


def _single_external_dca(
    output: Path,
    prefix: str,
    title: str,
    dca: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 8))
    handles = [
        axis.plot(dca["threshold"] * 100, dca["model_net_benefit"], color=color, lw=2.2)[0],
        axis.plot(
            dca["threshold"] * 100,
            dca["treat_all_net_benefit"],
            ls="--",
            color="#777777",
            lw=1.2,
        )[0],
        axis.axhline(0, color="#111111", lw=1.0),
    ]
    axis.set(
        xlabel="10-year risk threshold (%)",
        ylabel="Net benefit",
        title=title,
        xlim=(float(dca["threshold"].min() * 100), float(dca["threshold"].max() * 100)),
    )
    clean_axis(axis)
    axis.set_box_aspect(1)
    fig.legend(
        handles,
        [label, "Screen all", "Screen none"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=3,
        frameon=False,
        handlelength=2.7,
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.22)
    save_square(fig, output, f"Figure_{prefix}3_DCA")


def _single_external_forest(
    output: Path,
    prefix: str,
    title: str,
    row: pd.Series,
    color: str,
) -> None:
    metrics = [("c_index", "C-index"), ("auc", "10-year AUC"), ("pr_auc", "PR-AUC")]
    estimates = np.array([float(row[key]) for key, _ in metrics])
    lower = np.array([float(row[f"{key}_ci_lower"]) for key, _ in metrics])
    upper = np.array([float(row[f"{key}_ci_upper"]) for key, _ in metrics])
    y = np.arange(len(metrics))
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        markersize=6,
        color=color,
        ecolor="#555555",
        elinewidth=1.3,
        capsize=3.5,
    )
    for y_value, estimate, lo, hi in zip(y, estimates, lower, upper):
        axis.text(hi + 0.018, y_value, f"{estimate:.3f} ({lo:.3f}-{hi:.3f})", va="center", fontsize=8.5)
    axis.set_yticks(y, [label for _, label in metrics])
    axis.invert_yaxis()
    xmin = max(0.0, float(lower.min()) - 0.08)
    xmax = min(1.25, float(upper.max()) + 0.27)
    axis.set_xlim(xmin, xmax)
    axis.set_xlabel("Estimate (95% participant-bootstrap CI)")
    axis.set_title(title)
    clean_axis(axis)
    axis.set_box_aspect(1)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.14)
    save_square(fig, output, f"Figure_{prefix}4_performance_forest")


def _single_external_calibration(
    output: Path,
    prefix: str,
    title: str,
    calibration: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    limit = max(0.16, float(calibration[["predicted", "observed"]].max().max()) * 1.06)
    fig, axis = plt.subplots(figsize=(8, 8))
    handles = [
        axis.plot(
            calibration["predicted"],
            calibration["observed"],
            marker="o",
            markersize=4.5,
            lw=1.8,
            color=color,
        )[0],
        axis.plot([0, limit], [0, limit], ls="--", color="#777777", lw=1.1)[0],
    ]
    axis.set(
        xlabel="Mean predicted 10-year risk",
        ylabel="Observed 10-year risk",
        title=title,
        xlim=(0, limit),
        ylim=(0, limit),
    )
    clean_axis(axis)
    axis.set_aspect("equal", adjustable="box")
    fig.legend(
        handles,
        [label, "Ideal"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
        handlelength=2.7,
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.22)
    save_square(fig, output, f"Figure_{prefix}5_calibration")


def rebuild_wales_teacher_student(study_root: Path) -> None:
    base = study_root / "9_wales_external_validation" / "3_teacher_student_only"
    performance = pd.read_csv(base / "1_tables" / "Table_1_Teacher_Student_performance_95ci.csv")
    specs = [
        ("teacher_maps", "Teacher-MAPS", "T", "2_teacher_maps", "#0072B2"),
        ("student_distilled", "Student-MAPS", "S", "3_student_maps", "#D55E00"),
    ]
    for model_id, label, prefix, folder, color in specs:
        model_root = base / folder
        output = model_root / "2_figures"
        pred = pd.read_csv(model_root / "3_plot_data" / "external_predictions.csv")
        dca = pd.read_csv(model_root / "3_plot_data" / "dca_curve_data.csv")
        calibration = pd.read_csv(model_root / "3_plot_data" / "calibration_curve_data.csv")
        row = performance.loc[performance["model_id"].eq(model_id)].iloc[0]
        risk_column = next(column for column in pred.columns if column.endswith("_risk_10y"))
        y = pred["event_10y"].to_numpy(int)
        risk = pred[risk_column].to_numpy(float)
        heading = f"Wales external validation: {label}"
        _single_external_roc(output, prefix, f"{heading} ROC curve", y, risk, row, color)
        _single_external_pr(output, prefix, f"{heading} precision-recall curve", y, risk, row, color)
        _single_external_dca(output, prefix, f"{heading} decision curve", dca, label, color)
        _single_external_forest(output, prefix, f"{heading} performance", row, color)
        _single_external_calibration(
            output,
            prefix,
            f"{heading} calibration by risk decile",
            calibration,
            label,
            color,
        )
    manifest = {
        "style": "journal-ready square figures",
        "canvas_inches": [8.0, 8.0],
        "png_dpi": 600,
        "pdf": "vector",
        "gridlines": False,
        "legend_policy": "outside the plotting axes",
        "reporting_scope": ["Teacher-MAPS", "Student-MAPS"],
        "visualization_policy": "Teacher-MAPS and Student-MAPS shown separately",
    }
    (base / "publication_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bootstrap_auc(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    case_index = np.flatnonzero(y == 1)
    control_index = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in scores}
    for _ in range(repetitions):
        sampled = np.concatenate(
            (
                rng.choice(case_index, size=len(case_index), replace=True),
                rng.choice(control_index, size=len(control_index), replace=True),
            )
        )
        y_boot = y[sampled]
        for name, score in scores.items():
            values[name].append(float(roc_auc_score(y_boot, score[sampled])))
    point = {name: float(roc_auc_score(y, score)) for name, score in scores.items()}
    interval = {
        name: (float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975)))
        for name, result in values.items()
    }
    return point, interval, repetitions


def build_lag_sensitivity(study_root: Path, repetitions: int = 2000) -> Path:
    source = study_root / "5_teacher_student_multiseed" / "seed_007" / "predictions_test.csv"
    output = study_root / "12_supplementary_lag_sensitivity_seed7"
    tables = output / "1_tables"
    figures = output / "2_figures"
    plot_data = output / "3_plot_data"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plot_data.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(source)
    required = {
        "sample",
        "event_10y",
        "observed_time_years",
        "teacher_risk_10y",
        "student_distilled_risk_10y",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    model_columns = {
        "teacher_maps": "teacher_risk_10y",
        "student_distilled": "student_distilled_risk_10y",
    }
    rows: list[dict[str, object]] = []
    kept_frames: list[pd.DataFrame] = []
    baseline_events = int(data["event_10y"].sum())
    for lag in (0, 1, 2, 3):
        excluded_early_case = data["event_10y"].eq(1) & data["observed_time_years"].le(lag)
        subset = data.loc[~excluded_early_case].reset_index(drop=True)
        y = subset["event_10y"].to_numpy(int)
        scores = {name: subset[column].to_numpy(float) for name, column in model_columns.items()}
        point, interval, valid = bootstrap_auc(y, scores, repetitions, seed=2026 + lag)
        for model_id in model_columns:
            lo, hi = interval[model_id]
            rows.append(
                {
                    "model_id": model_id,
                    "model": model_label(model_id),
                    "lag_years": lag,
                    "n": int(len(subset)),
                    "controls": int((y == 0).sum()),
                    "events_retained": int((y == 1).sum()),
                    "events_excluded": baseline_events - int((y == 1).sum()),
                    "auc": point[model_id],
                    "auc_ci_lower": lo,
                    "auc_ci_upper": hi,
                    "valid_bootstrap": valid,
                }
            )
        keep = subset[["sample", "event_10y", "observed_time_years", *model_columns.values()]].copy()
        keep.insert(0, "lag_years", lag)
        kept_frames.append(keep)

    result = pd.DataFrame(rows)
    result.to_csv(tables / "Table_S_lag_sensitivity_auc_95ci.csv", index=False)
    pd.concat(kept_frames, ignore_index=True).to_csv(
        plot_data / "lag_sensitivity_analysis_sets.csv", index=False
    )

    fig, axis = plt.subplots(figsize=(8, 8))
    markers = {"teacher_maps": "o", "student_distilled": "s"}
    for index, model_id in enumerate(model_columns):
        frame = result.loc[result["model_id"].eq(model_id)].sort_values("lag_years")
        x = frame["lag_years"].to_numpy(float)
        y = frame["auc"].to_numpy(float)
        lower = frame["auc_ci_lower"].to_numpy(float)
        upper = frame["auc_ci_upper"].to_numpy(float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            color=model_color(model_id, index),
            marker=markers[model_id],
            markersize=6,
            lw=2.0,
            elinewidth=1.2,
            capsize=4,
            label=model_label(model_id),
        )
    event_counts = (
        result.loc[result["model_id"].eq("teacher_maps")]
        .sort_values("lag_years")["events_retained"]
        .astype(int)
        .tolist()
    )
    axis.set_xticks([0, 1, 2, 3], [f"{lag}\n({events} events)" for lag, events in zip((0, 1, 2, 3), event_counts)])
    ymin = max(0.5, float(result["auc_ci_lower"].min()) - 0.035)
    ymax = min(1.0, float(result["auc_ci_upper"].max()) + 0.035)
    axis.set(
        xlabel="Lag period: cases diagnosed within the first N years excluded",
        ylabel="10-year ROC-AUC",
        title="Lag sensitivity analysis in the fixed internal test set",
        xlim=(-0.18, 3.18),
        ylim=(ymin, ymax),
    )
    clean_axis(axis)
    axis.set_box_aspect(1)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )
    fig.text(
        0.5,
        0.145,
        "Error bars show stratified participant-bootstrap 95% confidence intervals (2,000 repetitions).",
        ha="center",
        fontsize=8.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.25)
    save_square(fig, figures, "Figure_S_lag_sensitivity_AUC")

    manifest = {
        "models": ["Teacher-MAPS seed 7", "paired Student-MAPS seed 7"],
        "source": str(source),
        "evaluation_split": "fixed test10",
        "lag_definition": "remove incident prostate-cancer cases with observed onset time <= lag; retain all non-cases and later cases",
        "lag_years": [0, 1, 2, 3],
        "auc_definition": "10-year fixed-horizon ROC-AUC among retained participants",
        "bootstrap": {
            "method": "stratified participant bootstrap, sampling cases and controls separately with replacement",
            "repetitions": repetitions,
            "confidence_interval": "percentile 95%",
            "base_seed": 2026,
        },
        "days_per_year": 365,
        "selection_warning": "Seed 7 was selected by internal test-set AUC; results are exploratory and test-selection biased.",
        "figure_style": {
            "canvas_inches": [8.0, 8.0],
            "png_dpi": 600,
            "gridlines": False,
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Seed 7 lag sensitivity analysis",
        "",
        "The fixed internal test set was re-evaluated after excluding prostate-cancer cases diagnosed within the first 1, 2, or 3 years. Non-cases and cases diagnosed after each lag were retained. AUC intervals use 2,000 stratified participant-bootstrap repetitions.",
        "",
        "| Model | Lag (years) | Retained events | Excluded events | AUC (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.itertuples():
        lines.append(
            f"| {row.model} | {row.lag_years} | {row.events_retained} | {row.events_excluded} | "
            f"{row.auc:.3f} ({row.auc_ci_lower:.3f}-{row.auc_ci_upper:.3f}) |"
        )
    lines.extend(
        [
            "",
            "Interpretation: a stable AUC after progressively excluding early diagnoses argues against performance being driven only by cancers already close to clinical detection at baseline. Seed 7 was selected using the internal test set, so these estimates remain exploratory.",
        ]
    )
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild square publication figures and run seed 7 lag sensitivity analysis."
    )
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    study_root = args.study_root.resolve()
    set_publication_style()
    rebuild_internal_analysis(study_root / "7_model_comparison_1", "Model comparison 1")
    rebuild_internal_analysis(study_root / "8_model_comparison_2_clinical", "Model comparison 2")
    rebuild_wales_teacher_student(study_root)
    lag_output = build_lag_sensitivity(study_root, args.bootstrap_repetitions)
    print(json.dumps({"status": "completed", "lag_output": str(lag_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
