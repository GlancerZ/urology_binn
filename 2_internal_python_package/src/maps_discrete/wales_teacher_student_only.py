from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


MODEL_SPECS = {
    "teacher_maps": {
        "label": "Teacher-MAPS",
        "source_label": "Teacher MAPS (seed 7)",
        "risk_column": "teacher_maps_risk_10y",
        "color": "#176B87",
        "prefix": "T",
        "folder": "2_teacher_maps",
    },
    "student_distilled": {
        "label": "Student-MAPS",
        "source_label": "Distilled Student (seed 98)",
        "risk_column": "student_distilled_risk_10y",
        "color": "#D97706",
        "prefix": "S",
        "folder": "3_student_maps",
    },
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output: Path, stem: str, *, right: float = 0.96, bottom: float = 0.18) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.04, bottom, right, 0.94))
    # Keep a fixed square canvas. Using bbox_inches="tight" here would expand
    # the page around an external legend and turn the exported figure into a
    # rectangle even when the plotting canvas is square.
    fig.savefig(output / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def _legend_below(axis: plt.Axes, *, columns: int = 1) -> None:
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        ncol=columns,
        borderaxespad=0,
        fontsize=9,
    )


def _plot_roc(
    y: np.ndarray,
    risk: np.ndarray,
    performance: pd.Series,
    spec: dict[str, str],
    figures: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y, risk)
    fig, axis = plt.subplots(figsize=(7.4, 7.4))
    axis.plot(
        fpr,
        tpr,
        color=spec["color"],
        lw=2.6,
        label=(
            f"{spec['label']}: AUC {performance['auc']:.3f} "
            f"(95% CI {performance['auc_ci_lower']:.3f}-{performance['auc_ci_upper']:.3f})"
        ),
    )
    axis.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.2, label="Chance")
    axis.set(
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        title=f"Wales external validation: {spec['label']} ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.22)
    axis.set_aspect("equal", adjustable="box")
    _legend_below(axis)
    _save(fig, figures, f"Figure_{spec['prefix']}1_ROC_curve", bottom=0.24)


def _plot_pr(
    y: np.ndarray,
    risk: np.ndarray,
    performance: pd.Series,
    spec: dict[str, str],
    figures: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y, risk)
    prevalence = float(np.mean(y))
    fig, axis = plt.subplots(figsize=(7.4, 7.4))
    axis.plot(
        recall,
        precision,
        color=spec["color"],
        lw=2.6,
        label=(
            f"{spec['label']}: PR-AUC {performance['pr_auc']:.3f} "
            f"(95% CI {performance['pr_auc_ci_lower']:.3f}-{performance['pr_auc_ci_upper']:.3f})"
        ),
    )
    axis.axhline(prevalence, ls="--", color="#6B7280", lw=1.2, label=f"Event prevalence {prevalence:.3f}")
    axis.set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
        title=f"Wales external validation: {spec['label']} precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.22)
    axis.set_aspect("equal", adjustable="box")
    _legend_below(axis)
    _save(fig, figures, f"Figure_{spec['prefix']}2_PR_curve", bottom=0.24)


def _plot_dca(dca: pd.DataFrame, spec: dict[str, str], figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 7.4))
    axis.plot(
        dca["threshold"] * 100,
        dca["model_net_benefit"],
        color=spec["color"],
        lw=2.6,
        label=spec["label"],
    )
    axis.plot(
        dca["threshold"] * 100,
        dca["treat_all_net_benefit"],
        "--",
        color="#6B7280",
        lw=1.4,
        label="Screen all",
    )
    axis.axhline(0, color="#111827", lw=1.2, label="Screen none")
    axis.set(
        xlabel="10-year risk threshold (%)",
        ylabel="Net benefit",
        title=f"Wales external validation: {spec['label']} decision curve",
        xlim=(float(dca["threshold"].min() * 100), float(dca["threshold"].max() * 100)),
    )
    axis.grid(alpha=0.22)
    axis.set_box_aspect(1)
    _legend_below(axis, columns=3)
    _save(fig, figures, f"Figure_{spec['prefix']}3_DCA", bottom=0.24)


def _plot_performance_forest(performance: pd.Series, spec: dict[str, str], figures: Path) -> None:
    metrics = [
        ("c_index", "C-index"),
        ("auc", "10-year AUC"),
        ("pr_auc", "PR-AUC"),
    ]
    estimates = np.array([float(performance[key]) for key, _ in metrics])
    lower = np.array([float(performance[f"{key}_ci_lower"]) for key, _ in metrics])
    upper = np.array([float(performance[f"{key}_ci_upper"]) for key, _ in metrics])
    y = np.arange(len(metrics))
    fig, axis = plt.subplots(figsize=(7.4, 7.4))
    axis.errorbar(
        estimates,
        y,
        xerr=[estimates - lower, upper - estimates],
        fmt="o",
        markersize=7,
        color=spec["color"],
        ecolor="#64748B",
        elinewidth=1.8,
        capsize=4,
    )
    for row, estimate, lo, hi in zip(y, estimates, lower, upper):
        axis.text(min(1.035, hi + 0.025), row, f"{estimate:.3f} ({lo:.3f}-{hi:.3f})", va="center", fontsize=9)
    axis.set_yticks(y, [label for _, label in metrics])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.16)
    axis.set_xlabel("Estimate (95% participant-bootstrap CI)")
    axis.set_title(f"Wales external validation: {spec['label']} performance")
    axis.grid(axis="x", alpha=0.22)
    axis.set_box_aspect(1)
    _save(fig, figures, f"Figure_{spec['prefix']}4_performance_forest", bottom=0.14)


def _plot_calibration(calibration: pd.DataFrame, spec: dict[str, str], figures: Path) -> None:
    limit = max(0.16, float(calibration[["predicted", "observed"]].max().max()) * 1.08)
    fig, axis = plt.subplots(figsize=(7.4, 7.4))
    axis.plot(
        calibration["predicted"],
        calibration["observed"],
        marker="o",
        markersize=5.5,
        lw=2.2,
        color=spec["color"],
        label=spec["label"],
    )
    axis.plot([0, limit], [0, limit], "--", color="#6B7280", lw=1.3, label="Ideal")
    axis.set(
        xlabel="Mean predicted 10-year risk",
        ylabel="Observed 10-year risk",
        title=f"Wales external validation: {spec['label']} calibration by risk decile",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axis.grid(alpha=0.22)
    axis.set_aspect("equal", adjustable="box")
    _legend_below(axis, columns=2)
    _save(fig, figures, f"Figure_{spec['prefix']}5_calibration", bottom=0.24)


def _formatted_interval(row: pd.Series, metric: str) -> str:
    return f"{row[metric]:.3f} ({row[f'{metric}_ci_lower']:.3f}, {row[f'{metric}_ci_upper']:.3f})"


def generate(study_root: Path) -> Path:
    source = study_root / "9_wales_external_validation"
    source_analysis = source / "1_model_comparison_1"
    output = source / "3_teacher_student_only"
    tables = output / "1_tables"
    tables.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(source / "wales_predictions_all_models.csv")
    performance_all = pd.read_csv(source_analysis / "1_tables" / "Table_1_model_performance_95ci.csv")
    comparison_all = pd.read_csv(source_analysis / "1_tables" / "Table_2_model_differences_nri_idi_95ci.csv")
    recovery_all = pd.read_csv(source_analysis / "1_tables" / "Table_4_student_performance_recovery_95ci.csv")
    dca_all = pd.read_csv(source_analysis / "3_plot_data" / "dca_curve_data.csv")
    calibration_all = pd.read_csv(source_analysis / "3_plot_data" / "calibration_curve_data.csv")

    model_ids = list(MODEL_SPECS)
    performance = performance_all.loc[performance_all["model_id"].isin(model_ids)].copy()
    performance["model"] = performance["model_id"].map({key: value["label"] for key, value in MODEL_SPECS.items()})
    performance.to_csv(tables / "Table_1_Teacher_Student_performance_95ci.csv", index=False)

    comparison = comparison_all.loc[
        comparison_all["model_id"].eq("student_distilled")
        & comparison_all["reference_model_id"].eq("teacher_maps")
    ].copy()
    comparison["model"] = "Student-MAPS"
    comparison["reference_model"] = "Teacher-MAPS"
    comparison.to_csv(tables / "Table_2_Student_vs_Teacher_NRI_IDI_95ci.csv", index=False)
    recovery_all.to_csv(tables / "Table_3_Student_performance_recovery_95ci.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    comparison_row = comparison.iloc[0]
    for model_id in model_ids:
        row = performance.loc[performance["model_id"].eq(model_id)].iloc[0]
        result: dict[str, object] = {
            "model": MODEL_SPECS[model_id]["label"],
            "n": int(row["n"]),
            "events": int(row["events"]),
            "c_index_95ci": _formatted_interval(row, "c_index"),
            "auc_95ci": _formatted_interval(row, "auc"),
            "pr_auc_95ci": _formatted_interval(row, "pr_auc"),
            "brier_95ci": _formatted_interval(row, "brier"),
            "nri_reference": "Teacher-MAPS" if model_id == "student_distilled" else "Not applicable",
            "continuous_nri_95ci": (
                _formatted_interval(comparison_row, "nri") if model_id == "student_distilled" else "Not applicable"
            ),
            "idi_95ci": (
                _formatted_interval(comparison_row, "idi") if model_id == "student_distilled" else "Not applicable"
            ),
        }
        summary_rows.append(result)
    pd.DataFrame(summary_rows).to_csv(tables / "Table_0_Wales_Teacher_Student_summary.csv", index=False)

    y = predictions["event_10y"].to_numpy(int)
    for model_id, spec in MODEL_SPECS.items():
        model_output = output / spec["folder"]
        figures = model_output / "2_figures"
        plot_data = model_output / "3_plot_data"
        plot_data.mkdir(parents=True, exist_ok=True)
        risk = predictions[spec["risk_column"]].to_numpy(float)
        model_performance = performance.loc[performance["model_id"].eq(model_id)].iloc[0]
        model_dca = dca_all.loc[dca_all["model_id"].eq(model_id)].copy()
        model_calibration = calibration_all.loc[calibration_all["model_id"].eq(model_id)].copy()
        predictions[["sample", "event_10y", "observed_time_years", spec["risk_column"]]].to_csv(
            plot_data / "external_predictions.csv", index=False
        )
        model_dca.to_csv(plot_data / "dca_curve_data.csv", index=False)
        model_calibration.to_csv(plot_data / "calibration_curve_data.csv", index=False)
        _plot_roc(y, risk, model_performance, spec, figures)
        _plot_pr(y, risk, model_performance, spec, figures)
        _plot_dca(model_dca, spec, figures)
        _plot_performance_forest(model_performance, spec, figures)
        _plot_calibration(model_calibration, spec, figures)

    manifest = {
        "reporting_scope": ["Teacher-MAPS", "Student-MAPS"],
        "visualization_policy": "Teacher-MAPS and Student-MAPS are shown in separate figures.",
        "n": int(len(predictions)),
        "events": int(y.sum()),
        "horizon_years": 10,
        "days_per_year": 365,
        "bootstrap_repetitions": 2000,
        "dca_threshold_percent_range": [0.5, 15.0],
        "nri_idi_reference": "Student-MAPS is compared directly with Teacher-MAPS; Teacher-MAPS has no NRI/IDI row because these are comparative metrics.",
        "external_refitting": False,
        "external_seed_selection": False,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Wales external validation - reporting set\n\n"
        "This reporting directory intentionally contains only Teacher-MAPS and distilled Student-MAPS. "
        "Their visualizations are stored separately. NRI and IDI are comparative metrics, so Student-MAPS "
        "is evaluated against Teacher-MAPS in Table 2; they are not displayed as standalone Teacher metrics.\n\n"
        "Cohort: n=910, 42 ten-year events. Internal preprocessing, weights, selected seeds and risk definition "
        "were frozen. Wales was not used for refitting, calibration, hyperparameter tuning or seed selection. "
        "All intervals use 2,000 participant-level bootstrap repetitions.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create separate Wales Teacher-MAPS and Student-MAPS reporting outputs.")
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.study_root.resolve())
    print(result)


if __name__ == "__main__":
    main()
