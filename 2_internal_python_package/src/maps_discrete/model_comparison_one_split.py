from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .config import load_config
from .final_evaluation import ModelPrediction, evaluate_models


TEACHER_TOP_N = 5
ML_BOTTOM_N = 5
ML_SEED_ANCHOR = "nmr_olink_prs__xgboost"

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6F4E7C",
    "#8C6D31",
    "#4D4D4D",
]

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]

FULL_OMICS_SPECS = [
    {
        "model_id": "teacher_maps",
        "label": "Teacher-MAPS",
        "source": "maps",
        "risk_column": "teacher_risk_10y",
        "ranking_column": "teacher_log_risk",
        "metric_model": "teacher",
    },
    {
        "model_id": "full_omics_xgboost",
        "label": "XGBoost",
        "source": "ml",
        "risk_column": "nmr_olink_prs__xgboost_risk_10y",
        "metric_model": "nmr_olink_prs__xgboost",
    },
    {
        "model_id": "full_omics_lightgbm",
        "label": "LightGBM",
        "source": "ml",
        "risk_column": "nmr_olink_prs__lightgbm_risk_10y",
        "metric_model": "nmr_olink_prs__lightgbm",
    },
    {
        "model_id": "full_omics_random_forest",
        "label": "Random Forest",
        "source": "ml",
        "risk_column": "nmr_olink_prs__random_forest_risk_10y",
        "metric_model": "nmr_olink_prs__random_forest",
    },
    {
        "model_id": "full_omics_logistic",
        "label": "Logistic regression",
        "source": "ml",
        "risk_column": "nmr_olink_prs__logistic_risk_10y",
        "metric_model": "nmr_olink_prs__logistic",
    },
    {
        "model_id": "full_omics_elastic_net",
        "label": "Elastic Net",
        "source": "ml",
        "risk_column": "nmr_olink_prs__elastic_net_risk_10y",
        "metric_model": "nmr_olink_prs__elastic_net",
    },
    {
        "model_id": "full_omics_svm_rbf",
        "label": "SVM-RBF",
        "source": "ml",
        "risk_column": "nmr_olink_prs__svm_rbf_risk_10y",
        "metric_model": "nmr_olink_prs__svm_rbf",
    },
]

NO_OLINK_SPECS = [
    {
        "model_id": "student_distilled",
        "label": "Student-MAPS",
        "source": "maps",
        "risk_column": "student_distilled_risk_10y",
        "ranking_column": "student_distilled_log_risk",
        "metric_model": "student_distilled",
    },
    {
        "model_id": "nmr_only_logistic",
        "label": "NMR-only Logistic",
        "source": "ml",
        "risk_column": "nmr_only__logistic_risk_10y",
        "metric_model": "nmr_only__logistic",
    },
    {
        "model_id": "prs_only_logistic",
        "label": "PRS-only Logistic",
        "source": "ml",
        "risk_column": "prs_only__logistic_risk_10y",
        "metric_model": "prs_only__logistic",
    },
    {
        "model_id": "nmr_prs_logistic",
        "label": "NMR+PRS Logistic",
        "source": "ml",
        "risk_column": "nmr_prs__logistic_risk_10y",
        "metric_model": "nmr_prs__logistic",
    },
    {
        "model_id": "no_olink_xgboost",
        "label": "NMR+PRS XGBoost",
        "source": "ml",
        "risk_column": "nmr_prs__xgboost_risk_10y",
        "metric_model": "nmr_prs__xgboost",
    },
    {
        "model_id": "no_olink_lightgbm",
        "label": "NMR+PRS LightGBM",
        "source": "ml",
        "risk_column": "nmr_prs__lightgbm_risk_10y",
        "metric_model": "nmr_prs__lightgbm",
    },
    {
        "model_id": "no_olink_random_forest",
        "label": "NMR+PRS Random Forest",
        "source": "ml",
        "risk_column": "nmr_prs__random_forest_risk_10y",
        "metric_model": "nmr_prs__random_forest",
    },
    {
        "model_id": "no_olink_elastic_net",
        "label": "NMR+PRS Elastic Net",
        "source": "ml",
        "risk_column": "nmr_prs__elastic_net_risk_10y",
        "metric_model": "nmr_prs__elastic_net",
    },
    {
        "model_id": "no_olink_svm_rbf",
        "label": "NMR+PRS SVM-RBF",
        "source": "ml",
        "risk_column": "nmr_prs__svm_rbf_risk_10y",
        "metric_model": "nmr_prs__svm_rbf",
    },
]


def _style() -> None:
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
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.set_facecolor("white")
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.tick_params(colors="#333333")


def _save_square(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(8.0, 8.0, forward=True)
    fig.savefig(output / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def _prediction_path(root: Path, source: str, seed: int) -> Path:
    if source == "maps":
        return root / "5_teacher_student_multiseed" / f"seed_{seed:03d}" / "predictions_test.csv"
    return root / "6_traditional_ml_multiseed" / f"seed_{seed:03d}" / "ml_predictions_test.csv"


def _select_seeds(root: Path) -> tuple[list[int], list[int], pd.DataFrame, pd.DataFrame]:
    teacher_metrics = pd.read_csv(
        root / "5_teacher_student_multiseed" / "101_multiseed_analysis" / "all_100_seed_test_metrics.csv"
    )
    ml_metrics = pd.read_csv(
        root / "6_traditional_ml_multiseed" / "101_multiseed_analysis" / "all_100_seed_ml_test_metrics.csv"
    )
    teacher_seeds = (
        teacher_metrics.loc[teacher_metrics["model"].eq("teacher")]
        .sort_values(["auc", "training_seed"], ascending=[False, True])
        .head(TEACHER_TOP_N)["training_seed"]
        .astype(int)
        .tolist()
    )
    ml_seeds = (
        ml_metrics.loc[ml_metrics["model"].eq(ML_SEED_ANCHOR)]
        .sort_values(["auc", "training_seed"], ascending=[True, True])
        .head(ML_BOTTOM_N)["training_seed"]
        .astype(int)
        .tolist()
    )
    if len(teacher_seeds) != TEACHER_TOP_N or len(ml_seeds) != ML_BOTTOM_N:
        raise RuntimeError("Failed to select the required five Teacher and ML seeds")
    return teacher_seeds, ml_seeds, teacher_metrics, ml_metrics


def _load_group_predictions(
    root: Path,
    specs: list[dict[str, str]],
    teacher_seeds: list[int],
    ml_seeds: list[int],
) -> tuple[pd.DataFrame, list[ModelPrediction], dict[str, list[np.ndarray]], list[dict[str, object]]]:
    base: pd.DataFrame | None = None
    predictions: list[ModelPrediction] = []
    seed_risks: dict[str, list[np.ndarray]] = {}
    sources: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        seeds = teacher_seeds if spec["source"] == "maps" else ml_seeds
        risk_arrays: list[np.ndarray] = []
        ranking_arrays: list[np.ndarray] = []
        paths: list[str] = []
        for seed in seeds:
            path = _prediction_path(root, spec["source"], seed)
            frame = pd.read_csv(path, dtype={"sample": str}).set_index("sample")
            if base is None:
                base = frame[["event_10y", "observed_time_years"]].copy()
            else:
                frame = frame.loc[base.index]
                if not np.array_equal(frame["event_10y"].to_numpy(int), base["event_10y"].to_numpy(int)):
                    raise RuntimeError(f"Endpoint mismatch: {path}")
                if not np.allclose(
                    frame["observed_time_years"].to_numpy(float),
                    base["observed_time_years"].to_numpy(float),
                ):
                    raise RuntimeError(f"Duration mismatch: {path}")
            risk_arrays.append(frame[spec["risk_column"]].to_numpy(float))
            ranking_column = spec.get("ranking_column")
            if ranking_column:
                ranking_arrays.append(frame[ranking_column].to_numpy(float))
            paths.append(str(path.resolve()))
        risk_matrix = np.vstack(risk_arrays)
        ranking_matrix = np.vstack(ranking_arrays) if ranking_arrays else None
        seed_risks[spec["model_id"]] = risk_arrays
        predictions.append(
            ModelPrediction(
                model_id=spec["model_id"],
                label=spec["label"],
                risk=risk_matrix.mean(axis=0),
                log_risk=ranking_matrix.mean(axis=0) if ranking_matrix is not None else None,
                color=PALETTE[index],
            )
        )
        sources.append(
            {
                **spec,
                "seeds": seeds,
                "prediction_paths": paths,
                "ensemble_definition": "participant-level arithmetic mean of five seed-specific risks",
            }
        )
    assert base is not None
    return base, predictions, seed_risks, sources


def _seed_metric_summary(
    specs: list[dict[str, str]],
    teacher_seeds: list[int],
    ml_seeds: list[int],
    teacher_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        seeds = teacher_seeds if spec["source"] == "maps" else ml_seeds
        source = teacher_metrics if spec["source"] == "maps" else ml_metrics
        selected = source.loc[
            source["model"].eq(spec["metric_model"])
            & source["training_seed"].astype(int).isin(seeds)
        ]
        if len(selected) != len(seeds):
            raise RuntimeError(f"Incomplete metric rows for {spec['metric_model']}")
        for metric in ("c_index", "auc", "pr_auc", "brier"):
            values = selected[metric].to_numpy(float)
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "model": spec["label"],
                    "metric": metric,
                    "seeds": ",".join(map(str, seeds)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _remove_default_figures(figures: Path) -> None:
    for stem in (
        "Figure_1_ROC_PR_curves",
        "Figure_2_DCA",
        "Figure_3_model_difference_forest",
        "Figure_4_AUC_PR_AUC_forest",
        "Figure_5_calibration",
    ):
        for suffix in (".png", ".pdf"):
            path = figures / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def _plot_mean_curve(
    *,
    kind: str,
    output: Path,
    title: str,
    specs: list[dict[str, str]],
    seed_risks: dict[str, list[np.ndarray]],
    event: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 8))
    grid = np.linspace(0.0, 1.0, 401)
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for index, spec in enumerate(specs):
        curves: list[np.ndarray] = []
        metric_values: list[float] = []
        for score in seed_risks[spec["model_id"]]:
            if kind == "roc":
                fpr, tpr, _ = roc_curve(event, score)
                curves.append(np.interp(grid, fpr, tpr))
                metric_values.append(float(roc_auc_score(event, score)))
            else:
                precision, recall, _ = precision_recall_curve(event, score)
                order = np.argsort(recall)
                curves.append(np.interp(grid, recall[order], precision[order]))
                metric_values.append(float(average_precision_score(event, score)))
        matrix = np.vstack(curves)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        if kind == "roc":
            mean[0], mean[-1] = 0.0, 1.0
        else:
            mean[-1] = float(event.mean())
        color = PALETTE[index]
        line = axis.plot(grid, mean, color=color, ls=LINESTYLES[index], lw=2.0)[0]
        axis.fill_between(
            grid,
            np.clip(mean - sd, 0, 1),
            np.clip(mean + sd, 0, 1),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        handles.append(line)
        metric_label = "AUC" if kind == "roc" else "PR-AUC"
        labels.append(f"{spec['label']} ({metric_label} {np.mean(metric_values):.3f})")
    if kind == "roc":
        axis.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1.0)
        axis.set(xlabel="1 - Specificity", ylabel="Sensitivity")
        stem = "Figure_1_ROC_curve"
    else:
        axis.axhline(float(event.mean()), ls="--", color="#777777", lw=1.0)
        axis.set(xlabel="Recall (Sensitivity)", ylabel="Precision")
        stem = "Figure_2_PR_curve"
    axis.set(title=title, xlim=(0, 1), ylim=(0, 1.01))
    _clean_axis(axis)
    axis.set_aspect("equal", adjustable="box")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        ncol=2,
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.text(
        0.5,
        0.155,
        "Line: pointwise mean across 5 seeds; shading: +/-1 SD.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.27)
    _save_square(fig, output, stem)


def _plot_dca(
    output: Path,
    title: str,
    specs: list[dict[str, str]],
    dca: pd.DataFrame,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 8))
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for index, spec in enumerate(specs):
        frame = dca.loc[dca["model_id"].eq(spec["model_id"])]
        handles.append(
            axis.plot(
                frame["threshold"] * 100,
                frame["model_net_benefit"],
                color=PALETTE[index],
                ls=LINESTYLES[index],
                lw=1.9,
            )[0]
        )
        labels.append(spec["label"])
    first = dca.loc[dca["model_id"].eq(specs[0]["model_id"])]
    handles.extend(
        [
            axis.plot(
                first["threshold"] * 100,
                first["treat_all_net_benefit"],
                color="#777777",
                ls="--",
                lw=1.1,
            )[0],
            axis.axhline(0, color="#111111", lw=1.0),
        ]
    )
    labels.extend(["Screen all", "Screen none"])
    axis.set(
        xlabel="10-year risk threshold (%)",
        ylabel="Net benefit",
        title=title,
        xlim=(float(first["threshold"].min() * 100), float(first["threshold"].max() * 100)),
    )
    _clean_axis(axis)
    axis.set_box_aspect(1)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
        ncol=3,
        handlelength=2.5,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.27)
    _save_square(fig, output, "Figure_3_DCA")


def _plot_performance_reclassification_forest(
    output: Path,
    title: str,
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Plot absolute discrimination and reference-based reclassification.

    C-index and fixed-horizon AUC are model-level absolute performance
    measures. Continuous NRI and IDI are pairwise measures and therefore retain
    their explicitly named reference model.
    """
    panels = [
        (performance, "c_index", "C-index", False),
        (performance, "auc", "10-year AUC", False),
        (comparison, "nri", "Continuous NRI", True),
        (comparison, "idi", "IDI", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=False)
    for index, (axis, (table, metric, heading, needs_reference)) in enumerate(
        zip(axes.flat, panels)
    ):
        y = np.arange(len(table))
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
        if needs_reference:
            axis.axvline(0, color="#B2182B", ls="--", lw=1.0)
        axis.set_title(heading)
        axis.set_yticks(y)
        if index % 2 == 0:
            if needs_reference:
                labels = [
                    f"{row.model} vs\n{row.reference_model}"
                    for row in table.itertuples()
                ]
            else:
                labels = table["model"].tolist()
            axis.set_yticklabels(labels, fontsize=7.0)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        _clean_axis(axis)
    fig.suptitle(title, y=0.975, fontweight="bold")
    fig.supxlabel(
        "Absolute performance (top) or reference-based estimate (bottom), with 95% participant-bootstrap CI",
        y=0.035,
        fontsize=8.7,
    )
    fig.subplots_adjust(left=0.32, right=0.98, top=0.90, bottom=0.10, hspace=0.28, wspace=0.22)
    _save_square(fig, output, "Figure_4_Cindex_AUC_NRI_IDI_forest")


def _plot_auc_pr_forest(
    output: Path,
    title: str,
    specs: list[dict[str, str]],
    performance: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    y = np.arange(len(performance))
    labels = performance["model"].tolist()
    for panel, (axis, metric, heading) in enumerate(
        zip(axes, ("auc", "pr_auc"), ("10-year AUC", "PR-AUC"))
    ):
        estimate = performance[metric].to_numpy(float)
        lower = performance[f"{metric}_ci_lower"].to_numpy(float)
        upper = performance[f"{metric}_ci_upper"].to_numpy(float)
        for index, (row, est, lo, hi) in enumerate(zip(y, estimate, lower, upper)):
            axis.errorbar(
                est,
                row,
                xerr=np.array([[est - lo], [hi - est]]),
                fmt="o",
                markersize=5.2,
                color=PALETTE[index],
                ecolor="#555555",
                elinewidth=1.1,
                capsize=2.5,
            )
        axis.set_title(heading)
        axis.set_yticks(y)
        if panel == 0:
            axis.set_yticklabels(labels, fontsize=7.8)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        _clean_axis(axis)
    fig.suptitle(title, y=0.955, fontweight="bold")
    fig.supxlabel("Estimate (95% participant-bootstrap CI)", y=0.055, fontsize=9.5)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.87, bottom=0.13, wspace=0.20)
    _save_square(fig, output, "Figure_5_AUC_PR_AUC_forest")


def _primary_table(performance: pd.DataFrame, comparison: pd.DataFrame, reference: str) -> pd.DataFrame:
    comparison_by_id = comparison.set_index("model_id")
    rows: list[dict[str, object]] = []
    for row in performance.itertuples():
        result: dict[str, object] = {
            "model_id": row.model_id,
            "model": row.model,
            "c_index": row.c_index,
            "c_index_ci_lower": row.c_index_ci_lower,
            "c_index_ci_upper": row.c_index_ci_upper,
            "auc": row.auc,
            "auc_ci_lower": row.auc_ci_lower,
            "auc_ci_upper": row.auc_ci_upper,
            "nri_idi_reference_model": (
                "Not applicable (reference model)"
                if row.model_id == reference
                else comparison_by_id.loc[row.model_id, "reference_model"]
            ),
            "nri": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "nri"],
            "nri_ci_lower": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "nri_ci_lower"],
            "nri_ci_upper": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "nri_ci_upper"],
            "idi": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "idi"],
            "idi_ci_lower": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "idi_ci_lower"],
            "idi_ci_upper": np.nan if row.model_id == reference else comparison_by_id.loc[row.model_id, "idi_ci_upper"],
        }
        rows.append(result)
    return pd.DataFrame(rows)


def _run_group(
    *,
    root: Path,
    output: Path,
    specs: list[dict[str, str]],
    group_title: str,
    reference_model_id: str,
    teacher_seeds: list[int],
    ml_seeds: list[int],
    teacher_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    repetitions: int,
    bootstrap_seed: int,
    horizon: float,
    dca_min: float,
    dca_max: float,
    dca_points: int,
) -> None:
    base, predictions, seed_risks, sources = _load_group_predictions(
        root, specs, teacher_seeds, ml_seeds
    )
    evaluate_models(
        samples=base.index.to_numpy(str),
        duration=base["observed_time_years"].to_numpy(float),
        event=base["event_10y"].to_numpy(int),
        predictions=predictions,
        output_dir=output,
        reference_model_id=reference_model_id,
        horizon=horizon,
        repetitions=repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=0.95,
        dca_threshold_min=dca_min,
        dca_threshold_max=dca_max,
        dca_threshold_points=dca_points,
        title_prefix=group_title,
    )
    figures = output / "2_figures"
    tables = output / "1_tables"
    plot_data = output / "3_plot_data"
    _remove_default_figures(figures)

    performance = pd.read_csv(tables / "Table_1_model_performance_95ci.csv")
    comparison = pd.read_csv(tables / "Table_2_model_differences_nri_idi_95ci.csv")
    dca = pd.read_csv(plot_data / "dca_curve_data.csv")
    seed_metrics = _seed_metric_summary(
        specs, teacher_seeds, ml_seeds, teacher_metrics, ml_metrics
    )
    seed_metrics.to_csv(tables / "Table_3_selected_seed_mean_metrics.csv", index=False)
    _primary_table(performance, comparison, reference_model_id).to_csv(
        tables / "Table_0_primary_Cindex_AUC_NRI_IDI_95CI.csv", index=False
    )

    event = base["event_10y"].to_numpy(int)
    _plot_mean_curve(
        kind="roc",
        output=figures,
        title=f"{group_title}: mean 10-year ROC curves",
        specs=specs,
        seed_risks=seed_risks,
        event=event,
    )
    _plot_mean_curve(
        kind="pr",
        output=figures,
        title=f"{group_title}: mean precision-recall curves",
        specs=specs,
        seed_risks=seed_risks,
        event=event,
    )
    _plot_dca(figures, f"{group_title}: decision curve analysis", specs, dca)
    _plot_performance_reclassification_forest(
        figures,
        f"{group_title}: performance and reclassification",
        performance,
        comparison,
    )
    _plot_auc_pr_forest(
        figures, f"{group_title}: discrimination estimates", specs, performance
    )

    manifest = {
        "group_title": group_title,
        "reference_model_id_for_nri_idi": reference_model_id,
        "reference_model": next(spec["label"] for spec in specs if spec["model_id"] == reference_model_id),
        "models": sources,
        "teacher_top5_seeds": teacher_seeds,
        "ml_bottom5_seeds_shared_by_all_ml_models": ml_seeds,
        "ml_seed_anchor": ML_SEED_ANCHOR,
        "curve_summary": "pointwise mean of five seed-specific curves; shading is +/-1 SD; legend reports arithmetic mean of seed-specific AUC or PR-AUC",
        "other_figure_summary": "participant-level arithmetic mean-risk ensemble across the corresponding five seeds",
        "bootstrap_repetitions": repetitions,
        "figure_style": {
            "canvas_inches": [8.0, 8.0],
            "png_dpi": 600,
            "pdf": "vector",
            "gridlines": False,
            "legend": "outside plotting axes",
            "one_figure_per_file": True,
        },
        "warning": "Teacher Top-5 and ML Bottom-5 seed selection used the fixed test set; results are exploratory and selection-biased.",
    }
    (output / "split_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {group_title}",
        "",
        f"NRI and IDI reference: {manifest['reference_model']}.",
        "",
        f"Teacher/Student seeds: {', '.join(map(str, teacher_seeds))}.",
        f"All ML seeds: {', '.join(map(str, ml_seeds))}, selected as the five lowest fixed-test ROC-AUC seeds of full-omics XGBoost.",
        "",
        "ROC and PR curves are pointwise means of five seed-specific curves with +/-1 SD shading. DCA and bootstrap tables use participant-level mean-risk ensembles.",
        "",
        "Outputs include separate ROC, PR, DCA, combined C-index/AUC/NRI/IDI forest, and AUC/PR-AUC forest files. C-index and 10-year AUC are absolute model performance; only NRI and IDI use the stated reference model. All figures use a square 8 x 8 inch canvas, white background, no gridlines, and vector PDF plus 600 dpi PNG.",
        "",
        "Important: test-set-directed seed selection makes these comparisons exploratory and selection-biased.",
    ]
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    config = load_config(config_path)
    root = Path(config.path).resolve().parents[1]
    output = Path(output_dir) if output_dir else root / "7_model_comparison_1"
    teacher_seeds, ml_seeds, teacher_metrics, ml_metrics = _select_seeds(root)
    repetitions = int(config.raw["evaluation"]["bootstrap_repetitions"])
    common = {
        "root": root,
        "teacher_seeds": teacher_seeds,
        "ml_seeds": ml_seeds,
        "teacher_metrics": teacher_metrics,
        "ml_metrics": ml_metrics,
        "repetitions": repetitions,
        "bootstrap_seed": int(config.test_holdout_seed),
        "horizon": float(config.primary_horizon),
        "dca_min": float(config.raw["evaluation"]["dca_threshold_min"]),
        "dca_max": float(config.raw["evaluation"]["dca_threshold_max"]),
        "dca_points": int(config.raw["evaluation"]["dca_threshold_points"]),
    }
    _run_group(
        output=output / "1_full_omics_NMR_Olink_PRS",
        specs=FULL_OMICS_SPECS,
        group_title="NMR + Olink + PRS",
        reference_model_id="full_omics_logistic",
        **common,
    )
    _run_group(
        output=output / "2_no_Olink_NMR_PRS",
        specs=NO_OLINK_SPECS,
        group_title="No Olink protein input",
        reference_model_id="nmr_prs_logistic",
        **common,
    )
    root_manifest = {
        "official_split_model_comparison_1": True,
        "parts": {
            "1_full_omics_NMR_Olink_PRS": {
                "models": [spec["label"] for spec in FULL_OMICS_SPECS],
                "nri_idi_reference": "Logistic regression using NMR + Olink + PRS",
            },
            "2_no_Olink_NMR_PRS": {
                "models": [spec["label"] for spec in NO_OLINK_SPECS],
                "nri_idi_reference": "NMR + PRS Logistic",
            },
        },
        "supersedes": "the previous combined model-comparison-1 visualization",
        "student_grouping_rationale": "Student-MAPS uses NMR + PRS and therefore belongs in the no-Olink comparison.",
    }
    (output / "OFFICIAL_SPLIT_COMPARISON_MANIFEST.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README_RESULTS.md").write_text(
        "# 模型比较1 - 按Olink输入拆分的正式可视化\n\n"
        "本目录下的正式模型比较1已拆为两部分：\n\n"
        "1. `1_full_omics_NMR_Olink_PRS`：Teacher-MAPS与六种NMR+Olink+PRS传统ML。\n"
        "2. `2_no_Olink_NMR_PRS`：Student-MAPS、NMR-only Logistic、PRS-only Logistic、NMR+PRS Logistic及五种NMR+PRS无蛋白ML。\n\n"
        "Student-MAPS实际输入不含Olink，因此归入第二部分。旧的根目录合并图已被本次拆分结果取代，仅为审计追溯而保留。\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split model comparison 1 by Olink input.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(result)


if __name__ == "__main__":
    _style()
    main()
