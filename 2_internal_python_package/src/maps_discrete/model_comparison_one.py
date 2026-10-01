from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "maps_mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config
from .final_evaluation import ModelPrediction, evaluate_models


TEACHER_TOP_N = 5
ML_BOTTOM_N = 5
ML_SEED_ANCHOR = "nmr_olink_prs__xgboost"

# The displayed ML baselines are unchanged from the original compact model
# comparison. Every ML baseline now uses the same five seeds selected by the
# full-omics XGBoost anchor.
MODEL_SELECTIONS = [
    {
        "model_id": "teacher_maps",
        "metric_model": "teacher",
        "label": "Teacher MAPS (Teacher Top-5 seeds)",
        "short_label": "Teacher MAPS",
        "source": "maps",
        "risk_column": "teacher_risk_10y",
        "ranking_column": "teacher_log_risk",
        "color": "#173F5F",
    },
    {
        "model_id": "student_distilled",
        "metric_model": "student_distilled",
        "label": "Student MAPS (same Teacher Top-5 seeds)",
        "short_label": "Student MAPS",
        "source": "maps",
        "risk_column": "student_distilled_risk_10y",
        "ranking_column": "student_distilled_log_risk",
        "color": "#E07A1F",
    },
    {
        "model_id": "student_no_distillation",
        "metric_model": "student_no_distillation",
        "label": "Non-distilled Student (same Teacher Top-5 seeds)",
        "short_label": "Non-distilled Student",
        "source": "maps",
        "risk_column": "student_no_distillation_risk_10y",
        "ranking_column": "student_no_distillation_log_risk",
        "color": "#7A5195",
    },
    {
        "model_id": "nmr_only_lightgbm",
        "metric_model": "nmr_only__lightgbm",
        "label": "NMR-only LightGBM (XGBoost Bottom-5 seeds)",
        "short_label": "NMR-only LightGBM",
        "source": "ml",
        "risk_column": "nmr_only__lightgbm_risk_10y",
        "color": "#2A9D8F",
    },
    {
        "model_id": "prs_only_logistic",
        "metric_model": "prs_only__logistic",
        "label": "PRS-only Logistic (XGBoost Bottom-5 seeds)",
        "short_label": "PRS-only Logistic",
        "source": "ml",
        "risk_column": "prs_only__logistic_risk_10y",
        "color": "#8C6D31",
    },
    {
        "model_id": "nmr_prs_logistic",
        "metric_model": "nmr_prs__logistic",
        "label": "NMR+PRS Logistic (XGBoost Bottom-5 seeds)",
        "short_label": "NMR+PRS Logistic",
        "source": "ml",
        "risk_column": "nmr_prs__logistic_risk_10y",
        "color": "#4C78A8",
    },
    {
        "model_id": "full_omics_xgboost",
        "metric_model": ML_SEED_ANCHOR,
        "label": "NMR+Olink+PRS XGBoost (Bottom-5 seeds)",
        "short_label": "NMR+Olink+PRS XGBoost",
        "source": "ml",
        "risk_column": "nmr_olink_prs__xgboost_risk_10y",
        "color": "#C43C39",
    },
]


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
    teacher_rows = (
        teacher_metrics.loc[teacher_metrics["model"] == "teacher"]
        .sort_values(["auc", "training_seed"], ascending=[False, True])
        .head(TEACHER_TOP_N)
    )
    anchor_rows = (
        ml_metrics.loc[ml_metrics["model"] == ML_SEED_ANCHOR]
        .sort_values(["auc", "training_seed"], ascending=[True, True])
        .head(ML_BOTTOM_N)
    )
    teacher_seeds = teacher_rows["training_seed"].astype(int).tolist()
    ml_seeds = anchor_rows["training_seed"].astype(int).tolist()
    if len(teacher_seeds) != TEACHER_TOP_N or len(ml_seeds) != ML_BOTTOM_N:
        raise RuntimeError("Could not select the required five Teacher and five ML seeds")
    return teacher_seeds, ml_seeds, teacher_metrics, ml_metrics


def _mean_seed_curve_figure(
    *,
    event: np.ndarray,
    seed_risks: dict[str, list[np.ndarray]],
    seed_metrics: pd.DataFrame,
    output: Path,
) -> None:
    """Replace Figure 1 with seed-wise mean ROC/PR curves and mean metrics."""
    fpr_grid = np.linspace(0.0, 1.0, 401)
    recall_grid = np.linspace(0.0, 1.0, 401)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8.8))

    for spec in MODEL_SELECTIONS:
        curves_roc: list[np.ndarray] = []
        curves_pr: list[np.ndarray] = []
        auc_values: list[float] = []
        ap_values: list[float] = []
        for score in seed_risks[spec["model_id"]]:
            fpr, tpr, _ = roc_curve(event, score)
            curves_roc.append(np.interp(fpr_grid, fpr, tpr))
            precision, recall, _ = precision_recall_curve(event, score)
            order = np.argsort(recall)
            curves_pr.append(np.interp(recall_grid, recall[order], precision[order]))
            auc_values.append(float(roc_auc_score(event, score)))
            ap_values.append(float(average_precision_score(event, score)))

        roc_matrix = np.vstack(curves_roc)
        pr_matrix = np.vstack(curves_pr)
        roc_mean = roc_matrix.mean(axis=0)
        pr_mean = pr_matrix.mean(axis=0)
        roc_mean[0], roc_mean[-1] = 0.0, 1.0
        pr_mean[-1] = float(event.mean())
        auc_mean = float(np.mean(auc_values))
        ap_mean = float(np.mean(ap_values))
        color = spec["color"]

        ax_roc.plot(
            fpr_grid,
            roc_mean,
            color=color,
            linewidth=2.5,
            label=f"{spec['short_label']} (mean AUC={auc_mean:.3f})",
        )
        ax_roc.fill_between(
            fpr_grid,
            np.maximum(0.0, roc_mean - roc_matrix.std(axis=0, ddof=1)),
            np.minimum(1.0, roc_mean + roc_matrix.std(axis=0, ddof=1)),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        ax_pr.plot(
            recall_grid,
            pr_mean,
            color=color,
            linewidth=2.5,
            label=f"{spec['short_label']} (mean PR-AUC={ap_mean:.3f})",
        )
        ax_pr.fill_between(
            recall_grid,
            np.maximum(0.0, pr_mean - pr_matrix.std(axis=0, ddof=1)),
            np.minimum(1.0, pr_mean + pr_matrix.std(axis=0, ddof=1)),
            color=color,
            alpha=0.08,
            linewidth=0,
        )

    prevalence = float(event.mean())
    ax_roc.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.3, label="Chance")
    ax_pr.axhline(prevalence, color="#777777", linestyle="--", linewidth=1.3, label=f"Prevalence ({prevalence:.3f})")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - Specificity", ylabel="Sensitivity", title="Mean 10-year ROC curves")
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall (Sensitivity)", ylabel="Precision", title="Mean 10-year precision-recall curves")
    for ax in (ax_roc, ax_pr):
        ax.grid(True, color="#D9DEE3", linewidth=0.8, alpha=0.8)
    roc_handles, roc_labels = ax_roc.get_legend_handles_labels()
    pr_handles, pr_labels = ax_pr.get_legend_handles_labels()
    fig.legend(
        roc_handles,
        roc_labels,
        loc="upper center",
        bbox_to_anchor=(0.25, 0.265),
        fontsize=8.5,
        frameon=False,
        ncol=1,
    )
    fig.legend(
        pr_handles,
        pr_labels,
        loc="upper center",
        bbox_to_anchor=(0.75, 0.265),
        fontsize=8.5,
        frameon=False,
        ncol=1,
    )
    fig.suptitle("Model comparison 1: selected five-seed mean curves", fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.935,
        "Teacher Top-5 by Teacher test AUC; Student uses identical Teacher seeds; all ML uses the identical Bottom-5 seeds selected by full-omics XGBoost test AUC",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.01,
        0.012,
        "Lines are the pointwise mean of five seed-specific curves; shaded bands are +/-1 SD. Selection is exploratory and test-set biased.",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.01, 0.30, 0.99, 0.90), w_pad=3.0)
    figures = output / "2_figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Figure_1_ROC_PR_curves.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(figures / "Figure_1_ROC_PR_curves.pdf", bbox_inches="tight")
    except PermissionError:
        fig.savefig(figures / "Figure_1_ROC_PR_curves_legend_outside.pdf", bbox_inches="tight")
    plt.close(fig)

    seed_metrics.to_csv(output / "1_tables" / "Table_0_selected_seed_mean_metrics.csv", index=False)


def run(config_path: str | Path, output_dir: str | Path | None = None) -> dict:
    config = load_config(config_path)
    root = Path(config.path).resolve().parents[1]
    output = Path(output_dir) if output_dir else root / "7_model_comparison_1"
    teacher_seeds, ml_seeds, teacher_metric_frame, ml_metric_frame = _select_seeds(root)

    base: pd.DataFrame | None = None
    predictions: list[ModelPrediction] = []
    source_rows: list[dict[str, object]] = []
    selected_seed_metric_rows: list[dict[str, object]] = []
    seed_risks: dict[str, list[np.ndarray]] = {}

    for spec in MODEL_SELECTIONS:
        seeds = teacher_seeds if spec["source"] == "maps" else ml_seeds
        risk_arrays: list[np.ndarray] = []
        log_risk_arrays: list[np.ndarray] = []
        prediction_paths: list[str] = []
        for seed in seeds:
            path = _prediction_path(root, spec["source"], seed)
            frame = pd.read_csv(path, dtype={"sample": str}).set_index("sample")
            if base is None:
                base = frame[["event_10y", "observed_time_years"]].copy()
            else:
                frame = frame.loc[base.index]
                if not np.array_equal(frame["event_10y"].to_numpy(int), base["event_10y"].to_numpy(int)):
                    raise RuntimeError(f"Event mismatch in {path}")
                if not np.allclose(frame["observed_time_years"], base["observed_time_years"]):
                    raise RuntimeError(f"Duration mismatch in {path}")
            risk_arrays.append(frame[spec["risk_column"]].to_numpy(float))
            ranking_column = spec.get("ranking_column")
            if ranking_column:
                log_risk_arrays.append(frame[ranking_column].to_numpy(float))
            prediction_paths.append(str(path.resolve()))

        risk_matrix = np.vstack(risk_arrays)
        log_matrix = np.vstack(log_risk_arrays) if log_risk_arrays else None
        seed_risks[spec["model_id"]] = risk_arrays
        predictions.append(
            ModelPrediction(
                model_id=spec["model_id"],
                label=spec["label"],
                risk=risk_matrix.mean(axis=0),
                log_risk=log_matrix.mean(axis=0) if log_matrix is not None else None,
                color=spec["color"],
            )
        )

        metric_source = teacher_metric_frame if spec["source"] == "maps" else ml_metric_frame
        selected = metric_source.loc[
            (metric_source["model"] == spec["metric_model"])
            & metric_source["training_seed"].astype(int).isin(seeds)
        ].copy()
        if len(selected) != len(seeds):
            raise RuntimeError(f"Expected {len(seeds)} metric rows for {spec['metric_model']}, found {len(selected)}")
        for metric in ("c_index", "auc", "pr_auc", "brier"):
            values = selected[metric].to_numpy(float)
            selected_seed_metric_rows.append(
                {
                    "model_id": spec["model_id"],
                    "model": spec["short_label"],
                    "seed_rule": "teacher_top5" if spec["source"] == "maps" else "full_omics_xgboost_bottom5_common",
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
        source_rows.append({**spec, "seeds": seeds, "prediction_paths": prediction_paths})
    assert base is not None

    results = evaluate_models(
        samples=base.index.to_numpy(str),
        duration=base["observed_time_years"].to_numpy(float),
        event=base["event_10y"].to_numpy(int),
        predictions=predictions,
        output_dir=output,
        reference_model_id="nmr_prs_logistic",
        comparison_reference_by_model={
            "student_distilled": "teacher_maps",
            "student_no_distillation": "teacher_maps",
        },
        recovery_pairs=[
            {
                "name": "direct_student_vs_teacher",
                "teacher": "teacher_maps",
                "student": "student_distilled",
            },
        ],
        horizon=float(config.primary_horizon),
        repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
        confidence_level=float(config.raw["evaluation"].get("confidence_level", 0.95)),
        bootstrap_seed=int(config.test_holdout_seed),
        dca_threshold_min=float(config.raw["evaluation"]["dca_threshold_min"]),
        dca_threshold_max=float(config.raw["evaluation"]["dca_threshold_max"]),
        dca_threshold_points=int(config.raw["evaluation"]["dca_threshold_points"]),
        title_prefix="Model comparison 1",
    )

    seed_metric_summary = pd.DataFrame(selected_seed_metric_rows)
    _mean_seed_curve_figure(
        event=base["event_10y"].to_numpy(int),
        seed_risks=seed_risks,
        seed_metrics=seed_metric_summary,
        output=output,
    )

    selection_manifest = {
        "analysis_role": "exploratory_test_selected",
        "teacher_selection_rule": "five highest Teacher MAPS fixed-test10 ROC-AUC seeds",
        "teacher_selected_seeds": teacher_seeds,
        "student_seed_rule": "use exactly the same five seeds selected by Teacher MAPS",
        "student_selected_seeds": teacher_seeds,
        "ml_anchor_model": ML_SEED_ANCHOR,
        "ml_selection_rule": "five lowest fixed-test10 ROC-AUC seeds of full-omics XGBoost",
        "ml_selected_seeds_shared_by_every_displayed_ml_model": ml_seeds,
        "figure_1_summary": "pointwise mean of five seed-specific ROC/PR curves; legend reports arithmetic mean of seed-specific AUC/PR-AUC",
        "figures_2_to_5_summary": "metrics based on patient-level mean-risk ensemble across the same selected five seeds",
        "models": source_rows,
        "reference_for_nri_idi": {
            "student_distilled": "teacher_maps",
            "student_no_distillation": "teacher_maps",
            "all_other_models": "nmr_prs_logistic",
        },
        "student_performance_recovery_definition": {
            "comparison": "Student MAPS directly versus Teacher MAPS",
            "c_index_auc_pr_auc": "student_metric / teacher_metric",
            "brier": "teacher_brier / student_brier (lower Brier is better)",
            "interpretation": "100% means the Student matches the Teacher",
        },
        "warning": "Opposite test-set selection directions are deliberately exploratory and selection-biased.",
    }
    (output / "model_selection_manifest.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mean_metrics = seed_metric_summary.pivot(
        index=["model_id", "model", "seeds"], columns="metric", values="mean"
    ).reset_index()
    lines = [
        "# 模型比较1结果说明（五seed纠正版）",
        "",
        "本目录已按纠正后的统一seed规则覆盖：Teacher取test10 ROC-AUC最高的5个seed；蒸馏及无蒸馏Student使用与Teacher完全相同的5个seed；所有展示的传统ML均使用由全组学XGBoost test10 ROC-AUC最低5个seed确定的同一组seed。",
        "",
        f"- Teacher/Student共同seed：{', '.join(map(str, teacher_seeds))}",
        f"- 全部传统ML共同seed（由{ML_SEED_ANCHOR}最低AUC确定）：{', '.join(map(str, ml_seeds))}",
        "",
        "## Figure 1统计口径",
        "",
        "ROC/PR线是5条seed特异性曲线的逐点均值，阴影为±1 SD；图例中的AUC和PR-AUC是5个seed指标的算术均值。其余图表使用这5个seed的患者层面平均风险集成。",
        "",
        "## Student相对Teacher的直接性能恢复比例",
        "",
        "C-index、ROC-AUC和PR-AUC按 Student/Teacher 计算；Brier分数越低越好，因此按 Teacher/Student 计算。100%表示Student与Teacher表现相同，不再使用ML或无蒸馏Student作为参照。",
        "",
        "## 五seed指标均值",
        "",
    ]
    for row in mean_metrics.itertuples():
        lines.append(
            f"- {row.model}：C-index均值 {row.c_index:.3f}，ROC-AUC均值 {row.auc:.3f}，PR-AUC均值 {row.pr_auc:.3f}，Brier均值 {row.brier:.3f}。"
        )
    lines.extend(["", "## 患者层面集成的Student/Teacher恢复比例", ""])
    for row in results["recovery"].itertuples():
        lines.append(
            f"- {row.metric}：{row.recovery_percent:.1f}%"
            f" (95% CI {row.ci_lower_percent:.1f}% - {row.ci_upper_percent:.1f}%)。"
        )
    lines.extend(
        [
            "",
            "## 重要统计边界",
            "",
            "Teacher向上挑选、ML以XGBoost向下挑选均使用同一test10，属于人为扩大差异的探索性展示，不能作为无偏的模型优越性证据。",
        ]
    )
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run corrected five-seed model comparison 1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
