from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation import (
    ModelPrediction,
    _plot_auc_pr_forest,
    _plot_calibration,
    _plot_dca,
    _plot_difference_forest,
    _plot_roc_pr,
)
from .model_comparison_one import MODEL_SELECTIONS


def run(study_root: str | Path) -> None:
    root = Path(study_root)
    output = root / "7_model_comparison_1"
    figures = output / "2_figures"
    tables = output / "1_tables"
    plot_data = output / "3_plot_data"
    aligned = pd.read_csv(plot_data / "aligned_predictions.csv", dtype={"sample": str})

    predictions = [
        ModelPrediction(
            model_id=spec["model_id"],
            label=spec["label"],
            risk=aligned[f"{spec['model_id']}_risk_10y"].to_numpy(float),
            log_risk=aligned[f"{spec['model_id']}_ranking_score"].to_numpy(float),
            color=spec["color"],
        )
        for spec in MODEL_SELECTIONS
    ]
    performance = pd.read_csv(tables / "Table_1_model_performance_95ci.csv")
    comparison = pd.read_csv(tables / "Table_2_model_differences_nri_idi_95ci.csv")
    dca = pd.read_csv(plot_data / "dca_curve_data.csv")
    calibration = pd.read_csv(plot_data / "calibration_curve_data.csv")
    y = aligned["event_10y"].to_numpy(int)
    evaluation_index = np.arange(len(aligned))

    _plot_roc_pr(figures, "Model comparison 1", predictions, y, evaluation_index, performance)
    _plot_dca(figures, "Model comparison 1", predictions, dca)
    _plot_difference_forest(figures, "Model comparison 1", comparison)
    _plot_auc_pr_forest(figures, "Model comparison 1", performance)
    _plot_calibration(figures, "Model comparison 1", predictions, calibration)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Re-render model comparison 1 figures from saved statistics")
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()
    run(args.study_root)


if __name__ == "__main__":
    main()
