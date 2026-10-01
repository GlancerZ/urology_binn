"""Summarise the improved Teacher variants on the repeated splits.

Reads the per-run 10-year risks that repeated_split_binn_improvements.py saves for every training
seed, averages the 5 seeds per split, and writes
  <variant>_ensemble5_by_split.csv               ensemble metrics on test10 and val10 log-loss per split
  final_comparison_random_splits.csv             means over the 20 random splits, with the frozen
                                                 Teacher and XGBoost (default and val10-tuned)
  teacher_vs_references_paired_differences.csv   paired differences on the same splits with
                                                 Nadeau-Bengio corrected p-values and 95% CIs
Variant selection uses val10 log-loss only.

Run after (all on the same splits, results under 14_repeated_split_validation/):
  repeated_split_validation.py, repeated_split_ml_baselines.py, repeated_split_xgboost_tuned.py,
  repeated_split_binn_improvements.py --variants binary --training-seeds 7 1 2 3 4
  repeated_split_binn_improvements.py --variants binary_relaxed --training-seeds 7 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.metrics import harrell_c_index
from maps_discrete.splits import stratified_split

STUDY = Path(__file__).resolve().parents[2]
ROOT = STUDY / "14_repeated_split_validation"
VARIANTS = ("binary", "binary_relaxed")
SEEDS = (7, 1, 2, 3, 4)
METRICS = ["auc_10y", "pr_auc_10y", "brier_10y", "c_index"]


def corrected_test(d: np.ndarray, n_test: int = 1726, n_train: int = 13807) -> tuple[float, float, float]:
    """Nadeau-Bengio corrected resampled t-test: two-sided p and 95% CI of the mean difference."""
    se = np.sqrt((1 / len(d) + n_test / n_train) * d.var(ddof=1))
    half = stats.t.ppf(0.975, len(d) - 1) * se
    return float(2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)), float(d.mean() - half), float(d.mean() + half)


def ensemble(variant: str, splits: dict, y: np.ndarray, duration: np.ndarray, risks: Path) -> pd.DataFrame:
    rows = []
    for name, (_, val, test) in splits.items():
        members = [np.load(risks / f"{variant}_{name}_seed{s}.npz") for s in SEEDS]
        t, v = np.mean([m["test"] for m in members], 0), np.mean([m["val"] for m in members], 0)
        rows.append({
            "split": name, "auc_10y": roc_auc_score(y[test], t), "pr_auc_10y": average_precision_score(y[test], t),
            "brier_10y": brier_score_loss(y[test], t), "c_index": harrell_c_index(duration[test], y[test], t),
            "val_logloss_10y": log_loss(y[val], v),
            "single_auc": np.mean([roc_auc_score(y[test], m["test"]) for m in members]),
            "single_pr": np.mean([average_precision_score(y[test], m["test"]) for m in members]),
        })
    return pd.DataFrame(rows).set_index("split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(ROOT / "binn_improvements"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    data = load_internal_cohort(config, include_reactome=False)
    y, duration = data.event, data.duration_years
    splits = {"frozen": (data.idx_train, data.idx_val, data.idx_test)}
    splits.update({f"random_{s:02d}": stratified_split(data.event, s) for s in range(1, 21)})
    random = [name for name in splits if name != "frozen"]

    ens = {v: ensemble(v, splits, y, duration, ROOT / "binn_improvements" / "risks") for v in VARIANTS}
    for variant, frame in ens.items():
        frame.to_csv(output / f"{variant}_ensemble5_by_split.csv")
    val_loss = {v: frame.loc[random, "val_logloss_10y"].mean() for v, frame in ens.items()}
    print("5-seed ensembles, mean val10 log-loss over the 20 random splits: "
          + ", ".join(f"{v} {loss:.5f}" for v, loss in val_loss.items())
          + f" -> selected {min(val_loss, key=val_loss.get)}")

    teacher = pd.read_csv(ROOT / "test10_metrics_by_split.csv").set_index("split")
    frozen_teacher = teacher.rename(columns={f"teacher_{m}": m for m in METRICS})
    klk3_prs = teacher.rename(columns={f"klk3_prs_logistic_{m}": m for m in METRICS})
    ml = pd.read_csv(ROOT / "ml_baselines" / "ml_test10_metrics_by_split.csv")
    xgb = ml[(ml.algorithm == "xgboost") & (ml.modality == "nmr_olink_prs")].set_index("split")
    xgb_tuned = pd.read_csv(ROOT / "xgboost_tuned" / "xgboost_tuned_selected_by_split.csv").set_index("split")

    binary, relaxed = ens["binary"], ens["binary_relaxed"]
    table = pd.DataFrame({
        "frozen Teacher": frozen_teacher.loc[random, METRICS].mean(),
        "BINN 10y, 5-seed ens": binary.loc[random, METRICS].mean(),
        "BINN relaxed, single seed": pd.Series({"auc_10y": relaxed.loc[random, "single_auc"].mean(),
                                                "pr_auc_10y": relaxed.loc[random, "single_pr"].mean()}),
        "BINN relaxed, 5-seed ens": relaxed.loc[random, METRICS].mean(),
        "XGBoost default": xgb.loc[random, METRICS].mean(),
        "XGBoost tuned": xgb_tuned.loc[random, METRICS].mean(),
    }).T
    table.to_csv(output / "final_comparison_random_splits.csv")
    print("\nmeans over the 20 random splits (test10):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    references = {"frozen Teacher": frozen_teacher, "KLK3 + PRS logistic": klk3_prs,
                  "XGBoost default": xgb, "XGBoost tuned": xgb_tuned}
    rows = []
    for variant, frame in ens.items():
        for name, ref in references.items():
            for metric in ("auc_10y", "pr_auc_10y"):
                d = (frame.loc[random, metric] - ref.loc[random, metric]).to_numpy()
                p, lower, upper = corrected_test(d)
                rows.append({"candidate": f"{variant} 5-seed ensemble", "reference": name, "metric": metric,
                             "mean_diff": d.mean(), "sd": d.std(ddof=1), "candidate_better_%": 100 * (d > 0).mean(),
                             "corrected_ci_lower": lower, "corrected_ci_upper": upper, "corrected_p": p})
    paired = pd.DataFrame(rows)
    paired.to_csv(output / "teacher_vs_references_paired_differences.csv", index=False)
    print("\npaired differences (ensemble - reference), Nadeau-Bengio corrected:")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
