"""Summarise Student variants on the repeated splits: single seeds, 5-seed ensembles, and paired
comparisons with the NMR + PRS / PRS-only ML baselines (same splits). Selection uses val10 log-loss;
p-values use the Nadeau-Bengio corrected resampled t-test."""

from __future__ import annotations

import glob
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
VARIANTS = ("frozen_nodistill", "frozen_distilled", "binary", "binary_distilled")
SEEDS = (7, 1, 2, 3, 4)


def corrected_p(d: np.ndarray, n_test: int = 1726, n_train: int = 13807) -> float:
    t = d.mean() / np.sqrt((1 / len(d) + n_test / n_train) * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), len(d) - 1))


def main() -> None:
    config = load_config(STUDY / "3_frozen_configs" / "analysis_config.toml")
    data = load_internal_cohort(config, include_reactome=False)
    y, duration = data.event, data.duration_years
    splits = {"frozen": (data.idx_train, data.idx_val, data.idx_test)}
    splits.update({f"random_{s:02d}": stratified_split(data.event, s) for s in range(1, 21)})
    rows = []
    for variant in VARIANTS:
        for name, (_, val, test) in splits.items():
            members = [np.load(ROOT / "students" / "risks" / f"{variant}_{name}_seed{s}.npz") for s in SEEDS]
            t, v = np.mean([m["test"] for m in members], 0), np.mean([m["val"] for m in members], 0)
            rows.append({
                "variant": variant, "split": name,
                "single_auc": np.mean([roc_auc_score(y[test], m["test"]) for m in members]),
                "single_pr": np.mean([average_precision_score(y[test], m["test"]) for m in members]),
                "auc_10y": roc_auc_score(y[test], t), "pr_auc_10y": average_precision_score(y[test], t),
                "brier_10y": brier_score_loss(y[test], t), "c_index": harrell_c_index(duration[test], y[test], t),
                "val_logloss_10y": log_loss(y[val], np.clip(v, 1e-7, 1 - 1e-7)),
            })
    ens = pd.DataFrame(rows)
    ens.to_csv(ROOT / "students" / "student_ensembles_by_split.csv", index=False)
    random = ens[ens.split != "frozen"]
    summary = random.groupby("variant")[["val_logloss_10y", "single_auc", "single_pr", "auc_10y", "pr_auc_10y",
                                         "brier_10y", "c_index"]].mean().sort_values("val_logloss_10y")
    print("5-seed Student ensembles, 20 random splits (sorted by val10 log-loss):")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    selected = summary.index[0]
    print(f"\nselected on val10 log-loss: {selected}")

    ml = pd.read_csv(ROOT / "ml_baselines" / "ml_test10_metrics_by_split.csv")
    refs = {
        "logistic NMR+PRS": ml[(ml.algorithm == "logistic") & (ml.modality == "nmr_prs")],
        "elastic_net NMR+PRS": ml[(ml.algorithm == "elastic_net") & (ml.modality == "nmr_prs")],
        "xgboost NMR+PRS": ml[(ml.algorithm == "xgboost") & (ml.modality == "nmr_prs")],
        "logistic PRS only": ml[(ml.algorithm == "logistic") & (ml.modality == "prs_only")],
    }
    print("\nreferences (20 random splits):")
    for name, frame in refs.items():
        r = frame[frame.split != "frozen"]
        print(f"  {name:22s} AUC {r.auc_10y.mean():.4f}  PR {r.pr_auc_10y.mean():.4f}")
    out = []
    for variant in VARIANTS:
        cand = random[random.variant == variant].set_index("split")
        for name, frame in refs.items():
            ref = frame.set_index("split").loc[cand.index]
            for metric in ("auc_10y", "pr_auc_10y"):
                d = (cand[metric] - ref[metric]).to_numpy()
                out.append({"student": variant, "reference": name, "metric": metric, "mean_diff": d.mean(),
                            "sd": d.std(ddof=1), "student_better_%": 100 * (d > 0).mean(), "corrected_p": corrected_p(d)})
    comparison = pd.DataFrame(out)
    comparison.to_csv(ROOT / "students" / "student_vs_ml_paired_differences.csv", index=False)
    print("\npaired differences (Student ensemble - reference), AUC:")
    print(comparison[comparison.metric == "auc_10y"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = random[random.variant == "frozen_nodistill"].set_index("split")
    print("\ndistillation effect (ensemble AUC, distilled - no distillation):")
    for a, b in (("frozen_distilled", "frozen_nodistill"), ("binary_distilled", "binary")):
        d = (random[random.variant == a].set_index("split").auc_10y - random[random.variant == b].set_index("split").auc_10y).to_numpy()
        print(f"  {a} - {b}: {d.mean():+.4f} (better in {100*(d>0).mean():.0f}%, corrected p {corrected_p(d):.2f})")


if __name__ == "__main__":
    main()
