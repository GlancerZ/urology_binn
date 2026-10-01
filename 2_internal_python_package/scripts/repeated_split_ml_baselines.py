"""Conventional ML baselines on the same event-stratified 80/10/10 splits as the Teacher.

Uses the package's frozen classifier definitions (ml.build_classifier) and modalities,
fits on train80 participants with known 10-year status, and evaluates on test10.
Split "frozen" is the protocol split; random_XX is splits.stratified_split(event, XX),
exactly as in repeated_split_validation.py. Tasks run in forked worker processes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.endpoint import fixed_horizon_status
from maps_discrete.metrics import harrell_c_index
from maps_discrete.ml import ALGORITHMS, MODALITIES, build_classifier, modality_features
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.splits import stratified_split

STUDY = Path(__file__).resolve().parents[2]
# scikit-learn >= 1.8 derives the penalty from l1_ratio; older versions ignore l1_ratio
# unless penalty="elasticnet", which would silently turn elastic_net into L2 logistic.
PENALTY_FROM_L1_RATIO = tuple(int(part) for part in sklearn.__version__.split(".")[:2]) >= (1, 8)
TASK_COST = {"svm_rbf": 6, "elastic_net": 5, "random_forest": 4, "xgboost": 3, "lightgbm": 2, "logistic": 1}
MODALITY_COST = {"nmr_olink_prs": 4, "nmr_prs": 2, "nmr_only": 2, "prs_only": 1}
_STATE: dict = {}  # filled in the parent before the worker processes are forked


def make_classifier(algorithm: str, seed: int, threads: int, svm_probability: bool):
    model = build_classifier(algorithm, seed)
    if algorithm == "elastic_net" and not PENALTY_FROM_L1_RATIO:
        model.set_params(penalty="elasticnet")
    if algorithm == "svm_rbf" and not svm_probability:
        # Platt scaling is a monotone map of the decision function, so AUC, PR-AUC and
        # C-index are unchanged; skipping its internal 5-fold CV makes SVM ~6x faster.
        model.set_params(probability=False)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=threads)
    return model


def run_task(task: tuple[str, str, str, int, int, bool]) -> dict:
    split, algorithm, modality, seed, threads, svm_probability = task
    data = _STATE["data"]
    train, _, test = _STATE["splits"][split]
    matrix = _STATE["matrices"][split]
    columns = [_STATE["column"][feature] for feature in modality_features(data, modality)]
    known, outcome = fixed_horizon_status(data.duration_years, data.event, 10)
    fit_rows = train[known[train]]
    started = time.time()
    with warnings.catch_warnings(), threadpool_limits(limits=threads):
        warnings.simplefilter("ignore")
        model = make_classifier(algorithm, seed, threads, svm_probability)
        model.fit(matrix[np.ix_(fit_rows, columns)], outcome[fit_rows])
        probability = algorithm != "svm_rbf" or svm_probability
        test_matrix = matrix[np.ix_(test, columns)]
        risk = model.predict_proba(test_matrix)[:, 1] if probability else model.decision_function(test_matrix)
    fit_seconds = time.time() - started
    duration, event = data.duration_years[test], data.event[test]
    test_known, test_outcome = fixed_horizon_status(duration, event, 10)
    return {
        "split": split, "algorithm": algorithm, "modality": modality, "seed": seed,
        "score": "probability" if probability else "decision_function",
        "n_features": len(columns), "train_n": int(len(fit_rows)), "test_n": int(len(test)),
        "c_index": harrell_c_index(duration, event, risk),
        "auc_10y": float(roc_auc_score(test_outcome[test_known], risk[test_known])),
        "pr_auc_10y": float(average_precision_score(test_outcome[test_known], risk[test_known])),
        "brier_10y": float(brier_score_loss(test_outcome[test_known], risk[test_known])) if probability else np.nan,
        "observed_expected_10y": (
            float(test_outcome[test_known].sum() / risk[test_known].sum()) if probability else np.nan
        ),
        "fit_seconds": fit_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "ml_baselines"))
    parser.add_argument("--teacher-results", default=str(STUDY / "14_repeated_split_validation" / "test10_metrics_by_split.csv"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--svm-probability", action="store_true",
        help="keep the frozen Platt-scaled SVM probabilities (about 6x slower; adds Brier for SVM)",
    )
    args = parser.parse_args()

    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    data = load_internal_cohort(config, include_reactome=False)
    omics = list(dict.fromkeys([*data.nmr_features, *data.protein_features]))
    features = [*omics, "PRS"]
    splits = {"frozen": (data.idx_train, data.idx_val, data.idx_test)}
    splits.update({f"random_{seed:02d}": stratified_split(data.event, seed) for seed in args.split_seeds})
    matrices = {}
    for name, (train, val, test) in splits.items():
        fitted = fit_preprocessor(replace(data, idx_train=train, idx_val=val, idx_test=test), omics)
        matrices[name] = fitted.matrix(features)
    _STATE.update(data=data, splits=splits, matrices=matrices, column={f: i for i, f in enumerate(features)})
    print(f"[{(time.time() - started) / 60:.1f} min] preprocessed {len(splits)} splits; "
          f"sklearn {sklearn.__version__}, elastic net penalty set explicitly: {not PENALTY_FROM_L1_RATIO}", flush=True)

    tasks = [
        (split, algorithm, modality, args.seed, args.threads, args.svm_probability)
        for split in splits for algorithm in args.algorithms for modality in args.modalities
    ]
    tasks.sort(key=lambda t: TASK_COST[t[1]] * MODALITY_COST[t[2]], reverse=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "ml_test10_metrics_by_split.csv", index=False)
            print(f"[{(time.time() - started) / 60:.1f} min] {len(rows)}/{len(tasks)} {row['split']} "
                  f"{row['algorithm']}/{row['modality']} AUC10={row['auc_10y']:.4f} "
                  f"PR10={row['pr_auc_10y']:.4f} fit={row['fit_seconds']:.0f}s", flush=True)

    frame = pd.DataFrame(rows)
    random = frame.loc[frame["split"].ne("frozen")]
    metrics = ["auc_10y", "pr_auc_10y", "c_index", "brier_10y", "observed_expected_10y", "fit_seconds"]
    summary = random.groupby(["modality", "algorithm"])[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    frozen = frame.loc[frame["split"].eq("frozen")].set_index(["modality", "algorithm"])[["auc_10y", "pr_auc_10y"]]
    summary = summary.join(frozen.add_prefix("frozen_split_")).sort_values("auc_10y_mean", ascending=False)
    summary.to_csv(output / "ml_summary_random_splits.csv")

    teacher_path = Path(args.teacher_results)
    if teacher_path.is_file():
        teacher = pd.read_csv(teacher_path).set_index("split")
        paired = []
        for (modality, algorithm), group in random.groupby(["modality", "algorithm"]):
            group = group.set_index("split")
            for metric in ("auc_10y", "pr_auc_10y"):
                delta = teacher.loc[group.index, f"teacher_{metric}"] - group[metric]
                paired.append({
                    "modality": modality, "algorithm": algorithm, "metric": metric,
                    "teacher_minus_model_mean": float(delta.mean()), "sd": float(delta.std()),
                    "share_teacher_better": float((delta > 0).mean()), "n_splits": int(len(delta)),
                })
        pd.DataFrame(paired).to_csv(output / "teacher_vs_ml_paired_differences.csv", index=False)
    (output / "ml_manifest.json").write_text(json.dumps({
        "purpose": "frozen ML baselines on the Teacher's repeated splits (test10 evaluation)",
        "split_seeds": args.split_seeds,
        "classifier_seed": args.seed,
        "sklearn": sklearn.__version__,
        "elastic_net_penalty_set_explicitly": not PENALTY_FROM_L1_RATIO,
        "svm_score": "Platt probability" if args.svm_probability else "decision_function (no Brier/calibration)",
        "fit_population": "train80 with known 10-year status",
        "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["auc_10y_mean", "auc_10y_std", "pr_auc_10y_mean", "pr_auc_10y_std", "brier_10y_mean", "fit_seconds_mean"]
    print("\nRandom splits, test10 (sorted by mean AUC):")
    print(summary[show].to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
