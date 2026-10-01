"""XGBoost tuned on each split's val10, as a fairness check for BINN comparisons.

For every split and max_depth in {2, 3, 4}: up to 2,000 trees at learning rate 0.03 with
early stopping (50 rounds) on val10 log-loss; the depth with the lowest val10 log-loss is
selected. Other settings follow the frozen baseline (subsample 0.8, colsample_bytree 0.8,
reg_lambda 1, no class weighting). Features: NMR + Olink + PRS. test10 is reported.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.metrics import harrell_c_index
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.splits import stratified_split

STUDY = Path(__file__).resolve().parents[2]
_STATE: dict = {}


def run_task(task: tuple[str, int, int]) -> dict:
    split, depth, threads = task
    train, val, test = _STATE["splits"][split]
    x, y = _STATE["x"][split], _STATE["y"]
    model = XGBClassifier(
        n_estimators=2000, max_depth=depth, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=50,
        random_state=7, n_jobs=threads,
    )
    started = time.time()
    model.fit(x[train], y[train], eval_set=[(x[val], y[val])], verbose=False)
    val_risk = model.predict_proba(x[val])[:, 1]
    risk = model.predict_proba(x[test])[:, 1]
    return {
        "split": split, "max_depth": depth, "best_iteration": int(model.best_iteration),
        "val_logloss_10y": float(log_loss(y[val], val_risk)), "val_auc_10y": float(roc_auc_score(y[val], val_risk)),
        "auc_10y": float(roc_auc_score(y[test], risk)), "pr_auc_10y": float(average_precision_score(y[test], risk)),
        "brier_10y": float(brier_score_loss(y[test], risk)),
        "c_index": harrell_c_index(_STATE["duration"][test], y[test], risk),
        "fit_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "xgboost_tuned"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
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
    _STATE.update(
        splits=splits, y=data.event.astype(int), duration=data.duration_years,
        x={name: fit_preprocessor(replace(data, idx_train=a, idx_val=b, idx_test=c), omics).matrix(features)
           for name, (a, b, c) in splits.items()},
    )
    print(f"[{(time.time() - started) / 60:.1f} min] preprocessed {len(splits)} splits", flush=True)
    tasks = [(split, depth, args.threads) for split in splits for depth in sorted(args.depths, reverse=True)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for future in as_completed([pool.submit(run_task, task) for task in tasks]):
            rows.append(future.result())
            pd.DataFrame(rows).to_csv(output / "xgboost_tuned_all_depths_by_split.csv", index=False)
            r = rows[-1]
            print(f"[{(time.time() - started) / 60:.1f} min] {len(rows)}/{len(tasks)} {r['split']} depth={r['max_depth']} "
                  f"trees={r['best_iteration'] + 1} valLL={r['val_logloss_10y']:.5f} AUC10={r['auc_10y']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    selected = frame.sort_values("val_logloss_10y").groupby("split").head(1).sort_values("split")
    selected.to_csv(output / "xgboost_tuned_selected_by_split.csv", index=False)
    random = selected.loc[selected["split"].ne("frozen")]
    print("\nselected depth counts:", random["max_depth"].value_counts().to_dict())
    print(random[["auc_10y", "pr_auc_10y", "brier_10y", "c_index", "val_logloss_10y"]].agg(["mean", "std"]).to_string(
        float_format=lambda v: f"{v:.4f}"))
    (output / "xgboost_tuned_manifest.json").write_text(json.dumps({
        "purpose": "XGBoost tuned on val10 (depth and early stopping) as a fairness check",
        "depths": args.depths, "split_seeds": args.split_seeds, "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
