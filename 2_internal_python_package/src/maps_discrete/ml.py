from __future__ import annotations

import json
from pathlib import Path
import time
import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import AnalysisConfig
from .data import CohortData, load_internal_cohort
from .endpoint import fixed_horizon_status
from .metrics import decision_curve, evaluate_model
from .preprocessing import fit_preprocessor
from .seeding import set_global_seed
from .splits import audit_split_manifest


ALGORITHMS = ("xgboost", "lightgbm", "random_forest", "logistic", "elastic_net", "svm_rbf")
MODALITIES = ("nmr_only", "prs_only", "nmr_prs", "nmr_olink_prs")


def modality_features(data: CohortData, modality: str) -> list[str]:
    groups = {
        "nmr_only": list(data.nmr_features),
        "prs_only": ["PRS"],
        "nmr_prs": [*data.nmr_features, "PRS"],
        "nmr_olink_prs": [*data.nmr_features, *data.protein_features, "PRS"],
    }
    if modality not in groups:
        raise ValueError(f"Unknown modality {modality!r}; choose from {sorted(groups)}")
    return list(dict.fromkeys(groups[modality]))


def build_classifier(algorithm: str, seed: int):
    """Frozen conventional classifiers; no test information enters fitting."""
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install the package with the [ml] extra for XGBoost") from exc
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("Install the package with the [ml] extra for LightGBM") from exc
        return LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1,
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if algorithm == "logistic":
        return LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", class_weight="balanced",
            max_iter=5000,
        )
    if algorithm == "elastic_net":
        return LogisticRegression(
            C=1.0, l1_ratio=0.5, solver="saga",
            class_weight="balanced", max_iter=5000, random_state=seed,
        )
    if algorithm == "svm_rbf":
        return SVC(
            C=1.0, gamma="scale", kernel="rbf", probability=True,
            class_weight="balanced", cache_size=4096, random_state=seed,
        )
    raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}")


def _feature_matrix(data: CohortData, features: list[str]) -> np.ndarray:
    """Use the same train-only omics preprocessing as MAPS."""
    required_omics = [feature for feature in features if feature != "PRS"]
    fitted = fit_preprocessor(data, required_omics)
    return fitted.matrix(features)


def run_ml_seed(
    config: AnalysisConfig,
    training_seed: int,
    output_dir: str | Path,
    modalities: tuple[str, ...] = MODALITIES,
    algorithms: tuple[str, ...] = ALGORITHMS,
    quick: bool = False,
) -> dict:
    """Fit all frozen ML baselines for one training seed and save test metrics."""
    started = time.time()
    training_seed = int(training_seed)
    if training_seed not in config.training_seeds:
        raise ValueError(f"Training seed {training_seed} is absent from the frozen grid")
    set_global_seed(training_seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_internal_cohort(config, include_reactome=False)
    horizon = config.primary_horizon
    known, y_all = fixed_horizon_status(data.duration_years, data.event, horizon)
    train_index = data.idx_train[known[data.idx_train]]
    validation_index = data.idx_val[known[data.idx_val]]
    test_index = data.idx_test
    if len(validation_index) == 0:
        raise RuntimeError("No known-status validation participants")

    evaluation = config.raw["evaluation"]
    thresholds = np.linspace(
        float(evaluation["dca_threshold_min"]),
        float(evaluation["dca_threshold_max"]),
        int(evaluation["dca_threshold_points"]),
    )
    metrics_rows: list[dict] = []
    prediction_frame = pd.DataFrame({
        "sample": data.samples,
        "split": data.split_labels,
        "event_10y": data.event,
        "observed_time_years": data.duration_years,
    })
    dca_frames = []
    model_parameters: dict[str, dict] = {}
    for modality in modalities:
        features = modality_features(data, modality)
        matrix = _feature_matrix(data, features)
        for algorithm in algorithms:
            model_id = f"{modality}__{algorithm}"
            print(f"ML seed={training_seed} model={model_id}", flush=True)
            model_started = time.time()
            classifier = build_classifier(algorithm, training_seed)
            classifier.fit(matrix[train_index], y_all[train_index])
            risk = classifier.predict_proba(matrix)[:, 1]
            model_parameters[model_id] = {
                key: value for key, value in classifier.get_params(deep=False).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            prediction_frame[f"{model_id}_risk_10y"] = risk
            result = evaluate_model(
                data.duration_years[test_index], data.event[test_index],
                risk[test_index], risk[test_index], horizon,
            )
            result.update({
                "training_seed": training_seed,
                "model": model_id,
                "modality": modality,
                "algorithm": algorithm,
                "dataset": "test10",
                "n_features": len(features),
            })
            metrics_rows.append(result)
            curve = decision_curve(
                data.duration_years[test_index], data.event[test_index],
                risk[test_index], thresholds, horizon,
            )
            curve.insert(0, "model", model_id)
            curve.insert(0, "training_seed", training_seed)
            dca_frames.append(curve)
            print(
                f"ML seed={training_seed} model={model_id} completed "
                f"minutes={(time.time() - model_started) / 60.0:.2f}",
                flush=True,
            )
            del classifier
            gc.collect()
    pd.DataFrame(metrics_rows).to_csv(output / "ml_test_metrics.csv", index=False)
    prediction_frame.to_csv(output / "ml_predictions_all.csv", index=False)
    prediction_frame.loc[prediction_frame["split"].eq("test10")].to_csv(
        output / "ml_predictions_test.csv", index=False
    )
    pd.concat(dca_frames, ignore_index=True).to_csv(output / "ml_decision_curve_test.csv", index=False)
    split_audit = audit_split_manifest(config.raw["data"]["source_split_manifest"])
    manifest = {
        "status": "quick_smoke" if quick else "completed",
        "training_seed": training_seed,
        "test_holdout_seed": config.test_holdout_seed,
        "train_validation_split_seed": config.train_validation_split_seed,
        "split_sha256": split_audit.sha256,
        "horizon_years": horizon,
        "modalities": list(modalities),
        "algorithms": list(algorithms),
        "model_parameters": model_parameters,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output / "ml_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
