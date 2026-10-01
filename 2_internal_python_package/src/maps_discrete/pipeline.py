from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import AnalysisConfig
from .data import CohortData, load_internal_cohort
from .metrics import decision_curve, evaluate_model
from .modeling import build_model, pathway_layer_sizes, routing_frame
from .prediction import PredictionBundle, predict
from .preprocessing import fit_preprocessor
from .seeding import set_global_seed
from .splits import audit_split_manifest
from .training import FitResult, TrainingSettings, fit_model


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def settings_from_config(config: AnalysisConfig, quick: bool = False) -> TrainingSettings:
    train = config.raw["training"]
    distill = config.raw["distillation"]
    return TrainingSettings(
        batch_size=int(train["batch_size"]),
        max_epochs=2 if quick else int(train["max_epochs"]),
        patience=1 if quick else int(train["patience"]),
        min_delta=float(train["min_delta"]),
        learning_rate=float(train["learning_rate"]),
        binn_weight_decay=float(train["binn_weight_decay"]),
        gradient_clip_norm=float(train["gradient_clip_norm"]),
        distillation_alpha=float(distill["alpha"]),
        distillation_temperature=float(distill["temperature"]),
    )


def _save_preprocessing(output: Path, fitted) -> None:
    frame = pd.DataFrame({
        "feature": fitted.means.index,
        "feature_type": [fitted.feature_types.get(x, "Unknown") for x in fitted.means.index],
        "fill_value_train": [fitted.fill_values[x] for x in fitted.means.index],
        "mean_train": fitted.means.to_numpy(),
        "standard_deviation_train": fitted.standard_deviations.to_numpy(),
    })
    frame.to_csv(output / "preprocessing_parameters.csv", index=False)


def _save_prediction_frame(
    output: Path,
    data: CohortData,
    predictions: dict[str, PredictionBundle],
) -> None:
    frame = pd.DataFrame({
        "sample": data.samples,
        "split": data.split_labels,
        "event_10y": data.event,
        "observed_time_years": data.duration_years,
    })
    for name, bundle in predictions.items():
        frame[f"{name}_log_risk"] = bundle.log_risk
        for year in range(1, bundle.cumulative_incidence.shape[1] + 1):
            frame[f"{name}_risk_{year}y"] = bundle.cumulative_incidence[:, year - 1]
    frame.to_csv(output / "predictions_all.csv", index=False)
    frame.loc[frame["split"].eq("test10")].to_csv(output / "predictions_test.csv", index=False)


def _test_metrics(
    data: CohortData,
    predictions: dict[str, PredictionBundle],
    horizon: int,
) -> pd.DataFrame:
    index = data.idx_test
    rows = []
    for name, bundle in predictions.items():
        result = evaluate_model(
            data.duration_years[index],
            data.event[index],
            bundle.log_risk[index],
            bundle.cumulative_incidence[index, horizon - 1],
            horizon,
        )
        result.update({"model": name, "dataset": "test10"})
        rows.append(result)
    return pd.DataFrame(rows)


def _save_dca(
    output: Path,
    data: CohortData,
    predictions: dict[str, PredictionBundle],
    config: AnalysisConfig,
) -> None:
    evaluation = config.raw["evaluation"]
    horizon = config.primary_horizon
    thresholds = np.linspace(
        float(evaluation["dca_threshold_min"]),
        float(evaluation["dca_threshold_max"]),
        int(evaluation["dca_threshold_points"]),
    )
    index = data.idx_test
    frames = []
    for name, bundle in predictions.items():
        frame = decision_curve(
            data.duration_years[index],
            data.event[index],
            bundle.cumulative_incidence[index, horizon - 1],
            thresholds,
            horizon,
        )
        frame.insert(0, "model", name)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output / "decision_curve_test.csv", index=False)


def run_maps_seed(
    config: AnalysisConfig,
    training_seed: int,
    output_dir: str | Path,
    quick: bool = False,
    include_supervised_student: bool = True,
) -> dict:
    """Train the discrete Teacher, distilled Student and optional no-KD Student."""
    started = time.time()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_seed = int(training_seed)
    if training_seed not in config.training_seeds:
        raise ValueError(f"Training seed {training_seed} is absent from the frozen seed grid")
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, quick)
    data = load_internal_cohort(config)
    (output / "endpoint_audit.json").write_text(
        json.dumps(data.audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    teacher_matrix = data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy()
    student_matrix = data.raw.loc[data.raw[data.entity_col].isin(data.nmr_features)].copy()
    model_kwargs = {
        "mapping": data.mapping,
        "pathways": data.pathways,
        "entity_col": data.entity_col,
        "binn_root": config.raw["data"]["binn_root"],
        "device": device,
        "n_layers": int(config.raw["training"]["n_layers"]),
        "n_intervals": config.n_intervals,
        "attention_dim": int(config.raw["training"]["attention_dim"]),
    }

    set_global_seed(training_seed)
    teacher = build_model(data_matrix=teacher_matrix, **model_kwargs)
    teacher_features = list(teacher.binn.inputs)
    fitted = fit_preprocessor(data, teacher_features)
    _save_preprocessing(output, fitted)
    teacher_x = fitted.matrix(teacher_features)
    prs = fitted.prs()
    teacher_dir = output / "1_teacher"
    teacher_fit = fit_model(
        teacher, teacher_x, prs, data.targets, data.interval_mask,
        data.idx_train, data.idx_val, device, teacher_dir, "teacher",
        training_seed, settings,
    )
    teacher_prediction = predict(teacher, teacher_x, prs, device)
    np.save(teacher_dir / "hazard_logits_all.npy", teacher_prediction.hazard_logits)
    np.save(teacher_dir / "attention_all.npy", teacher_prediction.attention)
    routing_frame(teacher).to_csv(teacher_dir / "input_routing.csv", index=False)

    set_global_seed(training_seed)
    student = build_model(data_matrix=student_matrix, **model_kwargs)
    student_features = list(student.binn.inputs)
    student_x = fitted.matrix(student_features)
    student_dir = output / "2_student_distilled"
    student_fit = fit_model(
        student, student_x, prs, data.targets, data.interval_mask,
        data.idx_train, data.idx_val, device, student_dir, "student_distilled",
        training_seed, settings, teacher_logits=teacher_prediction.hazard_logits,
    )
    student_prediction = predict(student, student_x, prs, device)
    np.save(student_dir / "hazard_logits_all.npy", student_prediction.hazard_logits)
    np.save(student_dir / "attention_all.npy", student_prediction.attention)
    routing_frame(student).to_csv(student_dir / "input_routing.csv", index=False)

    predictions = {
        "teacher": teacher_prediction,
        "student_distilled": student_prediction,
    }
    fits: dict[str, FitResult] = {"teacher": teacher_fit, "student_distilled": student_fit}
    layer_sizes = {
        "teacher": pathway_layer_sizes(teacher),
        "student_distilled": pathway_layer_sizes(student),
    }
    if include_supervised_student:
        set_global_seed(training_seed)
        supervised = build_model(data_matrix=student_matrix, **model_kwargs)
        supervised_dir = output / "3_student_no_distillation"
        supervised_fit = fit_model(
            supervised, student_x, prs, data.targets, data.interval_mask,
            data.idx_train, data.idx_val, device, supervised_dir, "student_no_distillation",
            training_seed, settings,
        )
        supervised_prediction = predict(supervised, student_x, prs, device)
        np.save(supervised_dir / "hazard_logits_all.npy", supervised_prediction.hazard_logits)
        np.save(supervised_dir / "attention_all.npy", supervised_prediction.attention)
        routing_frame(supervised).to_csv(supervised_dir / "input_routing.csv", index=False)
        predictions["student_no_distillation"] = supervised_prediction
        fits["student_no_distillation"] = supervised_fit
        layer_sizes["student_no_distillation"] = pathway_layer_sizes(supervised)

    metrics = _test_metrics(data, predictions, config.primary_horizon)
    metrics.insert(0, "training_seed", training_seed)
    metrics.to_csv(output / "test_metrics.csv", index=False)
    _save_prediction_frame(output, data, predictions)
    _save_dca(output, data, predictions, config)
    split_audit = audit_split_manifest(config.raw["data"]["source_split_manifest"])
    manifest = {
        "status": "quick_smoke" if quick else "completed",
        "training_seed": training_seed,
        "test_holdout_seed": config.test_holdout_seed,
        "train_validation_split_seed": config.train_validation_split_seed,
        "split_sha256": split_audit.sha256,
        "device": str(device),
        "config_path": str(config.path),
        "settings": asdict(settings),
        "fits": {name: asdict(fit) for name, fit in fits.items()},
        "pathway_layer_sizes": layer_sizes,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
