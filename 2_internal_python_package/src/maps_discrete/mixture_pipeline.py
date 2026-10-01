"""Single-seed runs and evaluation of the mixture cure Teacher.

Development runs never evaluate test10: the model is fitted on 90% of train80,
early-stopped on the remaining 10% and evaluated on val10. The final run
(final_test=True) follows the frozen protocol - fit on train80, early-stop on val10 -
and is the only mode that evaluates, or saves predictions for, test10.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import AnalysisConfig
from .data import CohortData, load_internal_cohort
from .endpoint import fixed_horizon_status
from .metrics import fixed_horizon_metrics, harrell_c_index
from .mixture_cure import (
    TIMING_HEADS,
    build_mixture_model,
    case_interval_distribution,
    fit_mixture_model,
    predict_mixture,
)
from .modeling import build_model
from .pipeline import _save_preprocessing, resolve_device, settings_from_config
from .prediction import predict
from .preprocessing import fit_preprocessor
from .seeding import set_global_seed
from .splits import audit_split_manifest
from .training import FitResult, TrainingSettings, fit_model

DEVELOPMENT_SPLIT_SEED = 20260924
TIMING_WINDOWS = ((0, 3), (3, 7), (7, 10))
# "ph" is the frozen proportional-hazards Teacher, kept for like-for-like comparison.
VARIANTS = ("ph", *TIMING_HEADS)


def development_split(data: CohortData, seed: int = DEVELOPMENT_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 90/10 split of train80; the 10% is used only for early stopping."""
    rng = np.random.default_rng(seed)
    held_out: list[int] = []
    for label in (0, 1):
        members = data.idx_train[data.event[data.idx_train] == label]
        held_out.extend(rng.choice(members, size=int(round(0.1 * len(members))), replace=False))
    stop = np.sort(np.asarray(held_out, dtype=int))
    return np.setdiff1d(data.idx_train, stop), stop


def onset_distribution(cumulative: np.ndarray) -> np.ndarray:
    """f(t | onset within the horizon) implied by a cumulative-incidence matrix."""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    increments = np.diff(np.column_stack([np.zeros(len(cumulative)), cumulative]), axis=1)
    return np.clip(increments, 0.0, None) / cumulative[:, -1:]


def _quantile_year(onset_probability: np.ndarray, q: float) -> np.ndarray:
    """Quantile of onset time, assuming a uniform onset time within each yearly interval."""
    cdf = np.cumsum(onset_probability, axis=1)
    k = np.minimum((cdf < q).sum(axis=1), onset_probability.shape[1] - 1)
    rows = np.arange(len(k))
    before = np.where(k > 0, cdf[rows, np.maximum(k - 1, 0)], 0.0)
    within = np.maximum(onset_probability[rows, k], 1e-12)
    return k + np.clip((q - before) / within, 0.0, 1.0)


def timing_summary(onset_probability: np.ndarray) -> pd.DataFrame:
    """Per-participant summaries of the onset-year distribution, given onset."""
    midpoints = np.arange(onset_probability.shape[1]) + 0.5
    frame = pd.DataFrame({
        "expected_onset_year": onset_probability @ midpoints,
        "median_onset_year": _quantile_year(onset_probability, 0.5),
        "onset_year_q10": _quantile_year(onset_probability, 0.1),
        "onset_year_q90": _quantile_year(onset_probability, 0.9),
    })
    for start, end in TIMING_WINDOWS:
        frame[f"p_onset_{start}_{end}y"] = onset_probability[:, start:end].sum(axis=1)
    return frame


def _pairwise_concordance(predicted_time: np.ndarray, true_time: np.ndarray) -> float:
    """Among cases: probability that the earlier onset also has the earlier predicted onset."""
    i, j = np.triu_indices(len(true_time), 1)
    keep = true_time[i] != true_time[j]
    agreement = (
        np.sign(true_time[i][keep] - true_time[j][keep])
        * np.sign(predicted_time[i][keep] - predicted_time[j][keep])
    )
    return float(np.mean(np.where(agreement == 0, 0.5, agreement > 0)))


def timing_metrics(
    duration: np.ndarray,
    event: np.ndarray,
    onset_probability: np.ndarray,
    reference_distribution: np.ndarray,
    horizon: int,
) -> dict[str, float | int]:
    """Onset-timing accuracy among participants with onset within the horizon.

    The reference is one onset-year distribution shared by everyone (training cases).
    """
    case = (np.asarray(event) == 1) & (np.asarray(duration) <= horizon)
    time_years = np.asarray(duration, dtype=float)[case]
    probability = np.asarray(onset_probability, dtype=float)[case]
    true_bin = np.clip(np.ceil(time_years).astype(int) - 1, 0, probability.shape[1] - 1)
    log_score = np.log(np.maximum(probability[np.arange(len(time_years)), true_bin], 1e-12))
    reference_log_score = np.log(reference_distribution[true_bin])
    summary = timing_summary(probability)
    reference_median = float(timing_summary(reference_distribution[None, :])["median_onset_year"].iloc[0])
    early = time_years <= 3
    either = early | (time_years > 7)
    lower = summary["onset_year_q10"].to_numpy()
    upper = summary["onset_year_q90"].to_numpy()
    return {
        "timing_cases": int(case.sum()),
        "timing_log_score": float(log_score.mean()),
        "timing_log_score_gain": float((log_score - reference_log_score).mean()),
        "timing_concordance": _pairwise_concordance(summary["expected_onset_year"].to_numpy(), time_years),
        "early_vs_late_auc": float(roc_auc_score(early[either], summary["p_onset_0_3y"].to_numpy()[either])),
        "median_year_mae": float(np.abs(summary["median_onset_year"].to_numpy() - time_years).mean()),
        "reference_median_year_mae": float(np.abs(reference_median - time_years).mean()),
        "interval80_coverage": float(np.mean((lower <= time_years) & (time_years <= upper))),
        "interval80_mean_width": float(np.mean(upper - lower)),
    }


def evaluate_cumulative(
    data: CohortData,
    cumulative: np.ndarray,
    index: np.ndarray,
    reference_distribution: np.ndarray,
    horizons: list[int],
) -> dict[str, float | int]:
    """Incidence discrimination/calibration and onset-timing metrics on one split."""
    duration = data.duration_years[index]
    event = data.event[index]
    risk = np.asarray(cumulative, dtype=float)[index]
    n_intervals = risk.shape[1]
    result: dict[str, float | int] = {
        "n": int(len(index)),
        "events": int(event.sum()),
        # Area under the cumulative-incidence curve ranks both "whether" and "when".
        "c_index": harrell_c_index(duration, event, risk.sum(axis=1)),
    }
    for horizon in horizons:
        metrics = fixed_horizon_metrics(duration, event, risk[:, horizon - 1], horizon)
        result.update({
            f"auc_{horizon}y": metrics["auc"],
            f"pr_auc_{horizon}y": metrics["pr_auc"],
            f"brier_{horizon}y": metrics["brier"],
        })
    known, outcome = fixed_horizon_status(duration, event, n_intervals)
    result[f"observed_expected_{n_intervals}y"] = float(outcome[known].sum() / risk[known, -1].sum())
    result.update(timing_metrics(duration, event, onset_distribution(risk), reference_distribution, n_intervals))
    return result


def build_teacher(
    config: AnalysisConfig,
    data: CohortData,
    variant: str,
    training_seed: int,
    device: torch.device,
):
    """Seeded construction of the proportional-hazards or a mixture cure Teacher."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    kwargs = {
        "data_matrix": data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy(),
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
    if variant == "ph":
        return build_model(**kwargs)
    return build_mixture_model(timing_head=variant, **kwargs)


def train_teacher(
    variant: str,
    model,
    x: np.ndarray,
    prs: np.ndarray,
    data: CohortData,
    fit_index: np.ndarray,
    stop_index: np.ndarray,
    device: torch.device,
    output_dir: str | Path,
    training_seed: int,
    settings: TrainingSettings,
) -> tuple[FitResult, np.ndarray, dict[str, np.ndarray]]:
    """Fit one Teacher; return the fit, cumulative incidence for everyone and extra outputs."""
    if variant == "ph":
        fit = fit_model(
            model, x, prs, data.targets, data.interval_mask, fit_index, stop_index,
            device, output_dir, "teacher", training_seed, settings,
        )
        bundle = predict(model, x, prs, device)
        return fit, bundle.cumulative_incidence.astype(np.float64), {"attention": bundle.attention}
    fit = fit_mixture_model(
        model, x, prs, data.targets, data.interval_mask, fit_index, stop_index,
        device, output_dir, f"mixture_{variant}", training_seed, settings,
    )
    prediction = predict_mixture(model, x, prs, device)
    extras = {"attention": prediction.attention}
    if prediction.timing_score is not None:
        extras["timing_score"] = prediction.timing_score
        extras["timing_attention"] = prediction.timing_attention
    return fit, prediction.cumulative_incidence, extras


def prediction_frame(
    data: CohortData,
    cumulative: np.ndarray,
    index: np.ndarray,
    timing_score: np.ndarray | None = None,
) -> pd.DataFrame:
    risk = np.asarray(cumulative, dtype=float)[index]
    n_intervals = risk.shape[1]
    onset = onset_distribution(risk)
    frame = pd.DataFrame({
        "sample": np.asarray(data.samples, dtype=object)[index],
        "split": data.split_labels[index],
        "event_10y": data.event[index],
        "observed_time_years": data.duration_years[index],
        f"onset_probability_{n_intervals}y": risk[:, -1],
    })
    for year in range(1, n_intervals + 1):
        frame[f"cumulative_incidence_{year}y"] = risk[:, year - 1]
    for year in range(1, n_intervals + 1):
        frame[f"onset_year_{year}_given_onset"] = onset[:, year - 1]
    frame = pd.concat([frame, timing_summary(onset)], axis=1)
    if timing_score is not None:
        frame["early_onset_score"] = timing_score[index]
    return frame


def run_mixture_seed(
    config: AnalysisConfig,
    training_seed: int,
    output_dir: str | Path,
    timing_head: str = "ordinal",
    final_test: bool = False,
    quick: bool = False,
) -> dict:
    """Train and evaluate one mixture cure Teacher; development mode unless final_test."""
    started = time.time()
    if timing_head not in TIMING_HEADS:
        raise ValueError(f"timing_head must be one of {TIMING_HEADS}, got {timing_head!r}")
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
    if final_test:
        fit_index, stop_index, eval_index, eval_name = data.idx_train, data.idx_val, data.idx_test, "test10"
    else:
        fit_index, stop_index = development_split(data)
        eval_index, eval_name = data.idx_val, "val10"

    model = build_teacher(config, data, timing_head, training_seed, device)
    features = list(model.binn.inputs)
    fitted = fit_preprocessor(data, features)
    _save_preprocessing(output, fitted)
    x = fitted.matrix(features)
    prs = fitted.prs()
    teacher_dir = output / "1_mixture_teacher"
    fit, cumulative, extras = train_teacher(
        timing_head, model, x, prs, data, fit_index, stop_index, device, teacher_dir, training_seed, settings,
    )
    reference = case_interval_distribution(data.targets, fit_index)
    horizons = [int(h) for h in config.raw["endpoint"]["horizons_years"]]
    metrics = evaluate_cumulative(data, cumulative, eval_index, reference, horizons)
    metrics.update({"training_seed": training_seed, "timing_head": timing_head, "dataset": eval_name})
    pd.DataFrame([metrics]).to_csv(output / f"{eval_name}_metrics.csv", index=False)

    saved = np.arange(len(data.samples)) if final_test else eval_index
    prediction_frame(data, cumulative, saved, extras.get("timing_score")).to_csv(
        output / "predictions.csv", index=False
    )
    np.savez(teacher_dir / "attention.npz", **{
        name: values[saved] for name, values in extras.items() if name.endswith("attention")
    })
    split_audit = audit_split_manifest(config.raw["data"]["source_split_manifest"])
    manifest = {
        "status": "quick_smoke" if quick else ("final_test" if final_test else "development"),
        "model": "mixture_cure_teacher",
        "timing_head": timing_head,
        "training_seed": training_seed,
        "test_holdout_seed": config.test_holdout_seed,
        "train_validation_split_seed": config.train_validation_split_seed,
        "split_sha256": split_audit.sha256,
        "fit_split": "train80" if final_test else "train80 minus development holdout",
        "early_stopping_split": (
            "val10" if final_test
            else f"10% of train80 (stratified, seed {DEVELOPMENT_SPLIT_SEED})"
        ),
        "evaluation_split": eval_name,
        "fit_n": int(len(fit_index)),
        "early_stopping_n": int(len(stop_index)),
        "evaluation_n": int(len(eval_index)),
        "device": str(device),
        "config_path": str(config.path),
        "settings": asdict(settings),
        "fit": asdict(fit),
        "metrics": metrics,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
