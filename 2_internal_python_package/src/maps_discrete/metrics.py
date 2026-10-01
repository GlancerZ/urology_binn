from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .endpoint import fixed_horizon_status


def harrell_c_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index; larger scores must indicate higher risk."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in np.flatnonzero(event == 1):
        later = np.flatnonzero(duration > duration[i])
        comparable += len(later)
        if len(later):
            concordant += float(np.sum(risk[i] > risk[later]))
            concordant += 0.5 * float(np.sum(risk[i] == risk[later]))
    return concordant / comparable if comparable else float("nan")


def fixed_horizon_metrics(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_risk: np.ndarray,
    horizon: float = 10.0,
) -> dict[str, float | int]:
    known, y_all = fixed_horizon_status(duration, event, horizon)
    y = y_all[known]
    score = np.asarray(predicted_risk, dtype=float)[known]
    if len(np.unique(y)) < 2:
        raise ValueError("Both cases and controls are required for AUC/PR-AUC")
    return {
        "auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "known_n": int(known.sum()),
        "cases": int(y.sum()),
        "controls": int((y == 0).sum()),
        "early_censored_excluded": int((~known).sum()),
    }


def continuous_nri_idi(
    duration: np.ndarray,
    event: np.ndarray,
    reference_risk: np.ndarray,
    new_risk: np.ndarray,
    horizon: float = 10.0,
) -> dict[str, float | int]:
    """Category-free NRI and IDI at one fixed horizon; ties are neutral."""
    known, y_all = fixed_horizon_status(duration, event, horizon)
    y = y_all[known]
    old = np.asarray(reference_risk, dtype=float)[known]
    new = np.asarray(new_risk, dtype=float)[known]
    case = y == 1
    control = ~case
    if not case.any() or not control.any():
        raise ValueError("NRI/IDI require both cases and controls")
    delta = new - old
    nri_event = float(np.mean(delta[case] > 0) - np.mean(delta[case] < 0))
    nri_nonevent = float(np.mean(delta[control] < 0) - np.mean(delta[control] > 0))
    old_slope = float(old[case].mean() - old[control].mean())
    new_slope = float(new[case].mean() - new[control].mean())
    return {
        "nri_event": nri_event,
        "nri_nonevent": nri_nonevent,
        "nri": nri_event + nri_nonevent,
        "idi": new_slope - old_slope,
        "reference_discrimination_slope": old_slope,
        "new_discrimination_slope": new_slope,
        "known_n": int(known.sum()),
    }


def decision_curve(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_risk: np.ndarray,
    thresholds: np.ndarray,
    horizon: float = 10.0,
) -> pd.DataFrame:
    """Standard binary net benefit on participants with known horizon status."""
    known, y_all = fixed_horizon_status(duration, event, horizon)
    y = y_all[known]
    risk = np.asarray(predicted_risk, dtype=float)[known]
    n = len(y)
    prevalence = float(y.mean())
    rows: list[dict[str, float | int]] = []
    for threshold in np.asarray(thresholds, dtype=float):
        if not 0 < threshold < 1:
            raise ValueError("DCA thresholds must be strictly between 0 and 1")
        positive = risk >= threshold
        tp = int(np.sum(positive & (y == 1)))
        fp = int(np.sum(positive & (y == 0)))
        odds = threshold / (1.0 - threshold)
        rows.append({
            "threshold": float(threshold),
            "model_net_benefit": float(tp / n - fp / n * odds),
            "treat_all_net_benefit": float(prevalence - (1.0 - prevalence) * odds),
            "treat_none_net_benefit": 0.0,
            "known_n": n,
            "cases": int(y.sum()),
        })
    return pd.DataFrame(rows)


def evaluate_model(
    duration: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
    cumulative_risk: np.ndarray,
    horizon: float = 10.0,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "n": int(len(event)),
        "events": int(np.sum(event)),
        "c_index": harrell_c_index(duration, event, log_risk),
    }
    result.update(fixed_horizon_metrics(duration, event, cumulative_risk, horizon))
    return result


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    valid_repetitions: int


def participant_bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> BootstrapInterval:
    """Participant-level percentile bootstrap; same indices can be paired across models."""
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    estimate = float(statistic(np.arange(n)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        index = rng.integers(0, n, size=n)
        try:
            value = float(statistic(index))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(50, repetitions // 2):
        raise RuntimeError("Too few valid bootstrap repetitions")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(estimate, float(lower), float(upper), len(values))


def _interval_table(
    estimates: dict[str, float],
    draws: dict[str, list[float]],
    confidence_level: float,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows = []
    for metric, estimate in estimates.items():
        values = np.asarray(draws[metric], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        rows.append({
            "metric": metric,
            "estimate": float(estimate),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "valid_bootstrap_repetitions": int(len(values)),
        })
    return pd.DataFrame(rows)


def bootstrap_model_performance(
    duration: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
    cumulative_risk: np.ndarray,
    horizon: float = 10.0,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> pd.DataFrame:
    """Performance estimates and participant-level percentile bootstrap CIs."""
    duration = np.asarray(duration)
    event = np.asarray(event)
    log_risk = np.asarray(log_risk)
    cumulative_risk = np.asarray(cumulative_risk)
    point = evaluate_model(duration, event, log_risk, cumulative_risk, horizon)
    metrics = ("c_index", "auc", "pr_auc", "brier")
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = rng.integers(0, len(event), size=len(event))
        try:
            result = evaluate_model(
                duration[index], event[index], log_risk[index], cumulative_risk[index], horizon
            )
        except ValueError:
            continue
        for metric in metrics:
            draws[metric].append(float(result[metric]))
    return _interval_table({metric: float(point[metric]) for metric in metrics}, draws, confidence_level)


def bootstrap_model_comparison(
    duration: np.ndarray,
    event: np.ndarray,
    reference_log_risk: np.ndarray,
    reference_cumulative_risk: np.ndarray,
    new_log_risk: np.ndarray,
    new_cumulative_risk: np.ndarray,
    horizon: float = 10.0,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> pd.DataFrame:
    """Paired bootstrap for discrimination differences, continuous NRI and IDI."""
    arrays = [
        np.asarray(duration), np.asarray(event), np.asarray(reference_log_risk),
        np.asarray(reference_cumulative_risk), np.asarray(new_log_risk),
        np.asarray(new_cumulative_risk),
    ]

    def compare(index: np.ndarray) -> dict[str, float]:
        d, e, ref_log, ref_risk, new_log, new_risk = [array[index] for array in arrays]
        ref = evaluate_model(d, e, ref_log, ref_risk, horizon)
        new = evaluate_model(d, e, new_log, new_risk, horizon)
        reclassification = continuous_nri_idi(d, e, ref_risk, new_risk, horizon)
        return {
            "delta_c_index": float(new["c_index"] - ref["c_index"]),
            "delta_auc": float(new["auc"] - ref["auc"]),
            "delta_pr_auc": float(new["pr_auc"] - ref["pr_auc"]),
            "nri_event": float(reclassification["nri_event"]),
            "nri_nonevent": float(reclassification["nri_nonevent"]),
            "nri": float(reclassification["nri"]),
            "idi": float(reclassification["idi"]),
        }

    estimates = compare(np.arange(len(arrays[1])))
    draws = {metric: [] for metric in estimates}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = rng.integers(0, len(arrays[1]), size=len(arrays[1]))
        try:
            result = compare(index)
        except ValueError:
            continue
        for metric, value in result.items():
            draws[metric].append(value)
    return _interval_table(estimates, draws, confidence_level)
