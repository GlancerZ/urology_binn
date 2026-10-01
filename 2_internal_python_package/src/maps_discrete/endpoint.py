from __future__ import annotations

import math

import numpy as np


def years_from_days(days: np.ndarray, days_per_year: int = 365) -> np.ndarray:
    """Convert follow-up days to years using the protocol-frozen denominator."""
    if days_per_year not in {365, 366}:
        raise ValueError("days_per_year must be 365 or 366")
    return np.asarray(days, dtype=float) / float(days_per_year)


def discrete_targets(
    duration_years: np.ndarray,
    event: np.ndarray,
    n_intervals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build interval event targets and the observable-interval mask."""
    duration = np.asarray(duration_years, dtype=float)
    status = np.asarray(event, dtype=int)
    if duration.shape != status.shape:
        raise ValueError("duration_years and event must have identical shape")
    targets = np.zeros((len(status), n_intervals), dtype=np.float32)
    mask = np.zeros_like(targets)
    for i, (time, occurred) in enumerate(zip(duration, status)):
        if occurred:
            interval = min(max(int(math.ceil(float(time))) - 1, 0), n_intervals - 1)
            mask[i, : interval + 1] = 1.0
            targets[i, interval] = 1.0
        else:
            complete = min(max(int(math.floor(float(time) + 1e-6)), 0), n_intervals)
            mask[i, :complete] = 1.0
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("At least one participant contributes no observable interval")
    return targets, mask


def fixed_horizon_status(
    duration_years: np.ndarray,
    event: np.ndarray,
    horizon_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return known-status mask and binary outcome at a fixed horizon.

    Cases have an event on/before the horizon. Controls are observed event-free
    through the horizon. Event-free participants censored earlier are excluded.
    """
    duration = np.asarray(duration_years, dtype=float)
    status = np.asarray(event, dtype=int)
    cases = (status == 1) & (duration <= horizon_years)
    controls = duration >= horizon_years
    known = cases | controls
    outcome = cases.astype(np.int8)
    return known, outcome
