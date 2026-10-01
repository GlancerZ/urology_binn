from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AnalysisConfig
from .selection import rank_seeds


def collect_metrics(root: str | Path, filename: str) -> pd.DataFrame:
    files = sorted(Path(root).glob(f"seed_*/{filename}"))
    if not files:
        raise FileNotFoundError(f"No {filename!r} under seed_* directories in {root}")
    frames = [pd.read_csv(path).assign(source_file=str(path)) for path in files]
    return pd.concat(frames, ignore_index=True)


def exploratory_seed_selection(
    config: AnalysisConfig,
    maps_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """Apply the explicitly approved exploratory, test-selected seed policy."""
    selection = config.raw["selection"]
    if selection.get("analysis_role") != "exploratory":
        raise ValueError("This selector is only available for exploratory analyses")
    n = int(selection["n_selected"])
    teacher = maps_metrics.loc[maps_metrics["model"].eq("teacher")]
    rank_kwargs = {
        "metric": "auc",
        "n": n,
        "dataset_role": "test",
        "analysis_role": "exploratory",
        "allow_test_based_selection": True,
    }
    teacher_seeds = rank_seeds(teacher, higher_is_better=True, **rank_kwargs)
    ml_bottom: dict[str, list[int]] = {}
    for model, group in ml_metrics.groupby("model", sort=True):
        ml_bottom[str(model)] = rank_seeds(group, higher_is_better=False, **rank_kwargs)
    result = {
        "analysis_role": "exploratory",
        "warning": "Seeds were selected on test performance; estimates are selection-biased.",
        "metric": "test_10y_auc",
        "teacher_top_seeds": teacher_seeds,
        "ml_bottom_seeds_by_model": ml_bottom,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "exploratory_seed_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
