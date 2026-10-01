from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AnalysisConfig


def write_batch_plan(
    config: AnalysisConfig,
    maps_root: str | Path,
    ml_root: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Create the auditable 100-seed run matrix without launching computation."""
    maps_root = Path(maps_root).resolve()
    ml_root = Path(ml_root).resolve()
    rows = []
    for seed in config.training_seeds:
        rows.extend([
            {
                "job_type": "maps_teacher_student",
                "training_seed": seed,
                "test_holdout_seed": config.test_holdout_seed,
                "train_validation_split_seed": config.train_validation_split_seed,
                "output_directory": str(maps_root / f"seed_{seed:03d}"),
                "status": "planned",
            },
            {
                "job_type": "traditional_ml",
                "training_seed": seed,
                "test_holdout_seed": config.test_holdout_seed,
                "train_validation_split_seed": config.train_validation_split_seed,
                "output_directory": str(ml_root / f"seed_{seed:03d}"),
                "status": "planned",
            },
        ])
    frame = pd.DataFrame(rows)
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
