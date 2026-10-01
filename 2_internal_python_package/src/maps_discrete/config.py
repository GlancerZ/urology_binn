from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    raw: dict

    @property
    def approved(self) -> bool:
        return bool(self.raw["protocol"]["approved"])

    @property
    def test_holdout_seed(self) -> int:
        return int(self.raw["seeds"]["test_holdout_seed"])

    @property
    def train_validation_split_seed(self) -> int:
        return int(self.raw["seeds"]["train_validation_split_seed"])

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return tuple(map(int, self.raw["seeds"]["training_seeds"]))

    @property
    def confirmatory_seeds(self) -> tuple[int, ...]:
        return tuple(map(int, self.raw["seeds"]["confirmatory_seeds"]))

    @property
    def days_per_year(self) -> int:
        return int(self.raw["endpoint"]["days_per_year"])

    @property
    def n_intervals(self) -> int:
        return int(self.raw["endpoint"]["n_intervals"])

    @property
    def primary_horizon(self) -> int:
        return int(self.raw["evaluation"]["primary_horizon_years"])

    def validate(self) -> None:
        seeds = self.training_seeds
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("training_seeds must be non-empty and unique")
        selection = self.raw["selection"]
        if selection["selection_dataset"] == "test" and not selection["allow_test_based_selection"]:
            raise ValueError("test-based seed selection is disabled")
        if selection["selection_dataset"] == "test" and selection.get("analysis_role") != "exploratory":
            raise ValueError("test-based seed selection must be labelled exploratory")
        if self.approved and not self.confirmatory_seeds and selection["primary_policy"] == "predeclared_ensemble":
            raise ValueError("approved predeclared ensemble requires confirmatory_seeds")
        if int(self.raw["endpoint"]["days_per_year"]) not in {365, 366}:
            raise ValueError("days_per_year must be explicitly fixed to 365 or 366")
        methods = {
            "c_index_method": "harrell_c",
            "auc_method": "fixed_horizon_known_status_roc_auc",
            "pr_auc_method": "fixed_horizon_known_status_average_precision",
            "nri_method": "continuous_category_free_nri",
            "idi_method": "discrimination_slope_difference",
            "dca_method": "standard_binary_net_benefit",
        }
        evaluation = self.raw["evaluation"]
        for key, expected in methods.items():
            if evaluation.get(key) != expected:
                raise ValueError(f"{key} must be frozen as {expected!r}")
        t_min = float(evaluation["dca_threshold_min"])
        t_max = float(evaluation["dca_threshold_max"])
        if not 0 < t_min < t_max < 1:
            raise ValueError("DCA thresholds must satisfy 0 < min < max < 1")
        if int(evaluation["bootstrap_repetitions"]) < 100:
            raise ValueError("bootstrap_repetitions must be at least 100")


_UNRESOLVED_PATH_VALUES = {"", "PENDING"}


def _resolve_path(value: str, base: Path) -> str:
    if value in _UNRESOLVED_PATH_VALUES or Path(value).is_absolute():
        return value
    return str((base / value).resolve())


def _resolve_relative_paths(raw: dict, base: Path) -> None:
    """Resolve relative data and checkpoint paths against the config directory."""
    data = raw.get("data", {})
    for key, value in data.items():
        data[key] = _resolve_path(str(value), base)
    selected = raw.get("selected_models", {})
    for key, value in selected.items():
        if key.endswith("_checkpoint"):
            selected[key] = _resolve_path(str(value), base)


def load_config(path: str | Path) -> AnalysisConfig:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    _resolve_relative_paths(raw, path.parent)
    config = AnalysisConfig(path=path, raw=raw)
    config.validate()
    return config
