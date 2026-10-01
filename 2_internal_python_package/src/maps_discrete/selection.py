from __future__ import annotations

import pandas as pd


class UnsafeSeedSelectionError(RuntimeError):
    pass


def rank_seeds(
    results: pd.DataFrame,
    *,
    metric: str,
    n: int,
    higher_is_better: bool,
    dataset_role: str,
    analysis_role: str = "confirmatory",
    allow_test_based_selection: bool = False,
) -> list[int]:
    """Rank seeds with an explicit guard against selecting on final test data."""
    if dataset_role == "test" and not allow_test_based_selection:
        raise UnsafeSeedSelectionError(
            "Selecting seeds on test performance is disabled by the draft protocol"
        )
    if dataset_role == "test" and analysis_role != "exploratory":
        raise UnsafeSeedSelectionError(
            "Test-ranked seeds may only be used in an explicitly exploratory analysis"
        )
    required = {"training_seed", metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")
    one_per_seed = results[["training_seed", metric]].dropna()
    if one_per_seed["training_seed"].duplicated().any():
        raise ValueError("expected one result per training seed")
    ordered = one_per_seed.sort_values(metric, ascending=not higher_is_better)
    return ordered.head(int(n))["training_seed"].astype(int).tolist()
