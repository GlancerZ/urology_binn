from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .endpoint import discrete_targets, years_from_days


@dataclass
class CohortData:
    raw: pd.DataFrame
    raw_num: pd.DataFrame
    entity_col: str
    samples: list[str]
    event: np.ndarray
    duration_years: np.ndarray
    targets: np.ndarray
    interval_mask: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    nmr_features: list[str]
    protein_features: list[str]
    mapping: pd.DataFrame
    pathways: pd.DataFrame
    audit: dict

    @property
    def split_labels(self) -> np.ndarray:
        labels = np.full(len(self.samples), "train80", dtype=object)
        labels[self.idx_val] = "val10"
        labels[self.idx_test] = "test10"
        return labels


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_reactome(config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    binn_root = Path(config.raw["data"]["binn_root"])
    project_root = Path(config.raw["data"]["project_root"])
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    if not binn_root.is_dir():
        raise FileNotFoundError(f"BINN root not found: {binn_root}")
    if str(binn_root) not in sys.path:
        sys.path.insert(0, str(binn_root))
    from binn.model.util import load_reactome_db

    with _working_directory(binn_root):
        reactome = load_reactome_db(input_source="uniprot&chebi")
    override = str(config.raw["data"].get("reactome_mapping_override", "")).strip()
    if override:
        mapping = Path(override)
        if not mapping.is_file():
            raise FileNotFoundError(mapping)
        reactome["mapping"] = pd.read_csv(
            mapping,
            sep="\t",
            header=None,
            names=["input", "translation", "url", "name", "x", "species"],
        )
        source = str(mapping.resolve())
    else:
        source = str(
            (binn_root / "binn" / "data" / "downloads" / "UniProt&ChEBI&DIY2Reactome_Human.txt").resolve()
        )
    return reactome["mapping"], reactome["pathways"], source


def _read_feature_list(path: str | Path) -> list[str]:
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


def load_internal_cohort(config: AnalysisConfig, include_reactome: bool = True) -> CohortData:
    """Load the fixed non-Wales cohort and its immutable 80/10/10 assignment."""
    raw = pd.read_csv(config.raw["data"]["internal_matrix"])
    raw.columns = raw.columns.astype(str)
    entity_col = str(raw.columns[0])
    design = pd.read_csv(config.raw["data"]["internal_design"], dtype={"sample": str})
    available = set(design["sample"])
    samples = [column for column in raw.columns[1:] if column in available]
    design = design.set_index("sample").loc[samples]
    design_event = pd.to_numeric(design["group"], errors="raise").to_numpy(np.int64)

    health = pd.read_csv(
        config.raw["data"]["internal_survival"], dtype={"eid": str}, low_memory=False
    ).set_index("eid")
    missing = sorted(set(samples) - set(health.index))
    if missing:
        raise RuntimeError(f"Survival endpoint missing for {len(missing)} cohort members")
    health = health.loc[samples].copy()
    event = pd.to_numeric(health["occur_in_10y"], errors="raise").to_numpy(np.int64)
    model_label = pd.to_numeric(health["model_label"], errors="raise").to_numpy(np.int64)
    if not (np.array_equal(event, design_event) and np.array_equal(event, model_label)):
        raise RuntimeError("Survival event status does not match the binary endpoint")

    baseline = pd.to_datetime(health["baseline_date"], errors="coerce")
    deadline = pd.to_datetime(health["deadline_10y"], errors="coerce")
    event_date = pd.to_datetime(health["t_event"], errors="coerce")
    censor_date = pd.to_datetime(health["t_censored"], errors="coerce")
    if baseline.isna().any() or deadline.isna().any() or (event.astype(bool) & event_date.isna()).any():
        raise RuntimeError("Invalid baseline, deadline, or event date")
    prebaseline = (event == 1) & ((event_date - baseline).dt.days.to_numpy(float) <= 0)
    excluded_ids = [sample for sample, invalid in zip(samples, prebaseline) if invalid]
    if prebaseline.any():
        keep = ~prebaseline
        positions = np.flatnonzero(keep)
        samples = [sample for sample, valid in zip(samples, keep) if valid]
        health = health.iloc[positions]
        event = event[keep]
        baseline = baseline.iloc[positions]
        deadline = deadline.iloc[positions]
        event_date = event_date.iloc[positions]
        censor_date = censor_date.iloc[positions]

    baseline = baseline.reset_index(drop=True)
    deadline = deadline.reset_index(drop=True)
    event_date = event_date.reset_index(drop=True)
    censor_date = censor_date.reset_index(drop=True)
    censor_end = pd.concat([censor_date, deadline], axis=1).min(axis=1)
    end_date = censor_end.copy()
    event_positions = np.flatnonzero(event == 1)
    end_date.iloc[event_positions] = event_date.iloc[event_positions].to_numpy()
    duration_days = (end_date - baseline).dt.days.to_numpy(float)
    duration_years = years_from_days(duration_days, config.days_per_year)
    reached_deadline = (censor_end >= (deadline - pd.Timedelta(days=1))).to_numpy()
    duration_years[(event == 0) & reached_deadline] = float(config.n_intervals)
    if np.any(~np.isfinite(duration_years)) or np.any(duration_years <= 0):
        raise RuntimeError("Survival duration must be finite and positive")
    if np.any(duration_years > config.n_intervals + 0.01):
        raise RuntimeError("Survival duration exceeds the model horizon")

    targets, interval_mask = discrete_targets(duration_years, event, config.n_intervals)
    split = pd.read_csv(config.raw["data"]["source_split_manifest"], dtype={"sample": str})
    split = split.loc[split["sample"].isin(samples)].set_index("sample").loc[samples].reset_index()
    if split["sample"].tolist() != samples:
        raise RuntimeError("Frozen split does not align with cohort order")
    if not np.array_equal(pd.to_numeric(split["group"]).to_numpy(), event):
        raise RuntimeError("Frozen split event labels do not align with endpoint")
    allowed = {"train80", "val10", "test10"}
    if set(split["split"]) != allowed:
        raise RuntimeError(f"Frozen split labels must be {sorted(allowed)}")
    idx_train = np.flatnonzero(split["split"].eq("train80"))
    idx_val = np.flatnonzero(split["split"].eq("val10"))
    idx_test = np.flatnonzero(split["split"].eq("test10"))

    if include_reactome:
        mapping, pathways, mapping_source = load_reactome(config)
    else:
        mapping = pd.DataFrame()
        pathways = pd.DataFrame()
        mapping_source = "not_loaded_for_non_binn_model"
    raw_num = raw.set_index(entity_col)[samples].astype(float)
    nmr_features = _read_feature_list(config.raw["data"]["nmr_feature_list"])
    protein_features = _read_feature_list(config.raw["data"]["protein_feature_list"])
    audit = {
        "n": len(samples),
        "events_10y": int(event.sum()),
        "early_censored_before_10y": int(((event == 0) & ~reached_deadline).sum()),
        "excluded_prebaseline_event_n": len(excluded_ids),
        "excluded_prebaseline_event_ids": excluded_ids,
        "days_per_year": config.days_per_year,
        "train_n": len(idx_train),
        "val_n": len(idx_val),
        "test_n": len(idx_test),
        "train_events": int(event[idx_train].sum()),
        "val_events": int(event[idx_val].sum()),
        "test_events": int(event[idx_test].sum()),
        "mapping_source": mapping_source,
        "mapping_rows": int(len(mapping)),
    }
    return CohortData(
        raw=raw,
        raw_num=raw_num,
        entity_col=entity_col,
        samples=samples,
        event=event,
        duration_years=duration_years.astype(np.float32),
        targets=targets,
        interval_mask=interval_mask,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        nmr_features=nmr_features,
        protein_features=protein_features,
        mapping=mapping,
        pathways=pathways,
        audit=audit,
    )
