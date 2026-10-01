from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd


EXPECTED_SPLITS = {"train80", "val10", "test10"}


@dataclass(frozen=True)
class SplitAudit:
    path: str
    sha256: str
    n: int
    train_n: int
    val_n: int
    test_n: int
    train_events: int
    val_events: int
    test_events: int

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split_manifest(path: str | Path) -> SplitAudit:
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={"sample": str})
    required = {"sample", "group", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"split manifest missing columns: {sorted(missing)}")
    if frame["sample"].duplicated().any():
        raise ValueError("split manifest contains duplicate participant IDs")
    observed = set(frame["split"].dropna().astype(str))
    if observed != EXPECTED_SPLITS:
        raise ValueError(f"unexpected split labels: {sorted(observed)}")
    group = pd.to_numeric(frame["group"], errors="raise").astype(int)
    if not set(group.unique()).issubset({0, 1}):
        raise ValueError("group must be binary")
    counts = frame.assign(group=group).groupby("split")["group"].agg(["size", "sum"])
    return SplitAudit(
        path=str(path),
        sha256=file_sha256(path),
        n=int(len(frame)),
        train_n=int(counts.loc["train80", "size"]),
        val_n=int(counts.loc["val10", "size"]),
        test_n=int(counts.loc["test10", "size"]),
        train_events=int(counts.loc["train80", "sum"]),
        val_events=int(counts.loc["val10", "sum"]),
        test_events=int(counts.loc["test10", "sum"]),
    )


def stratified_split(event, seed: int):
    """Event-stratified random 80/10/10 split (train, val, test indices) for robustness checks."""
    import numpy as np

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in (0, 1):
        members = rng.permutation(np.flatnonzero(np.asarray(event) == label))
        n_hold = int(round(0.1 * len(members)))
        test.append(members[:n_hold])
        val.append(members[n_hold:2 * n_hold])
        train.append(members[2 * n_hold:])
    return tuple(np.sort(np.concatenate(parts)) for parts in (train, val, test))
