from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class RunManifest:
    experiment_id: str
    model_id: str
    test_holdout_seed: int
    train_validation_split_seed: int
    training_seed: int
    config_path: str
    split_sha256: str
    created_utc: str
    status: str = "planned"

    @classmethod
    def planned(
        cls,
        experiment_id: str,
        model_id: str,
        test_holdout_seed: int,
        train_validation_split_seed: int,
        training_seed: int,
        config_path: str,
        split_sha256: str,
    ) -> "RunManifest":
        return cls(
            experiment_id=experiment_id,
            model_id=model_id,
            test_holdout_seed=int(test_holdout_seed),
            train_validation_split_seed=int(train_validation_split_seed),
            training_seed=int(training_seed),
            config_path=str(Path(config_path).resolve()),
            split_sha256=split_sha256,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
