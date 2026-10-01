from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from .config import load_config
from .guard import require_approved


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_complete(seed_dir: Path, seed: int) -> bool:
    path = seed_dir / "ml_run_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return manifest.get("status") == "completed" and int(manifest.get("training_seed", -1)) == seed


def write_status(root: Path, records: dict[int, dict]) -> None:
    temporary = root / "ml_batch_status.tmp.csv"
    final = root / "ml_batch_status.csv"
    pd.DataFrame([records[seed] for seed in sorted(records)]).to_csv(temporary, index=False)
    os.replace(temporary, final)


def run_batch(config_path: str | Path, output_root: str | Path, seeds: list[int] | None = None) -> int:
    config = load_config(config_path)
    require_approved(config)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = list(config.training_seeds if seeds is None else map(int, seeds))
    unknown = sorted(set(selected) - set(config.training_seeds))
    if unknown:
        raise ValueError(f"Seeds absent from frozen grid: {unknown}")
    records = {
        seed: {
            "training_seed": seed,
            "status": "completed" if is_complete(root / f"seed_{seed:03d}", seed) else "pending",
            "started_utc": "",
            "finished_utc": "",
            "elapsed_minutes": "",
            "return_code": "",
            "output_directory": str(root / f"seed_{seed:03d}"),
            "console_log": str(root / f"seed_{seed:03d}_console.log"),
            "message": "",
        }
        for seed in selected
    }
    write_status(root, records)
    started = time.time()
    for seed in selected:
        seed_dir = root / f"seed_{seed:03d}"
        if is_complete(seed_dir, seed):
            print(f"skip completed ML seed={seed}", flush=True)
            continue
        log_path = root / f"seed_{seed:03d}_console.log"
        seed_started = time.time()
        records[seed].update({"status": "running", "started_utc": utc_now()})
        write_status(root, records)
        command = [
            sys.executable, "-m", "maps_discrete", "run-ml",
            "--config", str(Path(config_path).resolve()),
            "--seed", str(seed), "--output", str(seed_dir),
        ]
        print(f"start ML seed={seed} log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(config.raw["data"]["project_root"]),
                check=False,
            )
        elapsed = (time.time() - seed_started) / 60.0
        success = result.returncode == 0 and is_complete(seed_dir, seed)
        records[seed].update({
            "status": "completed" if success else "failed",
            "finished_utc": utc_now(),
            "elapsed_minutes": elapsed,
            "return_code": result.returncode,
            "message": "" if success else "Child failed or completion manifest missing",
        })
        write_status(root, records)
        print(f"finish ML seed={seed} status={records[seed]['status']} minutes={elapsed:.2f}", flush=True)
        if not success:
            print(f"ML batch stopped after seed={seed}; inspect {log_path}", flush=True)
            break
    counts = pd.Series([record["status"] for record in records.values()]).value_counts().to_dict()
    summary = {
        "config_path": str(Path(config_path).resolve()),
        "output_root": str(root),
        "requested_seeds": selected,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "elapsed_hours": (time.time() - started) / 3600.0,
        "updated_utc": utc_now(),
    }
    (root / "ml_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if counts.get("failed", 0) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe traditional ML multiseed runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    raise SystemExit(run_batch(args.config, args.output_root, args.seeds))


if __name__ == "__main__":
    main()
