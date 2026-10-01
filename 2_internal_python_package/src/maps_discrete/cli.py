from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import write_batch_plan
from .config import load_config
from .guard import require_approved
from .ml import ALGORITHMS, MODALITIES, run_ml_seed
from .mixture_cure import TIMING_HEADS
from .mixture_pipeline import run_mixture_seed
from .pipeline import run_maps_seed
from .splits import audit_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal discrete-time MAPS experiment controls")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="validate draft config and frozen split")
    audit.add_argument("--config", required=True)
    guard = sub.add_parser("guard-run", help="fail unless the protocol is approved")
    guard.add_argument("--config", required=True)
    plan = sub.add_parser("plan-batch", help="write the frozen 100-seed job matrix")
    plan.add_argument("--config", required=True)
    plan.add_argument("--maps-root", required=True)
    plan.add_argument("--ml-root", required=True)
    plan.add_argument("--output", required=True)
    maps = sub.add_parser("run-maps", help="run Teacher and Student for one training seed")
    maps.add_argument("--config", required=True)
    maps.add_argument("--seed", required=True, type=int)
    maps.add_argument("--output", required=True)
    maps.add_argument("--quick-smoke", action="store_true")
    maps.add_argument("--without-supervised-student", action="store_true")
    ml = sub.add_parser("run-ml", help="run frozen ML baselines for one training seed")
    ml.add_argument("--config", required=True)
    ml.add_argument("--seed", required=True, type=int)
    ml.add_argument("--output", required=True)
    ml.add_argument("--quick-smoke", action="store_true")
    ml.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    ml.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    mixture = sub.add_parser(
        "run-mixture",
        help="run the mixture cure Teacher (onset probability + onset year) for one seed; "
        "development mode (val10 evaluation) unless --final-test",
    )
    mixture.add_argument("--config", required=True)
    mixture.add_argument("--seed", required=True, type=int)
    mixture.add_argument("--output", required=True)
    mixture.add_argument("--timing-head", choices=TIMING_HEADS, default="ordinal")
    mixture.add_argument(
        "--final-test", action="store_true",
        help="fit on train80, early-stop on val10 and evaluate test10; use once, after freezing",
    )
    mixture.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "guard-run":
        require_approved(config)
        print("Protocol approved")
        return
    if args.command == "plan-batch":
        frame = write_batch_plan(config, args.maps_root, args.ml_root, args.output)
        print(json.dumps({"jobs": len(frame), "output": str(Path(args.output).resolve())}, indent=2))
        return
    if args.command == "run-maps":
        if not args.quick_smoke:
            require_approved(config)
        result = run_maps_seed(
            config, args.seed, args.output, quick=args.quick_smoke,
            include_supervised_student=not args.without_supervised_student,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "run-ml":
        if not args.quick_smoke:
            require_approved(config)
        result = run_ml_seed(
            config, args.seed, args.output,
            modalities=tuple(args.modalities), algorithms=tuple(args.algorithms),
            quick=args.quick_smoke,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "run-mixture":
        if args.final_test:
            require_approved(config)
        result = run_mixture_seed(
            config, args.seed, args.output, timing_head=args.timing_head,
            final_test=args.final_test, quick=args.quick_smoke,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    split = audit_split_manifest(config.raw["data"]["source_split_manifest"])
    print(json.dumps({
        "approved": config.approved,
        "test_holdout_seed": config.test_holdout_seed,
        "train_validation_split_seed": config.train_validation_split_seed,
        "split": split.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
