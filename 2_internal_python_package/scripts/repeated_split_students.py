"""Student (NMR + PRS) variants on the Teacher's repeated splits.

Per split, two seed-7 Teachers are trained on train80 with val10 early stopping:
  frozen Teacher    discrete-time survival recipe (distillation source of Student-MAPS)
  improved Teacher  10-year objective + relaxed direct path (stage-1 selection file)
Students use NMR + PRS only (BINN on the NMR features, same fusion), for every student seed:
  frozen_nodistill  frozen recipe without distillation
  frozen_distilled  frozen recipe distilled from the frozen Teacher (Student-MAPS)
  binary            10-year objective, no distillation
  binary_distilled  10-year objective distilled from the improved Teacher's 10-year logits
                    (alpha and temperature from the frozen config)
10-year val/test risks are saved per run for seed ensembles; select on val10 log-loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.modeling import build_model
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.prediction import predict
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.seeding import set_global_seed
from maps_discrete.splits import stratified_split
from maps_discrete.training import fit_model
from repeated_split_binn_improvements import ImprovedBINN, fit, risk_10y, stage1_selection
from repeated_split_validation import calibration, discrimination
from dataclasses import replace

STUDY = Path(__file__).resolve().parents[2]
VARIANTS = ("frozen_nodistill", "frozen_distilled", "binary", "binary_distilled")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "14_repeated_split_validation" / "students"))
    parser.add_argument("--splits", nargs="+", default=None, help="subset of split names (default: all 21)")
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[7, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    (output / "risks").mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config, args.quick_smoke)
    data = load_internal_cohort(config)
    y10 = data.event.astype(np.float32)
    splits = {"frozen": (data.idx_train, data.idx_val, data.idx_test)}
    splits.update({f"random_{seed:02d}": stratified_split(data.event, seed) for seed in range(1, 21)})
    names = args.splits or list(splits)
    selection = stage1_selection(STUDY / "14_repeated_split_validation" / "binn_improvements"
                                 / "binary_direct_l1_1e-2_seed7_by_split.csv")
    kwargs = dict(
        mapping=data.mapping, pathways=data.pathways, entity_col=data.entity_col,
        binn_root=config.raw["data"]["binn_root"], device=device,
        n_layers=int(config.raw["training"]["n_layers"]), n_intervals=config.n_intervals,
        attention_dim=int(config.raw["training"]["attention_dim"]),
    )
    teacher_matrix = data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy()
    student_matrix = data.raw.loc[data.raw[data.entity_col].isin(data.nmr_features)].copy()
    set_global_seed(7)
    teacher = build_model(data_matrix=teacher_matrix, **kwargs)
    teacher_init = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    teacher_features = list(teacher.binn.inputs)
    students = {}
    for seed in args.student_seeds:
        set_global_seed(seed)
        student = build_model(data_matrix=student_matrix, **kwargs)
        students[seed] = (student, {k: v.detach().clone() for k, v in student.state_dict().items()})
    student_features = list(students[args.student_seeds[0]][0].binn.inputs)

    rows: list[dict] = []
    tag = "_".join(names) if args.splits else "all"
    for split in names:
        train, val, test = splits[split]
        split_data = replace(data, idx_train=train, idx_val=val, idx_test=test)
        fitted = fit_preprocessor(split_data, teacher_features)
        x_t, x_s, prs = fitted.matrix(teacher_features), fitted.matrix(student_features), fitted.prs()
        teacher.load_state_dict(teacher_init)
        with tempfile.TemporaryDirectory() as tmp:
            fit_model(teacher, x_t, prs, data.targets, data.interval_mask, train, val, device, tmp,
                      "teacher", 7, settings)
        frozen_teacher_logits = predict(teacher, x_t, prs, device).hazard_logits
        teacher.load_state_dict(teacher_init)
        columns = [teacher_features.index(f) for f in selection[split]]
        improved = ImprovedBINN(teacher, len(teacher_features), direct=True, direct_columns=columns).to(device)
        fit(improved, x_t, prs, data, y10, train, val, device, 7, settings, binary=True, l1=None)
        improved_logit = logit(risk_10y(improved, x_t, prs, device, binary=True))
        print(f"[{(time.time() - started) / 60:.1f} min] {split} teachers done; improved teacher val AUC "
              f"{roc_auc_score(y10[val], improved_logit[val]):.4f}", flush=True)

        for seed in args.student_seeds:
            student, init = students[seed]
            for variant in args.variants:
                student.load_state_dict(init)
                if variant.startswith("frozen"):
                    with tempfile.TemporaryDirectory() as tmp:
                        fit_model(student, x_s, prs, data.targets, data.interval_mask, train, val, device, tmp,
                                  variant, seed, settings,
                                  teacher_logits=frozen_teacher_logits if variant == "frozen_distilled" else None)
                    risk = predict(student, x_s, prs, device).cumulative_incidence[:, 9].astype(float)
                else:
                    model = ImprovedBINN(student, len(student_features), direct=False).to(device)
                    fit(model, x_s, prs, data, y10, train, val, device, seed, settings, binary=True, l1=None,
                        soft_logits=improved_logit if variant == "binary_distilled" else None)
                    risk = risk_10y(model, x_s, prs, device, binary=True)
                np.savez(output / "risks" / f"{variant}_{split}_seed{seed}.npz", val=risk[val], test=risk[test])
                duration, event = data.duration_years[test], data.event[test]
                row = {"variant": variant, "split": split, "student_seed": seed,
                       "val_logloss_10y": float(log_loss(y10[val], np.clip(risk[val], 1e-7, 1 - 1e-7))),
                       "val_auc_10y": float(roc_auc_score(y10[val], risk[val]))}
                row.update({k.replace("m_", "", 1): v for k, v in discrimination("m", duration, event, risk[test]).items()})
                row.update({k.replace("m_", "", 1): v for k, v in calibration("m", duration, event, risk[test]).items()})
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / f"students_{tag}_by_split.csv", index=False)
            print(f"[{(time.time() - started) / 60:.1f} min] {split} seed {seed}: " + ", ".join(
                f"{r['variant']} {r['auc_10y']:.3f}" for r in rows[-len(args.variants):]), flush=True)
    (output / f"manifest_{tag}.json").write_text(json.dumps({
        "splits": names, "student_seeds": args.student_seeds, "variants": args.variants,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
