"""Wales external validation of Teacher candidates (and, with --part student, Student candidates).

All models are fitted on the internal frozen split only (train80 fit, val10 early stopping or
tuning); Wales is scored once. Wales features are imputed and standardised with the internal
train80 parameters. Metrics: 10-year AUC, PR-AUC, Harrell C, Brier and observed/expected,
with participant-level bootstrap CIs and paired differences against a reference model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from maps_discrete.config import load_config
from maps_discrete.data import load_internal_cohort
from maps_discrete.metrics import harrell_c_index
from maps_discrete.ml import build_classifier
from maps_discrete.modeling import build_model
from maps_discrete.pipeline import resolve_device, settings_from_config
from maps_discrete.prediction import predict
from maps_discrete.preprocessing import fit_preprocessor
from maps_discrete.seeding import set_global_seed
from maps_discrete.training import fit_model
from maps_discrete.wales_external_validation import _external_matrix, _load_wales
from repeated_split_binn_improvements import ImprovedBINN, fit, risk_10y, stage1_selection
import tempfile

STUDY = Path(__file__).resolve().parents[2]
ENSEMBLE_SEEDS = (7, 1, 2, 3, 4)


def metrics(event, duration, risk, probability=True) -> dict[str, float]:
    result = {
        "auc_10y": float(roc_auc_score(event, risk)),
        "pr_auc_10y": float(average_precision_score(event, risk)),
        "c_index": harrell_c_index(duration, event, risk),
    }
    if probability:
        result["brier_10y"] = float(brier_score_loss(event, risk))
        result["observed_expected_10y"] = float(event.sum() / risk.sum())
    return result


def bootstrap(event, risks: dict[str, np.ndarray], reference: str, repetitions=2000, seed=2026) -> pd.DataFrame:
    """Percentile CIs for AUC / PR-AUC of every model and paired differences to the reference."""
    rng = np.random.default_rng(seed)
    draws = {name: {"auc": [], "pr": []} for name in risks}
    n = len(event)
    for _ in range(repetitions):
        index = rng.integers(0, n, n)
        if event[index].sum() == 0 or event[index].sum() == n:
            continue
        for name, risk in risks.items():
            draws[name]["auc"].append(roc_auc_score(event[index], risk[index]))
            draws[name]["pr"].append(average_precision_score(event[index], risk[index]))
    rows = []
    for name in risks:
        auc, pr = np.array(draws[name]["auc"]), np.array(draws[name]["pr"])
        d_auc = auc - np.array(draws[reference]["auc"])
        d_pr = pr - np.array(draws[reference]["pr"])
        rows.append({
            "model": name,
            "auc_ci_lower": np.quantile(auc, 0.025), "auc_ci_upper": np.quantile(auc, 0.975),
            "pr_auc_ci_lower": np.quantile(pr, 0.025), "pr_auc_ci_upper": np.quantile(pr, 0.975),
            f"delta_auc_vs_{reference}": float(np.mean(d_auc)),
            "delta_auc_ci_lower": np.quantile(d_auc, 0.025), "delta_auc_ci_upper": np.quantile(d_auc, 0.975),
            f"delta_pr_vs_{reference}": float(np.mean(d_pr)),
            "delta_pr_ci_lower": np.quantile(d_pr, 0.025), "delta_pr_ci_upper": np.quantile(d_pr, 0.975),
        })
    return pd.DataFrame(rows)


def tuned_xgboost(x, y, train, val):
    """Depth in {2,3,4} with early stopping on val10 log-loss, as repeated_split_xgboost_tuned.py."""
    best = None
    for depth in (2, 3, 4):
        model = XGBClassifier(
            n_estimators=2000, max_depth=depth, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=50,
            random_state=7, n_jobs=8,
        )
        model.fit(x[train], y[train], eval_set=[(x[val], y[val])], verbose=False)
        loss = log_loss(y[val], model.predict_proba(x[val])[:, 1])
        if best is None or loss < best[0]:
            best = (loss, depth, model)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(STUDY / "3_frozen_configs" / "analysis_config.toml"))
    parser.add_argument("--output", default=str(STUDY / "15_external_validation_wales"))
    parser.add_argument("--part", choices=("teacher", "student"), default="teacher")
    args = parser.parse_args()
    if args.part == "student":
        run_student(args)
        return

    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config)
    data = load_internal_cohort(config)
    train, val = data.idx_train, data.idx_val
    y10 = data.event.astype(np.float32)
    raw_w, entity_w, samples_w, event_w, duration_w, audit_w = _load_wales(config)
    raw_num_w = raw_w.set_index(entity_w)[samples_w].apply(pd.to_numeric, errors="coerce")

    teacher_matrix = data.raw.loc[data.raw[data.entity_col].ne("PRS")].copy()
    base_kwargs = dict(
        mapping=data.mapping, pathways=data.pathways, entity_col=data.entity_col,
        binn_root=config.raw["data"]["binn_root"], device=device,
        n_layers=int(config.raw["training"]["n_layers"]), n_intervals=config.n_intervals,
        attention_dim=int(config.raw["training"]["attention_dim"]),
    )
    set_global_seed(7)
    frozen = build_model(data_matrix=teacher_matrix, **base_kwargs)
    features = list(frozen.binn.inputs)
    fitted = fit_preprocessor(data, features)
    x, prs = fitted.matrix(features), fitted.prs()
    x_w, coverage = _external_matrix(raw_num_w, samples_w, features, fitted)
    prs_w, _ = _external_matrix(raw_num_w, samples_w, ["PRS"], fitted)
    print(f"[{(time.time() - started) / 60:.1f} min] Wales n={len(samples_w)} events={int(event_w.sum())}; "
          f"observed value fraction {coverage['observed_value_fraction']:.3f}", flush=True)

    risks: dict[str, np.ndarray] = {}
    probability = {}
    frozen.load_state_dict(torch.load(config.raw["selected_models"]["teacher_checkpoint"], map_location=device,
                                      weights_only=True))
    risks["teacher_frozen_seed7"] = predict(frozen, x_w, prs_w, device).cumulative_incidence[:, 9].astype(float)
    probability["teacher_frozen_seed7"] = True

    selection = stage1_selection(STUDY / "14_repeated_split_validation" / "binn_improvements"
                                 / "binary_direct_l1_1e-2_seed7_by_split.csv")["frozen"]
    for name, relaxed in (("binn_10y_ensemble5", False), ("binn_relaxed_ensemble5", True)):
        member_risks, val_risks = [], []
        for seed in ENSEMBLE_SEEDS:
            set_global_seed(seed)
            base = build_model(data_matrix=teacher_matrix, **base_kwargs)
            columns = [features.index(f) for f in selection] if relaxed else None
            model = ImprovedBINN(base, len(features), direct=relaxed, direct_columns=columns).to(device)
            fit(model, x, prs, data, y10, train, val, device, seed, settings, binary=True, l1=None)
            member_risks.append(risk_10y(model, x_w, prs_w, device, binary=True))
            val_risks.append(risk_10y(model, x, prs, device, binary=True)[val])
            print(f"[{(time.time() - started) / 60:.1f} min] {name} seed {seed} done", flush=True)
        risks[name] = np.mean(member_risks, axis=0)
        probability[name] = True
        print(f"   internal val10 AUC of {name}: {roc_auc_score(y10[val], np.mean(val_risks, axis=0)):.4f}", flush=True)

    omics = list(dict.fromkeys([*data.nmr_features, *data.protein_features]))
    ml_features = [*omics, "PRS"]
    ml_fitted = fit_preprocessor(data, omics)
    x_ml = ml_fitted.matrix(ml_features)
    x_ml_w, _ = _external_matrix(raw_num_w, samples_w, ml_features, ml_fitted)
    y = data.event.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb = build_classifier("xgboost", 7).fit(x_ml[train], y[train])
        risks["xgboost_default"] = xgb.predict_proba(x_ml_w)[:, 1]
        loss, depth, xgb_t = tuned_xgboost(x_ml, y, train, val)
        risks["xgboost_tuned"] = xgb_t.predict_proba(x_ml_w)[:, 1]
        print(f"   tuned XGBoost: depth {depth}, trees {xgb_t.best_iteration + 1}, val log-loss {loss:.4f}", flush=True)
        lgbm = build_classifier("lightgbm", 7).fit(x_ml[train], y[train])
        risks["lightgbm_default"] = lgbm.predict_proba(x_ml_w)[:, 1]
        k, p = ml_features.index("P07288"), ml_features.index("PRS")
        logistic = LogisticRegression(C=1.0, max_iter=5000).fit(x_ml[np.ix_(train, [k, p])], y[train])
        risks["logistic_klk3_prs"] = logistic.predict_proba(x_ml_w[:, [k, p]])[:, 1]
        risks["klk3_alone"] = x_ml_w[:, k].astype(float)
    probability.update(xgboost_default=True, xgboost_tuned=True, lightgbm_default=False,
                       logistic_klk3_prs=True, klk3_alone=False)

    event = np.asarray(event_w, dtype=int)
    duration = np.asarray(duration_w, dtype=float)
    table = pd.DataFrame([{"model": name, **metrics(event, duration, risk, probability[name])}
                          for name, risk in risks.items()])
    table = table.merge(bootstrap(event, risks, "xgboost_default"), on="model")
    table.to_csv(output / "wales_teacher_metrics.csv", index=False)
    pd.DataFrame({"sample": samples_w, "event_10y": event, "observed_time_years": duration, **risks}).to_csv(
        output / "wales_teacher_predictions.csv", index=False)
    (output / "wales_teacher_manifest.json").write_text(json.dumps({
        "wales_audit": audit_w, "coverage": {k: v for k, v in coverage.items() if k != "missing_features"},
        "fit": "internal frozen split: train80 fit, val10 early stopping / tuning",
        "relaxed_direct_features": selection, "ensemble_seeds": ENSEMBLE_SEEDS,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 260)
    show = ["model", "auc_10y", "auc_ci_lower", "auc_ci_upper", "pr_auc_10y", "c_index", "brier_10y",
            "observed_expected_10y", "delta_auc_vs_xgboost_default", "delta_auc_ci_lower", "delta_auc_ci_upper",
            "delta_pr_vs_xgboost_default"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def run_student(args) -> None:
    """Student variants (NMR + PRS) fitted on the internal frozen split, scored once on Wales."""
    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = resolve_device(str(config.raw["training"]["device"]))
    settings = settings_from_config(config)
    data = load_internal_cohort(config)
    train, val = data.idx_train, data.idx_val
    y10 = data.event.astype(np.float32)
    raw_w, entity_w, samples_w, event_w, duration_w, audit_w = _load_wales(config)
    raw_num_w = raw_w.set_index(entity_w)[samples_w].apply(pd.to_numeric, errors="coerce")
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
    fitted = fit_preprocessor(data, teacher_features)
    x_t, prs = fitted.matrix(teacher_features), fitted.prs()
    prs_w, _ = _external_matrix(raw_num_w, samples_w, ["PRS"], fitted)

    # Distillation sources, as in repeated_split_students.py: frozen seed-7 Teacher and improved Teacher
    teacher.load_state_dict(torch.load(config.raw["selected_models"]["teacher_checkpoint"], map_location=device,
                                       weights_only=True))
    frozen_teacher_logits = predict(teacher, x_t, prs, device).hazard_logits
    teacher.load_state_dict(teacher_init)
    selection = stage1_selection(STUDY / "14_repeated_split_validation" / "binn_improvements"
                                 / "binary_direct_l1_1e-2_seed7_by_split.csv")["frozen"]
    improved = ImprovedBINN(teacher, len(teacher_features), direct=True,
                            direct_columns=[teacher_features.index(f) for f in selection]).to(device)
    fit(improved, x_t, prs, data, y10, train, val, device, 7, settings, binary=True, l1=None)
    p = np.clip(risk_10y(improved, x_t, prs, device, binary=True), 1e-6, 1 - 1e-6)
    improved_logit = np.log(p / (1 - p)).astype(np.float32)
    print(f"[{(time.time() - started) / 60:.1f} min] distillation Teachers ready", flush=True)

    risks: dict[str, np.ndarray] = {}
    probability: dict[str, bool] = {}
    per_seed: dict[str, list[np.ndarray]] = {v: [] for v in ("frozen_nodistill", "frozen_distilled", "binary", "binary_distilled")}
    for seed in ENSEMBLE_SEEDS:
        set_global_seed(seed)
        student = build_model(data_matrix=student_matrix, **kwargs)
        init = {k: v.detach().clone() for k, v in student.state_dict().items()}
        features = list(student.binn.inputs)
        x_s = fitted.matrix(features)
        x_s_w, _ = _external_matrix(raw_num_w, samples_w, features, fitted)
        for variant in per_seed:
            student.load_state_dict(init)
            if variant.startswith("frozen"):
                with tempfile.TemporaryDirectory() as tmp:
                    fit_model(student, x_s, prs, data.targets, data.interval_mask, train, val, device, tmp, variant,
                              seed, settings,
                              teacher_logits=frozen_teacher_logits if variant == "frozen_distilled" else None)
                per_seed[variant].append(predict(student, x_s_w, prs_w, device).cumulative_incidence[:, 9].astype(float))
            else:
                model = ImprovedBINN(student, len(features), direct=False).to(device)
                fit(model, x_s, prs, data, y10, train, val, device, seed, settings, binary=True, l1=None,
                    soft_logits=improved_logit if variant == "binary_distilled" else None)
                per_seed[variant].append(risk_10y(model, x_s_w, prs_w, device, binary=True))
        print(f"[{(time.time() - started) / 60:.1f} min] student seed {seed} done", flush=True)
    for variant, members in per_seed.items():
        risks[f"student_{variant}_seed7"] = members[0]
        risks[f"student_{variant}_ensemble5"] = np.mean(members, axis=0)
        probability[f"student_{variant}_seed7"] = probability[f"student_{variant}_ensemble5"] = True

    ml_features = [*data.nmr_features, "PRS"]
    ml_fitted = fit_preprocessor(data, list(data.nmr_features))
    x_ml = ml_fitted.matrix(ml_features)
    x_ml_w, _ = _external_matrix(raw_num_w, samples_w, ml_features, ml_fitted)
    y = data.event.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algorithm in ("logistic", "xgboost", "lightgbm"):
            model = build_classifier(algorithm, 7).fit(x_ml[train], y[train])
            risks[f"{algorithm}_nmr_prs"] = model.predict_proba(x_ml_w)[:, 1]
            probability[f"{algorithm}_nmr_prs"] = algorithm == "xgboost"
        loss, depth, xgb_t = tuned_xgboost(x_ml, y, train, val)
        risks["xgboost_tuned_nmr_prs"] = xgb_t.predict_proba(x_ml_w)[:, 1]
        probability["xgboost_tuned_nmr_prs"] = True
        p_col = ml_features.index("PRS")
        prs_model = LogisticRegression(C=1.0, max_iter=5000).fit(x_ml[train][:, [p_col]], y[train])
        risks["logistic_prs_only"] = prs_model.predict_proba(x_ml_w[:, [p_col]])[:, 1]
        probability["logistic_prs_only"] = True

    event = np.asarray(event_w, dtype=int)
    duration = np.asarray(duration_w, dtype=float)
    table = pd.DataFrame([{"model": name, **metrics(event, duration, risk, probability[name])}
                          for name, risk in risks.items()])
    table = table.merge(bootstrap(event, risks, "logistic_nmr_prs"), on="model")
    table.to_csv(output / "wales_student_metrics.csv", index=False)
    pd.DataFrame({"sample": samples_w, "event_10y": event, "observed_time_years": duration, **risks}).to_csv(
        output / "wales_student_predictions.csv", index=False)
    (output / "wales_student_manifest.json").write_text(json.dumps({
        "wales_audit": audit_w, "fit": "internal frozen split: train80 fit, val10 early stopping / tuning",
        "distillation_sources": {"frozen_distilled": "frozen seed-7 Teacher checkpoint",
                                 "binary_distilled": f"improved seed-7 Teacher (direct path {selection})"},
        "ensemble_seeds": ENSEMBLE_SEEDS, "elapsed_minutes": (time.time() - started) / 60.0,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 260)
    show = ["model", "auc_10y", "auc_ci_lower", "auc_ci_upper", "pr_auc_10y", "c_index", "brier_10y",
            "observed_expected_10y", "delta_auc_vs_logistic_nmr_prs", "delta_auc_ci_lower", "delta_auc_ci_upper"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
