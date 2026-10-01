from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import load_internal_cohort, load_reactome
from .endpoint import fixed_horizon_status, years_from_days
from .final_evaluation import ModelPrediction, evaluate_models
from .ml import build_classifier, modality_features
from .model_comparison_two import MODEL_DEFINITIONS, _clinical_frame
from .modeling import build_model
from .pipeline import resolve_device
from .prediction import predict
from .preprocessing import fit_preprocessor
from .seeding import set_global_seed


ML_SELECTIONS = [
    ("nmr_only", "lightgbm", 20, "nmr_only_lightgbm", "NMR-only LightGBM (seed 20)", "#2A9D8F"),
    ("prs_only", "logistic", 1, "prs_only_logistic", "PRS-only Logistic (seed 1)", "#8C6D31"),
    ("nmr_prs", "logistic", 1, "nmr_prs_logistic", "NMR+PRS Logistic (seed 1)", "#4C78A8"),
    ("nmr_olink_prs", "xgboost", 93, "full_omics_xgboost", "NMR+Olink+PRS XGBoost (seed 93)", "#C43C39"),
]


def _summary_table(results: dict, comparison_set: str) -> pd.DataFrame:
    performance = results["performance"].set_index("model_id")
    comparisons = results["comparison"].set_index("model_id")
    rows = []
    for model_id, perf in performance.iterrows():
        row = {
            "comparison_set": comparison_set,
            "model_id": model_id,
            "model": perf["model"],
            "reference_model": "",
        }
        for metric in ("c_index", "auc", "pr_auc", "brier"):
            row[metric] = float(perf[metric])
            row[f"{metric}_ci_lower"] = float(perf[f"{metric}_ci_lower"])
            row[f"{metric}_ci_upper"] = float(perf[f"{metric}_ci_upper"])
        if model_id in comparisons.index:
            comp = comparisons.loc[model_id]
            row["reference_model"] = comp["reference_model"]
            for metric in ("delta_c_index", "delta_auc", "nri", "idi"):
                row[metric] = float(comp[metric])
                row[f"{metric}_ci_lower"] = float(comp[f"{metric}_ci_lower"])
                row[f"{metric}_ci_upper"] = float(comp[f"{metric}_ci_upper"])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_wales(config) -> tuple[pd.DataFrame, str, list[str], np.ndarray, np.ndarray, dict]:
    raw = pd.read_csv(config.raw["data"]["wales_matrix"])
    raw.columns = raw.columns.astype(str)
    entity_col = str(raw.columns[0])
    design = pd.read_csv(config.raw["data"]["wales_design"], dtype={"sample": str})
    samples = [column for column in raw.columns[1:] if column in set(design["sample"])]
    design = design.set_index("sample").loc[samples]
    health = pd.read_csv(
        config.raw["data"]["wales_survival"], dtype={"eid": str}, low_memory=False
    ).set_index("eid").loc[samples]
    event = pd.to_numeric(health["occur_in_10y"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(event, pd.to_numeric(design["group"]).to_numpy(np.int64)):
        raise RuntimeError("Wales design and endpoint event labels do not match")

    baseline = pd.to_datetime(health["baseline_date"], errors="coerce").reset_index(drop=True)
    deadline = pd.to_datetime(health["deadline_10y"], errors="coerce").reset_index(drop=True)
    event_date = pd.to_datetime(health["t_event"], errors="coerce").reset_index(drop=True)
    censor_date = pd.to_datetime(health["t_censored"], errors="coerce").reset_index(drop=True)
    if baseline.isna().any() or deadline.isna().any() or (event.astype(bool) & event_date.isna()).any():
        raise RuntimeError("Invalid Wales endpoint dates")
    if np.any((event == 1) & ((event_date - baseline).dt.days.to_numpy(float) <= 0)):
        raise RuntimeError("Wales contains pre-baseline events")
    censor_end = pd.concat([censor_date, deadline], axis=1).min(axis=1)
    end_date = censor_end.copy()
    event_positions = np.flatnonzero(event == 1)
    end_date.iloc[event_positions] = event_date.iloc[event_positions].to_numpy()
    duration = years_from_days(
        (end_date - baseline).dt.days.to_numpy(float), config.days_per_year
    )
    reached_deadline = (censor_end >= (deadline - pd.Timedelta(days=1))).to_numpy()
    duration[(event == 0) & reached_deadline] = float(config.n_intervals)
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0):
        raise RuntimeError("Wales survival duration must be finite and positive")
    audit = {
        "n": len(samples),
        "events_10y": int(event.sum()),
        "early_censored_before_10y": int(((event == 0) & ~reached_deadline).sum()),
        "days_per_year": int(config.days_per_year),
    }
    return raw, entity_col, samples, event, duration.astype(np.float32), audit


def _external_matrix(raw_num: pd.DataFrame, samples: list[str], features: list[str], fitted) -> tuple[np.ndarray, dict]:
    rows = []
    missing_features = []
    observed_values = 0
    total_values = len(features) * len(samples)
    for feature in features:
        if feature in raw_num.index:
            values = pd.to_numeric(raw_num.loc[feature, samples], errors="coerce")
            observed_values += int(values.notna().sum())
        else:
            values = pd.Series(np.nan, index=samples, dtype=float)
            missing_features.append(feature)
        fill = float(fitted.fill_values[feature])
        mean = float(fitted.means[feature])
        std = float(fitted.standard_deviations[feature])
        rows.append(((values.fillna(fill).to_numpy(float) - mean) / std).astype(np.float32))
    matrix = np.column_stack(rows).astype(np.float32)
    coverage = {
        "requested_features": len(features),
        "missing_feature_rows": len(missing_features),
        "missing_features": missing_features,
        "observed_value_fraction": float(observed_values / total_values) if total_values else 1.0,
    }
    return matrix, coverage


def _load_checkpoint(model, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def _neural_predictions(config, raw, entity_col, samples, output: Path):
    mapping, pathways, mapping_source = load_reactome(config)
    device = resolve_device(str(config.raw["training"]["device"]))
    nmr_features = pd.read_csv(config.raw["data"]["nmr_feature_list"]).iloc[:, 0].astype(str).tolist()
    teacher_matrix = raw.loc[raw[entity_col].astype(str).ne("PRS")].copy()
    student_matrix = raw.loc[raw[entity_col].astype(str).isin(nmr_features)].copy()
    kwargs = {
        "mapping": mapping,
        "pathways": pathways,
        "entity_col": entity_col,
        "binn_root": config.raw["data"]["binn_root"],
        "device": device,
        "n_layers": int(config.raw["training"]["n_layers"]),
        "n_intervals": config.n_intervals,
        "attention_dim": int(config.raw["training"]["attention_dim"]),
    }
    teacher = build_model(data_matrix=teacher_matrix, **kwargs)
    student = build_model(data_matrix=student_matrix, **kwargs)
    nondistilled = build_model(data_matrix=student_matrix, **kwargs)
    selected = config.raw["selected_models"]
    _load_checkpoint(teacher, selected["teacher_checkpoint"], device)
    _load_checkpoint(student, selected["distilled_student_checkpoint"], device)
    _load_checkpoint(nondistilled, selected["non_distilled_student_reference_checkpoint"], device)

    parameters = pd.read_csv(
        Path(config.path).resolve().parents[1]
        / "5_teacher_student_multiseed" / "seed_007" / "preprocessing_parameters.csv"
    ).set_index("feature")
    raw_num = raw.set_index(entity_col)[samples].apply(pd.to_numeric, errors="coerce")

    class FrozenPreprocessor:
        pass

    fitted = FrozenPreprocessor()
    fitted.fill_values = parameters["fill_value_train"].astype(float).to_dict()
    fitted.means = parameters["mean_train"].astype(float)
    fitted.standard_deviations = parameters["standard_deviation_train"].astype(float)
    teacher_x, teacher_coverage = _external_matrix(raw_num, samples, list(teacher.binn.inputs), fitted)
    student_x, student_coverage = _external_matrix(raw_num, samples, list(student.binn.inputs), fitted)
    prs, prs_coverage = _external_matrix(raw_num, samples, ["PRS"], fitted)
    teacher_bundle = predict(teacher, teacher_x, prs, device)
    student_bundle = predict(student, student_x, prs, device)
    nondistilled_bundle = predict(nondistilled, student_x, prs, device)
    manifest = {
        "device": str(device),
        "mapping_source": mapping_source,
        "teacher_coverage": teacher_coverage,
        "student_coverage": student_coverage,
        "prs_coverage": prs_coverage,
        "checkpoints": {
            "teacher": selected["teacher_checkpoint"],
            "student_distilled": selected["distilled_student_checkpoint"],
            "student_no_distillation": selected["non_distilled_student_reference_checkpoint"],
        },
    }
    del teacher, student, nondistilled, teacher_x, student_x, prs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return teacher_bundle, student_bundle, nondistilled_bundle, raw_num, manifest


def _ml_predictions(config, raw_num, samples, output: Path):
    internal = load_internal_cohort(config, include_reactome=False)
    known, y_all = fixed_horizon_status(
        internal.duration_years, internal.event, config.primary_horizon
    )
    train_index = internal.idx_train[known[internal.idx_train]]
    predictions = []
    coverage_rows = []
    for modality, algorithm, seed, model_id, label, color in ML_SELECTIONS:
        set_global_seed(seed)
        features = modality_features(internal, modality)
        required_omics = [feature for feature in features if feature != "PRS"]
        fitted = fit_preprocessor(internal, required_omics)
        train_matrix = fitted.matrix(features)
        external, coverage = _external_matrix(raw_num, samples, features, fitted)
        classifier = build_classifier(algorithm, seed)
        classifier.fit(train_matrix[train_index], y_all[train_index])
        risk = classifier.predict_proba(external)[:, 1]
        model_path = output / "models" / f"{model_id}.joblib"
        joblib.dump(classifier, model_path)
        predictions.append(ModelPrediction(model_id, label, risk, risk, color))
        coverage_rows.append({"model_id": model_id, "modality": modality, **coverage})
        del classifier, train_matrix, external, fitted
        gc.collect()
    del internal
    gc.collect()
    pd.DataFrame(coverage_rows).to_json(
        output / "ml_external_feature_coverage.json", orient="records", indent=2
    )
    return predictions, coverage_rows


def _clinical_predictions(config, samples, event, duration, output: Path):
    cohort = pd.DataFrame({
        "sample": samples,
        "split": "external_wales",
        "event_10y": event,
        "observed_time_years": duration,
    })
    frame = _clinical_frame(config, cohort, matrix_path=config.raw["data"]["wales_matrix"])
    predictions = []
    for definition in MODEL_DEFINITIONS:
        model = joblib.load(
            Path(config.path).resolve().parents[1]
            / "8_model_comparison_2_clinical" / "models" / f"{definition['model_id']}.joblib"
        )
        features = [*definition["numeric"], *definition["categorical"]]
        risk = model.predict_proba(frame[features])[:, 1]
        predictions.append(ModelPrediction(
            definition["model_id"], definition["label"], risk, risk, definition["color"]
        ))
    frame.to_csv(output / "wales_clinical_covariates.csv", index=False)
    return predictions


def run(config_path: str | Path, output_dir: str | Path | None = None) -> dict:
    config = load_config(config_path)
    root = Path(config.path).resolve().parents[1]
    output = Path(output_dir) if output_dir else root / "9_wales_external_validation"
    (output / "models").mkdir(parents=True, exist_ok=True)
    raw, entity_col, samples, event, duration, endpoint_audit = _load_wales(config)
    teacher_bundle, student_bundle, nondistilled_bundle, raw_num, neural_manifest = _neural_predictions(
        config, raw, entity_col, samples, output
    )
    ml_predictions, ml_coverage = _ml_predictions(config, raw_num, samples, output)
    clinical_predictions = _clinical_predictions(config, samples, event, duration, output)

    teacher_prediction = ModelPrediction(
        "teacher_maps", "Teacher MAPS (seed 7)", teacher_bundle.cumulative_incidence[:, 9],
        teacher_bundle.log_risk, "#173F5F",
    )
    student_prediction = ModelPrediction(
        "student_distilled", "Distilled Student (seed 98)", student_bundle.cumulative_incidence[:, 9],
        student_bundle.log_risk, "#E07A1F",
    )
    nondistilled_prediction = ModelPrediction(
        "student_no_distillation", "Non-distilled Student (seed 87)",
        nondistilled_bundle.cumulative_incidence[:, 9], nondistilled_bundle.log_risk, "#7A5195",
    )
    patient_predictions = pd.DataFrame({
        "sample": samples,
        "event_10y": event,
        "observed_time_years": duration,
        "teacher_maps_risk_10y": teacher_prediction.risk,
        "teacher_maps_ranking_score": teacher_prediction.log_risk,
        "student_distilled_risk_10y": student_prediction.risk,
        "student_distilled_ranking_score": student_prediction.log_risk,
        "student_no_distillation_risk_10y": nondistilled_prediction.risk,
        "student_no_distillation_ranking_score": nondistilled_prediction.log_risk,
    })
    for prediction in ml_predictions:
        patient_predictions[f"{prediction.model_id}_risk_10y"] = prediction.risk
    for prediction in clinical_predictions:
        patient_predictions[f"{prediction.model_id}_risk_10y"] = prediction.risk
    patient_predictions.to_csv(output / "wales_predictions_all_models.csv", index=False)

    common = dict(
        samples=np.asarray(samples), duration=duration, event=event,
        horizon=float(config.primary_horizon),
        repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
        confidence_level=float(config.raw["evaluation"].get("confidence_level", 0.95)),
        bootstrap_seed=int(config.test_holdout_seed),
        dca_threshold_min=float(config.raw["evaluation"]["dca_threshold_min"]),
        dca_threshold_max=float(config.raw["evaluation"]["dca_threshold_max"]),
        dca_threshold_points=int(config.raw["evaluation"]["dca_threshold_points"]),
    )
    comparison_one = evaluate_models(
        **common,
        predictions=[teacher_prediction, student_prediction, nondistilled_prediction, *ml_predictions],
        output_dir=output / "1_model_comparison_1",
        reference_model_id="nmr_prs_logistic",
        comparison_reference_by_model={
            "student_distilled": "teacher_maps",
            "student_no_distillation": "teacher_maps",
        },
        recovery_pairs=[
            {
                "name": "direct_student_vs_teacher",
                "teacher": "teacher_maps",
                "student": "student_distilled",
            },
        ],
        title_prefix="Wales external comparison 1",
    )
    comparison_two = evaluate_models(
        **common,
        predictions=[*clinical_predictions, teacher_prediction, student_prediction],
        output_dir=output / "2_model_comparison_2",
        reference_model_id="base",
        comparison_reference_by_model={
            "lifestyle": "base",
            "lifestyle_klk3": "lifestyle",
            "lifestyle_klk3_prs": "lifestyle_klk3",
            "prs_only": "base",
            "teacher_maps": "lifestyle_klk3_prs",
            "student_distilled": "lifestyle_klk3_prs",
        },
        recovery_pairs=[
            {
                "name": "direct_student_vs_teacher",
                "teacher": "teacher_maps",
                "student": "student_distilled",
            },
        ],
        title_prefix="Wales external comparison 2",
    )
    summary_one = _summary_table(comparison_one, "model_comparison_1")
    summary_two = _summary_table(comparison_two, "model_comparison_2")
    summary_one.to_csv(
        output / "1_model_comparison_1" / "1_tables" / "Table_0_Wales_summary.csv",
        index=False,
    )
    summary_two.to_csv(
        output / "2_model_comparison_2" / "1_tables" / "Table_0_Wales_summary.csv",
        index=False,
    )
    top_tables = output / "1_tables"
    top_tables.mkdir(parents=True, exist_ok=True)
    pd.concat([summary_one, summary_two], ignore_index=True).to_csv(
        top_tables / "Table_0_Wales_external_validation_all_models.csv", index=False
    )
    manifest = {
        "external_validation_policy": "all preprocessing, model weights, model definitions and selected seeds frozen from internal cohort; no Wales retraining or seed selection",
        "endpoint_audit": endpoint_audit,
        "neural_inference": neural_manifest,
        "ml_models": [
            {"modality": m, "algorithm": a, "seed": s, "model_id": mid, "coverage": cov}
            for (m, a, s, mid, _label, _color), cov in zip(ML_SELECTIONS, ml_coverage)
        ],
        "clinical_note": "education omitted; Olink KLK3 P07288 is a proxy and not clinical serum PSA",
        "bootstrap_repetitions": int(config.raw["evaluation"]["bootstrap_repetitions"]),
        "dca_threshold_percent_range": [
            float(config.raw["evaluation"]["dca_threshold_min"]) * 100.0,
            float(config.raw["evaluation"]["dca_threshold_max"]) * 100.0,
        ],
        "comparison_references": {
            "model_comparison_1": {
                "teacher_and_ml": "NMR+PRS Logistic",
                "distilled_and_non_distilled_student": "Teacher MAPS",
            },
            "model_comparison_2": {
                "lifestyle": "Base",
                "lifestyle_klk3": "Lifestyle",
                "lifestyle_klk3_prs": "Lifestyle+BMI+KLK3",
                "prs_only": "Base",
                "teacher_and_student": "Lifestyle+BMI+KLK3+PRS",
            },
        },
        "student_teacher_recovery": "direct direction-aware ratio; no ML reference",
        "internal_seed_selection_note": "single checkpoints were selected by highest internal test AUC before Wales evaluation; Wales was not used for selection",
    }
    (output / "external_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Wales外部验证结果说明",
        "",
        f"Wales冻结队列共{endpoint_audit['n']}人、10年事件{endpoint_audit['events_10y']}例。所有预处理参数、模型权重、seed选择和风险定义均来自内部队列；Wales未重新训练或选择seed。",
        "",
        "所有点估计附2,000次受试者层bootstrap 95%CI。由于仅42例事件，NRI/IDI和模型间差值应谨慎解释。",
        "",
        "教育水平不可用并省略；Olink KLK3（P07288）仅作为PSA代理，不等同于临床血清PSA。",
        "",
        "DCA阈值固定为0.5%-15%。Student性能恢复比例按Student vs Teacher直接计算，不使用ML参照。",
        "",
        "## 模型比较1外部性能",
        "",
    ]
    for row in comparison_one["performance"].itertuples():
        lines.append(
            f"- {row.model}：C-index {row.c_index:.3f}，AUC {row.auc:.3f}，"
            f"PR-AUC {row.pr_auc:.3f}，Brier {row.brier:.3f}。"
        )
    lines.extend(["", "## 模型比较2外部性能", ""])
    for row in comparison_two["performance"].itertuples():
        lines.append(
            f"- {row.model}：C-index {row.c_index:.3f}，AUC {row.auc:.3f}，"
            f"PR-AUC {row.pr_auc:.3f}，Brier {row.brier:.3f}。"
        )
    recovery = comparison_two["recovery"]
    if len(recovery):
        lines.extend(["", "## Student相对Teacher的外部直接恢复比例", ""])
        for row in recovery.itertuples():
            lines.append(
                f"- {row.metric}：{row.recovery_percent:.1f}%"
                f" (95% CI {row.ci_lower_percent:.1f}% - {row.ci_upper_percent:.1f}%)。"
            )
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"comparison_one": comparison_one, "comparison_two": comparison_two, "manifest": manifest}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run frozen Wales external validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
