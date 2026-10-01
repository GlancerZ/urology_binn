from __future__ import annotations

"""Rebuild the final seed-7 MAPS/Student-MAPS Sankey and GradientSHAP outputs.

The visual and attribution definitions intentionally match
``17_discrete_time_survival_sankey_shap_seed10`` so the seed-7 figures are
directly comparable with the exploratory seed-10 figures.
"""

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


STUDY = Path(__file__).resolve().parents[2]
PROJECT = STUDY.parent
PACKAGE_SRC = STUDY / "2_internal_python_package" / "src"
CONFIG_PATH = STUDY / "3_frozen_configs" / "analysis_config.toml"
SEED_DIR = STUDY / "5_teacher_student_multiseed" / "seed_007"
TEACHER_CHECKPOINT = SEED_DIR / "1_teacher" / "teacher_best_seed7.pt"
STUDENT_CHECKPOINT = SEED_DIR / "2_student_distilled" / "student_distilled_best_seed7.pt"
PREPROCESS_PATH = SEED_DIR / "preprocessing_parameters.csv"
OUTPUT_ROOT = STUDY / "11_seed7_frozen_maps_student_maps_interpretability"

MAPS_SANKEY_DIR = OUTPUT_ROOT / "1_MAPS_hierarchy_sankey"
STUDENT_SANKEY_DIR = OUTPUT_ROOT / "2_Student-MAPS_hierarchy_sankey"
MAPS_SHAP_DIR = OUTPUT_ROOT / "3_MAPS_SHAP"
STUDENT_SHAP_DIR = OUTPUT_ROOT / "4_Student-MAPS_SHAP"
COMPARISON_DIR = OUTPUT_ROOT / "5_MAPS_Student-MAPS_comparison"

REFERENCE_SCRIPT = PROJECT / "5_scripts" / "4_reports_and_visualizations" / "18_build_discrete_survival_sankey_shap.py"
SANKEY_SCRIPT = PROJECT / "5_scripts" / "4_reports_and_visualizations" / "12_build_simple_dual_path_sankey.py"

BACKGROUND_SIZE = 128
EXPLAIN_TEST_SIZE = 512
GRADIENT_SHAP_DRAWS = 64


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def standardize_feature_by_sample(data, features: list[str]) -> np.ndarray:
    params = pd.read_csv(PREPROCESS_PATH, dtype={"feature": str}).set_index("feature")
    ordered = features + ["PRS"]
    missing = [feature for feature in ordered if feature not in params.index]
    if missing:
        raise RuntimeError(f"Missing preprocessing parameters for {len(missing)} features")
    matrix = data.raw_num.loc[ordered, data.samples]
    values = matrix.to_numpy(dtype=np.float32, copy=True)
    fill = params.loc[ordered, "fill_value_train"].to_numpy(dtype=np.float32)
    fill = np.nan_to_num(fill, nan=0.0)
    missing_row, missing_col = np.where(np.isnan(values))
    if len(missing_row):
        values[missing_row, missing_col] = fill[missing_row]
    mean = params.loc[ordered, "mean_train"].to_numpy(dtype=np.float32)
    std = params.loc[ordered, "standard_deviation_train"].to_numpy(dtype=np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    values -= mean[:, None]
    values /= std[:, None]
    values[~np.isfinite(values)] = 0.0
    return values


def data_view(data) -> dict:
    return {
        "samples": data.samples,
        "event": data.event,
        "duration_years": data.duration_years,
        "idx_train": data.idx_train,
        "idx_test": data.idx_test,
    }


def save_sankey(
    sankey,
    model_key: str,
    model_name: str,
    model: torch.nn.Module,
    checkpoint: Path,
    out_dir: Path,
    contributions: np.ndarray,
    feature_types: dict[str, str],
    names: dict[str, str],
) -> None:
    network = sankey.extract_network(model)
    h6_count = len(network["nodes"][5])
    h6_importance = np.abs(contributions[:, :h6_count]).mean(axis=0)
    prs_importance = float(np.abs(contributions[:, -1]).mean())
    edges, input_importance, layer_importance, _ = sankey.backward_information_flow(
        network, h6_importance, feature_types, nmr_only=False
    )
    if model_key == "maps":
        input_note = "Input命名Top 15血浆蛋白和Top 5 NMR代谢物"
    else:
        input_note = "Input命名Top 15 NMR代谢物"
    sankey.save_view(
        out_dir,
        f"{model_key}_seed7_discrete_survival_hierarchy_sankey",
        f"{model_name}（seed 7）：层级通路信息流",
        f"H1-H6每层展示Top 10通路且顶部对齐；{input_note}；节点高度随重要性单调变化并设上限；PRS置于H6列底部。",
        edges,
        input_importance,
        layer_importance,
        network,
        feature_types,
        names,
        prs_importance,
        False,
        {
            "model": model_name,
            "training_seed": 7,
            "checkpoint": str(checkpoint),
            "outcome": "shared log-risk driving ten discrete annual hazards",
            "importance_method": "mean absolute H6/PRS contribution across the full cohort, allocated backward by trained masked absolute weights",
            "attention_regularization_lambda": 0.0,
            "display_rule": f"Top 10 pathways per layer with aligned tops; {input_note}; capped monotonic node-height mapping; PRS at bottom of H6 column",
            "comparison_reference": str(REFERENCE_SCRIPT),
        },
    )


def save_shap_outputs(
    reference,
    model_key: str,
    model_name: str,
    model: torch.nn.Module,
    checkpoint: Path,
    out_dir: Path,
    data: dict,
    features: list[str],
    feature_types: dict[str, str],
    names: dict[str, str],
    standardized: np.ndarray,
    background_idx: np.ndarray,
    explain_idx: np.ndarray,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = reference.SharedLogRiskWrapper(model).to(device).eval()
    batch_size = reference.TEACHER_SHAP_BATCH if model_key == "teacher" else reference.STUDENT_SHAP_BATCH
    draw_chunk = reference.TEACHER_DRAW_CHUNK if model_key == "teacher" else reference.STUDENT_DRAW_CHUNK
    shap_values, quality, predicted_log_risk = reference.expected_gradients(
        wrapper,
        standardized,
        background_idx,
        explain_idx,
        device,
        batch_size,
        draw_chunk,
        model_key,
    )
    all_features = features + ["PRS"]
    explained_values = np.ascontiguousarray(standardized[:, explain_idx].T)
    file_stem = "maps" if model_key == "teacher" else "student_maps"
    np.savez_compressed(
        out_dir / f"{file_stem}_seed7_gradientshap_test_subset.npz",
        sample=np.asarray(data["samples"])[explain_idx],
        feature=np.asarray(all_features),
        shap_values=shap_values,
        standardized_values=explained_values,
        event_10y=data["event"][explain_idx],
        observed_time_years=data["duration_years"][explain_idx],
        predicted_log_risk=predicted_log_risk,
    )
    summary = reference.feature_summary(
        all_features, feature_types, names, explained_values, shap_values
    )
    summary.to_csv(out_dir / "shap_feature_summary_all.csv", index=False, encoding="utf-8-sig")
    summary.head(30).to_csv(out_dir / "shap_top30_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "sample": np.asarray(data["samples"])[explain_idx],
            "event_10y": data["event"][explain_idx],
            "observed_time_years": data["duration_years"][explain_idx],
            "predicted_log_risk": predicted_log_risk,
        }
    ).to_csv(out_dir / "shap_explained_test_subset.csv", index=False, encoding="utf-8-sig")

    reference.plot_beeswarm(model_name, out_dir, summary, all_features, explained_values, shap_values)
    reference.plot_importance_direction(model_name, out_dir, summary)
    reference.plot_dependence(
        model_name, out_dir, summary, all_features, explained_values, shap_values, explained_values[:, -1]
    )
    modality = reference.plot_modality_and_horizons(
        model_name,
        out_dir,
        summary,
        predicted_log_risk,
        model.baseline_logits.detach().cpu().numpy().astype(float),
    )
    metadata = {
        "model": model_name,
        "training_seed": 7,
        "checkpoint": str(checkpoint),
        "shap_method": "GradientSHAP / expected gradients",
        "shap_target": "shared continuous log-risk score used by all ten annual hazards",
        "interpretation": "positive SHAP raises cumulative disease risk at 1, 3, 5 and 10 years; negative SHAP lowers it",
        "test_subset_sampling": "seed-7 stratified proportional sample from the fixed test10 set",
        "n_test_total": int(len(data["idx_test"])),
        "n_test_explained": int(len(explain_idx)),
        "n_test_events_explained": int(data["event"][explain_idx].sum()),
        "n_background": int(len(background_idx)),
        "n_features_including_prs": int(len(all_features)),
        "quality": quality,
        "omics_summary": modality.to_dict("records"),
        "top10_features": summary.head(10).to_dict("records"),
        "comparison_reference": str(REFERENCE_SCRIPT),
    }
    (out_dir / "shap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if str(PACKAGE_SRC) not in sys.path:
        sys.path.insert(0, str(PACKAGE_SRC))
    from maps_discrete.config import load_config
    from maps_discrete.data import load_internal_cohort
    from maps_discrete.modeling import build_model

    reference = load_module("seed7_interpretability_reference", REFERENCE_SCRIPT)
    sankey = load_module("seed7_sankey_helpers", SANKEY_SCRIPT)
    reference.SEED = 7
    reference.BACKGROUND_SIZE = BACKGROUND_SIZE
    reference.EXPLAIN_TEST_SIZE = EXPLAIN_TEST_SIZE
    reference.GRADIENT_SHAP_DRAWS = GRADIENT_SHAP_DRAWS
    reference.COMPARISON_DIR = COMPARISON_DIR
    sankey.SAVE_PDF = True
    # Match the exact display convention currently stored in directory 17.
    sankey.COLLAPSE_SAME_ID_CHAINS = False
    reference.add_fonts()

    for folder in [MAPS_SANKEY_DIR, STUDENT_SANKEY_DIR, MAPS_SHAP_DIR, STUDENT_SHAP_DIR, COMPARISON_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    cohort = load_internal_cohort(config)
    data = data_view(cohort)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type != "cuda":
        raise RuntimeError("This analysis is configured to run on GPU, but CUDA is unavailable")

    model_kwargs = {
        "mapping": cohort.mapping,
        "pathways": cohort.pathways,
        "entity_col": cohort.entity_col,
        "binn_root": config.raw["data"]["binn_root"],
        "device": device,
        "n_layers": 6,
        "n_intervals": 10,
        "attention_dim": 32,
    }
    feature_types = {str(x): "NMR" for x in cohort.nmr_features} | {
        str(x): "Protein" for x in cohort.protein_features
    } | {"PRS": "PRS"}
    names = sankey.build_name_map(cohort.mapping, SEED_DIR / "1_teacher")
    names["PRS"] = "前列腺癌多基因风险评分（PRS）"

    background_idx = reference.stratified_sample(
        cohort.idx_train, cohort.event, BACKGROUND_SIZE, 7 + 101
    )
    explain_idx = reference.stratified_sample(
        cohort.idx_test, cohort.event, EXPLAIN_TEST_SIZE, 7 + 202
    )
    pd.DataFrame(
        {
            "sample": np.asarray(cohort.samples)[explain_idx],
            "event_10y": cohort.event[explain_idx],
            "observed_time_years": cohort.duration_years[explain_idx],
        }
    ).to_csv(OUTPUT_ROOT / "shap_test_subset_seed7.csv", index=False, encoding="utf-8-sig")

    teacher_matrix = cohort.raw.loc[cohort.raw[cohort.entity_col].ne("PRS")].copy()
    teacher = build_model(data_matrix=teacher_matrix, **model_kwargs)
    teacher.load_state_dict(
        torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=True), strict=True
    )
    teacher.eval()
    teacher_features = list(map(str, teacher.binn.inputs))
    teacher_standardized = standardize_feature_by_sample(cohort, teacher_features)
    teacher_contributions = reference.recompute_contributions(teacher, teacher_standardized, device)
    np.save(MAPS_SANKEY_DIR / "maps_h6_prs_contributions_all.npy", teacher_contributions)
    save_sankey(
        sankey, "maps", "MAPS", teacher, TEACHER_CHECKPOINT, MAPS_SANKEY_DIR,
        teacher_contributions, feature_types, names,
    )
    teacher_summary_path = MAPS_SHAP_DIR / "shap_feature_summary_all.csv"
    if args.resume and teacher_summary_path.exists():
        teacher_summary = pd.read_csv(teacher_summary_path, dtype={"feature_id": str})
        teacher_metadata = json.loads((MAPS_SHAP_DIR / "shap_metadata.json").read_text(encoding="utf-8"))
    else:
        teacher_summary, teacher_metadata = save_shap_outputs(
            reference, "teacher", "MAPS", teacher, TEACHER_CHECKPOINT, MAPS_SHAP_DIR,
            data, teacher_features, feature_types, names, teacher_standardized,
            background_idx, explain_idx, device,
        )
    del teacher_contributions, teacher_standardized, teacher, teacher_matrix
    gc.collect()
    torch.cuda.empty_cache()

    student_matrix = cohort.raw.loc[cohort.raw[cohort.entity_col].isin(cohort.nmr_features)].copy()
    student = build_model(data_matrix=student_matrix, **model_kwargs)
    student.load_state_dict(
        torch.load(STUDENT_CHECKPOINT, map_location=device, weights_only=True), strict=True
    )
    student.eval()
    student_features = list(map(str, student.binn.inputs))
    student_standardized = standardize_feature_by_sample(cohort, student_features)
    student_contributions = reference.recompute_contributions(student, student_standardized, device)
    np.save(STUDENT_SANKEY_DIR / "student_maps_h6_prs_contributions_all.npy", student_contributions)
    save_sankey(
        sankey, "student_maps", "Student-MAPS", student, STUDENT_CHECKPOINT,
        STUDENT_SANKEY_DIR, student_contributions, feature_types, names,
    )
    student_summary_path = STUDENT_SHAP_DIR / "shap_feature_summary_all.csv"
    if args.resume and student_summary_path.exists():
        student_summary = pd.read_csv(student_summary_path, dtype={"feature_id": str})
        student_metadata = json.loads((STUDENT_SHAP_DIR / "shap_metadata.json").read_text(encoding="utf-8"))
    else:
        student_summary, student_metadata = save_shap_outputs(
            reference, "student", "Student-MAPS", student, STUDENT_CHECKPOINT,
            STUDENT_SHAP_DIR, data, student_features, feature_types, names,
            student_standardized, background_idx, explain_idx, device,
        )

    comparison = reference.plot_shared_nmr_comparison(teacher_summary, student_summary)
    summary = {
        "analysis": "Frozen seed-7 MAPS and paired Student-MAPS Sankey/GradientSHAP",
        "training_seed": 7,
        "device": str(device),
        "fixed_test_n": int(len(cohort.idx_test)),
        "fixed_test_events": int(cohort.event[cohort.idx_test].sum()),
        "shap_explained_test_n": int(len(explain_idx)),
        "shap_explained_events": int(cohort.event[explain_idx].sum()),
        "maps_checkpoint": str(TEACHER_CHECKPOINT),
        "student_maps_checkpoint": str(STUDENT_CHECKPOINT),
        "maps": teacher_metadata,
        "student_maps": student_metadata,
        "shared_nmr_prs_comparison": comparison,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# 冻结seed 7模型：MAPS与Student-MAPS桑基图及SHAP\n\n"
        "- MAPS和Student-MAPS均来自training seed 7；Student-MAPS由seed 7 MAPS蒸馏。\n"
        "- 桑基图与目录17采用同一算法和视觉参数：MAPS输入显示Top 15蛋白+Top 5 NMR，Student-MAPS显示Top 15 NMR；H1-H6各显示Top 10通路；PRS位于H6列底部。\n"
        "- `Other`是未显示节点的绘图聚合，`Residual channel`是网络真实路由，二者不能混同。\n"
        "- SHAP采用GradientSHAP/expected gradients，训练集背景128人，固定测试集分层抽样512人，每人64次插值。\n"
        "- 正SHAP表示提高共享log-risk，因此同时提高1/3/5/10年累计发病风险；SHAP不等于因果效应。\n"
        "- 每套SHAP包含蜂群图、全局重要性与方向、依赖图、组学来源及时间窗映射，并保存完整逐样本数组和汇总表。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT_ROOT), "runtime_minutes": summary["runtime_minutes"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
