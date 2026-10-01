from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config
from .endpoint import fixed_horizon_status
from .final_evaluation import ModelPrediction, evaluate_models


BASELINE_COLUMNS = {
    "Participant ID": "sample",
    "At or above moderate/vigorous/walking recommendation | Instance 0": "physical_activity",
    "Smoking status | Instance 0": "smoking",
    "Alcohol intake frequency. | Instance 0": "alcohol",
}
ETHNICITY_COLUMNS = {
    "eid": "sample",
    "Age": "age",
    "Townsend": "townsend",
    "Ethnic": "ethnicity_raw",
}
BMI_COLUMNS = {
    "Participant ID": "sample",
    "Body mass index (BMI) | Instance 0": "bmi",
}


MODEL_DEFINITIONS = [
    {
        "model_id": "base",
        "label": "Base (age + ethnicity + Townsend)",
        "numeric": ["age", "townsend"],
        "categorical": ["ethnicity"],
        "color": "#4C78A8",
    },
    {
        "model_id": "lifestyle",
        "label": "Lifestyle",
        "numeric": ["age", "townsend"],
        "categorical": ["ethnicity", "smoking", "alcohol", "physical_activity"],
        "color": "#2A9D8F",
    },
    {
        "model_id": "lifestyle_klk3",
        "label": "Lifestyle + BMI + Olink KLK3 proxy",
        "numeric": ["age", "townsend", "bmi", "klk3_olink"],
        "categorical": ["ethnicity", "smoking", "alcohol", "physical_activity"],
        "color": "#8C6D31",
    },
    {
        "model_id": "lifestyle_klk3_prs",
        "label": "Lifestyle + BMI + Olink KLK3 proxy + PRS",
        "numeric": ["age", "townsend", "bmi", "klk3_olink", "prs"],
        "categorical": ["ethnicity", "smoking", "alcohol", "physical_activity"],
        "color": "#7A5195",
    },
    {
        "model_id": "prs_only",
        "label": "PRS-only",
        "numeric": ["prs"],
        "categorical": [],
        "color": "#E15759",
    },
]


def _read_matrix_rows(path: str | Path, wanted: set[str]) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        samples = [str(value) for value in header[1:]]
        rows: dict[str, list[str]] = {}
        for row in reader:
            if row and row[0] in wanted:
                rows[row[0]] = row[1:]
                if wanted.issubset(rows):
                    break
    missing = sorted(wanted - set(rows))
    if missing:
        raise KeyError(f"Matrix rows not found: {missing}")
    return pd.DataFrame(
        {feature: pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float) for feature, values in rows.items()},
        index=pd.Index(samples, name="sample"),
    )


def _clean_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "Prefer not to answer": pd.NA, "Do not know": pd.NA})
    return cleaned.fillna("Missing").astype(str)


def _collapse_ethnicity(value: str) -> str:
    if value == "Missing":
        return "Missing"
    lower = value.lower()
    if value in {"British", "Irish", "White", "Any other white background"}:
        return "White"
    if value in {"Indian", "Pakistani", "Bangladeshi", "Chinese", "Any other Asian background"}:
        return "Asian"
    if value in {"Caribbean", "African", "Any other Black background"}:
        return "Black"
    if "mixed" in lower or "white and" in lower:
        return "Mixed"
    return "Other"


def _clinical_frame(
    config,
    cohort: pd.DataFrame,
    matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    data_config = config.raw["data"]
    baseline = pd.read_csv(
        data_config["clinical_baseline_metadata"],
        usecols=list(BASELINE_COLUMNS),
        dtype={"Participant ID": str},
    ).rename(columns=BASELINE_COLUMNS)
    ethnicity = pd.read_csv(
        data_config["clinical_ethnicity_metadata"],
        usecols=list(ETHNICITY_COLUMNS),
        dtype={"eid": str},
    ).rename(columns=ETHNICITY_COLUMNS)
    bmi = pd.read_csv(
        data_config["clinical_bmi_metadata"],
        usecols=list(BMI_COLUMNS),
        dtype={"Participant ID": str},
    ).rename(columns=BMI_COLUMNS)
    molecular = _read_matrix_rows(
        matrix_path or data_config["internal_matrix"], {"P07288", "PRS"}
    )
    molecular = molecular.rename(columns={"P07288": "klk3_olink", "PRS": "prs"}).reset_index()

    frame = cohort.merge(ethnicity, on="sample", how="left", validate="one_to_one")
    frame = frame.merge(baseline, on="sample", how="left", validate="one_to_one")
    frame = frame.merge(bmi, on="sample", how="left", validate="one_to_one")
    frame = frame.merge(molecular, on="sample", how="left", validate="one_to_one")
    for column in ("age", "townsend", "bmi", "klk3_olink", "prs"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("smoking", "alcohol", "physical_activity"):
        frame[column] = _clean_category(frame[column])
    frame["ethnicity"] = _clean_category(frame["ethnicity_raw"]).map(_collapse_ethnicity)
    return frame


def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, class_weight=None)),
    ])


def run(config_path: str | Path, output_dir: str | Path | None = None) -> dict:
    config = load_config(config_path)
    root = Path(config.path).resolve().parents[1]
    output = Path(output_dir) if output_dir else root / "8_model_comparison_2_clinical"
    models_dir = output / "models"
    predictions_dir = output / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = root / "5_teacher_student_multiseed" / "seed_007" / "predictions_all.csv"
    cohort = pd.read_csv(
        cohort_path,
        usecols=["sample", "split", "event_10y", "observed_time_years"],
        dtype={"sample": str},
    )
    frame = _clinical_frame(config, cohort)
    known, status = fixed_horizon_status(
        frame["observed_time_years"].to_numpy(float),
        frame["event_10y"].to_numpy(int),
        float(config.primary_horizon),
    )
    train_mask = frame["split"].eq("train80").to_numpy() & known
    test_mask = frame["split"].eq("test10").to_numpy()

    clinical_predictions: list[ModelPrediction] = []
    prediction_table = frame[["sample", "split", "event_10y", "observed_time_years"]].copy()
    manifest_models = []
    for definition in MODEL_DEFINITIONS:
        features = [*definition["numeric"], *definition["categorical"]]
        model = _pipeline(definition["numeric"], definition["categorical"])
        model.fit(frame.loc[train_mask, features], status[train_mask])
        risk = model.predict_proba(frame[features])[:, 1]
        prediction_table[f"{definition['model_id']}_risk_10y"] = risk
        clinical_predictions.append(ModelPrediction(
            definition["model_id"], definition["label"], risk[test_mask], risk[test_mask], definition["color"]
        ))
        model_path = models_dir / f"{definition['model_id']}.joblib"
        joblib.dump(model, model_path)
        manifest_models.append({
            **definition,
            "features": features,
            "model_path": str(model_path.resolve()),
            "training_n": int(train_mask.sum()),
            "training_events": int(status[train_mask].sum()),
        })

    prediction_table.to_csv(predictions_dir / "clinical_predictions_all.csv", index=False)
    prediction_table.loc[test_mask].to_csv(predictions_dir / "clinical_predictions_test.csv", index=False)

    test_samples = frame.loc[test_mask, "sample"].astype(str).to_numpy()
    teacher = pd.read_csv(
        root / "5_teacher_student_multiseed" / "seed_007" / "predictions_test.csv", dtype={"sample": str}
    ).set_index("sample").loc[test_samples]
    student = pd.read_csv(
        root / "5_teacher_student_multiseed" / "seed_098" / "predictions_test.csv", dtype={"sample": str}
    ).set_index("sample").loc[test_samples]
    predictions = [
        *clinical_predictions,
        ModelPrediction(
            "teacher_maps", "Teacher MAPS (seed 7)", teacher["teacher_risk_10y"].to_numpy(float),
            teacher["teacher_log_risk"].to_numpy(float), "#173F5F",
        ),
        ModelPrediction(
            "student_distilled", "Distilled Student (seed 98)", student["student_distilled_risk_10y"].to_numpy(float),
            student["student_distilled_log_risk"].to_numpy(float), "#E07A1F",
        ),
    ]

    results = evaluate_models(
        samples=test_samples,
        duration=frame.loc[test_mask, "observed_time_years"].to_numpy(float),
        event=frame.loc[test_mask, "event_10y"].to_numpy(int),
        predictions=predictions,
        output_dir=output,
        reference_model_id="base",
        comparison_reference_by_model={
            "lifestyle": "base",
            "lifestyle_klk3": "base",
            "lifestyle_klk3_prs": "base",
            "prs_only": "base",
            "teacher_maps": "base",
            "student_distilled": "base",
        },
        recovery_pairs=[
            {
                "name": "direct_student_vs_teacher",
                "teacher": "teacher_maps",
                "student": "student_distilled",
            }
        ],
        horizon=float(config.primary_horizon),
        repetitions=int(config.raw["evaluation"]["bootstrap_repetitions"]),
        confidence_level=float(config.raw["evaluation"].get("confidence_level", 0.95)),
        bootstrap_seed=int(config.test_holdout_seed),
        dca_threshold_min=float(config.raw["evaluation"]["dca_threshold_min"]),
        dca_threshold_max=float(config.raw["evaluation"]["dca_threshold_max"]),
        dca_threshold_points=int(config.raw["evaluation"]["dca_threshold_points"]),
        title_prefix="Model comparison 2",
    )

    performance = results["performance"].set_index("model_id")
    comparisons = results["comparison"].set_index("model_id")
    # C-index and 10-year AUC are stand-alone discrimination measures. Keep
    # pairwise reference definitions only for NRI/IDI in the published table.
    comparison_table = results["comparison"]
    reclassification_columns = [
        "model_id", "model", "reference_model_id", "reference_model",
        *[column for column in comparison_table.columns if column.startswith(("nri", "idi"))],
    ]
    comparison_table[reclassification_columns].to_csv(
        output / "1_tables" / "Table_2_model_differences_nri_idi_95ci.csv", index=False
    )
    summary_rows = []
    for prediction in predictions:
        perf_row = performance.loc[prediction.model_id]
        row = {
            "model_id": prediction.model_id,
            "model": prediction.label,
            "nri_idi_reference_model": "",
        }
        for metric in ("c_index", "auc"):
            row[metric] = float(perf_row[metric])
            row[f"{metric}_ci_lower"] = float(perf_row[f"{metric}_ci_lower"])
            row[f"{metric}_ci_upper"] = float(perf_row[f"{metric}_ci_upper"])
        if prediction.model_id in comparisons.index:
            comp_row = comparisons.loc[prediction.model_id]
            row["nri_idi_reference_model"] = comp_row["reference_model"]
            for metric in ("nri", "idi"):
                row[metric] = float(comp_row[metric])
                row[f"{metric}_ci_lower"] = float(comp_row[f"{metric}_ci_lower"])
                row[f"{metric}_ci_upper"] = float(comp_row[f"{metric}_ci_upper"])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output / "1_tables" / "Table_0_model_comparison_2_summary.csv", index=False
    )
    analysis_manifest_path = output / "analysis_manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    analysis_manifest["metric_reference_policy"] = {
        "c_index": "absolute model performance; no reference model",
        "auc": "absolute model performance; no reference model",
        "nri": "paired against Base for every non-Base model",
        "idi": "paired against Base for every non-Base model",
    }
    analysis_manifest_path.write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    missing_masks = {
        "age": frame["age"].isna(),
        "townsend": frame["townsend"].isna(),
        "ethnicity": frame["ethnicity"].eq("Missing"),
        "smoking": frame["smoking"].eq("Missing"),
        "alcohol": frame["alcohol"].eq("Missing"),
        "physical_activity": frame["physical_activity"].eq("Missing"),
        "bmi": frame["bmi"].isna(),
        "olink_klk3_proxy": frame["klk3_olink"].isna(),
        "prs": frame["prs"].isna(),
    }
    missingness = [
        {
            "variable": variable,
            "status": "available",
            "missing_n": int(mask.sum()),
            "missing_percent": float(mask.mean() * 100),
        }
        for variable, mask in missing_masks.items()
    ]
    missingness.extend([
        {
            "variable": "education",
            "status": "unavailable_omitted_by_user_decision",
            "missing_n": int(len(frame)),
            "missing_percent": 100.0,
        },
        {
            "variable": "clinical_serum_psa",
            "status": "unavailable_replaced_by_olink_klk3_proxy",
            "missing_n": int(len(frame)),
            "missing_percent": 100.0,
        },
    ])
    pd.DataFrame(missingness).to_csv(output / "1_tables" / "Table_5_covariate_missingness.csv", index=False)

    manifest = {
        "education": "unavailable; omitted from all clinical model definitions",
        "psa": "clinical serum PSA unavailable; Olink KLK3 protein P07288 used as a proxy",
        "klk3_proxy_is_not_clinical_psa": True,
        "prs_only_model_included": True,
        "fit_subset": "fixed train80 participants with known 10-year status only",
        "evaluation_subset": "fixed test10; early-censored participants excluded from fixed-horizon binary metrics",
        "numeric_missingness": "train80 median imputation plus missing indicator",
        "categorical_missingness": "explicit Missing category; Prefer not to answer and Do not know treated as missing",
        "dca_threshold_percent_range": [
            float(config.raw["evaluation"]["dca_threshold_min"]) * 100.0,
            float(config.raw["evaluation"]["dca_threshold_max"]) * 100.0,
        ],
        "incremental_comparison_references": {
            "lifestyle": "base",
            "lifestyle_klk3": "base",
            "lifestyle_klk3_prs": "base",
            "prs_only": "base",
            "teacher_maps": "base",
            "student_distilled": "base",
        },
        "student_teacher_recovery": "direct direction-aware ratio; no ML reference",
        "metric_reference_policy": {
            "c_index": "absolute; no comparison model",
            "auc": "absolute; no comparison model",
            "nri": "paired against Base for every non-Base model",
            "idi": "paired against Base for every non-Base model",
        },
        "models": manifest_models,
    }
    (output / "clinical_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    perf = performance
    lines = [
        "# 模型比较2结果说明",
        "",
        "教育水平数据缺失，按用户决定从全部临床模型中省略。临床血清PSA缺失，按用户决定使用Olink KLK3（UniProt P07288）作为代理；KLK3代理不能解释为临床PSA测量。",
        "",
        "全部临床模型及PRS-only仅在固定Train80的已知10年状态样本中拟合，统一在固定Test10评价。数值变量在Train80中位数填补并标准化，分类变量保留Missing类别并独热编码。DCA风险阈值预先固定为0.5%-15%。",
        "",
        "C-index和10-year AUC均为各模型在固定Test10上的独立绝对指标，不使用对照模型。所有非Base模型的NRI/IDI统一以Base为参照；Base自身不计算相对自身的NRI/IDI。",
        "",
        "## 主要点估计",
        "",
    ]
    for prediction in predictions:
        row = perf.loc[prediction.model_id]
        lines.append(
            f"- {prediction.label}：C-index {row.c_index:.3f}，AUC {row.auc:.3f}，PR-AUC {row.pr_auc:.3f}，Brier {row.brier:.3f}。"
        )
    lines.extend([
        "",
        "注意：此目录现有Student结果仍来自历史的seed 98，而项目后续冻结的配对Student-MAPS为seed 7。若用于最终配对模型比较，需用seed 7重新评价。",
    ])
    recovery = results["recovery"]
    if len(recovery):
        lines.extend(["", "## Student相对Teacher的直接性能恢复比例", ""])
        for row in recovery.itertuples():
            lines.append(
                f"- {row.metric}：{row.recovery_percent:.1f}%"
                f" (95% CI {row.ci_lower_percent:.1f}% - {row.ci_upper_percent:.1f}%)。"
            )
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 模型比较2：临床递增模型\n\n"
        "状态：已完成。\n\n"
        "教育水平缺失并已按用户决定省略；临床血清PSA以Olink KLK3 (P07288) 代理。"
        "输出包含统一表格、ROC/PR、DCA、C-index/AUC绝对值加全部相对Base的NRI/IDI的Forest plot、AUC/PR-AUC Forest plot和校准曲线。\n",
        "注意：现有Student结果仍来自历史的seed 98，非后续冻结的配对seed 7 Student-MAPS。\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run frozen clinical model comparison 2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
