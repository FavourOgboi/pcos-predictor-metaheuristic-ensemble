import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config.app_config import (
    COLOR_SCHEME,
    DATA_PATHS,
    FEATURE_MEANINGS_M1,
    FEATURE_MEANINGS_M2,
    MODEL_PATHS,
    MODEL1_LABELS,
    MODEL2_LABELS,
    NORMAL_REFERENCE_M1,
    NORMAL_REFERENCE_M2,
)


@st.cache_data(show_spinner=False)
def load_dataframe(path_key: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PATHS[path_key])


@st.cache_data(show_spinner=False)
def load_json_data(path: Path) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_model_bundle(bundle_key: str) -> Dict:
    return joblib.load(MODEL_PATHS[bundle_key])


@st.cache_resource(show_spinner=False)
def load_scaler(scaler_key: str):
    return joblib.load(MODEL_PATHS[scaler_key])


def bmi_to_category(bmi: float) -> int:
    if bmi < 18.5:
        return 0
    if bmi < 25:
        return 1
    if bmi < 30:
        return 2
    return 3


def bmi_label(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bmi_color(bmi: float) -> str:
    if bmi < 25:
        return COLOR_SCHEME["secondary"]
    if bmi < 30:
        return COLOR_SCHEME["accent_soft"]
    return COLOR_SCHEME["accent"]


def classify_bp(systolic: float, diastolic: float) -> Tuple[str, str]:
    if systolic >= 140 or diastolic >= 90:
        return "High Stage 2", COLOR_SCHEME["accent"]
    if systolic >= 130 or diastolic >= 80:
        return "High Stage 1", "#F4A261"
    if 120 <= systolic < 130 and diastolic < 80:
        return "Elevated", COLOR_SCHEME["accent_soft"]
    return "Normal", COLOR_SCHEME["secondary"]


def compute_model1_features(inputs: Dict[str, float]) -> Dict[str, float]:
    bmi = round(inputs["weight_kg"] / ((inputs["height_cm"] / 100) ** 2), 4)
    symptom_burden = int(
        inputs["weight_gain_y_n"]
        + inputs["hair_growth_y_n"]
        + inputs["skin_darkening_y_n"]
        + inputs["hair_loss_y_n"]
        + inputs["pimples_y_n"]
    )
    return {
        "age_yrs": float(inputs["age_yrs"]),
        "bmi": float(bmi),
        "cycle_regularity_binary": float(inputs["cycle_regularity_binary"]),
        "cycle_length_days": float(inputs["cycle_length_days"]),
        "weight_gain_y_n": float(inputs["weight_gain_y_n"]),
        "hair_growth_y_n": float(inputs["hair_growth_y_n"]),
        "skin_darkening_y_n": float(inputs["skin_darkening_y_n"]),
        "hair_loss_y_n": float(inputs["hair_loss_y_n"]),
        "pimples_y_n": float(inputs["pimples_y_n"]),
        "fast_food_y_n": float(inputs["fast_food_y_n"]),
        "regular_exercise_y_n": float(inputs["regular_exercise_y_n"]),
        "systolic_bp_mmhg": float(inputs["systolic_bp_mmhg"]),
        "diastolic_bp_mmhg": float(inputs["diastolic_bp_mmhg"]),
        "waist_hip_ratio": float(inputs["waist_hip_ratio"]),
        "rbs_mg_dl": float(inputs["rbs_mg_dl"]),
        "symptom_burden": float(symptom_burden),
        "bmi_category": float(bmi_to_category(bmi)),
        "bp_elevated_flag": float(int(inputs["systolic_bp_mmhg"] >= 130 or inputs["diastolic_bp_mmhg"] >= 80)),
        "lifestyle_risk": float(inputs["fast_food_y_n"] + (1 - inputs["regular_exercise_y_n"])),
    }


def compute_model2_features(inputs: Dict[str, float]) -> Dict[str, float]:
    ratio = float(inputs["fsh_miu_ml"] / inputs["lh_miu_ml"]) if inputs["lh_miu_ml"] else 0.0
    return {
        "amh_ng_ml": float(inputs["amh_ng_ml"]),
        "fsh_miu_ml": float(inputs["fsh_miu_ml"]),
        "lh_miu_ml": float(inputs["lh_miu_ml"]),
        "fsh_lh_ratio": float(ratio),
        "tsh_miu_l": float(inputs["tsh_miu_l"]),
        "prl_ng_ml": float(inputs["prl_ng_ml"]),
        "vit_d3_ng_ml": float(inputs["vit_d3_ng_ml"]),
        "prg_ng_ml": float(inputs["prg_ng_ml"]),
        "beta_hcg_i_miu_ml": float(inputs["beta_hcg_i_miu_ml"]),
        "beta_hcg_ii_miu_ml": float(inputs["beta_hcg_ii_miu_ml"]),
        "follicle_no_left": float(inputs["follicle_no_left"]),
        "follicle_no_right": float(inputs["follicle_no_right"]),
        "avg_follicle_size_left_mm": float(inputs["avg_follicle_size_left_mm"]),
        "avg_follicle_size_right_mm": float(inputs["avg_follicle_size_right_mm"]),
        "endometrium_mm": float(inputs["endometrium_mm"]),
    }


def bundle_predict(bundle: Dict, scaler, feature_values: Dict[str, float], feature_names: List[str]) -> Dict:
    x_df = pd.DataFrame([[feature_values[name] for name in feature_names]], columns=feature_names)
    x_scaled = scaler.transform(x_df)
    prob_lookup = {
        "LR_prob": float(bundle["base_models"]["lr"].predict_proba(x_scaled)[0, 1]),
        "RF_prob": float(bundle["base_models"]["rf"].predict_proba(x_scaled)[0, 1]),
        "XGB_prob": float(bundle["base_models"]["xgb"].predict_proba(x_scaled)[0, 1]),
    }
    meta_input = np.array([[prob_lookup[name] for name in bundle.get("meta_feature_order", ["LR_prob", "RF_prob", "XGB_prob"])]], dtype=float)
    probability = float(bundle["meta_model"].predict_proba(meta_input)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    return {
        "probability": probability,
        "prediction": int(probability >= threshold),
        "threshold": threshold,
        "base_probabilities": prob_lookup,
    }


def risk_band(probability: float) -> Tuple[str, str, str]:
    if probability >= 0.60:
        return "HIGH RISK", "risk-high", "High probability of a PCOS-like pattern"
    if probability >= 0.40:
        return "MODERATE RISK", "risk-medium", "Borderline pattern that deserves follow-up"
    return "LOW RISK", "risk-low", "Lower probability of a PCOS-like pattern"


def best_model_rows(final_results: pd.DataFrame) -> Dict[str, pd.Series]:
    return {model_set: subset.loc[subset["auc"].idxmax()] for model_set, subset in final_results.groupby("model_set")}


def best_individual_rows(final_results: pd.DataFrame) -> Dict[str, pd.Series]:
    grouped = {}
    for model_set in ["Model 1", "Model 2"]:
        subset = final_results[(final_results["model_set"] == model_set) & (final_results["approach"] == "Individual")]
        grouped[model_set] = subset.loc[subset["auc"].idxmax()]
    return grouped


def get_metric_deltas(final_results: pd.DataFrame) -> Dict[str, float]:
    best_overall = best_model_rows(final_results)
    best_individual = best_individual_rows(final_results)
    return {
        "model1_auc_delta": best_overall["Model 1"]["auc"] - best_individual["Model 1"]["auc"],
        "model1_recall_delta": best_overall["Model 1"]["recall"] - best_individual["Model 1"]["recall"],
        "model2_auc_delta": best_overall["Model 2"]["auc"] - best_individual["Model 2"]["auc"],
        "gap_delta": (best_overall["Model 2"]["auc"] - best_overall["Model 1"]["auc"]) - (best_individual["Model 2"]["auc"] - best_individual["Model 1"]["auc"]),
    }


def make_model1_reason_summary(features: Dict[str, float], ranking_df: pd.DataFrame) -> List[Dict[str, str]]:
    reasons = []
    for feature in ranking_df["feature"].tolist():
        if feature == "cycle_regularity_binary" and features[feature] == 0:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Irregular cycles are a strong routine screening sign in PCOS."})
        elif feature == "bmi" and features[feature] >= 25:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": f"BMI is in the {bmi_label(features['bmi']).lower()} range, which raises metabolic concern."})
        elif feature == "cycle_length_days" and features[feature] > 35:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Cycle length is above the usual 21–35 day pattern."})
        elif feature == "symptom_burden" and features[feature] >= 2:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Several visible symptoms are appearing together."})
        elif feature == "skin_darkening_y_n" and features[feature] == 1:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Skin darkening can reflect insulin-resistance related change."})
        elif feature == "hair_growth_y_n" and features[feature] == 1:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Hair growth pattern is fitting an androgen-related signal."})
        elif feature == "rbs_mg_dl" and features[feature] > 140:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Random blood sugar is above the usual reference band."})
        elif feature == "bp_elevated_flag" and features[feature] == 1:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Blood pressure is crossing an elevated threshold."})
        elif feature == "lifestyle_risk" and features[feature] >= 1:
            reasons.append({"feature": MODEL1_LABELS[feature], "reason": "Lifestyle pattern is adding extra background risk."})
        if len(reasons) == 5:
            break
    return reasons or [{"feature": "Overall pattern", "reason": "Most entered values are not strongly matching the high-risk PCOS pattern."}]


def make_model2_reason_summary(features: Dict[str, float], ranking_df: pd.DataFrame) -> List[Dict[str, str]]:
    reasons = []
    for feature in ranking_df["feature"].tolist():
        if feature == "amh_ng_ml" and features[feature] > 3.5:
            reasons.append({"feature": MODEL2_LABELS[feature], "reason": "AMH is above a commonly discussed PCOS marker threshold."})
        elif feature == "fsh_lh_ratio" and features[feature] < 1:
            reasons.append({"feature": MODEL2_LABELS[feature], "reason": "FSH/LH ratio is below 1, which often appears in PCOS discussions."})
        elif feature in {"follicle_no_left", "follicle_no_right"} and features[feature] >= 12:
            reasons.append({"feature": MODEL2_LABELS[feature], "reason": "Follicle count is meeting a Rotterdam-style threshold."})
        elif feature == "lh_miu_ml" and features[feature] >= 10:
            reasons.append({"feature": MODEL2_LABELS[feature], "reason": "LH is in a higher range that supports a PCOS-like endocrine pattern."})
        elif feature == "endometrium_mm" and features[feature] >= 8:
            reasons.append({"feature": MODEL2_LABELS[feature], "reason": "Endometrium thickness is adding to the reproductive pattern."})
        if len(reasons) == 5:
            break
    return reasons or [{"feature": "Overall pattern", "reason": "Most invasive measurements are not strongly matching the high-risk benchmark pattern."}]


def model1_input_summary(raw_inputs: Dict[str, float], features: Dict[str, float]) -> pd.DataFrame:
    rows = []
    display_values = {
        "Age (years)": raw_inputs["age_yrs"],
        "Weight (kg)": raw_inputs["weight_kg"],
        "Height (cm)": raw_inputs["height_cm"],
        "BMI": round(features["bmi"], 2),
        "Waist-hip ratio": raw_inputs["waist_hip_ratio"],
        "Random blood sugar": raw_inputs["rbs_mg_dl"],
        "Systolic BP": raw_inputs["systolic_bp_mmhg"],
        "Diastolic BP": raw_inputs["diastolic_bp_mmhg"],
        "Cycle length (days)": raw_inputs["cycle_length_days"],
        "Cycle regularity": "Regular" if raw_inputs["cycle_regularity_binary"] == 1 else "Irregular",
        "Weight gain": "Present" if raw_inputs["weight_gain_y_n"] else "Absent",
        "Hair growth": "Present" if raw_inputs["hair_growth_y_n"] else "Absent",
        "Skin darkening": "Present" if raw_inputs["skin_darkening_y_n"] else "Absent",
        "Hair loss": "Present" if raw_inputs["hair_loss_y_n"] else "Absent",
        "Pimples": "Present" if raw_inputs["pimples_y_n"] else "Absent",
        "Fast food": "Yes" if raw_inputs["fast_food_y_n"] else "No",
        "Regular exercise": "Yes" if raw_inputs["regular_exercise_y_n"] else "No",
        "Symptom burden": int(features["symptom_burden"]),
        "BMI category": bmi_label(features["bmi"]),
        "BP elevated flag": "Yes" if features["bp_elevated_flag"] else "No",
        "Lifestyle risk": int(features["lifestyle_risk"]),
    }
    mapping = {
        "Age (years)": "age_yrs",
        "BMI": "bmi",
        "Waist-hip ratio": "waist_hip_ratio",
        "Random blood sugar": "rbs_mg_dl",
        "Systolic BP": "systolic_bp_mmhg",
        "Diastolic BP": "diastolic_bp_mmhg",
        "Cycle length (days)": "cycle_length_days",
    }
    for label, value in display_values.items():
        status, reference = "Context value", "-"
        if label in mapping:
            lower, upper, note = NORMAL_REFERENCE_M1[mapping[label]]
            status = "Within usual range" if lower <= float(value) <= upper else "Outside usual range"
            reference = note
        elif label == "Cycle regularity":
            status = "Regular pattern" if value == "Regular" else "Outside usual monthly pattern"
        elif label in {"Weight gain", "Hair growth", "Skin darkening", "Hair loss", "Pimples"}:
            status = "Symptom present" if value == "Present" else "Symptom not reported"
        elif label == "Fast food":
            status = "Lifestyle concern" if value == "Yes" else "Lower lifestyle concern"
        elif label == "Regular exercise":
            status = "Protective habit" if value == "Yes" else "Lower activity support"
        elif label == "Symptom burden":
            status = "Higher" if value >= 2 else "Lower"
        elif label == "BP elevated flag":
            status = "Elevated" if value == "Yes" else "Not elevated"
        elif label == "Lifestyle risk":
            status = "Higher" if value >= 1 else "Lower"
        elif label in {"Weight (kg)", "Height (cm)"}:
            status = "Raw input"
        rows.append({"Feature": label, "Value": value, "Status": status, "Reference": reference})
    return pd.DataFrame(rows)


def model2_input_summary(features: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, label in MODEL2_LABELS.items():
        value = round(float(features[key]), 4)
        status, reference = "Measured", "-"
        if key in NORMAL_REFERENCE_M2:
            lower, upper, note = NORMAL_REFERENCE_M2[key]
            status = "Within reference" if lower <= value <= upper else "Flagged pattern"
            reference = note
        rows.append({"Feature": label, "Value": value, "Status": status, "Reference": reference})
    return pd.DataFrame(rows)


def build_text_report(title: str, probability: float, risk_label: str, feature_rows: pd.DataFrame, reasons: List[Dict[str, str]]) -> str:
    lines = [title, "=" * len(title), "", f"Predicted probability: {probability:.1%}", f"Risk band: {risk_label}", "", "Top plain-language reasons:"]
    for item in reasons:
        lines.append(f"- {item['feature']}: {item['reason']}")
    lines.extend(["", "Entered values:"])
    for _, row in feature_rows.iterrows():
        lines.append(f"- {row['Feature']}: {row['Value']} ({row['Status']})")
    lines.extend(["", "This report is for research and educational use only.", "It is not a clinical diagnosis."])
    return "\n".join(lines)


def feature_meanings(model_key: str) -> Dict[str, str]:
    return FEATURE_MEANINGS_M1 if model_key == "model1" else FEATURE_MEANINGS_M2
