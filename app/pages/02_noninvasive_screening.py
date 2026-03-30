import streamlit as st

from components.shap_display import render_patient_reason_summary, render_shap_section
from components.styling import (
    apply_global_styles,
    configure_page,
    render_footer,
    render_info_banner,
    render_metric_card,
    render_sidebar,
)
from components.utils import (
    bmi_color,
    bmi_label,
    bundle_predict,
    build_text_report,
    classify_bp,
    compute_model1_features,
    load_dataframe,
    load_model_bundle,
    load_scaler,
    make_model1_reason_summary,
    model1_input_summary,
    risk_band,
)
from config.app_config import IMAGE_PATHS


def render_badge(label: str, css_class: str, helper_text: str) -> None:
    st.markdown(f"<div class='{css_class}'>{label}<br><span style='font-size:0.9rem; font-weight:500;'>{helper_text}</span></div>", unsafe_allow_html=True)


configure_page("Non-Invasive Screening", "🟢")
apply_global_styles()
render_sidebar()

st.markdown(
    """
    <div class='page-header'>
        <h1>🟢 Non-Invasive PCOS Screening</h1>
        <p>Enter routine clinical and lifestyle information — no laboratory tests required</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_info_banner(
    "This screening tool is using 19 routine clinical and lifestyle features. "
    "It is powered by the final metaheuristic-optimized stacked ensemble model. "
    "The result is for screening support only and is not a clinical diagnosis."
)

try:
    final_bundle = load_model_bundle("model1_final")
    scaler = load_scaler("scaler1")
    shap_rank_m1 = load_dataframe("shap_rank_m1")
except Exception as exc:
    st.error(f"Unable to load the Model 1 screening resources: {exc}")
    render_footer()
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Demographics and Body")
    age_yrs = st.slider("Age (years)", 14, 50, 26, 1)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=62.0, step=0.5)
    height_cm = st.number_input("Height (cm)", min_value=130.0, max_value=195.0, value=160.0, step=0.5)
    waist_hip_ratio = st.number_input("Waist-Hip Ratio", min_value=0.60, max_value=1.20, value=0.84, step=0.01)
    rbs_mg_dl = st.number_input("Random Blood Sugar (mg/dL)", min_value=60.0, max_value=300.0, value=96.0, step=1.0)
    live_bmi = weight_kg / ((height_cm / 100) ** 2)
    st.markdown(
        f"""
        <div class='section-card'>
            <strong>Live BMI</strong><br>
            <span style='font-size:2rem; color:{bmi_color(live_bmi)}; font-weight:700;'>{live_bmi:.2f}</span><br>
            <span style='color:#5B708A;'>{bmi_label(live_bmi)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("### Cardiovascular and Cycle")
    systolic_bp_mmhg = st.slider("Systolic BP (mmHg)", 70, 200, 118, 1)
    diastolic_bp_mmhg = st.slider("Diastolic BP (mmHg)", 40, 130, 76, 1)
    bp_label, bp_color = classify_bp(systolic_bp_mmhg, diastolic_bp_mmhg)
    st.markdown(
        f"""
        <div class='section-card'>
            <strong>Live BP Classification</strong><br>
            <span style='font-size:1.5rem; color:{bp_color}; font-weight:700;'>{bp_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cycle_regularity_text = st.radio("Cycle regularity", ["Regular (every month)", "Irregular"], horizontal=False)
    cycle_regularity_binary = 1 if cycle_regularity_text == "Regular (every month)" else 0
    cycle_length_days = st.slider("Cycle length (days)", 20, 90, 31, 1)

with col3:
    st.markdown("### Symptoms and Lifestyle")
    weight_gain_y_n = int(st.checkbox("Recent unexplained weight gain"))
    hair_growth_y_n = int(st.checkbox("Excess body or facial hair growth"))
    skin_darkening_y_n = int(st.checkbox("Skin darkening (neck, armpits, groin)"))
    hair_loss_y_n = int(st.checkbox("Hair thinning or hair loss"))
    pimples_y_n = int(st.checkbox("Pimples or acne (face/jawline)"))
    fast_food_y_n = int(st.checkbox("Eat fast food regularly"))
    regular_exercise_y_n = int(st.checkbox("Exercise regularly"))

raw_inputs = {
    "age_yrs": age_yrs,
    "weight_kg": weight_kg,
    "height_cm": height_cm,
    "waist_hip_ratio": waist_hip_ratio,
    "rbs_mg_dl": rbs_mg_dl,
    "systolic_bp_mmhg": systolic_bp_mmhg,
    "diastolic_bp_mmhg": diastolic_bp_mmhg,
    "cycle_regularity_binary": cycle_regularity_binary,
    "cycle_length_days": cycle_length_days,
    "weight_gain_y_n": weight_gain_y_n,
    "hair_growth_y_n": hair_growth_y_n,
    "skin_darkening_y_n": skin_darkening_y_n,
    "hair_loss_y_n": hair_loss_y_n,
    "pimples_y_n": pimples_y_n,
    "fast_food_y_n": fast_food_y_n,
    "regular_exercise_y_n": regular_exercise_y_n,
}

feature_values = compute_model1_features(raw_inputs)

st.markdown("### Prediction")
if st.button("Run Non-Invasive Screening"):
    result = bundle_predict(final_bundle, scaler, feature_values, final_bundle["feature_names"])
    risk_label, risk_class, helper_text = risk_band(result["probability"])
    render_badge(risk_label, risk_class, helper_text)
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.markdown(f"**Predicted PCOS probability:** {result['probability']:.1%}")
    base_cols = st.columns(3)
    for col, (name, value) in zip(base_cols, result["base_probabilities"].items()):
        with col:
            render_metric_card(name.replace("_prob", " base").replace("_", " "), f"{value:.3f}")

    metric_cols = st.columns(5)
    breakdown = [
        ("Symptom burden", str(int(feature_values["symptom_burden"]))),
        ("BMI category", bmi_label(feature_values["bmi"])),
        ("BP flag", "Elevated" if feature_values["bp_elevated_flag"] else "Not elevated"),
        ("Lifestyle risk", str(int(feature_values["lifestyle_risk"]))),
        ("Cycle pattern", "Irregular" if feature_values["cycle_regularity_binary"] == 0 else "Regular"),
    ]
    for col, (label, value) in zip(metric_cols, breakdown):
        with col:
            render_metric_card(label, value)

    reasons = make_model1_reason_summary(feature_values, shap_rank_m1)
    render_patient_reason_summary("Why the model is giving this result", reasons)

    render_shap_section(
        "Research SHAP Evidence — Model 1",
        IMAGE_PATHS["shap_m1_bar"],
        IMAGE_PATHS["shap_m1_beeswarm"],
        "This global SHAP bar chart is showing which non-invasive features mattered most across the research test set.",
        "This beeswarm plot is showing the direction of each feature effect and how it varied across patients.",
        shap_rank_m1,
        "model1",
    )

    summary_df = model1_input_summary(raw_inputs, feature_values)
    st.markdown("### Feature Input Summary")
    st.dataframe(summary_df, use_container_width=True)

    report_text = build_text_report(
        "PCOS Early Detection System — Non-Invasive Screening Report",
        result["probability"],
        risk_label,
        summary_df,
        reasons,
    )
    st.download_button(
        "Download screening report",
        data=report_text,
        file_name="pcos_noninvasive_screening_report.txt",
        mime="text/plain",
    )

render_footer()
