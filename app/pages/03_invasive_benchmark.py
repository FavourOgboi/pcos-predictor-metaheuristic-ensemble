import streamlit as st

from app.components.shap_display import render_patient_reason_summary, render_shap_section
from app.components.styling import (
    apply_global_styles,
    configure_page,
    render_footer,
    render_info_banner,
    render_metric_card,
    render_sidebar,
)
from app.components.utils import (
    bundle_predict,
    build_text_report,
    compute_model2_features,
    load_dataframe,
    load_model_bundle,
    load_scaler,
    make_model2_reason_summary,
    model2_input_summary,
    risk_band,
)
from app.config.app_config import IMAGE_PATHS


def render_badge(label: str, css_class: str, helper_text: str) -> None:
    st.markdown(f"<div class='{css_class}'>{label}<br><span style='font-size:0.9rem; font-weight:500;'>{helper_text}</span></div>", unsafe_allow_html=True)


configure_page("Clinical Benchmark", "🔵")
apply_global_styles()
render_sidebar()

st.markdown(
    """
    <div class='page-header'>
        <h1>🔵 Clinical Benchmark</h1>
        <p>Enter hormonal and ultrasound findings for the invasive benchmark model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_info_banner(
    "This benchmark tool is using 15 invasive clinical features. "
    "It is designed as the richer reference model in the thesis. "
    "It supports clinical comparison, not standalone diagnosis."
)

try:
    final_bundle = load_model_bundle("model2_final")
    scaler = load_scaler("scaler2")
    shap_rank_m2 = load_dataframe("shap_rank_m2")
except Exception as exc:
    st.error(f"Unable to load the Model 2 benchmark resources: {exc}")
    render_footer()
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Hormonal Markers")
    amh_ng_ml = st.number_input("AMH (ng/mL)", 0.1, 20.0, 4.2, 0.1, help="Anti-Müllerian Hormone. Values above 3.5 ng/mL are associated with PCOS.")
    fsh_miu_ml = st.number_input("FSH (mIU/mL)", 0.5, 30.0, 6.0, 0.1)
    lh_miu_ml = st.number_input("LH (mIU/mL)", 0.5, 40.0, 8.2, 0.1)
    live_ratio = fsh_miu_ml / lh_miu_ml if lh_miu_ml else 0.0
    st.markdown(f"<div class='section-card'><strong>Live FSH/LH ratio</strong><br><span style='font-size:1.8rem; color:#2176AE; font-weight:700;'>{live_ratio:.2f}</span></div>", unsafe_allow_html=True)
    tsh_miu_l = st.number_input("TSH (mIU/L)", 0.1, 10.0, 2.5, 0.1)
    prl_ng_ml = st.number_input("Prolactin (ng/mL)", 1.0, 100.0, 18.0, 0.5)
    vit_d3_ng_ml = st.number_input("Vitamin D3 (ng/mL)", 1.0, 100.0, 28.0, 0.5)
    prg_ng_ml = st.number_input("Progesterone (ng/mL)", 0.1, 30.0, 1.2, 0.1)
    beta_hcg_i_miu_ml = st.number_input("Beta-HCG I (mIU/mL)", 0.1, 500.0, 2.0, 0.5)
    beta_hcg_ii_miu_ml = st.number_input("Beta-HCG II (mIU/mL)", 0.1, 500.0, 2.0, 0.5)

with col2:
    st.markdown("### Ultrasound Measurements")
    follicle_no_left = st.slider("Follicle Count — Left Ovary", 0, 30, 12)
    follicle_no_right = st.slider("Follicle Count — Right Ovary", 0, 30, 14)
    if follicle_no_left >= 12 or follicle_no_right >= 12:
        st.markdown("<div class='warning-banner'>Rotterdam follicle threshold met (≥12 follicles).</div>", unsafe_allow_html=True)
    avg_follicle_size_left_mm = st.number_input("Average Follicle Size — Left (mm)", 1.0, 30.0, 8.0, 0.5)
    avg_follicle_size_right_mm = st.number_input("Average Follicle Size — Right (mm)", 1.0, 30.0, 8.5, 0.5)
    endometrium_mm = st.number_input("Endometrium (mm)", 1.0, 20.0, 7.5, 0.1)

raw_inputs = {
    "amh_ng_ml": amh_ng_ml,
    "fsh_miu_ml": fsh_miu_ml,
    "lh_miu_ml": lh_miu_ml,
    "tsh_miu_l": tsh_miu_l,
    "prl_ng_ml": prl_ng_ml,
    "vit_d3_ng_ml": vit_d3_ng_ml,
    "prg_ng_ml": prg_ng_ml,
    "beta_hcg_i_miu_ml": beta_hcg_i_miu_ml,
    "beta_hcg_ii_miu_ml": beta_hcg_ii_miu_ml,
    "follicle_no_left": follicle_no_left,
    "follicle_no_right": follicle_no_right,
    "avg_follicle_size_left_mm": avg_follicle_size_left_mm,
    "avg_follicle_size_right_mm": avg_follicle_size_right_mm,
    "endometrium_mm": endometrium_mm,
}
feature_values = compute_model2_features(raw_inputs)

st.markdown("### Prediction")
if st.button("Run Clinical Benchmark Prediction"):
    result = bundle_predict(final_bundle, scaler, feature_values, final_bundle["feature_names"])
    risk_label, risk_class, helper_text = risk_band(result["probability"])
    render_badge(risk_label, risk_class, helper_text)
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.markdown(f"**Predicted PCOS probability:** {result['probability']:.1%}")
    base_cols = st.columns(3)
    for col, (name, value) in zip(base_cols, result["base_probabilities"].items()):
        with col:
            render_metric_card(name.replace("_prob", " base").replace("_", " "), f"{value:.3f}")

    reasons = make_model2_reason_summary(feature_values, shap_rank_m2)
    render_patient_reason_summary("Why the benchmark model is giving this result", reasons)

    render_shap_section(
        "Research SHAP Evidence — Model 2",
        IMAGE_PATHS["shap_m2_bar"],
        IMAGE_PATHS["shap_m2_beeswarm"],
        "This global SHAP bar chart is showing which hormonal and ultrasound features mattered most across the research test set.",
        "This beeswarm plot is showing the direction of each invasive feature effect and how it varied across patients.",
        shap_rank_m2,
        "model2",
    )

    summary_df = model2_input_summary(feature_values)
    st.markdown("### Feature Input Summary")
    st.dataframe(summary_df, use_container_width=True)

    report_text = build_text_report(
        "PCOS Early Detection System — Invasive Benchmark Report",
        result["probability"],
        risk_label,
        summary_df,
        reasons,
    )
    st.download_button(
        "Download benchmark report",
        data=report_text,
        file_name="pcos_invasive_benchmark_report.txt",
        mime="text/plain",
    )

render_footer()
