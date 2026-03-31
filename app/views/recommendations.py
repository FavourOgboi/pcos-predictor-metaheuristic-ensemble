import streamlit as st

from app.components.styling import render_section_card
from app.components.utils import best_model_rows, load_dataframe


def render() -> None:
    st.markdown(
        """
        <div class='page-header'>
            <h1>Clinical Recommendations</h1>
            <p>Evidence-based guidance derived from the study findings</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    final_results = load_dataframe("final_results")
    best_rows = best_model_rows(final_results)
    non_invasive_auc = best_rows["Model 1"]["auc"] * 100

    cards = [
        (
            "For Women (Self-Screening)",
            "Preventive",
            "If you experience irregular periods combined with unexplained weight gain, excess hair growth, or skin darkening, you may be at elevated PCOS risk. Use the Non-Invasive Screening Tool on this platform as a first step. This tool does not replace clinical diagnosis, but it can help you decide whether to seek professional evaluation. Early screening allows earlier intervention.",
        ),
        (
            "For Primary Care Clinicians",
            "Clinical Action",
            f"Patients presenting with BMI above 25, menstrual cycles longer than 35 days or clearly irregular timing, and any two additional symptoms such as hirsutism, skin darkening, or acne should be considered for PCOS evaluation. This study shows that these non-invasive markers alone can achieve about {non_invasive_auc:.1f}% AUC in separating PCOS from non-PCOS patients. Referral for hormonal testing is recommended for high-risk screening results.",
        ),
        (
            "For Resource-Limited Settings",
            "Public Health",
            "In settings where laboratory testing and ultrasound are unavailable, the Non-Invasive Model provides a validated screening framework requiring only age, BMI, blood pressure, menstrual cycle information, and symptom presence. This study shows that this reduced data still reaches clinically meaningful performance. Community health workers can use this as a structured checklist.",
        ),
        (
            "Lifestyle Interventions",
            "Modifiable Risk",
            "This study identified fast food intake and low exercise as contributors to the lifestyle risk score. Both are modifiable. Regular exercise and better diet quality remain strong first-line interventions for PCOS management and for reducing cardiometabolic strain.",
        ),
        (
            "Future Research Directions",
            "Research",
            "Future work can include external validation on an independent clinical cohort, longitudinal risk tracking, federated learning across hospitals, wearable-device integration, and a mobile app for community-level self-screening.",
        ),
    ]

    icons = ["👩", "🩺", "🌍", "🥗", "🔭"]
    for icon, (title, badge, body) in zip(icons, cards):
        st.markdown(
            f"""
            <div class='section-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-size:1.4rem; font-weight:700;'>{icon} {title}</div>
                    <div style='background:#EBF4FF; color:#1B4F8A; border-radius:999px; padding:0.35rem 0.8rem; font-size:0.8rem; font-weight:700;'>{badge}</div>
                </div>
                <p style='margin-top:1rem; color:#40566F; line-height:1.8;'>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_card(
        "Supporting References",
        """
        1. Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group. Revised diagnostic criteria for PCOS.<br>
        2. Lundberg, S. M., and Lee, S.-I. A Unified Approach to Interpreting Model Predictions.<br>
        3. Wekker, V. et al. Cardiovascular risk profile in women with PCOS.<br>
        4. Henney, N. et al. Metabolic burden and obesity-linked risk in PCOS.
        """,
    )
