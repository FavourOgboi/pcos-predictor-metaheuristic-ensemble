import streamlit as st

from app.components.styling import render_warning_banner


def render() -> None:
    st.markdown(
        """
        <div class='page-header' style='background: linear-gradient(135deg, #78350F, #92400E);'>
            <h1>Disclaimer</h1>
            <p>Important information about the limitations and appropriate use of this system</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sections = [
        (
            "Research Purpose",
            "This system was developed as part of an MSc research thesis. It is intended exclusively for academic research, educational purposes, and demonstration of machine learning methodology. It has not been reviewed, approved, or certified by any medical regulatory authority.",
        ),
        (
            "Not a Medical Device",
            "This tool does not constitute a medical device under any jurisdiction. The predictions produced by this system are probabilistic outputs of a machine learning model trained on a research dataset. They are not clinical diagnoses and must not be used as the basis for any medical decision, treatment, or intervention.",
        ),
        (
            "Data Limitations",
            "The models were trained on a publicly available dataset of 541 patients from a specific clinical population. Performance may vary across different populations, demographics, and healthcare settings. The survey external validation used imputed values for several clinical features, so users should not expect identical performance in every setting.",
        ),
        (
            "Seek Professional Advice",
            "Any person who is concerned about PCOS symptoms should consult a qualified healthcare professional. Symptoms of PCOS overlap with other medical conditions and require proper clinical evaluation for accurate diagnosis. This system is a screening tool, not a diagnostic tool.",
        ),
    ]

    for title, body in sections:
        st.markdown(
            f"""
            <div class='section-card'>
                <h3 style='margin-top:0; color:#7C2D12;'>{title}</h3>
                <p style='line-height:1.8; color:#5C4033;'>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    consent = st.checkbox(
        "I have read and understood the disclaimer and agree to use this system for research and educational purposes only."
    )
    if not consent:
        render_warning_banner("Please acknowledge the disclaimer before using this research application.")
