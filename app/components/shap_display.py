from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from app.config.app_config import FEATURE_MEANINGS_M1, FEATURE_MEANINGS_M2


def show_image_or_placeholder(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.markdown(
            f"""
            <div class='placeholder-image'>
                <div style='font-size:2rem;'>🖼️</div>
                <div style='font-weight:600; margin-top:0.5rem;'>{caption}</div>
                <div style='font-size:0.9rem; margin-top:0.4rem;'>Image file is not available yet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ranked_feature_summary(rank_df: pd.DataFrame, model_key: str, top_n: int = 3) -> None:
    meanings = FEATURE_MEANINGS_M1 if model_key == "model1" else FEATURE_MEANINGS_M2
    value_column = "mean_abs_shap" if "mean_abs_shap" in rank_df.columns else rank_df.columns[1]
    for idx, row in rank_df.head(top_n).iterrows():
        meaning = meanings.get(row["feature"], "This feature is part of the learned PCOS pattern.")
        st.markdown(
            f"""
            <div class='risk-reason'>
                <strong>Top feature {idx + 1}: {row['feature']}</strong><br>
                Mean SHAP importance: <strong>{row[value_column]:.4f}</strong><br>
                {meaning}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_patient_reason_summary(title: str, reasons: List[Dict[str, str]]) -> None:
    st.markdown(f"### {title}")
    for reason in reasons:
        st.markdown(
            f"""
            <div class='risk-reason'>
                <strong>{reason['feature']}</strong><br>
                {reason['reason']}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_shap_section(title: str, bar_path: Path, beeswarm_path: Path, bar_text: str, beeswarm_text: str, rank_df: pd.DataFrame, model_key: str) -> None:
    st.markdown(f"### {title}")
    col1, col2 = st.columns(2)
    with col1:
        show_image_or_placeholder(bar_path, "Global SHAP bar chart")
        st.markdown(bar_text)
    with col2:
        show_image_or_placeholder(beeswarm_path, "SHAP beeswarm plot")
        st.markdown(beeswarm_text)
    render_ranked_feature_summary(rank_df, model_key=model_key, top_n=3)
