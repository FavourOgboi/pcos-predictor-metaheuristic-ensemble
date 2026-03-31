from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.charts import apply_plotly_style, overlapping_histogram
from app.components.styling import (
    apply_global_styles,
    configure_page,
    render_footer,
    render_info_banner,
    render_section_card,
    render_sidebar,
    render_warning_banner,
)
from app.components.utils import load_dataframe


configure_page("Heart & PCOS Study", "❤️")
apply_global_styles()
render_sidebar()

st.markdown(
    """
    <div class='page-header'>
        <h1>❤️ Heart Disease & PCOS - Association Study</h1>
        <p>Investigating cardiometabolic overlap between PCOS and cardiovascular risk markers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    pcos_df = load_dataframe("pcos_clinical").copy()
    heart_df = load_dataframe("heart_cleaned").copy()
except Exception as exc:
    st.error(f"Unable to load one of the study datasets: {exc}")
    render_footer()
    st.stop()

pcos_df["PCOS status"] = pcos_df["pcos_y_n"].map({0: "PCOS negative", 1: "PCOS positive"})
pcos_df["elevated_bp_flag"] = (
    (pcos_df["systolic_bp_mmhg"] >= 130) | (pcos_df["diastolic_bp_mmhg"] >= 80)
).astype(int)
heart_df["Heart status"] = heart_df["heart_disease_target"].map({0: "Heart negative", 1: "Heart positive"})
heart_df["high_resting_bp_flag"] = (heart_df["resting_bp"] >= 130).astype(int)

pcos_neg = pcos_df[pcos_df["pcos_y_n"] == 0]
pcos_pos = pcos_df[pcos_df["pcos_y_n"] == 1]
heart_pos = heart_df[heart_df["heart_disease_target"] == 1]

render_info_banner(
    "This page is showing a group-level comparison, not a patient-level linkage. "
    "The PCOS cohort and the heart cohort are different sets of women, so the findings here support discussion about cardiovascular overlap and monitoring need, not direct causation."
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("PCOS cohort", f"{len(pcos_df)} patients")
with metric_cols[1]:
    st.metric("Heart cohort", f"{len(heart_df)} patients")
with metric_cols[2]:
    age_gap = heart_df["age"].mean() - pcos_df["age_yrs"].mean()
    st.metric("Mean age gap", f"{age_gap:.1f} years")
with metric_cols[3]:
    st.metric("Heart-positive share", f"{(heart_df['heart_disease_target'].mean() * 100):.1f}%")

render_section_card(
    "Study Context",
    f"""
    This study is using the PCOS clinical cohort to ask whether routine PCOS-related features are showing overlap with cardiovascular risk patterns.
    The heart dataset is acting as a small female reference cohort with {len(heart_df)} patients, not as a matched control group.
    This means the comparison is useful for discussion, screening relevance, and clinical caution, but it is not proving that the same women with PCOS later developed heart disease.
    """,
)

cohort_summary = pd.DataFrame(
    [
        ["PCOS negative", len(pcos_neg), round(pcos_neg["age_yrs"].mean(), 2), round(pcos_neg["systolic_bp_mmhg"].mean(), 2), round(pcos_neg["diastolic_bp_mmhg"].mean(), 2), round(pcos_neg["bmi"].mean(), 2), round(pcos_neg["rbs_mg_dl"].mean(), 2), "No"],
        ["PCOS positive", len(pcos_pos), round(pcos_pos["age_yrs"].mean(), 2), round(pcos_pos["systolic_bp_mmhg"].mean(), 2), round(pcos_pos["diastolic_bp_mmhg"].mean(), 2), round(pcos_pos["bmi"].mean(), 2), round(pcos_pos["rbs_mg_dl"].mean(), 2), "No"],
        ["Heart female reference", len(heart_df), round(heart_df["age"].mean(), 2), round(heart_df["resting_bp"].mean(), 2), "N/A", "N/A", "N/A", "Yes"],
    ],
    columns=[
        "Cohort",
        "n",
        "Mean age",
        "Mean systolic/resting BP",
        "Mean diastolic BP",
        "Mean BMI",
        "Mean random blood sugar",
        "Cholesterol available",
    ],
)
st.dataframe(cohort_summary, use_container_width=True)
st.markdown(
    f"""
**Cohort insight:** The first thing to notice is the age difference.
The PCOS cohort has a mean age of {pcos_df['age_yrs'].mean():.2f} years, while the heart reference cohort has a mean age of {heart_df['age'].mean():.2f} years.
This means age is the biggest confound in the whole comparison and every later chart must be read with that caution in mind.
"""
)

bp_fig = overlapping_histogram(
    {
        "PCOS negative": {"values": pcos_neg["systolic_bp_mmhg"], "color": "#2176AE"},
        "PCOS positive": {"values": pcos_pos["systolic_bp_mmhg"], "color": "#E63946"},
        "Heart female reference": {"values": heart_df["resting_bp"], "color": "#2A9D8F"},
    },
    "Blood Pressure Comparison Across Cohorts",
    "Systolic / resting blood pressure (mmHg)",
    threshold_lines=[(130, "130 mmHg threshold")],
)
st.plotly_chart(bp_fig, use_container_width=True)

bp_compare = pd.DataFrame(
    [
        ["PCOS negative", round(pcos_neg["systolic_bp_mmhg"].mean(), 2), round(pcos_neg["diastolic_bp_mmhg"].mean(), 2), round((pcos_neg["elevated_bp_flag"].mean() * 100), 1)],
        ["PCOS positive", round(pcos_pos["systolic_bp_mmhg"].mean(), 2), round(pcos_pos["diastolic_bp_mmhg"].mean(), 2), round((pcos_pos["elevated_bp_flag"].mean() * 100), 1)],
        ["Heart female reference", round(heart_df["resting_bp"].mean(), 2), "N/A", round((heart_df["high_resting_bp_flag"].mean() * 100), 1)],
    ],
    columns=["Group", "Mean systolic/resting BP", "Mean diastolic BP", "Above threshold %"],
)
st.dataframe(bp_compare, use_container_width=True)
st.markdown(
    f"""
**Blood pressure insight:** The PCOS-positive and PCOS-negative groups are very close on mean blood pressure, but both sit below the older heart reference cohort.
Still, {bp_compare.loc[1, 'Above threshold %']:.1f}% of the PCOS-positive group crosses the elevated blood-pressure rule used in this study.
This means blood pressure is not the main PCOS separator on its own, but it still adds to the cardiometabolic burden story.
"""
)

age_fig = overlapping_histogram(
    {
        "PCOS negative": {"values": pcos_neg["age_yrs"], "color": "#2176AE"},
        "PCOS positive": {"values": pcos_pos["age_yrs"], "color": "#E63946"},
        "Heart positive": {"values": heart_pos["age"], "color": "#2A9D8F"},
    },
    "Age Overlap Between PCOS and Heart Cohorts",
    "Age (years)",
)
age_fig.add_vrect(
    x0=30,
    x1=45,
    fillcolor="#E9C46A",
    opacity=0.16,
    line_width=0,
    annotation_text="Shared age window 30-45",
)
st.plotly_chart(age_fig, use_container_width=True)

age_compare = pd.DataFrame(
    [
        ["PCOS negative", round(pcos_neg["age_yrs"].mean(), 2), round(((pcos_neg["age_yrs"].between(30, 45)).mean() * 100), 1)],
        ["PCOS positive", round(pcos_pos["age_yrs"].mean(), 2), round(((pcos_pos["age_yrs"].between(30, 45)).mean() * 100), 1)],
        ["Heart positive", round(heart_pos["age"].mean(), 2), round(((heart_pos["age"].between(30, 45)).mean() * 100), 1)],
    ],
    columns=["Group", "Mean age", "Share in age 30-45 %"],
)
st.dataframe(age_compare, use_container_width=True)
st.markdown(
    f"""
**Age insight:** The shared age window between 30 and 45 years exists, but it is much smaller in the heart-positive group.
The heart-positive mean age is {age_compare.loc[2, 'Mean age']:.2f} years, which is far above the PCOS-positive mean age of {age_compare.loc[1, 'Mean age']:.2f}.
This means the heart page should be read as a trajectory discussion: PCOS may sit earlier on a cardiometabolic pathway, but this dataset cannot prove the later transition.
"""
)

pcos_risk_df = pd.DataFrame(
    {
        "Group": ["PCOS negative", "PCOS positive"],
        "BMI": [pcos_neg["bmi"].mean(), pcos_pos["bmi"].mean()],
        "Random blood sugar": [pcos_neg["rbs_mg_dl"].mean(), pcos_pos["rbs_mg_dl"].mean()],
        "Pulse rate": [pcos_neg["pulse_rate_bpm"].mean(), pcos_pos["pulse_rate_bpm"].mean()],
        "Cycle length": [pcos_neg["cycle_length_days"].mean(), pcos_pos["cycle_length_days"].mean()],
    }
).round(2)
st.dataframe(pcos_risk_df, use_container_width=True)

pcos_melt = pcos_risk_df.melt(id_vars="Group", var_name="Marker", value_name="Mean value")
bridge_fig = px.bar(
    pcos_melt,
    x="Marker",
    y="Mean value",
    color="Group",
    barmode="group",
    color_discrete_map={"PCOS negative": "#2176AE", "PCOS positive": "#E63946"},
)
st.plotly_chart(apply_plotly_style(bridge_fig, "Within-PCOS Cardiometabolic Bridge Markers"), use_container_width=True)
st.markdown(
    f"""
**Within-PCOS insight:** The clearest bridge to cardiovascular discussion is actually inside the PCOS dataset itself.
The PCOS-positive group has higher BMI ({pcos_risk_df.loc[1, 'BMI']:.2f} vs {pcos_risk_df.loc[0, 'BMI']:.2f}) and slightly higher random blood sugar ({pcos_risk_df.loc[1, 'Random blood sugar']:.2f} vs {pcos_risk_df.loc[0, 'Random blood sugar']:.2f}).
This matters because the metabolic burden in PCOS is already visible before any external heart comparison is made.
"""
)

marker_table = pd.DataFrame(
    [
        ["Age", round(pcos_pos["age_yrs"].mean(), 2), round(heart_df["age"].mean(), 2), "Directly comparable"],
        ["Systolic/resting blood pressure", round(pcos_pos["systolic_bp_mmhg"].mean(), 2), round(heart_df["resting_bp"].mean(), 2), "Comparable with caution"],
        ["Random blood sugar", round(pcos_pos["rbs_mg_dl"].mean(), 2), "N/A", "PCOS only"],
        ["BMI", round(pcos_pos["bmi"].mean(), 2), "N/A", "PCOS only"],
        ["Cholesterol", "N/A", round(heart_df["cholesterol"].mean(), 2), "Heart only"],
        ["Pulse / max heart rate", round(pcos_pos["pulse_rate_bpm"].mean(), 2), round(heart_df["max_heart_rate"].mean(), 2), "Not directly equivalent"],
    ],
    columns=["Marker", "PCOS positive", "Heart reference", "Interpretation"],
)
st.dataframe(marker_table, use_container_width=True)
st.markdown(
    """
**Marker comparison insight:** Some markers can be compared directly, some only partly, and some not at all.
Age and blood pressure are the cleanest shared markers.
BMI and glucose strengthen the PCOS risk story, while cholesterol strengthens the heart-cohort story, but those one-sided markers should not be over-read as direct matches.
"""
)

render_section_card(
    "Key Findings",
    f"""
    <strong>1.</strong> The strongest cardiovascular discussion point begins inside the PCOS dataset itself: the PCOS-positive group carries higher BMI and higher blood sugar than the PCOS-negative group.<br><br>
    <strong>2.</strong> Blood pressure in the PCOS-positive group is not dramatically higher than the PCOS-negative group, but the elevated-BP share is still high at {bp_compare.loc[1, 'Above threshold %']:.1f}%.<br><br>
    <strong>3.</strong> The heart cohort is much older, with a mean age gap of about {age_gap:.1f} years, so age is the single biggest limitation in the cross-dataset comparison.<br><br>
    <strong>4.</strong> The shared window between ages 30 and 45 suggests some overlap in life stage, but it is not large enough to remove the confounding issue.<br><br>
    <strong>5.</strong> The study therefore supports cardiovascular monitoring in PCOS, but it does not show that the same women later became heart patients.
    """,
)

render_warning_banner(
    "Important caution: the heart-positive subgroup shows some counterintuitive internal patterns, including lower mean resting blood pressure than the heart-negative subgroup. "
    "This reflects the small cohort size and dataset composition, so the heart page should be used for thesis discussion and context, not for strong causal claims."
)

render_section_card(
    "Study Summary",
    f"""
    This page is supporting the thesis in a careful way.
    First, it shows that the PCOS-positive group already carries a heavier metabolic pattern than the PCOS-negative group.
    Second, it shows that the female heart reference cohort is older and generally more cardiovascularly loaded, especially on age and resting blood pressure.
    Third, it shows that there is enough overlap to justify cardiovascular attention in PCOS, but not enough to claim a direct one-to-one disease pathway from these datasets alone.
    In simple terms, the study is saying this: PCOS is not just a reproductive issue in this dataset. It is also sitting next to a broader cardiometabolic pattern, which is why screening and follow-up matter.
    """,
)

render_footer()
