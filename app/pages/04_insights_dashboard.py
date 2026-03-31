import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.charts import (
    age_histogram,
    apply_plotly_style,
    bmi_boxplot,
    bp_scatter,
    class_distribution_pie,
    correlation_heatmap,
    cycle_violin,
    selected_feature_histogram,
    symptom_prevalence_chart,
)
from app.components.styling import (
    apply_global_styles,
    configure_page,
    render_footer,
    render_info_banner,
    render_section_card,
    render_sidebar,
)
from app.components.utils import load_dataframe


configure_page("PCOS Insights Dashboard", "📊")
apply_global_styles()
render_sidebar()

st.markdown(
    """
    <div class='page-header'>
        <h1>📊 PCOS Insights Dashboard</h1>
        <p>Exploratory analysis and clinical patterns from the PCOS clinical dataset (n=541)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_info_banner(
    "This dashboard is separating the PCOS story into two clear views. "
    "The first view is showing the non-invasive clinical picture that can be used for routine screening. "
    "The second view is showing the invasive hormone and ultrasound picture that supports formal clinical confirmation."
)

try:
    clinical_df = load_dataframe("pcos_clinical").copy()
except Exception as exc:
    st.error(f"Unable to load the clinical PCOS dataset: {exc}")
    render_footer()
    st.stop()

clinical_df["PCOS status"] = clinical_df["pcos_y_n"].map({0: "No PCOS", 1: "PCOS"})
clinical_df["symptom_burden"] = clinical_df[
    ["weight_gain_y_n", "hair_growth_y_n", "skin_darkening_y_n", "hair_loss_y_n", "pimples_y_n"]
].sum(axis=1)
clinical_df["bp_elevated_flag"] = (
    (clinical_df["systolic_bp_mmhg"] >= 130) | (clinical_df["diastolic_bp_mmhg"] >= 80)
).astype(int)
clinical_df["cycle_delay_flag"] = (clinical_df["cycle_length_days"] > 35).astype(int)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Total patients", len(clinical_df))
with metric_cols[1]:
    st.metric(
        "PCOS positive",
        int((clinical_df["pcos_y_n"] == 1).sum()),
        f"{(clinical_df['pcos_y_n'].mean() * 100):.1f}%",
    )
with metric_cols[2]:
    st.metric(
        "PCOS negative",
        int((clinical_df["pcos_y_n"] == 0).sum()),
        f"{((1 - clinical_df['pcos_y_n'].mean()) * 100):.1f}%",
    )
with metric_cols[3]:
    st.metric(
        "Median symptom burden",
        f"{clinical_df.groupby('pcos_y_n')['symptom_burden'].median().loc[1]:.0f}",
        "PCOS positive group",
    )

row_top_left, row_top_right = st.columns(2)
with row_top_left:
    st.plotly_chart(class_distribution_pie(clinical_df), use_container_width=True)
with row_top_right:
    st.plotly_chart(age_histogram(clinical_df), use_container_width=True)

age_summary = clinical_df.groupby("pcos_y_n")["age_yrs"].agg(["mean", "median", "std"]).round(2).rename(index={0: "No PCOS", 1: "PCOS"})
st.dataframe(age_summary, use_container_width=True)
st.markdown(
    f"""
**Overall insight:** The class split is giving us a clear comparison base, with {(clinical_df['pcos_y_n'].mean() * 100):.1f}% of patients in the PCOS-positive group.
The age pattern is also easy to read: the PCOS-positive group is slightly younger on average ({age_summary.loc['PCOS', 'mean']:.2f} years) than the PCOS-negative group ({age_summary.loc['No PCOS', 'mean']:.2f} years).
This means later charts should be read mainly as phenotype differences, not as a result of a large age shift.
"""
)

tab_noninvasive, tab_invasive = st.tabs(
    ["Non-Invasive Clinical Patterns", "Invasive Hormonal and Ultrasound Patterns"]
)

with tab_noninvasive:
    render_section_card(
        "What this section is showing",
        """
        This section is focusing on the features that can be collected during a routine clinical visit without specialist laboratory or ultrasound support.
        It is helping the user see why the non-invasive model works: the pattern is not coming from one single variable.
        It is coming from body size, cycle disturbance, visible symptoms, sugar level, and blood-pressure related strain appearing together.
        """,
    )

    non_invasive_features = [
        "bmi",
        "cycle_length_days",
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "rbs_mg_dl",
        "waist_hip_ratio",
        "symptom_burden",
    ]
    non_inv_summary = (
        clinical_df.groupby("PCOS status")[non_invasive_features]
        .agg(["mean", "median"])
        .round(2)
    )
    st.dataframe(non_inv_summary, use_container_width=True)
    st.markdown(
        """
**Non-invasive summary:** This table is showing that the PCOS-positive group is carrying a heavier routine screening pattern overall.
BMI, blood sugar, waist-hip ratio, and symptom burden are all moving in the high-risk direction, while cycle disturbance remains one of the clearest reproductive signs.
This matters because these are the same kinds of features that a primary-care or community screening workflow can actually collect.
"""
    )

    st.plotly_chart(bmi_boxplot(clinical_df), use_container_width=True)
    bmi_summary = clinical_df.groupby("PCOS status")["bmi"].agg(["mean", "median", "std"]).round(2)
    st.dataframe(bmi_summary, use_container_width=True)
    st.markdown(
        f"""
**BMI insight:** The BMI distribution is shifting upward in the PCOS group.
The median BMI is {bmi_summary.loc['PCOS', 'median']:.2f} in the PCOS-positive group compared with {bmi_summary.loc['No PCOS', 'median']:.2f} in the PCOS-negative group.
This means excess body weight is acting as one of the clearest non-invasive metabolic signals in the dataset.
"""
    )

    symptom_cols = ["weight_gain_y_n", "hair_growth_y_n", "skin_darkening_y_n", "hair_loss_y_n", "pimples_y_n"]
    st.plotly_chart(symptom_prevalence_chart(clinical_df, symptom_cols), use_container_width=True)
    symptom_gap = (
        clinical_df.groupby("pcos_y_n")[symptom_cols]
        .mean()
        .T.assign(gap=lambda df: df[1] - df[0])
        .sort_values("gap", ascending=False)
        .mul(100)
        .round(1)
    )
    st.dataframe(symptom_gap.rename(columns={0: "No PCOS %", 1: "PCOS %", "gap": "Gap %"}), use_container_width=True)
    top_symptoms = [name.replace("_y_n", "").replace("_", " ") for name in symptom_gap.index[:3]]
    st.markdown(
        f"""
**Symptom insight:** The strongest visible symptom gaps are coming from {top_symptoms[0]}, {top_symptoms[1]}, and {top_symptoms[2]}.
Skin darkening alone is separating the groups by about {symptom_gap.iloc[0]['Gap %']:.1f} percentage points, which is a very strong screening clue.
This tells us the non-invasive model is not guessing. It is learning a symptom pattern that is clearly more common in the PCOS-positive group.
"""
    )

    st.plotly_chart(bp_scatter(clinical_df), use_container_width=True)
    bp_table = (
        clinical_df.groupby("PCOS status")[["systolic_bp_mmhg", "diastolic_bp_mmhg", "bp_elevated_flag"]]
        .agg({"systolic_bp_mmhg": "mean", "diastolic_bp_mmhg": "mean", "bp_elevated_flag": "mean"})
        .round(2)
    )
    bp_table["bp_elevated_flag"] = (bp_table["bp_elevated_flag"] * 100).round(1)
    bp_table = bp_table.rename(columns={"bp_elevated_flag": "Elevated BP %"})
    st.dataframe(bp_table, use_container_width=True)
    st.markdown(
        f"""
**Blood pressure insight:** The mean blood-pressure values are only slightly different between the two groups, but the elevated-BP share is still high in both.
About {bp_table.loc['PCOS', 'Elevated BP %']:.1f}% of the PCOS-positive group crosses the elevated blood-pressure rule used in this study.
This means blood pressure is better read as part of a broader cardiometabolic pattern, not as a stand-alone PCOS marker.
"""
    )

    st.plotly_chart(cycle_violin(clinical_df), use_container_width=True)
    cycle_table = clinical_df.groupby("PCOS status")[["cycle_length_days", "cycle_delay_flag"]].agg(
        {"cycle_length_days": ["mean", "median"], "cycle_delay_flag": "mean"}
    ).round(2)
    cycle_delay_pct = (clinical_df.groupby("PCOS status")["cycle_delay_flag"].mean() * 100).round(1)
    st.dataframe(cycle_table, use_container_width=True)
    st.markdown(
        f"""
**Cycle insight:** Cycle length is one of the easiest reproductive signs to understand in this dataset.
The median cycle length is {clinical_df.groupby('PCOS status')['cycle_length_days'].median().loc['PCOS']:.2f} in the PCOS-positive group and {clinical_df.groupby('PCOS status')['cycle_length_days'].median().loc['No PCOS']:.2f} in the PCOS-negative group.
Also, {cycle_delay_pct.loc['PCOS']:.1f}% of the PCOS-positive group shows cycle length above 35 days, which strengthens the menstrual-disruption story.
"""
    )

    corr_inputs = [
        "age_yrs",
        "bmi",
        "cycle_length_days",
        "waist_hip_ratio",
        "rbs_mg_dl",
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "weight_gain_y_n",
        "hair_growth_y_n",
        "skin_darkening_y_n",
        "hair_loss_y_n",
        "pimples_y_n",
        "symptom_burden",
    ]
    st.plotly_chart(correlation_heatmap(clinical_df, corr_inputs), use_container_width=True)
    target_corr = (
        clinical_df[corr_inputs + ["pcos_y_n"]]
        .corr()["pcos_y_n"]
        .drop("pcos_y_n")
        .abs()
        .sort_values(ascending=False)
    )
    corr_table = target_corr.reset_index()
    corr_table.columns = ["feature", "abs_correlation_with_pcos"]
    st.dataframe(corr_table.round(3), use_container_width=True)
    st.markdown(
        f"""
**Correlation insight:** The strongest simple non-invasive links with the target are {corr_table.iloc[0]['feature']}, {corr_table.iloc[1]['feature']}, and {corr_table.iloc[2]['feature']}.
This is showing that the non-invasive story is being driven by visible symptoms and metabolic features at the same time.
That combination is exactly why a non-invasive screening model can still perform well.
"""
    )

    feature = st.selectbox("Select a non-invasive feature to inspect", corr_inputs, key="non_inv_feature")
    st.plotly_chart(selected_feature_histogram(clinical_df, feature), use_container_width=True)
    feature_stats = (
        clinical_df.groupby("PCOS status")[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )
    st.dataframe(feature_stats, use_container_width=True)
    st.markdown(
        f"""
**Selected feature insight:** This view is helping you inspect {feature} directly.
Use it to see whether the two groups are separating clearly, overlapping heavily, or showing outliers.
If the overlap is small, the feature is likely carrying strong screening value. If the overlap is wide, the feature still matters, but mostly in combination with other features.
"""
    )

    render_section_card(
        "Non-Invasive Takeaway",
        f"""
        The non-invasive picture is strong and easy to explain.
        The clearest routine screening signs are the visible symptom cluster, higher BMI, longer cycle pattern, and modest cardiometabolic strain.
        The most striking symptom gaps are skin darkening ({symptom_gap.iloc[0]['Gap %']:.1f} percentage points), weight gain ({symptom_gap.loc['weight_gain_y_n', 'Gap %']:.1f} percentage points), and hair growth ({symptom_gap.loc['hair_growth_y_n', 'Gap %']:.1f} percentage points).
        This means a good screening workflow does not need to wait for hormone tests before flagging a patient for follow-up.
        """,
    )

with tab_invasive:
    render_section_card(
        "What this section is showing",
        """
        This section is focusing on the invasive part of the clinical picture.
        These features come from hormone tests and ultrasound findings.
        They are not needed for first-line screening, but they help show what the deeper clinical confirmation pattern looks like.
        """,
    )

    invasive_focus = [
        "amh_ng_ml",
        "lh_miu_ml",
        "fsh_lh_ratio",
        "follicle_no_left",
        "follicle_no_right",
        "endometrium_mm",
    ]
    invasive_summary = clinical_df.groupby("PCOS status")[invasive_focus].agg(["mean", "median"]).round(2)
    st.dataframe(invasive_summary, use_container_width=True)
    st.markdown(
        """
**Invasive summary:** This table is showing a much sharper biological separation.
AMH, LH, and follicle counts move strongly upward in the PCOS-positive group, while the reproductive profile becomes clearly different from the PCOS-negative group.
This is why the invasive benchmark model is expected to perform strongly: these markers are closer to formal diagnostic evidence.
"""
    )

    amh_fig = px.box(
        clinical_df,
        x="PCOS status",
        y="amh_ng_ml",
        color="PCOS status",
        points="outliers",
        color_discrete_map={"No PCOS": "#2176AE", "PCOS": "#E63946"},
    )
    amh_fig.add_hline(y=3.5, line_dash="dash", line_color="#E9C46A", annotation_text="AMH 3.5 ng/mL marker")
    st.plotly_chart(apply_plotly_style(amh_fig, "AMH by PCOS Status"), use_container_width=True)
    amh_table = clinical_df.groupby("PCOS status")["amh_ng_ml"].agg(["mean", "median", "std"]).round(2)
    st.dataframe(amh_table, use_container_width=True)
    st.markdown(
        f"""
**AMH insight:** AMH is one of the strongest invasive markers in the dataset.
The PCOS-positive mean is {amh_table.loc['PCOS', 'mean']:.2f} ng/mL compared with {amh_table.loc['No PCOS', 'mean']:.2f} ng/mL in the PCOS-negative group.
This is a large shift and it fits the clinical literature where elevated AMH is often linked to PCOS.
"""
    )

    ratio_fig = px.violin(
        clinical_df,
        x="PCOS status",
        y="fsh_lh_ratio",
        color="PCOS status",
        box=True,
        points="all",
        color_discrete_map={"No PCOS": "#2176AE", "PCOS": "#E63946"},
    )
    ratio_fig.add_hline(y=1.0, line_dash="dash", line_color="#E9C46A", annotation_text="Ratio = 1")
    st.plotly_chart(apply_plotly_style(ratio_fig, "FSH/LH Ratio by PCOS Status"), use_container_width=True)
    ratio_table = clinical_df.groupby("PCOS status")[["fsh_miu_ml", "lh_miu_ml", "fsh_lh_ratio"]].agg(["mean", "median"]).round(2)
    st.dataframe(ratio_table, use_container_width=True)
    st.markdown(
        f"""
**Hormone-ratio insight:** The hormone balance is clearly different across the two groups.
The PCOS-positive group shows higher LH and a lower FSH/LH ratio on average, which is a pattern often discussed in PCOS endocrinology.
This means the invasive model is learning a hormone profile that is clinically believable, not a random numerical split.
"""
    )

    follicle_df = (
        clinical_df.groupby("PCOS status")[["follicle_no_left", "follicle_no_right"]]
        .mean()
        .round(2)
        .reset_index()
        .melt(id_vars="PCOS status", var_name="ovary_side", value_name="mean_follicle_count")
    )
    st.dataframe(follicle_df, use_container_width=True)
    follicle_fig = px.bar(
        follicle_df,
        x="ovary_side",
        y="mean_follicle_count",
        color="PCOS status",
        barmode="group",
        color_discrete_map={"No PCOS": "#2176AE", "PCOS": "#E63946"},
    )
    follicle_fig.add_hline(y=12, line_dash="dash", line_color="#E9C46A", annotation_text="Rotterdam count threshold")
    st.plotly_chart(apply_plotly_style(follicle_fig, "Mean Follicle Counts by Ovary Side and PCOS Status"), use_container_width=True)
    follicle_threshold = (
        clinical_df.assign(
            either_ovary_ge_12=((clinical_df["follicle_no_left"] >= 12) | (clinical_df["follicle_no_right"] >= 12)).astype(int)
        )
        .groupby("PCOS status")["either_ovary_ge_12"]
        .mean()
        .mul(100)
        .round(1)
    )
    st.markdown(
        f"""
**Follicle-count insight:** The follicle picture is much heavier in the PCOS-positive group for both ovaries.
About {follicle_threshold.loc['PCOS']:.1f}% of the PCOS-positive group meets the >=12 follicle rule in at least one ovary, compared with only {follicle_threshold.loc['No PCOS']:.1f}% in the PCOS-negative group.
This is one of the clearest invasive signs in the dataset and strongly supports the benchmark model.
"""
    )

    endometrium_fig = px.box(
        clinical_df,
        x="PCOS status",
        y="endometrium_mm",
        color="PCOS status",
        points="outliers",
        color_discrete_map={"No PCOS": "#2176AE", "PCOS": "#E63946"},
    )
    st.plotly_chart(apply_plotly_style(endometrium_fig, "Endometrium Thickness by PCOS Status"), use_container_width=True)
    endometrium_table = clinical_df.groupby("PCOS status")["endometrium_mm"].agg(["mean", "median", "std"]).round(2)
    st.dataframe(endometrium_table, use_container_width=True)
    st.markdown(
        f"""
**Endometrium insight:** Endometrium thickness is showing a smaller shift than AMH or follicle count.
That makes it useful as a supporting marker rather than the main driver of separation.
This is important because not every invasive feature carries the same diagnostic weight.
"""
    )

    invasive_corr_inputs = [
        "amh_ng_ml",
        "fsh_miu_ml",
        "lh_miu_ml",
        "fsh_lh_ratio",
        "prl_ng_ml",
        "vit_d3_ng_ml",
        "prg_ng_ml",
        "follicle_no_left",
        "follicle_no_right",
        "avg_follicle_size_left_mm",
        "avg_follicle_size_right_mm",
        "endometrium_mm",
    ]
    st.plotly_chart(correlation_heatmap(clinical_df, invasive_corr_inputs), use_container_width=True)
    invasive_target_corr = (
        clinical_df[invasive_corr_inputs + ["pcos_y_n"]]
        .corr()["pcos_y_n"]
        .drop("pcos_y_n")
        .abs()
        .sort_values(ascending=False)
    )
    invasive_corr_table = invasive_target_corr.reset_index()
    invasive_corr_table.columns = ["feature", "abs_correlation_with_pcos"]
    st.dataframe(invasive_corr_table.round(3), use_container_width=True)
    st.markdown(
        f"""
**Invasive correlation insight:** The leading invasive links with the target are {invasive_corr_table.iloc[0]['feature']}, {invasive_corr_table.iloc[1]['feature']}, and {invasive_corr_table.iloc[2]['feature']}.
This is showing that the invasive benchmark is drawing its strength from hormone and ovarian morphology markers together.
That is exactly what we would expect from a clinically deeper model.
"""
    )

    invasive_feature = st.selectbox("Select an invasive feature to inspect", invasive_corr_inputs, key="inv_feature")
    st.plotly_chart(selected_feature_histogram(clinical_df, invasive_feature), use_container_width=True)
    invasive_feature_stats = (
        clinical_df.groupby("PCOS status")[invasive_feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )
    st.dataframe(invasive_feature_stats, use_container_width=True)
    st.markdown(
        f"""
**Selected invasive feature insight:** This view lets you inspect {invasive_feature} directly.
If the two groups separate clearly here, the feature is acting like a strong benchmark marker.
If the overlap is wider, the feature still matters, but mostly as part of the full invasive pattern rather than alone.
"""
    )

    render_section_card(
        "Invasive Takeaway",
        f"""
        The invasive story is sharper and more biological than the non-invasive story.
        AMH rises from {amh_table.loc['No PCOS', 'mean']:.2f} to {amh_table.loc['PCOS', 'mean']:.2f} ng/mL, LH rises strongly, and follicle counts almost double across both ovaries.
        These are the kinds of markers that support clinical confirmation rather than first-line community screening.
        Together they explain why the invasive benchmark performs so well: it is learning a hormone and ultrasound signature that is much closer to formal diagnostic evidence.
        """,
    )

render_section_card(
    "Dashboard Summary",
    """
    The main lesson from this dashboard is simple.
    The non-invasive view shows that visible symptoms, body size, sugar, and cycle disturbance already create a strong screening pattern.
    The invasive view then shows that hormone and ultrasound markers make that separation even sharper.
    This means the two parts of the thesis are working together: the non-invasive model is useful for early flagging, while the invasive benchmark shows what added clinical depth can confirm.
    """,
)

render_footer()
