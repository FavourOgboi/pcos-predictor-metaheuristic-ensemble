import streamlit as st

from components.styling import render_metric_card, render_section_card
from components.utils import best_individual_rows, best_model_rows, get_metric_deltas, load_dataframe
from config.app_config import DATASET_SUMMARY


def render_home_page() -> None:
    final_results = load_dataframe("final_results")
    best_rows = best_model_rows(final_results)
    deltas = get_metric_deltas(final_results)

    st.markdown(
        """
        <div class='page-header'>
            <div style='display:flex; align-items:center; gap:1.5rem;'>
                <div style='font-size:4rem;'>🔬</div>
                <div>
                    <h1>PCOS Early Detection System</h1>
                    <p>A Metaheuristic-Optimized Ensemble Learning Approach for Non-Invasive Polycystic Ovary Syndrome Screening</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_card(
        "About This Thesis",
        """
        This thesis is studying how PCOS can be detected earlier with machine learning.
        The main idea is simple: many women show useful warning signs before hormone tests or ultrasound are done.
        If those warning signs can be learned well, then screening can start earlier, with less cost and less delay.
        The study is therefore building a non-invasive PCOS screening system, comparing it with an invasive clinical benchmark,
        optimizing both with metaheuristic search, and then explaining the final models with SHAP so the decisions can be understood.
        """,
    )

    thesis_cols = st.columns(3)
    thesis_cards = [
        (
            "Main Aim",
            "The thesis is checking whether routine clinical and lifestyle features can support meaningful early PCOS screening without depending fully on laboratory or ultrasound tests.",
        ),
        (
            "Why This Matters",
            "PCOS is often diagnosed late. Earlier screening can help women seek follow-up sooner, reduce uncertainty, and support earlier lifestyle or clinical intervention.",
        ),
        (
            "What This System Shows",
            "The final app brings the full study together: screening models, benchmark comparison, optimization results, explainability, and the main clinical insights from the data.",
        ),
    ]
    for col, (title, body) in zip(thesis_cols, thesis_cards):
        with col:
            render_section_card(title, body)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Best Model 1 AUC", f"{best_rows['Model 1']['auc']:.3f}", f"{deltas['model1_auc_delta']:+.3f} vs best individual")
    with metric_cols[1]:
        render_metric_card("Best Model 1 Recall", f"{best_rows['Model 1']['recall']:.3f}", f"{deltas['model1_recall_delta']:+.3f} vs best individual")
    with metric_cols[2]:
        render_metric_card("Best Model 2 AUC", f"{best_rows['Model 2']['auc']:.3f}", f"{deltas['model2_auc_delta']:+.3f} vs best individual")
    with metric_cols[3]:
        gap = best_rows["Model 2"]["auc"] - best_rows["Model 1"]["auc"]
        render_metric_card("AUC Gap", f"{gap:+.3f}", f"{deltas['gap_delta']:+.3f} vs individual gap")

    st.markdown("### Study Overview")
    st.markdown(
        """
        <div class='section-card'>
            <div class='timeline-wrap'>
                <div class='timeline-step'><strong>1. Data Collection and Cleaning</strong><br><span style='color:#5B708A;'>Raw clinical, survey, and heart datasets were cleaned and audited.</span></div>
                <div class='timeline-step'><strong>2. Exploratory Analysis</strong><br><span style='color:#5B708A;'>Clinical, hormonal, survey, and heart-related patterns were studied in detail.</span></div>
                <div class='timeline-step'><strong>3. Model Development</strong><br><span style='color:#5B708A;'>Individual models and stacked ensembles were trained on locked modelling sets.</span></div>
                <div class='timeline-step'><strong>4. Optimization</strong><br><span style='color:#5B708A;'>WaO, RSO, and CSO were used to tune the final ensemble models.</span></div>
                <div class='timeline-step'><strong>5. Deployment</strong><br><span style='color:#5B708A;'>The final thesis tool is being packaged as a clinical research web application.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    study_detail_cols = st.columns(2)
    with study_detail_cols[0]:
        render_section_card(
            "What The Models Are Doing",
            """
            Model 1 is using non-invasive features such as BMI, cycle pattern, blood pressure, sugar level, symptoms, and lifestyle factors.
            It is answering the practical screening question: can useful PCOS risk prediction happen without specialist tests?
            Model 2 is using invasive hormone and ultrasound features as a benchmark, so the study can measure how much performance is gained when deeper clinical information is available.
            """,
        )
    with study_detail_cols[1]:
        render_section_card(
            "What Makes The Study Stronger",
            """
            The work is not stopping at single models.
            It moves from individual algorithms to stacking, then to metaheuristic optimization with WaO, RSO, and CSO, and finally to SHAP explainability and survey-style validation.
            This makes the final system easier to defend because it does not only predict well — it also shows how the result was improved and why the final prediction can be interpreted.
            """,
        )

    st.markdown("### Dataset Summary")
    dataset_cols = st.columns(3)
    for col, item in zip(dataset_cols, DATASET_SUMMARY):
        with col:
            render_metric_card(item["name"], f"{item['n']} patients", f"{item['features']} features · {item['note']}")
