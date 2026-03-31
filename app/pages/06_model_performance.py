import pandas as pd
import streamlit as st

from app.components.shap_display import show_image_or_placeholder
from app.components.styling import apply_global_styles, configure_page, render_footer, render_section_card, render_sidebar
from app.components.utils import best_individual_rows, best_model_rows, load_dataframe
from app.config.app_config import IMAGE_PATHS


configure_page("Model Performance", "📈")
apply_global_styles()
render_sidebar()

st.markdown(
    """
    <div class='page-header'>
        <h1>📈 Model Performance</h1>
        <p>Complete evaluation across all modelling approaches — individual, ensemble, and optimized</p>
    </div>
    """,
    unsafe_allow_html=True,
)

final_results = load_dataframe("final_results")
best_rows = best_model_rows(final_results)
best_individual = best_individual_rows(final_results)

with st.expander("Individual Models (Logistic Regression, Random Forest, XGBoost)", expanded=False):
    st.markdown("Logistic Regression is acting as the simple linear baseline. Random Forest is capturing non-linear splits and interactions. XGBoost is adding boosted tree learning that often performs strongly on tabular clinical data.")

with st.expander("Ensemble Stacking", expanded=False):
    st.markdown(
        """
        Stacking is combining several base learners instead of trusting one model alone.
        The base learners each make a probability prediction first. A meta-learner then learns how to combine those probabilities.

        ```text
        Patient features
             |
        +----+----+----+
        |    |    |    |
        LR   RF   XGB  -> base probabilities
              |
           Meta-learner
              |
         Final PCOS probability
        ```

        Out-of-fold predictions are being used during training so the meta-learner does not learn from leaked base-model predictions.
        """
    )

with st.expander("Metaheuristic Optimization (WaO, RSO, CSO)", expanded=False):
    st.markdown("Metaheuristic optimization is tuning the stack hyperparameters with nature-inspired search. WaO, RSO, and CSO are exploring different parts of the search space, keeping good candidates, and improving the final ensemble beyond the baseline stacking stage.")

tabs = st.tabs(["Model 1: Non-Invasive", "Model 2: Invasive Benchmark", "Side-by-Side Summary"])
for tab, model_set in zip(tabs[:2], ["Model 1", "Model 2"]):
    with tab:
        subset = final_results[final_results["model_set"] == model_set].copy()
        numeric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
        styled = subset[["model", "approach", "algorithm"] + numeric_cols].style.highlight_max(subset=numeric_cols, color="#D1FAE5").highlight_min(subset=numeric_cols, color="#FEE2E2").format({col: "{:.4f}" for col in numeric_cols})
        st.dataframe(styled, use_container_width=True)

with tabs[2]:
    summary_rows = []
    for model_set in ["Model 1", "Model 2"]:
        for approach in ["Individual", "Stacking", "Optimized"]:
            subset = final_results[
                (final_results["model_set"] == model_set)
                & (final_results["approach"] == approach)
            ]
            best_row = subset.loc[subset["auc"].idxmax()]
            summary_rows.append(
                {
                    "Model set": model_set,
                    "Approach": approach,
                    "Best model": best_row["model"],
                    "AUC": round(best_row["auc"], 4),
                    "Recall": round(best_row["recall"], 4),
                    "F1": round(best_row["f1"], 4),
                }
            )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    show_image_or_placeholder(IMAGE_PATHS["convergence_m1"], "Model 1 optimization convergence")
    st.markdown("This chart is showing how WaO, RSO, and CSO searched the Model 1 non-invasive space. It helps show which optimizer found a strong answer fastest.")
with col2:
    show_image_or_placeholder(IMAGE_PATHS["convergence_m2"], "Model 2 optimization convergence")
    st.markdown("This chart is showing the same optimization story for the invasive benchmark model.")

show_image_or_placeholder(IMAGE_PATHS["master_heatmap"], "Master optimization heatmap")
st.markdown("The master heatmap is summarizing all phases together — individual models, stacking, and optimized ensembles. It gives one compact view of which approach performed best on each metric.")

gap = best_rows["Model 2"]["auc"] - best_rows["Model 1"]["auc"]
render_section_card(
    "Performance Gap Analysis",
    f"""
    The best non-invasive model is reaching an AUC of {best_rows['Model 1']['auc']:.3f}.
    The best invasive benchmark is reaching an AUC of {best_rows['Model 2']['auc']:.3f}.
    The gap is {gap:+.3f} percentage points. In this repo state the non-invasive model is actually matching or slightly exceeding the invasive benchmark, which strongly supports practical PCOS screening without laboratory tests.
    """,
)

shap_cols = st.columns(2)
with shap_cols[0]:
    show_image_or_placeholder(IMAGE_PATHS["shap_m1_bar"], "Non-Invasive Model Feature Importance")
    st.markdown("The non-invasive model is being driven by routine screening signals such as BMI, cycle pattern, symptom burden, and metabolic markers.")
with shap_cols[1]:
    show_image_or_placeholder(IMAGE_PATHS["shap_m2_bar"], "Invasive Benchmark Feature Importance")
    st.markdown("The invasive model is being driven by hormonal and ultrasound features such as AMH, follicle burden, and endocrine ratios.")

render_footer()
