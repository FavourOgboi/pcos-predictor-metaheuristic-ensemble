from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "11_shap_explainability.ipynb"


def normalize_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    if normalized:
        normalized += "\n"
    return normalized.splitlines(keepends=True)


def normalize_markdown_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    cleaned_lines = [line.lstrip() if line.strip() else "" for line in normalized.splitlines()]
    normalized = "\n".join(cleaned_lines)
    if normalized:
        normalized += "\n"
    return normalized.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": normalize_markdown_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalize_source(text),
    }


def notebook_document(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook_document(cells), indent=2), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


OUTPUT_ROWS = [
    ("`images/shap/01_shap_concept_diagram.png`", "Main visual that introduces local and global SHAP ideas."),
    ("`images/shap/02_shap_game_theory_illustration.png`", "Game-theory visual showing how coalitions build the prediction credit idea."),
    ("`images/shap/model1/03-13_*.png`", "Model 1 SHAP figures: global, local, dependence, interaction, and meta-learner weight views."),
    ("`images/shap/model2/14-24_*.png`", "Model 2 SHAP figures: global, local, dependence, interaction, and meta-learner weight views."),
    ("`images/shap/survey/25-29_*.png`", "Survey validation figures: class balance, ROC, confusion matrix, and survey SHAP views."),
    ("`cleaned_data/modelling_sets/shap_values_model1.npy`", "Saved Model 1 RF-based SHAP matrix on the clinical test set."),
    ("`cleaned_data/modelling_sets/shap_values_model2.npy`", "Saved Model 2 RF-based SHAP matrix on the clinical test set."),
    ("`cleaned_data/modelling_sets/shap_values_survey.npy`", "Saved survey SHAP matrix using the Model 1 RF explainer."),
    ("`cleaned_data/modelling_sets/shap_feature_ranking_model1.csv`", "Saved Model 1 global SHAP ranking table."),
    ("`cleaned_data/modelling_sets/shap_feature_ranking_model2.csv`", "Saved Model 2 global SHAP ranking table."),
    ("`cleaned_data/modelling_sets/survey_validation_results.csv`", "Saved external-style survey validation metrics."),
    ("`cleaned_data/modelling_sets/survey_validation_predictions.csv`", "Saved row-level survey predictions and probabilities."),
]


SHAP_OUTPUT_TABLE = markdown_table(["Path", "Meaning"], OUTPUT_ROWS)


M1_TOP_TEXT = """
**Scientific meaning:** This section is using SHAP to explain the final non-invasive PCOS model. The analysis is showing which routine screening features are pushing the prediction up or down across the clinical test patients.

**Simple summary:** This part is showing what the final non-invasive model is paying attention to.

**Why this matters:** If the top features make clinical sense, the model is easier to trust and easier to explain to clinicians and patients.
"""


M2_TOP_TEXT = """
**Scientific meaning:** This section is using SHAP to explain the final invasive benchmark model. The analysis is showing which hormonal and ultrasound features are driving the final prediction.

**Simple summary:** This part is showing what the invasive benchmark model is relying on.

**Why this matters:** If the top features match known PCOS biomarkers, the model is learning real clinical patterns rather than noise.
"""


def model_section(
    prefix: str,
    model_name: str,
    model_num: str,
    feature_var: str,
    train_var: str,
    ytrain_var: str,
    test_var: str,
    ytest_var: str,
    bundle_var: str,
    best_var: str,
    shap_file: str,
    rank_file: str,
    folder: str,
    model_color: str,
    interaction_feature_a: str,
    interaction_feature_b: str,
    interaction_file: str,
    interaction_title: str,
    summary_note: str,
) -> list[dict]:
    title = f"## Section {model_num} — Explaining {model_name}"
    if prefix == "m1":
        intro = (
            "This section is explaining the final optimized non-invasive screening model. "
            "The model uses 19 routine features that can be collected without hormonal testing or ultrasound. "
            "The saved optimized bundle is the source of truth, and the saved JSON file is providing the optimizer name and the final reported test metrics."
        )
    else:
        intro = (
            "This section is explaining the final optimized invasive benchmark model. "
            "The model uses 15 hormonal and ultrasound features. "
            "The saved optimized bundle is the source of truth, and the SHAP results are being read against known PCOS biomarker literature rather than against Model 1."
        )

    cells: list[dict] = [
        markdown_cell(title + "\n\n" + intro),
        markdown_cell(summary_note),
        code_cell(
            f"""
            # Loading the saved final optimized {model_name} bundle is making the explainability section use the deployed model artifact directly.
            print('Saved best-parameter JSON:')
            print(json.dumps({best_var}, indent=2))
            print()
            print('Saved optimized model test results:')
            print({bundle_var}['test_results'])
            print()
            print('Saved stack family:')
            print({bundle_var}['stack_family'])
            print()
            print('Saved feature count:')
            print(len({bundle_var}['feature_names']))
            """
        ),
        code_cell(
            f"""
            # Running the saved final optimized {model_name} bundle on the test set is confirming that the loaded artifact reproduces its saved metrics.
            test_meta_{prefix}, y_{prefix}_proba = stack_predict_proba({bundle_var}, {test_var})
            threshold_{prefix} = float({bundle_var}.get('threshold', 0.5))
            y_{prefix}_pred = (y_{prefix}_proba >= threshold_{prefix}).astype(int)
            results_{prefix}_final = evaluate_prediction('{model_name} Final Optimized', {ytest_var}, y_{prefix}_pred, y_{prefix}_proba)
            results_{prefix}_check = pd.DataFrame([
                {{'source': 'Saved bundle', **{bundle_var}['test_results']}},
                {{'source': 'Notebook 11 rerun', **results_{prefix}_final}}
            ])
            print(results_{prefix}_check.to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Building test DataFrames is making the SHAP plotting cells easier to read and easier to label.
            {prefix}_test_df = pd.DataFrame({test_var}, columns={feature_var})
            {prefix}_train_df = pd.DataFrame({train_var}, columns={feature_var})
            print({prefix}_test_df.head().to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Computing TreeSHAP values from the saved Random Forest base learner is producing the main explanation layer for {model_name}.
            rf_{prefix} = {bundle_var}['base_models']['rf']
            explainer_{prefix} = shap.TreeExplainer(rf_{prefix})
            shap_values_raw_{prefix} = explainer_{prefix}.shap_values({test_var})
            sv_{prefix} = extract_positive_shap(shap_values_raw_{prefix})
            expected_value_{prefix} = extract_positive_expected_value(explainer_{prefix}.expected_value)
            explanation_{prefix} = shap.Explanation(
                values=sv_{prefix},
                base_values=np.repeat(expected_value_{prefix}, sv_{prefix}.shape[0]),
                data={test_var},
                feature_names={feature_var},
            )
            np.save(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / '{shap_file}', sv_{prefix})
            print('SHAP matrix shape:')
            print(sv_{prefix}.shape)
            print()
            print('Saved file:')
            print(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / '{shap_file}')
            """
        ),
        code_cell(
            f"""
            # Building the SHAP ranking table is summarizing how important each {model_name} feature is on average.
            mean_abs_shap_{prefix} = np.abs(sv_{prefix}).mean(axis=0)
            shap_rank_{prefix} = pd.DataFrame({{
                'feature': {feature_var},
                'mean_abs_shap': mean_abs_shap_{prefix},
                'mean_shap': sv_{prefix}.mean(axis=0),
                'positive_impact_share': (sv_{prefix} > 0).mean(axis=0),
                'negative_impact_share': (sv_{prefix} < 0).mean(axis=0),
            }}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            shap_rank_{prefix}.insert(0, 'rank', np.arange(1, len(shap_rank_{prefix}) + 1))
            shap_rank_{prefix}.to_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / '{rank_file}', index=False)
            print(shap_rank_{prefix}.to_string(index=False))
            """
        ),
        markdown_cell(
            f"""
            ### Global SHAP Explanation — {model_name}

            This section is answering one main question: across all test patients, which features are mattering most on average? The mean absolute SHAP value is showing average importance, while the sign of the SHAP values is showing whether a feature is usually pushing the prediction toward PCOS or away from PCOS.
            """
        ),
        code_cell(
            f"""
            # Printing the global SHAP ranking table is showing the exact data used for the bar chart.
            global_bar_{prefix} = shap_rank_{prefix}[['rank', 'feature', 'mean_abs_shap']].copy()
            print(global_bar_{prefix}.to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Plotting the global SHAP bar chart is showing the average absolute contribution of each {model_name} feature.
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            plot_df = global_bar_{prefix}.sort_values('mean_abs_shap', ascending=True)
            colors = sns.light_palette(COLORS['{model_color}'], n_colors=len(plot_df), reverse=True)
            bars = ax.barh(plot_df['feature'], plot_df['mean_abs_shap'], color=colors, edgecolor='black', linewidth=0.4)
            ax.axvline(0.05, color='#999999', linestyle='--', linewidth=1, label='0.05 reference')
            ax.axvline(0.10, color='#555555', linestyle=':', linewidth=1, label='0.10 reference')
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('{model_name} — Global SHAP Feature Importance')
            for bar, value in zip(bars, plot_df['mean_abs_shap']):
                ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f'{{value:.3f}}', va='center', fontsize=8)
            ax.legend(loc='lower right', frameon=False)
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{'03_m1_shap_bar_global.png' if prefix == 'm1' else '14_m2_shap_bar_global.png'}', bbox_inches='tight')
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Global Bar Chart Insight

            This chart is showing the features with the strongest average effect on the prediction. Features at the top of the chart are the ones the model is using most often. This matters because these are the features you would mention first when explaining how the model works.
            """
        ),
        code_cell(
            f"""
            # Printing the beeswarm support table is showing the average SHAP direction for high and low feature values before plotting.
            beeswarm_summary_{prefix} = []
            for feature in {feature_var}:
                values = {prefix}_test_df[feature].values
                shap_col = sv_{prefix}[:, {feature_var}.index(feature)]
                median_value = float(np.nanmedian(values))
                high_mask = values >= median_value
                low_mask = values < median_value
                beeswarm_summary_{prefix}.append({{
                    'feature': feature,
                    'median_value': median_value,
                    'mean_shap_high_values': float(np.nanmean(shap_col[high_mask])) if high_mask.any() else np.nan,
                    'mean_shap_low_values': float(np.nanmean(shap_col[low_mask])) if low_mask.any() else np.nan,
                }})
            beeswarm_summary_{prefix} = pd.DataFrame(beeswarm_summary_{prefix}).sort_values('feature')
            print(beeswarm_summary_{prefix}.round(4).to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Plotting the beeswarm view is showing both SHAP direction and patient-level spread for {model_name}.
            shap.summary_plot(sv_{prefix}, {prefix}_test_df, feature_names={feature_var}, show=False, plot_size=(9.5, 6.5))
            plt.title('{model_name} — SHAP Beeswarm Plot')
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{'04_m1_shap_beeswarm.png' if prefix == 'm1' else '15_m2_shap_beeswarm.png'}', bbox_inches='tight')
            plt.show()
            """
        ),
        markdown_cell(
            f"""
            ### Reading the Beeswarm Plot — {model_name}

            Each dot is one patient. The horizontal position is showing how much that feature pushed the prediction toward PCOS on the right or away from PCOS on the left. The colour is showing whether that patient's feature value was high or low. When red dots sit more on the right and blue dots sit more on the left, it means higher values of that feature are usually increasing the model's PCOS risk estimate.
            """
        ),
        code_cell(
            f"""
            # Printing SHAP summary statistics is showing the exact matrix summary used before the heatmap.
            shap_stats_{prefix} = pd.DataFrame({{
                'feature': {feature_var},
                'mean': sv_{prefix}.mean(axis=0),
                'std': sv_{prefix}.std(axis=0),
                'min': sv_{prefix}.min(axis=0),
                'max': sv_{prefix}.max(axis=0),
            }})
            print(shap_stats_{prefix}.round(4).to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Plotting the SHAP heatmap is showing whether groups of patients share similar explanation patterns.
            shap.plots.heatmap(explanation_{prefix}, show=False, max_display=min(12, len({feature_var})))
            plt.gcf().set_size_inches(10, 6.5)
            plt.title('{model_name} — SHAP Heatmap')
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{'05_m1_shap_heatmap.png' if prefix == 'm1' else '16_m2_shap_heatmap.png'}', bbox_inches='tight')
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Heatmap Insight

            This heatmap is showing whether some patients share similar explanation patterns. If blocks of colour appear together, that suggests the model is seeing clusters of patients with similar feature-driven risk patterns.
            """
        ),
        markdown_cell(
            f"""
            ### Local SHAP Explanation — {model_name}

            This section is moving from the population view to the patient view. Three cases are being checked: a true positive, a false negative, and a false positive. This is showing why the final stack got a patient right, why it missed a case, and why it raised a false alarm.
            """
        ),
        code_cell(
            f"""
            # Selecting one true positive, one false negative, and one false positive is preparing the local explanation cases.
            selected_cases_{prefix} = select_case_indices({ytest_var}, y_{prefix}_pred, y_{prefix}_proba)
            print('Case counts:')
            print(f"  TP count: {{selected_cases_{prefix}['tp_count']}}")
            print(f"  FN count: {{selected_cases_{prefix}['fn_count']}}")
            print(f"  FP count: {{selected_cases_{prefix}['fp_count']}}")
            print()
            print('Selected indices:')
            print(selected_cases_{prefix})
            """
        ),
        code_cell(
            f"""
            # Printing the selected patient feature values is showing the raw profile behind each local explanation.
            patient_cases_{prefix} = {prefix}_test_df.copy()
            for case_label in ['tp', 'fn', 'fp']:
                idx = selected_cases_{prefix}[f'{{case_label}}_idx']
                print(case_label.upper(), 'patient values')
                print(patient_cases_{prefix}.iloc[idx].to_string())
                print(f"Predicted probability: {{y_{prefix}_proba[idx]:.4f}}")
                print(f"Actual label: {{int({ytest_var}[idx])}} | Predicted label: {{int(y_{prefix}_pred[idx])}}")
                print('-' * 60)
            """
        ),
    ]
    dep_files = {
        "m1": ["09_m1_shap_dependence_feature1.png", "10_m1_shap_dependence_feature2.png", "11_m1_shap_dependence_feature3.png"],
        "m2": ["20_m2_shap_dependence_feature1.png", "21_m2_shap_dependence_feature2.png", "22_m2_shap_dependence_feature3.png"],
    }[prefix]
    water_files = {
        "m1": ["06_m1_shap_waterfall_true_positive.png", "07_m1_shap_waterfall_false_negative.png", "08_m1_shap_waterfall_false_positive.png"],
        "m2": ["17_m2_shap_waterfall_true_positive.png", "18_m2_shap_waterfall_false_negative.png", "19_m2_shap_waterfall_false_positive.png"],
    }[prefix]
    case_names = [("tp", "True Positive"), ("fn", "False Negative"), ("fp", "False Positive")]
    for idx, (case_key, case_label) in enumerate(case_names):
        cells.append(code_cell(
            f"""
            # Plotting the {case_label.lower()} waterfall is showing which features pushed this case up or down.
            {case_key}_idx_{prefix} = selected_cases_{prefix}['{case_key}_idx']
            {case_key}_explanation_{prefix} = shap.Explanation(
                values=sv_{prefix}[{case_key}_idx_{prefix}],
                base_values=expected_value_{prefix},
                data={test_var}[{case_key}_idx_{prefix}],
                feature_names={feature_var},
            )
            shap.plots.waterfall({case_key}_explanation_{prefix}, show=False)
            plt.gcf().set_size_inches(9, 5.8)
            plt.title('{model_name} — {case_label} Patient')
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{water_files[idx]}', bbox_inches='tight')
            plt.show()
            """
        ))
        cells.append(markdown_cell(
            f"""
            ### {case_label} Waterfall Insight

            This waterfall is showing how the final score moved from the base value to the patient's final prediction. Features pushing to the right are raising risk, and features pushing to the left are lowering risk. This matters because it shows the exact feature mix behind a correct call, a missed case, or a false alarm.
            """
        ))
    cells.append(markdown_cell(
        f"""
        ### SHAP Dependence Plots — {model_name}

        Dependence plots are showing how the value of a feature is relating to its SHAP contribution across patients. This helps answer whether the effect is roughly linear, threshold-based, or more mixed.
        """
    ))
    cells.append(code_cell(
        f"""
        # Printing the top three SHAP features is selecting the features used for the dependence plots.
        top3_{prefix} = shap_rank_{prefix}.head(3)['feature'].tolist()
        top3_table_{prefix} = shap_rank_{prefix}.head(3)[['rank', 'feature', 'mean_abs_shap']]
        print(top3_table_{prefix}.to_string(index=False))
        """
    ))
    for idx in range(3):
        cells.append(code_cell(
            f"""
            # Plotting dependence feature {idx + 1} is showing how one top feature changes its SHAP contribution across patients.
            feature_{prefix}_{idx + 1} = top3_{prefix}[{idx}]
            shap.dependence_plot(feature_{prefix}_{idx + 1}, sv_{prefix}, {prefix}_test_df, feature_names={feature_var}, interaction_index='auto', show=False)
            plt.gcf().set_size_inches(8.2, 5.4)
            plt.title('{model_name} — SHAP Dependence: ' + feature_{prefix}_{idx + 1})
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{dep_files[idx]}', bbox_inches='tight')
            plt.show()
            """
        ))
        cells.append(markdown_cell(
            """
            ### Dependence Plot Insight

            This dependence plot is showing how the feature value and the SHAP effect move together. A smooth upward or downward pattern suggests a more monotonic effect, while a scattered pattern suggests a more mixed or threshold-like effect.
            """
        ))
    cells.extend([
        code_cell(
            f"""
            # Printing the interaction support table is showing the exact grouped data used before the interaction plot.
            idx_a_{prefix} = {feature_var}.index('{interaction_feature_a}')
            idx_b_{prefix} = {feature_var}.index('{interaction_feature_b}')
            interaction_df_{prefix} = pd.DataFrame({{
                'feature_a_value': {test_var}[:, idx_a_{prefix}],
                'feature_a_shap': sv_{prefix}[:, idx_a_{prefix}],
                'feature_b_value': {test_var}[:, idx_b_{prefix}],
            }})
            if '{interaction_feature_b}' == 'cycle_regularity_binary':
                interaction_df_{prefix}['feature_b_group'] = interaction_df_{prefix}['feature_b_value'].map({{0: 'Irregular', 1: 'Regular'}}).fillna('Unknown')
            else:
                interaction_df_{prefix}['feature_b_group'] = pd.qcut(
                    interaction_df_{prefix}['feature_b_value'],
                    q=2,
                    labels=['Lower', 'Higher'],
                    duplicates='drop',
                )
            interaction_groups_{prefix} = interaction_df_{prefix}.groupby('feature_b_group')[['feature_a_value', 'feature_a_shap']].describe().round(3)
            print(interaction_groups_{prefix}.to_string())
            """
        ),
        code_cell(
            f"""
            # Plotting the main interaction view is checking whether feature A is acting differently at different levels of feature B.
            interaction_plot_df_{prefix} = interaction_df_{prefix}.copy()
            fig, ax = plt.subplots(figsize=(8.6, 5.4))
            if '{interaction_feature_b}' == 'cycle_regularity_binary':
                palette = {{'Irregular': COLORS['shap_pos'], 'Regular': COLORS['shap_neg'], 'Unknown': COLORS['neutral']}}
            else:
                palette = {{'Lower': COLORS['shap_neg'], 'Higher': COLORS['shap_pos']}}
            for group_name, group_df in interaction_plot_df_{prefix}.groupby('feature_b_group'):
                ax.scatter(
                    group_df['feature_a_value'],
                    group_df['feature_a_shap'],
                    s=30,
                    alpha=0.65,
                    color=palette[str(group_name)],
                    label=str(group_name),
                )
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel('{interaction_feature_a}')
            ax.set_ylabel('SHAP value for {interaction_feature_a}')
            ax.set_title('{interaction_title}')
            ax.legend(title='{interaction_feature_b} group', frameon=False)
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{interaction_file}', bbox_inches='tight')
            plt.show()
            """
        ),
        markdown_cell(
            f"""
            ### Interaction Plot Insight

            This interaction plot is checking whether `{interaction_feature_a}` becomes more or less important when `{interaction_feature_b}` changes. This matters because PCOS risk often comes from combined feature patterns rather than from isolated features.
            """
        ),
        code_cell(
            f"""
            # Printing the meta-learner coefficient table is showing how much the final LR meta-model trusted each base learner.
            meta_coefs_{prefix} = {bundle_var}['meta_model'].coef_[0]
            base_names_{prefix} = ['Logistic Regression', 'Random Forest', 'XGBoost']
            meta_weight_df_{prefix} = pd.DataFrame({{
                'base_learner': base_names_{prefix},
                'coefficient': meta_coefs_{prefix},
                'abs_weight': np.abs(meta_coefs_{prefix}),
            }}).sort_values('abs_weight', ascending=False).reset_index(drop=True)
            print(meta_weight_df_{prefix}.to_string(index=False))
            """
        ),
        code_cell(
            f"""
            # Plotting the meta-learner weight chart is showing which base learner the final stack trusted most.
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            meta_plot_df_{prefix} = meta_weight_df_{prefix}.sort_values('coefficient')
            colors_{prefix} = [COLORS['{model_color}'] if val >= 0 else COLORS['shap_neg'] for val in meta_plot_df_{prefix}['coefficient']]
            bars = ax.barh(meta_plot_df_{prefix}['base_learner'], meta_plot_df_{prefix}['coefficient'], color=colors_{prefix}, edgecolor='black', linewidth=0.4)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Meta-learner coefficient')
            ax.set_title('{model_name} — Meta-Learner Base Learner Weights')
            for bar, value in zip(bars, meta_plot_df_{prefix}['coefficient']):
                ax.text(value + (0.02 if value >= 0 else -0.02), bar.get_y() + bar.get_height()/2, f'{{value:.3f}}', va='center', ha='left' if value >= 0 else 'right', fontsize=8)
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '{folder}' / '{'13_m1_meta_base_learner_weights.png' if prefix == 'm1' else '24_m2_meta_base_learner_weights.png'}', bbox_inches='tight')
            plt.show()
            """
        ),
        markdown_cell(
            """
            ### Meta-Learner Weight Insight

            This chart is showing how strongly the final stack is weighting each base learner. A larger absolute coefficient means the meta-learner is trusting that base learner more when it forms the final probability.
            """
        ),
        code_cell(
            f"""
            # Running a small Kernel SHAP fidelity check is checking whether the RF-based SHAP story broadly matches the full final stack on a small subset.
            background_idx_{prefix} = stratified_take({ytrain_var}, 40, seed=RANDOM_STATE)
            explain_idx_{prefix} = stratified_take({ytest_var}, 30, seed=RANDOM_STATE + 1)
            background_{prefix} = {train_var}[background_idx_{prefix}]
            explain_subset_{prefix} = {test_var}[explain_idx_{prefix}]
            kernel_explainer_{prefix} = shap.KernelExplainer(lambda data: stack_predict_proba({bundle_var}, np.asarray(data))[1], background_{prefix})
            kernel_values_raw_{prefix} = kernel_explainer_{prefix}.shap_values(explain_subset_{prefix}, nsamples=100)
            kernel_sv_{prefix} = extract_positive_shap(kernel_values_raw_{prefix})
            kernel_rank_{prefix} = pd.DataFrame({{
                'feature': {feature_var},
                'kernel_mean_abs_shap': np.abs(kernel_sv_{prefix}).mean(axis=0),
            }}).sort_values('kernel_mean_abs_shap', ascending=False).reset_index(drop=True)
            fidelity_top_{prefix} = kernel_rank_{prefix}.head(5).copy()
            rf_top_{prefix} = shap_rank_{prefix}.head(5)[['feature', 'mean_abs_shap']].copy()
            overlap_{prefix} = sorted(set(fidelity_top_{prefix}['feature']) & set(rf_top_{prefix}['feature']))
            print('Kernel SHAP top five on the small stack subset:')
            print(fidelity_top_{prefix}.to_string(index=False))
            print()
            print('RF TreeSHAP top five on the full test set:')
            print(rf_top_{prefix}.to_string(index=False))
            print()
            print('Top-five overlap:')
            print(overlap_{prefix})
            """
        ),
        markdown_cell(
            """
            ### Full-Stack Fidelity Check

            This check is not replacing the main TreeSHAP analysis. It is only checking whether the top features from the fast Random Forest explanation broadly agree with the full final stack on a small subset. If the overlap is reasonable, the main SHAP story is easier to defend.
            """
        ),
        code_cell(
            f"""
            # Writing the clinical summary is turning the main SHAP findings into a short explanation that can be used in the thesis discussion.
            top_features_{prefix} = ', '.join([f"`{{name}}`" for name in shap_rank_{prefix}.head(5)['feature'].tolist()])
            if '{prefix}' == 'm2':
                marker_hits = [name for name in ['amh_ng_ml', 'fsh_lh_ratio', 'follicle_no_left', 'follicle_no_right'] if name in shap_rank_{prefix}.head(5)['feature'].tolist()]
                marker_text = ', '.join([f"`{{name}}`" for name in marker_hits]) if marker_hits else 'none of the expected marker group'
                extra_text = f" Literature-linked markers appearing in the top five are: {{marker_text}}."
            else:
                extra_text = ''
            render_markdown(
                '### {model_name} Clinical Summary\\n'
                f'**What the top features are:** {{top_features_{prefix}}}.\\n'
                f'**What the local cases show:** The selected true positive, false negative, and false positive cases are showing how the model gets a case right, misses a case, or raises a false alarm.\\n'
                f'**What the interaction shows:** The interaction plot is showing whether two clinically related features are making each other stronger or weaker.\\n'
                f'**Why this matters:** This gives a feature-level explanation of how {model_name.lower()} is making decisions.{{extra_text}}\\n'
                f'**Simple summary:** The SHAP results are showing the main features that the final {model_name.lower()} is relying on.'
            )
            """
        ),
    ])
    return cells


def build_cells() -> list[dict]:
    cells: list[dict] = []
    cells.append(markdown_cell(
        """
        # SHAP Explainability and External Validation — Final Optimized Models

        ## Introduction
        This notebook is doing four related jobs. First, it is teaching SHAP clearly enough that a reader can understand what SHAP is, why it is used, and what kind of answers it gives. Second, it is explaining the final optimized non-invasive model. Third, it is explaining the final optimized invasive model. Fourth, it is testing the final non-invasive model on the survey external validation set.

        Explainability is not optional for a clinical AI tool. A model that predicts risk without showing why it reached that result is hard to trust in practice. SHAP is giving feature-level explanations so clinicians can see which features are pushing a prediction upward or downward.

        The Model 1 SHAP section is revealing what the final non-invasive model is doing with routine screening features such as BMI, symptom burden, cycle features, blood pressure, and lifestyle signals. This is the most clinically deployable part of the study because these are the features that can be collected most easily in real screening work.

        The Model 2 SHAP section is revealing what the final invasive benchmark model is doing with hormonal and ultrasound features. This section is helping us check whether the model is learning the same biomarker patterns that the literature already describes in PCOS.

        The two SHAP sections are being treated as separate analyses. The models use different feature sets, so comparing the SHAP value of a non-invasive feature against the SHAP value of an invasive biomarker would not be a meaningful clinical comparison.

        The survey validation section is testing whether the final Model 1 pipeline can still produce useful results outside the clinical training setting. Because some survey features were placeholder-filled in Notebook 07, this is an external-style check rather than a perfect external clinical validation.

        ## Output Files
        """
        + SHAP_OUTPUT_TABLE
    ))

    cells.append(markdown_cell("## Section 1 — Imports and Configuration"))

    cells.append(code_cell(
        """
        # Importing the packages for SHAP, plotting, saved model loading, and validation is preparing the notebook for explainability work.
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        import os
        import warnings
        import json
        import joblib
        from pathlib import Path
        from IPython.display import display, Markdown

        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is required for Notebook 11. Install it with `%pip install shap` "
                "or `conda install -c conda-forge shap`, then restart the kernel."
            ) from exc

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
        )

        warnings.filterwarnings('ignore')
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        shap.initjs()

        RANDOM_STATE = 42
        TARGET = 'pcos_y_n'

        COLORS = {
            'pcos_pos': '#E63946',
            'pcos_neg': '#457B9D',
            'model1': '#2A9D8F',
            'model2': '#E9C46A',
            'shap_pos': '#E63946',
            'shap_neg': '#457B9D',
            'neutral': '#2C3E50',
            'survey': '#9B2226',
        }
        """
    ))

    cells.append(code_cell(
        """
        # Resolving the project root is making the notebook work from either the repo root or the notebooks folder.
        search_start = Path.cwd().resolve()
        PROJECT_ROOT = None
        for candidate in [search_start, *search_start.parents]:
            if (
                (candidate / 'cleaned_data').exists() and
                (candidate / 'models').exists() and
                (candidate / 'notebooks').exists() and
                (candidate / 'README.md').exists()
            ):
                PROJECT_ROOT = candidate
                break
        if PROJECT_ROOT is None:
            raise FileNotFoundError('Could not resolve the project root.')
        os.chdir(PROJECT_ROOT)
        print(f'Project root resolved to: {PROJECT_ROOT}')

        # Creating the SHAP output folders is making sure all saved figures have fixed locations.
        os.makedirs(PROJECT_ROOT / 'images' / 'shap' / 'model1', exist_ok=True)
        os.makedirs(PROJECT_ROOT / 'images' / 'shap' / 'model2', exist_ok=True)
        os.makedirs(PROJECT_ROOT / 'images' / 'shap' / 'survey', exist_ok=True)
        print('Directory ready: images/shap/model1')
        print('Directory ready: images/shap/model2')
        print('Directory ready: images/shap/survey')
        """
    ))

    cells.append(code_cell(
        """
        # Loading the locked modelling data and saved final model artifacts is grounding the notebook in the current repo state.
        train1 = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'model1_train.csv')
        test1 = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'model1_test.csv')
        train2 = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'model2_train.csv')
        test2 = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'model2_test.csv')
        survey = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'survey_external_validation.csv')

        with open(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'feature_names_model1.json', encoding='utf-8') as f:
            MODEL1_FEATURES = json.load(f)
        with open(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'feature_names_model2.json', encoding='utf-8') as f:
            MODEL2_FEATURES = json.load(f)

        with open(PROJECT_ROOT / 'models' / 'model1_best_params.json', encoding='utf-8') as f:
            m1_best = json.load(f)
        with open(PROJECT_ROOT / 'models' / 'model2_best_params.json', encoding='utf-8') as f:
            m2_best = json.load(f)

        final_bundle_m1 = joblib.load(PROJECT_ROOT / 'models' / 'model1_final_optimized.pkl')
        final_bundle_m2 = joblib.load(PROJECT_ROOT / 'models' / 'model2_final_optimized.pkl')
        final_results = pd.read_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'final_model_results.csv')

        print('Loaded data shapes:')
        print(f'train1: {train1.shape}')
        print(f'test1 : {test1.shape}')
        print(f'train2: {train2.shape}')
        print(f'test2 : {test2.shape}')
        print(f'survey: {survey.shape}')
        print()
        print('Model 1 best params JSON:')
        print(json.dumps(m1_best, indent=2))
        print()
        print('Model 2 best params JSON:')
        print(json.dumps(m2_best, indent=2))
        print()
        print('Final results table:')
        print(final_results.to_string(index=False))
        """
    ))

    cells.append(code_cell(
        """
        # Separating features and targets is preparing the train, test, and survey arrays used in the explainability and validation sections.
        X1_train = train1[MODEL1_FEATURES].values
        y1_train = train1[TARGET].values
        X1_test = test1[MODEL1_FEATURES].values
        y1_test = test1[TARGET].values

        X2_train = train2[MODEL2_FEATURES].values
        y2_train = train2[TARGET].values
        X2_test = test2[MODEL2_FEATURES].values
        y2_test = test2[TARGET].values

        X_survey = survey[MODEL1_FEATURES].values
        y_survey = survey[TARGET].values

        print(f'X1_train shape: {X1_train.shape}')
        print(f'X1_test shape : {X1_test.shape}')
        print(f'X2_train shape: {X2_train.shape}')
        print(f'X2_test shape : {X2_test.shape}')
        print(f'X_survey shape: {X_survey.shape}')
        print()
        print('Model 1 feature count:', len(MODEL1_FEATURES))
        print('Model 2 feature count:', len(MODEL2_FEATURES))
        """
    ))

    cells.append(code_cell(
        """
        # Defining shared helper functions is keeping the SHAP and validation sections readable and consistent.
        def render_markdown(text: str) -> None:
            display(Markdown(text))


        def evaluate_prediction(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
            return {
                'model': name,
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'auc': float(roc_auc_score(y_true, y_proba)),
            }


        def stack_predict_proba(bundle: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            base_order = bundle.get('meta_feature_order', ['LR_prob', 'RF_prob', 'XGB_prob'])
            key_map = {'LR_prob': 'lr', 'RF_prob': 'rf', 'XGB_prob': 'xgb'}
            meta_features = np.column_stack([
                bundle['base_models'][key_map[col]].predict_proba(X)[:, 1]
                for col in base_order
            ])
            proba = bundle['meta_model'].predict_proba(meta_features)[:, 1]
            return meta_features, proba


        def extract_positive_shap(shap_values):
            if isinstance(shap_values, list):
                return np.asarray(shap_values[1])
            arr = np.asarray(shap_values)
            if arr.ndim == 3 and arr.shape[-1] == 2:
                return arr[:, :, 1]
            return arr


        def extract_positive_expected_value(expected_value):
            if isinstance(expected_value, list):
                return float(expected_value[1])
            arr = np.asarray(expected_value)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 2:
                return float(arr.reshape(-1)[1])
            return float(arr.reshape(-1)[0])


        def select_case_indices(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
            tp = np.where((y_true == 1) & (y_pred == 1))[0]
            fn = np.where((y_true == 1) & (y_pred == 0))[0]
            fp = np.where((y_true == 0) & (y_pred == 1))[0]
            positives = np.where(y_true == 1)[0]
            negatives = np.where(y_true == 0)[0]
            tp_idx = int(tp[np.argmax(y_proba[tp])]) if len(tp) else int(positives[np.argmax(y_proba[positives])])
            fn_idx = int(fn[np.argmin(y_proba[fn])]) if len(fn) else int(positives[np.argmin(y_proba[positives])])
            fp_idx = int(fp[np.argmax(y_proba[fp])]) if len(fp) else int(negatives[np.argmax(y_proba[negatives])])
            return {
                'tp_idx': tp_idx,
                'fn_idx': fn_idx,
                'fp_idx': fp_idx,
                'tp_count': int(len(tp)),
                'fn_count': int(len(fn)),
                'fp_count': int(len(fp)),
            }


        def top_signed_contributors(shap_row: np.ndarray, feature_names: list[str], value_row: pd.Series, positive: bool = True, top_n: int = 5) -> list[dict]:
            frame = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_row,
                'feature_value': [value_row[name] for name in feature_names],
            })
            frame = frame.sort_values('shap_value', ascending=not positive)
            if positive:
                frame = frame[frame['shap_value'] > 0]
            else:
                frame = frame[frame['shap_value'] < 0]
            return frame.head(top_n).to_dict('records')


        def describe_dependence_shape(feature_values: np.ndarray, shap_values: np.ndarray) -> tuple[str, float]:
            if np.std(feature_values) == 0 or np.std(shap_values) == 0:
                return 'flat or unstable', 0.0
            rho = float(np.corrcoef(feature_values, shap_values)[0, 1])
            if np.isnan(rho):
                return 'mixed or irregular', 0.0
            abs_rho = abs(rho)
            if abs_rho >= 0.75:
                return 'strongly monotonic', rho
            if abs_rho >= 0.35:
                return 'moderately monotonic', rho
            return 'non-linear or mixed', rho


        def stratified_take(y: np.ndarray, total: int, seed: int = 42):
            rng = np.random.default_rng(seed)
            y = np.asarray(y)
            indices = []
            for cls in sorted(np.unique(y)):
                cls_idx = np.where(y == cls)[0]
                n_take = max(1, int(round(total * (len(cls_idx) / len(y)))))
                n_take = min(n_take, len(cls_idx))
                picked = rng.choice(cls_idx, size=n_take, replace=False)
                indices.extend(picked.tolist())
            indices = sorted(set(indices))
            if len(indices) > total:
                indices = list(rng.choice(indices, size=total, replace=False))
            while len(indices) < total:
                remaining = [idx for idx in range(len(y)) if idx not in indices]
                if not remaining:
                    break
                indices.append(int(rng.choice(remaining)))
            return np.array(sorted(indices), dtype=int)
        """
    ))

    cells.append(markdown_cell(
        """
        ## Section 2 — Understanding SHAP

        ### What Is SHAP?

        #### 2.1 The Explainability Problem
        Modern ensemble models can be accurate while still being hard to interpret. When a model predicts that a patient has high PCOS risk, a clinician needs to know why. Which features were pushing the prediction upward? Which features were pulling it downward? Without this explanation, the model is difficult to trust in real clinical use.

        #### 2.2 The Game Theory Foundation
        SHAP (SHapley Additive exPlanations) is based on cooperative game theory. In a game, several players work together to produce an outcome, and the Shapley value tells us how much credit each player deserves. In SHAP, the players are features and the outcome is the model prediction.

        The Shapley value for feature $j$ is:

        $$
        \\phi_j = \\sum_{S \\subseteq F \\setminus \\{j\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} \\left[f(S \\cup \\{j\\}) - f(S)\\right]
        $$

        where $F$ is the full set of features and $S$ is any subset that does not include feature $j$. This means SHAP is averaging the marginal contribution of feature $j$ across all possible subsets of the other features.

        In simple words, SHAP is asking: if we add this feature into many different possible feature combinations, how much does it change the prediction on average?

        #### 2.3 The Additive Property
        SHAP explanations are additive:

        $$
        f(x) = \\phi_0 + \\sum_{j=1}^{M} \\phi_j
        $$

        where $f(x)$ is the model output for one patient, $\\phi_0$ is the base value, and each $\\phi_j$ is the contribution of one feature.

        In simple words, SHAP is breaking every prediction into a starting point plus feature-by-feature pushes up or down.

        #### 2.4 SHAP for Tree-Based Models
        For tree-based models such as Random Forest and XGBoost, TreeSHAP computes exact Shapley values much faster than a brute-force coalition calculation. This is why the main explanation layer in this notebook is using the saved Random Forest base learner inside each final optimized stack.

        #### 2.5 Two Levels of Explanation
        SHAP gives two kinds of insight:
        1. **Global explanations** show which features matter most across the whole population.
        2. **Local explanations** show why one specific patient received one specific prediction.

        This notebook is using both.
        """
    ))

    cells.append(code_cell(
        """
        # Creating the main SHAP concept diagram is introducing the difference between local and global explanations.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        local_features = ['BMI', 'Cycle', 'BP', 'Exercise', 'RBS']
        local_values = [0.24, 0.18, 0.12, -0.08, -0.05]
        local_colors = [COLORS['shap_pos'] if v > 0 else COLORS['shap_neg'] for v in local_values]
        axes[0].barh(local_features, local_values, color=local_colors, edgecolor='black', linewidth=0.5)
        axes[0].axvline(0, linestyle='--', color='black', linewidth=1)
        axes[0].text(0.01, -0.6, 'Base value (mean prediction)', fontsize=9)
        axes[0].annotate('Final prediction', xy=(0.30, 4.2), xytext=(0.18, 4.7), arrowprops=dict(arrowstyle='->', color='black'))
        axes[0].set_title('Local SHAP Explanation — One Patient')
        axes[0].set_xlabel('Feature contribution')

        global_features = ['BMI', 'Cycle', 'Skin', 'Hair', 'BP']
        global_importance = [0.18, 0.15, 0.11, 0.09, 0.06]
        axes[1].barh(global_features[::-1], global_importance[::-1], color=COLORS['model1'], edgecolor='black', linewidth=0.5)
        axes[1].set_title('Global SHAP — Feature Importance Ranking')
        axes[1].set_xlabel('Mean |SHAP value|')

        plt.suptitle('SHAP Explanation Framework — Local and Global Views', y=1.02, fontsize=13)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '01_shap_concept_diagram.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(code_cell(
        """
        # Creating the game-theory illustration is showing how feature coalitions build the SHAP idea.
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis('off')

        circles = [('BMI', (0.15, 0.78)), ('Cycle', (0.30, 0.78)), ('BP', (0.45, 0.78))]
        for label, (x, y) in circles:
            circ = plt.Circle((x, y), 0.055, color=COLORS['model1'], alpha=0.85)
            ax.add_patch(circ)
            ax.text(x, y, label, color='white', ha='center', va='center', fontsize=10, fontweight='bold')

        coalition_boxes = [
            ('{BMI}', 0.10, 0.48, '0.38'),
            ('{Cycle}', 0.25, 0.48, '0.41'),
            ('{BP}', 0.40, 0.48, '0.35'),
            ('{BMI+Cycle}', 0.10, 0.28, '0.62'),
            ('{BMI+BP}', 0.28, 0.28, '0.53'),
            ('{Cycle+BP}', 0.46, 0.28, '0.50'),
            ('{BMI+Cycle+BP}', 0.28, 0.08, '0.74'),
        ]
        for label, x, y, score in coalition_boxes:
            rect = plt.Rectangle((x, y), 0.17, 0.10, fill=True, color='#F1FAEE', ec='#1D3557', lw=1.0)
            ax.add_patch(rect)
            ax.text(x + 0.085, y + 0.06, label, ha='center', va='center', fontsize=9)
            ax.text(x + 0.085, y + 0.025, f'Pred = {score}', ha='center', va='center', fontsize=8, color=COLORS['neutral'])

        ax.text(0.70, 0.36, 'SHAP averages the marginal contribution\\nacross all coalition orderings.', fontsize=11, color=COLORS['neutral'])
        ax.set_title('SHAP Game Theory Illustration — Feature Coalitions', fontsize=13)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / '02_shap_game_theory_illustration.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.extend(model_section(
        prefix="m1",
        model_name="Model 1",
        model_num="3",
        feature_var="MODEL1_FEATURES",
        train_var="X1_train",
        ytrain_var="y1_train",
        test_var="X1_test",
        ytest_var="y1_test",
        bundle_var="final_bundle_m1",
        best_var="m1_best",
        shap_file="shap_values_model1.npy",
        rank_file="shap_feature_ranking_model1.csv",
        folder="model1",
        model_color="model1",
        interaction_feature_a="bmi",
        interaction_feature_b="cycle_regularity_binary",
        interaction_file="12_m1_shap_interaction_bmi_cycle.png",
        interaction_title="Model 1 — BMI SHAP Interaction with Cycle Regularity",
        summary_note=M1_TOP_TEXT,
    ))

    cells.extend(model_section(
        prefix="m2",
        model_name="Model 2",
        model_num="4",
        feature_var="MODEL2_FEATURES",
        train_var="X2_train",
        ytrain_var="y2_train",
        test_var="X2_test",
        ytest_var="y2_test",
        bundle_var="final_bundle_m2",
        best_var="m2_best",
        shap_file="shap_values_model2.npy",
        rank_file="shap_feature_ranking_model2.csv",
        folder="model2",
        model_color="model2",
        interaction_feature_a="amh_ng_ml",
        interaction_feature_b="fsh_lh_ratio",
        interaction_file="23_m2_shap_interaction_amh_fsh.png",
        interaction_title="Model 2 — AMH SHAP Interaction with FSH/LH Ratio",
        summary_note=M2_TOP_TEXT,
    ))

    cells.append(markdown_cell(
        """
        ## Section 5 — Survey External Validation

        This section is testing how the final non-invasive optimized model behaves on a different data collection setting. The survey dataset was not used for training. It is self-reported, and some Model 1 features were missing there and had to be placeholder-filled in Notebook 07. This means the validation is useful, but it is still approximate and should be read carefully.
        """
    ))

    cells.append(code_cell(
        """
        # Printing the survey dataset profile is showing what kind of external-style validation set is being used.
        print(f'Survey validation set shape : {survey.shape}')
        print(f'Feature count               : {len(MODEL1_FEATURES)}')
        print()
        print('Class distribution in survey validation set:')
        print(survey[TARGET].value_counts().to_string())
        print()
        print(survey[TARGET].value_counts(normalize=True).mul(100).round(2).to_string())
        print()
        print('Feature alignment status from Notebook 07:')
        print('Direct survey features  : age_yrs, bmi, weight_gain_y_n, hair_growth_y_n,')
        print('                          skin_darkening_y_n, hair_loss_y_n, pimples_y_n,')
        print('                          fast_food_y_n, regular_exercise_y_n,')
        print('                          cycle_regularity_binary, cycle_length_days')
        print('Imputed with clinical')
        print('training medians        : systolic_bp_mmhg, diastolic_bp_mmhg,')
        print('                          waist_hip_ratio, rbs_mg_dl')
        print('Derived features        : symptom_burden, bmi_category, bp_elevated_flag, lifestyle_risk')
        """
    ))

    cells.append(code_cell(
        """
        # Printing the class-distribution comparison table is preparing the data behind the survey balance plot.
        survey_balance_df = pd.DataFrame({
            'dataset': ['Survey', 'Survey', 'Clinical Test', 'Clinical Test'],
            'label': ['No PCOS', 'PCOS', 'No PCOS', 'PCOS'],
            'count': [
                int((y_survey == 0).sum()),
                int((y_survey == 1).sum()),
                int((y1_test == 0).sum()),
                int((y1_test == 1).sum()),
            ]
        })
        print(survey_balance_df.to_string(index=False))
        """
    ))

    cells.append(code_cell(
        """
        # Plotting the survey class-distribution comparison is showing how the survey set differs from the clinical test set.
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        sns.barplot(data=survey_balance_df, x='dataset', y='count', hue='label', palette=[COLORS['pcos_neg'], COLORS['pcos_pos']], ax=ax)
        ax.set_title('Class Distribution — Survey Validation vs Clinical Test Set')
        ax.set_ylabel('Patient count')
        for patch in ax.patches:
            ax.annotate(f"{int(patch.get_height())}", (patch.get_x() + patch.get_width()/2, patch.get_height()), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / 'survey' / '25_survey_class_distribution.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(markdown_cell(
        """
        ### Survey Class Balance Insight

        This chart is showing whether the survey label mix looks similar to the clinical test set or more imbalanced. This matters because a very different class mix can change how easy or hard it is for the model to hold its performance outside the clinical setting.
        """
    ))

    cells.append(code_cell(
        """
        # Running the final non-invasive optimized stack on the survey data is testing external-style generalisation.
        survey_meta_features, y_survey_proba = stack_predict_proba(final_bundle_m1, X_survey)
        survey_threshold = float(final_bundle_m1.get('threshold', 0.5))
        y_survey_pred = (y_survey_proba >= survey_threshold).astype(int)
        survey_results = evaluate_prediction('Survey External Validation', y_survey, y_survey_pred, y_survey_proba)
        print('Survey external validation results:')
        for key, value in survey_results.items():
            if key == 'model':
                continue
            print(f'  {key:10s}: {value:.4f}')
        """
    ))

    cells.append(code_cell(
        """
        # Saving the survey validation outputs is preserving the external-style metrics and row-level predictions for later review.
        pd.DataFrame([{
            'dataset': 'Survey External Validation',
            'n_patients': len(y_survey),
            **{k: v for k, v in survey_results.items() if k != 'model'}
        }]).to_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'survey_validation_results.csv', index=False)

        survey_prediction_df = pd.DataFrame(X_survey, columns=MODEL1_FEATURES)
        survey_prediction_df['true_label'] = y_survey
        survey_prediction_df['predicted_label'] = y_survey_pred
        survey_prediction_df['predicted_proba'] = y_survey_proba
        survey_prediction_df.to_csv(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'survey_validation_predictions.csv', index=False)
        print('survey_validation_results.csv saved.')
        print('survey_validation_predictions.csv saved.')
        """
    ))

    cells.append(code_cell(
        """
        # Printing the ROC comparison table is preparing the exact values used in the survey ROC plot.
        roc_compare_df = pd.DataFrame([
            {'dataset': 'Clinical Test', 'auc': roc_auc_score(y1_test, y_m1_proba)},
            {'dataset': 'Survey Validation', 'auc': roc_auc_score(y_survey, y_survey_proba)},
        ])
        print(roc_compare_df.round(4).to_string(index=False))
        """
    ))

    cells.append(code_cell(
        """
        # Plotting the survey ROC curve against the clinical test ROC is showing how much discrimination holds outside the training context.
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        RocCurveDisplay.from_predictions(y1_test, y_m1_proba, ax=ax, name=f'Clinical Test (AUC = {roc_auc_score(y1_test, y_m1_proba):.3f})', color=COLORS['model1'])
        RocCurveDisplay.from_predictions(y_survey, y_survey_proba, ax=ax, name=f'Survey Validation (AUC = {roc_auc_score(y_survey, y_survey_proba):.3f})', color=COLORS['survey'])
        ax.plot([0, 1], [0, 1], linestyle='--', color='#999999', linewidth=1)
        ax.set_title('External Validation — Survey ROC Curve vs Clinical Test Set')
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / 'survey' / '26_survey_roc_curve.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(markdown_cell(
        """
        ### Survey ROC Insight

        This ROC chart is showing how well the final non-invasive model separates positive and negative cases on survey data compared with the clinical test data. If the survey curve stays reasonably close to the clinical curve, that suggests the learned pattern is travelling outside the original clinical setting.
        """
    ))

    cells.append(code_cell(
        """
        # Printing the survey confusion table is showing the exact classification outcomes before plotting.
        cm_survey = confusion_matrix(y_survey, y_survey_pred)
        cm_survey_df = pd.DataFrame(
            cm_survey,
            index=['Actual: No PCOS', 'Actual: PCOS'],
            columns=['Predicted: No PCOS', 'Predicted: PCOS'],
        )
        print(cm_survey_df.to_string())
        print()
        print(f"True Positives  : {cm_survey[1, 1]}")
        print(f"False Negatives : {cm_survey[1, 0]}")
        print(f"False Positives : {cm_survey[0, 1]}")
        print(f"True Negatives  : {cm_survey[0, 0]}")
        """
    ))

    cells.append(code_cell(
        """
        # Plotting the survey confusion matrix is showing what kinds of survey cases were caught and missed.
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        sns.heatmap(cm_survey_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title('External Validation — Survey Confusion Matrix')
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / 'survey' / '27_survey_confusion_matrix.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(markdown_cell(
        """
        ### Survey Confusion Matrix Insight

        This confusion matrix is showing how many survey cases were correctly flagged, missed, or wrongly flagged. The false negatives are especially important because they show the survey cases the model is not catching.
        """
    ))

    cells.append(code_cell(
        """
        # Computing survey SHAP values with the saved Model 1 RF explainer is showing what the survey predictions are leaning on.
        shap_values_survey_raw = explainer_m1.shap_values(X_survey)
        sv_survey = extract_positive_shap(shap_values_survey_raw)
        np.save(PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_values_survey.npy', sv_survey)
        print('Survey SHAP matrix shape:')
        print(sv_survey.shape)
        """
    ))

    cells.append(code_cell(
        """
        # Printing the survey SHAP ranking table is preparing the exact data behind the survey SHAP plots.
        mean_abs_shap_survey = np.abs(sv_survey).mean(axis=0)
        shap_rank_survey = pd.DataFrame({
            'feature': MODEL1_FEATURES,
            'mean_abs_shap': mean_abs_shap_survey,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        print('Survey SHAP ranking:')
        print(shap_rank_survey.to_string(index=False))
        print()
        print('Clinical Model 1 SHAP ranking:')
        print(shap_rank_m1[['feature', 'mean_abs_shap']].to_string(index=False))
        """
    ))

    cells.append(code_cell(
        """
        # Plotting the survey SHAP global bar chart is showing which features matter most in the external-style survey setting.
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plot_survey_rank = shap_rank_survey.sort_values('mean_abs_shap', ascending=True)
        bars = ax.barh(plot_survey_rank['feature'], plot_survey_rank['mean_abs_shap'], color=COLORS['survey'], alpha=0.85, edgecolor='black', linewidth=0.4)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Survey External Validation — Global SHAP Feature Importance')
        for bar, value in zip(bars, plot_survey_rank['mean_abs_shap']):
            ax.text(value + 0.002, bar.get_y() + bar.get_height()/2, f'{value:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / 'survey' / '28_survey_shap_bar_global.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(code_cell(
        """
        # Plotting the survey SHAP beeswarm is showing the direction and spread of the survey explanations patient by patient.
        survey_df_features = pd.DataFrame(X_survey, columns=MODEL1_FEATURES)
        shap.summary_plot(sv_survey, survey_df_features, feature_names=MODEL1_FEATURES, show=False, plot_size=(9.2, 6.2))
        plt.title('Survey External Validation — SHAP Beeswarm')
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'images' / 'shap' / 'survey' / '29_survey_shap_beeswarm.png', bbox_inches='tight')
        plt.show()
        """
    ))

    cells.append(markdown_cell(
        """
        ### Survey SHAP Insight

        These survey SHAP plots are showing whether the same non-invasive features are still dominating outside the clinical test setting. If the important features stay similar, that supports the idea that the model has learned a real non-invasive PCOS pattern rather than something highly specific to the training cohort.
        """
    ))

    cells.append(code_cell(
        """
        # Writing the survey interpretation is summarizing how the external-style validation compared with the clinical test results.
        clinical_m1_test = results_m1_final
        top_overlap_survey = sorted(set(shap_rank_survey.head(5)['feature']) & set(shap_rank_m1.head(5)['feature']))
        overlap_text = ', '.join([f'`{name}`' for name in top_overlap_survey]) if top_overlap_survey else 'limited on this run'
        survey_markdown = (
            '### Survey Validation Interpretation\\n'
            f'**What the result is showing:** The survey validation AUC is `{survey_results["auc"]:.3f}` and the recall is `{survey_results["recall"]:.3f}`, compared with the clinical test AUC of `{clinical_m1_test["auc"]:.3f}` and recall of `{clinical_m1_test["recall"]:.3f}`.\\n\\n'
            '**What this means:** This shows whether the non-invasive signal learned from the clinical cohort is still visible in the self-reported survey setting.\\n\\n'
            '**Why this matters:** The survey set is more like a real-world screening situation, but it is also noisier and partly placeholder-filled.\\n\\n'
            f'**Simple summary:** The survey validation gives an external-style check, not a perfect external clinical validation. The top-five SHAP overlap is {overlap_text}.'
        )
        render_markdown(survey_markdown)
        """
    ))

    cells.append(markdown_cell(
        """
        ## Section 6 — Final Verification
        """
    ))

    cells.append(code_cell(
        """
        # Checking every required output file is confirming that the full Notebook 11 output contract was completed.
        required_files = [
            PROJECT_ROOT / 'images' / 'shap' / '01_shap_concept_diagram.png',
            PROJECT_ROOT / 'images' / 'shap' / '02_shap_game_theory_illustration.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '03_m1_shap_bar_global.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '04_m1_shap_beeswarm.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '05_m1_shap_heatmap.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '06_m1_shap_waterfall_true_positive.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '07_m1_shap_waterfall_false_negative.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '08_m1_shap_waterfall_false_positive.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '09_m1_shap_dependence_feature1.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '10_m1_shap_dependence_feature2.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '11_m1_shap_dependence_feature3.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '12_m1_shap_interaction_bmi_cycle.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model1' / '13_m1_meta_base_learner_weights.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '14_m2_shap_bar_global.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '15_m2_shap_beeswarm.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '16_m2_shap_heatmap.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '17_m2_shap_waterfall_true_positive.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '18_m2_shap_waterfall_false_negative.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '19_m2_shap_waterfall_false_positive.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '20_m2_shap_dependence_feature1.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '21_m2_shap_dependence_feature2.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '22_m2_shap_dependence_feature3.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '23_m2_shap_interaction_amh_fsh.png',
            PROJECT_ROOT / 'images' / 'shap' / 'model2' / '24_m2_meta_base_learner_weights.png',
            PROJECT_ROOT / 'images' / 'shap' / 'survey' / '25_survey_class_distribution.png',
            PROJECT_ROOT / 'images' / 'shap' / 'survey' / '26_survey_roc_curve.png',
            PROJECT_ROOT / 'images' / 'shap' / 'survey' / '27_survey_confusion_matrix.png',
            PROJECT_ROOT / 'images' / 'shap' / 'survey' / '28_survey_shap_bar_global.png',
            PROJECT_ROOT / 'images' / 'shap' / 'survey' / '29_survey_shap_beeswarm.png',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_values_model1.npy',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_values_model2.npy',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_values_survey.npy',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_feature_ranking_model1.csv',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'shap_feature_ranking_model2.csv',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'survey_validation_results.csv',
            PROJECT_ROOT / 'cleaned_data' / 'modelling_sets' / 'survey_validation_predictions.csv',
        ]

        all_passed = True
        for path in required_files:
            exists = path.exists()
            non_empty = exists and path.stat().st_size > 0
            status = 'PASS' if (exists and non_empty) else 'FAIL'
            if status == 'FAIL':
                all_passed = False
            print(f'[{status}] {path}')

        print()
        if all_passed:
            print('All Notebook 11 outputs exist and are non-empty.')
        else:
            print('Some Notebook 11 outputs are missing or empty.')
        """
    ))

    cells.append(markdown_cell(
        """
        ## Notebook Summary and Next Steps

        This notebook is teaching SHAP, explaining the final optimized non-invasive model, explaining the final optimized invasive model, and testing the non-invasive model on the survey external validation set. The Model 1 SHAP results are showing which routine features are driving the final screening model. The Model 2 SHAP results are showing which hormonal and ultrasound features are driving the invasive benchmark. The survey section is showing how well the non-invasive model travels outside the clinical test setting.

        Notebook 12 will be building the final Streamlit application. It will expose both optimized models, the SHAP explanation views, patient input forms, model performance pages, and the study-level insights in a single interactive interface.
        """
    ))

    return cells


def main() -> None:
    write_notebook(NOTEBOOK_PATH, build_cells())
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
