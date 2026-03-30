from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_individual_models.ipynb"


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
    ("`models/model1_lr.pkl`", "Saved Logistic Regression model for Model 1."),
    ("`models/model1_rf.pkl`", "Saved Random Forest model for Model 1."),
    ("`models/model1_xgb.pkl`", "Saved XGBoost model for Model 1."),
    ("`models/model2_lr.pkl`", "Saved Logistic Regression model for Model 2."),
    ("`models/model2_rf.pkl`", "Saved Random Forest model for Model 2."),
    ("`models/model2_xgb.pkl`", "Saved XGBoost model for Model 2."),
    ("`cleaned_data/modelling_sets/individual_model_results.csv`", "Master results table for all six individual models."),
    ("`images/individual_models/01-14_*.png`", "Confusion matrices, ROC plots, CV plots, and master comparison figures."),
]


MODEL1_ALGOS = [
    {
        "label": "Logistic Regression",
        "short": "LR",
        "var": "lr1",
        "cv_var": "cv_results_m1_lr",
        "result_var": "results_m1_lr",
        "pred_var": "y1_pred_lr",
        "proba_var": "y1_proba_lr",
        "cm_var": "cm_m1_lr",
        "cm_df_var": "cm_m1_lr_df",
        "save_name": "model1_lr.pkl",
        "cm_png": "01_model1_lr_confusion_matrix.png",
        "markdown": (
            "Logistic Regression is serving as the interpretable linear baseline for Model 1. "
            "It is assuming a linear decision boundary between PCOS-negative and PCOS-positive cases in the non-invasive feature space. "
            "If this model performs well, it suggests the routine screening features already separate the classes in a simple way. "
            "If it underperforms the tree-based models, that suggests non-linear feature interactions matter."
        ),
        "definition": """
            lr1 = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='lbfgs'
            )
        """,
    },
    {
        "label": "Random Forest",
        "short": "RF",
        "var": "rf1",
        "cv_var": "cv_results_m1_rf",
        "result_var": "results_m1_rf",
        "pred_var": "y1_pred_rf",
        "proba_var": "y1_proba_rf",
        "cm_var": "cm_m1_rf",
        "cm_df_var": "cm_m1_rf_df",
        "save_name": "model1_rf.pkl",
        "cm_png": "02_model1_rf_confusion_matrix.png",
        "markdown": (
            "Random Forest is serving as the main tree-based ensemble baseline for Model 1. "
            "It is capturing non-linear interactions between symptoms, blood pressure, glucose, and body measures. "
            "This matters because PCOS screening patterns are often built from combinations of features rather than one linear shift. "
            "It is also producing native feature importance values that can later be compared with SHAP."
        ),
        "definition": """
            rf1 = RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )
        """,
    },
    {
        "label": "XGBoost",
        "short": "XGB",
        "var": "xgb1",
        "cv_var": "cv_results_m1_xgb",
        "result_var": "results_m1_xgb",
        "pred_var": "y1_pred_xgb",
        "proba_var": "y1_proba_xgb",
        "cm_var": "cm_m1_xgb",
        "cm_df_var": "cm_m1_xgb_df",
        "save_name": "model1_xgb.pkl",
        "cm_png": "03_model1_xgb_confusion_matrix.png",
        "markdown": (
            "XGBoost is serving as the strongest expected individual learner for Model 1. "
            "It is building trees one after another, with each new tree correcting earlier errors. "
            "This is often powerful for tabular clinical data, especially when feature interactions are uneven or non-linear. "
            "The scale_pos_weight value is still being printed, even though the Notebook 07 training set was already balanced with SMOTE."
        ),
        "definition": """
            xgb1 = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw1,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            )
        """,
    },
]


MODEL2_ALGOS = [
    {
        "label": "Logistic Regression",
        "short": "LR",
        "var": "lr2",
        "cv_var": "cv_results_m2_lr",
        "result_var": "results_m2_lr",
        "pred_var": "y2_pred_lr",
        "proba_var": "y2_proba_lr",
        "cm_var": "cm_m2_lr",
        "cm_df_var": "cm_m2_lr_df",
        "save_name": "model2_lr.pkl",
        "cm_png": "04_model2_lr_confusion_matrix.png",
        "markdown": (
            "Logistic Regression is serving as the interpretable linear baseline for the invasive benchmark set. "
            "It is testing whether the hormonal and ultrasound features separate PCOS status in a simple linear way. "
            "If it performs strongly here, it suggests invasive markers may already carry clean directional signal."
        ),
        "definition": """
            lr2 = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='lbfgs'
            )
        """,
    },
    {
        "label": "Random Forest",
        "short": "RF",
        "var": "rf2",
        "cv_var": "cv_results_m2_rf",
        "result_var": "results_m2_rf",
        "pred_var": "y2_pred_rf",
        "proba_var": "y2_proba_rf",
        "cm_var": "cm_m2_rf",
        "cm_df_var": "cm_m2_rf_df",
        "save_name": "model2_rf.pkl",
        "cm_png": "05_model2_rf_confusion_matrix.png",
        "markdown": (
            "Random Forest is testing whether the invasive benchmark features contain richer non-linear structure than the non-invasive model. "
            "Because the feature set includes hormonal and ultrasound markers, it may capture stronger branching patterns than the routine screening model."
        ),
        "definition": """
            rf2 = RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )
        """,
    },
    {
        "label": "XGBoost",
        "short": "XGB",
        "var": "xgb2",
        "cv_var": "cv_results_m2_xgb",
        "result_var": "results_m2_xgb",
        "pred_var": "y2_pred_xgb",
        "proba_var": "y2_proba_xgb",
        "cm_var": "cm_m2_xgb",
        "cm_df_var": "cm_m2_xgb_df",
        "save_name": "model2_xgb.pkl",
        "cm_png": "06_model2_xgb_confusion_matrix.png",
        "markdown": (
            "XGBoost is serving as the strongest expected learner on the invasive benchmark set. "
            "It is checking whether a boosted-tree approach can extract the most value from the hormonal and ultrasound features. "
            "This is important because the best invasive model becomes the main reference point for judging the cost of removing invasive tests."
        ),
        "definition": """
            xgb2 = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw2,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            )
        """,
    },
]


def add_algorithm_block(
    cells: list[dict],
    *,
    model_label: str,
    feature_var: str,
    x_train: str,
    y_train: str,
    x_test: str,
    y_test: str,
    algo: dict,
    scale_pos_weight_var: str | None = None,
    add_feature_importance: bool = False,
) -> None:
    cells.append(
        markdown_cell(
            f"""
## {model_label}: {algo["label"]}

{algo["markdown"]}
"""
        )
    )

    if scale_pos_weight_var:
        cells.append(
            code_cell(
                f"""
                # Computing scale_pos_weight is documenting the effective class weighting being passed into {model_label} {algo["label"]}.
                neg_count = int(({y_train} == 0).sum())
                pos_count = int(({y_train} == 1).sum())
                {scale_pos_weight_var} = neg_count / pos_count

                print(f"Negative class count : {{neg_count}}")
                print(f"Positive class count : {{pos_count}}")
                print(f"scale_pos_weight     : {{{scale_pos_weight_var}:.4f}}")
                print("Note: the value is expected to be close to 1.0 because Notebook 07 already balanced the training set with SMOTE.")
                """
            )
        )

    cells.append(
        code_cell(
            f"""
            # Defining and training {model_label} {algo["label"]} is fitting the individual model on the locked training set from Notebook 07.
            {algo["definition"].strip()}
            {algo["var"]}.fit({x_train}, {y_train})
            print("{algo['var']} trained successfully.")
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Running cross-validation for {model_label} {algo["label"]} is estimating training stability with the locked 10-fold strategy.
            {algo["cv_var"]} = run_cv("{model_label} - {algo['label']}", {algo["var"]}, {x_train}, {y_train}, cv)
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Evaluating {model_label} {algo["label"]} on the hold-out test set is producing the final test metrics used in the comparison tables.
            {algo["pred_var"]} = {algo["var"]}.predict({x_test})
            {algo["proba_var"]} = {algo["var"]}.predict_proba({x_test})[:, 1]
            {algo["result_var"]} = evaluate_model("{model_label} - {algo['short']}", {algo["var"]}, {x_test}, {y_test})

            {algo["result_var"]}_df = pd.DataFrame([{algo["result_var"]}]).round(4)
            print({algo["result_var"]}_df.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the simple metric insight is making the test-set numbers easier to explain in plain language.
            metric_markdown = "\\n".join([
                "### Simple Metric Reading",
                "",
                f"- Accuracy: **{{{algo['result_var']}['accuracy']:.4f}}**. This is the share of all test cases the model got right.",
                f"- Precision: **{{{algo['result_var']}['precision']:.4f}}**. This shows how often a positive prediction was actually correct.",
                f"- Recall: **{{{algo['result_var']}['recall']:.4f}}**. This is the most important measure when missing a real PCOS case is costly, because it shows how many true PCOS cases were caught.",
                f"- F1 score: **{{{algo['result_var']}['f1']:.4f}}**. This is the balance between precision and recall.",
                f"- AUC: **{{{algo['result_var']}['auc']:.4f}}**. This shows how well the model separates PCOS and non-PCOS cases overall.",
                "",
                f"**Simple meaning:** {model_label} {algo['label']} is giving a recall of **{{{algo['result_var']}['recall']*100:.1f}}%**, so it is catching about that share of PCOS-positive cases in the test set.",
            ])
            display(Markdown(metric_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Printing the confusion matrix table for {model_label} {algo["label"]} is documenting the exact counts behind the heatmap.
            {algo["cm_var"]} = confusion_matrix({y_test}, {algo["pred_var"]})
            {algo["cm_df_var"]} = pd.DataFrame(
                {algo["cm_var"]},
                index=["Actual: No PCOS", "Actual: PCOS"],
                columns=["Predicted: No PCOS", "Predicted: PCOS"],
            )

            print("Confusion Matrix - {model_label} {algo['label']}:")
            print({algo["cm_df_var"]})
            print()
            print(f"True Positives  (caught PCOS cases) : {{{algo['cm_var']}[1, 1]}}")
            print(f"False Negatives (missed PCOS cases) : {{{algo['cm_var']}[1, 0]}}")
            print(f"False Positives (false alarms)      : {{{algo['cm_var']}[0, 1]}}")
            print(f"True Negatives  (correctly cleared) : {{{algo['cm_var']}[0, 0]}}")
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Plotting the confusion matrix for {model_label} {algo["label"]} is turning the printed count table into an easy visual summary.
            fig, ax = plt.subplots(figsize=(5.4, 4.4))
            sns.heatmap(
                {algo["cm_df_var"]},
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                linewidths=0.8,
                linecolor="white",
                ax=ax,
            )
            ax.set_title("Confusion Matrix - {model_label} {algo['label']} (Test Set)")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            output_path = IMAGE_DIR / "{algo['cm_png']}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {{output_path}}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the confusion-matrix insight is explaining the most important error pattern in simple terms.
            tp = int({algo["cm_var"]}[1, 1])
            fn = int({algo["cm_var"]}[1, 0])
            fp = int({algo["cm_var"]}[0, 1])
            tn = int({algo["cm_var"]}[0, 0])

            confusion_markdown = "\\n".join([
                "### Confusion Matrix Insight",
                "",
                f"- True positives: **{{tp}}**. These are PCOS cases the model caught correctly.",
                f"- False negatives: **{{fn}}**. These are real PCOS cases the model missed.",
                f"- False positives: **{{fp}}**. These are false alarms where the model predicted PCOS but the patient was negative.",
                f"- True negatives: **{{tn}}**. These are non-PCOS cases the model cleared correctly.",
                "",
                f"**Simple meaning:** For {model_label} {algo['label']}, the most important number to watch is the false-negative count of **{{fn}}**, because missed PCOS cases matter more in a screening study.",
            ])
            display(Markdown(confusion_markdown))
            """
        )
    )

    if add_feature_importance:
        cells.append(
            code_cell(
                f"""
                # Printing the ranked feature importances for {model_label} Random Forest is giving an interpretable reference before later SHAP work.
                fi_df = pd.DataFrame({{
                    "feature": {feature_var},
                    "importance": {algo["var"]}.feature_importances_,
                }}).sort_values("importance", ascending=False).reset_index(drop=True)

                print("{model_label} Random Forest - Feature Importances (ranked):")
                print(fi_df.to_string(index=False))
                """
            )
        )

        cells.append(
            code_cell(
                """
                # Writing the feature-importance insight is making the ranking table easier to explain in a meeting.
                top_features = fi_df.head(3)['feature'].tolist()
                feature_markdown = "\\n".join([
                    "### Feature Importance Insight",
                    "",
                    f"- The top features in this Random Forest model are **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**.",
                    "- Higher feature importance means the model is using that feature more often when it builds its decision trees.",
                    "- This is only a quick model-based ranking. The stronger explanation step will come later with SHAP.",
                    "",
                    "**Simple meaning:** These top features are the ones this Random Forest model is leaning on the most when it decides whether a case looks like PCOS.",
                ])
                display(Markdown(feature_markdown))
                """
            )
        )

    cells.append(
        code_cell(
            f"""
            # Saving {model_label} {algo["label"]} is preserving the fitted model for later ensemble and explainability notebooks.
            model_path = MODELS_DIR / "{algo['save_name']}"
            joblib.dump({algo["var"]}, model_path)
            print(f"Saved model to: {{model_path}}")
            """
        )
    )


def add_comparison_block(
    cells: list[dict],
    *,
    model_label: str,
    set_short: str,
    results_vars: list[str],
    cv_vars: list[str],
    model_vars: list[str],
    x_test: str,
    y_test: str,
    roc_png: str,
    cv_png: str,
) -> None:
    cells.append(
        markdown_cell(
            f"""
## {model_label}: Algorithm Comparison

This section is comparing the three individual algorithms inside {model_label} before moving to the next model set. The table and plots below are defining the single-model baseline for this feature contract.
"""
            )
        )

    results_name = f"{set_short.lower()}_results"
    roc_table_name = f"{set_short.lower()}_roc_table"
    cv_auc_df_name = f"{set_short.lower()}_cv_auc_df"
    cv_auc_long_name = f"{set_short.lower()}_cv_auc_long"
    cv_auc_stats_name = f"{set_short.lower()}_cv_auc_stats"

    cells.append(
        code_cell(
            f"""
            # Printing the {model_label} comparison table is showing the final hold-out test metrics side by side.
            {results_name} = pd.DataFrame([{", ".join(results_vars)}])
            {results_name}["algorithm"] = ["LR", "RF", "XGB"]
            {results_name} = {results_name}.set_index("model").round(4)

            print("{model_label.upper()} - Algorithm Comparison (Test Set)")
            print("=" * 70)
            print({results_name}.to_string())
            print()
            print(f"Best Recall : {{{results_name}['recall'].idxmax()}} ({{{results_name}['recall'].max():.4f}})")
            print(f"Best AUC    : {{{results_name}['auc'].idxmax()}} ({{{results_name}['auc'].max():.4f}})")
            print(f"Best F1     : {{{results_name}['f1'].idxmax()}} ({{{results_name}['f1'].max():.4f}})")
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the {model_label} comparison insight is turning the metric table into a simple summary.
            best_auc_row = {results_name}.reset_index().sort_values("auc", ascending=False).iloc[0]
            best_recall_row = {results_name}.reset_index().sort_values("recall", ascending=False).iloc[0]
            best_f1_row = {results_name}.reset_index().sort_values("f1", ascending=False).iloc[0]

            comparison_markdown = "\\n".join([
                "### Comparison Insight",
                "",
                f"- The best AUC in {model_label} is coming from **{{best_auc_row['algorithm']}}** at **{{best_auc_row['auc']:.4f}}**.",
                f"- The best recall in {model_label} is coming from **{{best_recall_row['algorithm']}}** at **{{best_recall_row['recall']:.4f}}**.",
                f"- The best F1 score in {model_label} is coming from **{{best_f1_row['algorithm']}}** at **{{best_f1_row['f1']:.4f}}**.",
                "",
                f"**Simple meaning:** Inside {model_label}, this table is showing which algorithm is best at overall separation, which one catches the most PCOS cases, and which one gives the best balance between missing cases and raising false alarms.",
            ])
            display(Markdown(comparison_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Printing the ROC summary table is documenting the exact AUC values used before plotting the {model_label} ROC curves.
            {roc_table_name} = (
                {results_name}.reset_index()[["model", "algorithm", "auc"]]
                .sort_values("auc", ascending=False)
                .reset_index(drop=True)
            )
            print({roc_table_name}.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Plotting the {model_label} ROC curves is comparing discrimination across the three individual algorithms on the same test set.
            fig, ax = plt.subplots(figsize=(6.6, 4.8))
            RocCurveDisplay.from_estimator({model_vars[0]}, {x_test}, {y_test}, ax=ax, name="LR", color=COLORS["lr"])
            RocCurveDisplay.from_estimator({model_vars[1]}, {x_test}, {y_test}, ax=ax, name="RF", color=COLORS["rf"])
            RocCurveDisplay.from_estimator({model_vars[2]}, {x_test}, {y_test}, ax=ax, name="XGB", color=COLORS["xgb"])
            ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.2)
            ax.set_title("{model_label} - ROC Curves: LR vs RF vs XGBoost")
            ax.grid(alpha=0.25)

            output_path = IMAGE_DIR / "{roc_png}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {{output_path}}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the ROC insight is explaining what the curve comparison means in simple terms.
            best_roc_row = {roc_table_name}.iloc[0]
            roc_markdown = "\\n".join([
                "### ROC Insight",
                "",
                f"- The top ROC result in {model_label} is **{{best_roc_row['algorithm']}}** with an AUC of **{{best_roc_row['auc']:.4f}}**.",
                "- The curve that stays closer to the top-left corner is doing a better job of separating positive and negative cases.",
                "- The diagonal line is the no-skill line. A good model should stay above it.",
                "",
                f"**Simple meaning:** In {model_label}, the algorithm with the highest AUC is giving the cleanest overall separation between PCOS and non-PCOS cases.",
            ])
            display(Markdown(roc_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Printing the raw fold AUC table is documenting the exact cross-validation values used in the {model_label} CV distribution plot.
            {cv_auc_df_name} = pd.DataFrame({{
                "fold": range(1, len({cv_vars[0]}["fold_scores"]) + 1),
                "LR": {cv_vars[0]}["fold_scores"]["roc_auc"].values,
                "RF": {cv_vars[1]}["fold_scores"]["roc_auc"].values,
                "XGB": {cv_vars[2]}["fold_scores"]["roc_auc"].values,
            }})
            print({cv_auc_df_name}.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Plotting the {model_label} cross-validation AUC distributions is showing variation across the ten folds rather than only the mean score.
            {cv_auc_long_name} = {cv_auc_df_name}.melt(
                id_vars="fold",
                var_name="algorithm",
                value_name="auc"
            )

            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            sns.boxplot(
                data={cv_auc_long_name},
                x="algorithm",
                y="auc",
                palette=[COLORS["lr"], COLORS["rf"], COLORS["xgb"]],
                width=0.55,
                ax=ax,
            )
            sns.stripplot(
                data={cv_auc_long_name},
                x="algorithm",
                y="auc",
                color=COLORS["neutral"],
                alpha=0.55,
                size=4.5,
                ax=ax,
            )
            ax.set_title("{model_label} - Cross-Validation AUC Distribution (10 Folds)")
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("AUC-ROC")
            ax.grid(axis="y", alpha=0.25)

            output_path = IMAGE_DIR / "{cv_png}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {{output_path}}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the cross-validation insight is explaining model stability in simple terms.
            {cv_auc_stats_name} = (
                {cv_auc_long_name}
                .groupby("algorithm")["auc"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )

            best_cv_row = {cv_auc_stats_name}.iloc[0]
            most_stable_row = {cv_auc_stats_name}.sort_values("std", ascending=True).iloc[0]

            cv_markdown = "\\n".join([
                "### Cross-Validation Insight",
                "",
                f"- The highest mean cross-validation AUC in {model_label} is coming from **{{best_cv_row['algorithm']}}** at **{{best_cv_row['mean']:.4f}}**.",
                f"- The most stable algorithm across folds in {model_label} is **{{most_stable_row['algorithm']}}** with a standard deviation of **{{most_stable_row['std']:.4f}}**.",
                "- A smaller spread means the model is behaving more consistently from fold to fold.",
                "",
                f"**Simple meaning:** This chart is not only showing which algorithm is strong, it is also showing which one is steady and repeatable across different training folds in {model_label}.",
            ])
            display(Markdown(cv_markdown))
            """
        )
    )


def build_notebook() -> list[dict]:
    output_table = markdown_table(["Output File", "Purpose"], OUTPUT_ROWS)
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            f"""
# Individual Model Training and Evaluation - Logistic Regression, Random Forest, XGBoost

## Introduction
This notebook is training and evaluating the six individual models used in this phase of the thesis. The notebook is loading the prepared modelling sets from Notebook 07 and is fitting three algorithms on each model set:

- Model 1 + Logistic Regression
- Model 1 + Random Forest
- Model 1 + XGBoost
- Model 2 + Logistic Regression
- Model 2 + Random Forest
- Model 2 + XGBoost

Model 1 is representing the **non-invasive clinical screening model**. Model 2 is representing the **invasive benchmark model**. This notebook is comparing them fairly because both model sets were prepared earlier and are now being reused without any new splitting, scaling, or class balancing.

This notebook is coming before ensemble learning, optimization, and SHAP because the project first needs to understand how the individual base learners perform on their own. These six trained models become the baseline that later notebooks must improve on.

The same locked 10-fold stratified cross-validation object from Notebook 07 is being reused for all six models so the comparison stays consistent. A key caution is also being stated clearly here: the saved training sets were already SMOTEd in Notebook 07, so the cross-validation scores in this notebook may read slightly better than a fold-local SMOTE design. The hold-out test-set metrics remain the main external benchmark inside this notebook.

No new preprocessing is happening here. This notebook is **not** re-splitting data, re-scaling features, or re-applying SMOTE. It is only training, evaluating, saving models, and producing comparison visuals.

## Output Files Produced in This Notebook
{output_table}
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Section 1 - Imports and Configuration

This section is importing the libraries, resolving the project root, and loading the locked modelling files produced in Notebook 07.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Importing the modelling libraries is preparing the notebook for training, evaluation, plotting, and artifact export.
            from pathlib import Path
            import os
            import warnings
            import json
            import joblib

            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import seaborn as sns
            from IPython.display import Markdown, display

            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_validate, StratifiedKFold
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix,
                RocCurveDisplay, ConfusionMatrixDisplay
            )

            try:
                from xgboost import XGBClassifier
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "xgboost is required for Notebook 08. Install it in your current notebook environment with "
                    "`pip install xgboost` or `conda install -c conda-forge xgboost`, then restart the kernel and run the notebook again."
                ) from exc

            warnings.filterwarnings('ignore')
            sns.set_style('whitegrid')
            plt.rcParams['figure.dpi'] = 150
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

            RANDOM_STATE = 42
            TARGET = 'pcos_y_n'

            COLORS = {
                'pcos_pos': '#E63946',
                'pcos_neg': '#457B9D',
                'model1': '#2A9D8F',
                'model2': '#E9C46A',
                'lr': '#264653',
                'rf': '#2A9D8F',
                'xgb': '#E76F51',
                'neutral': '#2C3E50',
                'highlight': '#8D5A97'
            }
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Resolving the project root and creating the image output folder is keeping the notebook runnable from different working directories.
            def resolve_project_root() -> Path:
                current = Path.cwd().resolve()
                candidates = [current, current.parent, current.parent.parent]
                for candidate in candidates:
                    if (candidate / 'cleaned_data').exists() and (candidate / 'models').exists():
                        return candidate
                return current

            PROJECT_ROOT = resolve_project_root()
            MODELLING_DIR = PROJECT_ROOT / 'cleaned_data' / 'modelling_sets'
            MODELS_DIR = PROJECT_ROOT / 'models'
            IMAGE_DIR = PROJECT_ROOT / 'images' / 'individual_models'
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            print(f"Created or confirmed: {IMAGE_DIR}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Loading the prepared modelling sets and feature contracts is using Notebook 07 as the only source of truth for training data.
            train1 = pd.read_csv(MODELLING_DIR / 'model1_train.csv')
            test1 = pd.read_csv(MODELLING_DIR / 'model1_test.csv')
            train2 = pd.read_csv(MODELLING_DIR / 'model2_train.csv')
            test2 = pd.read_csv(MODELLING_DIR / 'model2_test.csv')

            with open(MODELLING_DIR / 'feature_names_model1.json', encoding='utf-8') as f:
                MODEL1_FEATURES = json.load(f)
            with open(MODELLING_DIR / 'feature_names_model2.json', encoding='utf-8') as f:
                MODEL2_FEATURES = json.load(f)

            cv = joblib.load(MODELS_DIR / 'cv_strategy.pkl')

            assert len(MODEL1_FEATURES) == 19, f"Expected 19 Model 1 features, found {len(MODEL1_FEATURES)}."
            assert len(MODEL2_FEATURES) == 15, f"Expected 15 Model 2 features, found {len(MODEL2_FEATURES)}."

            print("Model 1 feature list:")
            print(MODEL1_FEATURES)
            print()
            print("Model 2 feature list:")
            print(MODEL2_FEATURES)
            print()
            print("DataFrame shapes:")
            print(f"train1: {train1.shape}")
            print(f"test1 : {test1.shape}")
            print(f"train2: {train2.shape}")
            print(f"test2 : {test2.shape}")
            print()
            print("CV strategy:")
            print(cv)
            print(f"CV strategy type: {type(cv).__name__}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Separating features and target is locking the exact training arrays used by each algorithm.
            X1_train = train1[MODEL1_FEATURES].values
            y1_train = train1[TARGET].values
            X1_test = test1[MODEL1_FEATURES].values
            y1_test = test1[TARGET].values

            X2_train = train2[MODEL2_FEATURES].values
            y2_train = train2[TARGET].values
            X2_test = test2[MODEL2_FEATURES].values
            y2_test = test2[TARGET].values

            shape_table = pd.DataFrame(
                [
                    {"set": "X1_train", "rows": X1_train.shape[0], "columns": X1_train.shape[1]},
                    {"set": "X1_test", "rows": X1_test.shape[0], "columns": X1_test.shape[1]},
                    {"set": "X2_train", "rows": X2_train.shape[0], "columns": X2_train.shape[1]},
                    {"set": "X2_test", "rows": X2_test.shape[0], "columns": X2_test.shape[1]},
                ]
            )
            print(shape_table.to_string(index=False))
            print()

            class_distribution_table = pd.DataFrame(
                [
                    {"set": "y1_train", "no_pcos": int((y1_train == 0).sum()), "pcos": int((y1_train == 1).sum())},
                    {"set": "y1_test", "no_pcos": int((y1_test == 0).sum()), "pcos": int((y1_test == 1).sum())},
                    {"set": "y2_train", "no_pcos": int((y2_train == 0).sum()), "pcos": int((y2_train == 1).sum())},
                    {"set": "y2_test", "no_pcos": int((y2_test == 0).sum()), "pcos": int((y2_test == 1).sum())},
                ]
            )
            print(class_distribution_table.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Defining evaluate_model is keeping the hold-out metric calculation consistent across all six trained models.
            def evaluate_model(name, model, X_test, y_test):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                return {
                    'model': name,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                    'auc': roc_auc_score(y_test, y_prob),
                }
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Defining run_cv is keeping the ten-fold cross-validation summary consistent across all algorithms and both model sets.
            def run_cv(name, model, X_train, y_train, cv):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                cv_output = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=None,
                )

                fold_scores = pd.DataFrame({
                    'fold': range(1, len(cv_output['test_accuracy']) + 1),
                    'accuracy': cv_output['test_accuracy'],
                    'precision': cv_output['test_precision'],
                    'recall': cv_output['test_recall'],
                    'f1': cv_output['test_f1'],
                    'roc_auc': cv_output['test_roc_auc'],
                })

                summary_df = pd.DataFrame({
                    'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    'mean': [
                        fold_scores['accuracy'].mean(),
                        fold_scores['precision'].mean(),
                        fold_scores['recall'].mean(),
                        fold_scores['f1'].mean(),
                        fold_scores['roc_auc'].mean(),
                    ],
                    'std': [
                        fold_scores['accuracy'].std(),
                        fold_scores['precision'].std(),
                        fold_scores['recall'].std(),
                        fold_scores['f1'].std(),
                        fold_scores['roc_auc'].std(),
                    ],
                }).round(4)

                print(f"Cross-validation summary - {name}")
                print(summary_df.to_string(index=False))

                return {
                    'model': name,
                    'summary_table': summary_df,
                    'fold_scores': fold_scores,
                    'accuracy_mean': summary_df.loc[summary_df['metric'] == 'accuracy', 'mean'].iloc[0],
                    'accuracy_std': summary_df.loc[summary_df['metric'] == 'accuracy', 'std'].iloc[0],
                    'precision_mean': summary_df.loc[summary_df['metric'] == 'precision', 'mean'].iloc[0],
                    'precision_std': summary_df.loc[summary_df['metric'] == 'precision', 'std'].iloc[0],
                    'recall_mean': summary_df.loc[summary_df['metric'] == 'recall', 'mean'].iloc[0],
                    'recall_std': summary_df.loc[summary_df['metric'] == 'recall', 'std'].iloc[0],
                    'f1_mean': summary_df.loc[summary_df['metric'] == 'f1', 'mean'].iloc[0],
                    'f1_std': summary_df.loc[summary_df['metric'] == 'f1', 'std'].iloc[0],
                    'roc_auc_mean': summary_df.loc[summary_df['metric'] == 'roc_auc', 'mean'].iloc[0],
                    'roc_auc_std': summary_df.loc[summary_df['metric'] == 'roc_auc', 'std'].iloc[0],
                }
            """
        )
    )

    for algo in MODEL1_ALGOS:
        add_algorithm_block(
            cells,
            model_label="Model 1",
            feature_var="MODEL1_FEATURES",
            x_train="X1_train",
            y_train="y1_train",
            x_test="X1_test",
            y_test="y1_test",
            algo=algo,
            scale_pos_weight_var="spw1" if algo["short"] == "XGB" else None,
            add_feature_importance=algo["short"] == "RF",
        )

    add_comparison_block(
        cells,
        model_label="Model 1",
        set_short="M1",
        results_vars=["results_m1_lr", "results_m1_rf", "results_m1_xgb"],
        cv_vars=["cv_results_m1_lr", "cv_results_m1_rf", "cv_results_m1_xgb"],
        model_vars=["lr1", "rf1", "xgb1"],
        x_test="X1_test",
        y_test="y1_test",
        roc_png="07_model1_roc_curves_all_algorithms.png",
        cv_png="09_model1_cv_score_distributions.png",
    )

    cells.append(
        markdown_cell(
            """
## Section 6 - Model 2: All Three Algorithms

This section is repeating the same evaluation pattern on the invasive benchmark set so the non-invasive and invasive results can be compared fairly.
"""
        )
    )

    for algo in MODEL2_ALGOS:
        add_algorithm_block(
            cells,
            model_label="Model 2",
            feature_var="MODEL2_FEATURES",
            x_train="X2_train",
            y_train="y2_train",
            x_test="X2_test",
            y_test="y2_test",
            algo=algo,
            scale_pos_weight_var="spw2" if algo["short"] == "XGB" else None,
            add_feature_importance=False,
        )

    add_comparison_block(
        cells,
        model_label="Model 2",
        set_short="M2",
        results_vars=["results_m2_lr", "results_m2_rf", "results_m2_xgb"],
        cv_vars=["cv_results_m2_lr", "cv_results_m2_rf", "cv_results_m2_xgb"],
        model_vars=["lr2", "rf2", "xgb2"],
        x_test="X2_test",
        y_test="y2_test",
        roc_png="08_model2_roc_curves_all_algorithms.png",
        cv_png="10_model2_cv_score_distributions.png",
    )

    cells.append(
        markdown_cell(
            """
## Section 7 - Master Comparison: All Six Models

This section is combining the six individual results into one thesis-ready comparison table. It is answering two questions at once: which algorithm is best inside each model set, and how large the non-invasive versus invasive performance gap really is.
"""
        )
    )

    cells.extend(build_master_section())
    return cells


def build_master_section() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        code_cell(
            """
            # Building the master results table is combining the six hold-out test results into one comparison frame.
            all_results = pd.DataFrame([
                results_m1_lr, results_m1_rf, results_m1_xgb,
                results_m2_lr, results_m2_rf, results_m2_xgb,
            ])
            all_results['model_set'] = ['Model 1', 'Model 1', 'Model 1', 'Model 2', 'Model 2', 'Model 2']
            all_results['algorithm'] = ['LR', 'RF', 'XGB', 'LR', 'RF', 'XGB']
            all_results = all_results.round(4)

            print("MASTER RESULTS TABLE - All Six Individual Models")
            print("=" * 80)
            print(all_results.to_string(index=False))
            print()

            gap_rows = []
            for algo in ['LR', 'RF', 'XGB']:
                m1_auc = all_results.loc[
                    (all_results['algorithm'] == algo) & (all_results['model_set'] == 'Model 1'),
                    'auc'
                ].iloc[0]
                m2_auc = all_results.loc[
                    (all_results['algorithm'] == algo) & (all_results['model_set'] == 'Model 2'),
                    'auc'
                ].iloc[0]
                gap = m2_auc - m1_auc
                gap_rows.append({
                    'algorithm': algo,
                    'model1_auc': round(m1_auc, 4),
                    'model2_auc': round(m2_auc, 4),
                    'absolute_gap': round(gap, 4),
                    'percentage_point_gap': round(gap * 100, 2),
                })
                print(f"{algo}: Model 2 AUC = {m2_auc:.4f} | Model 1 AUC = {m1_auc:.4f} | Gap = {gap:.4f} ({gap*100:.2f}%)")

            gap_df = pd.DataFrame(gap_rows)
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Saving the master results CSV is making the six-model comparison reusable in later notebooks and reporting.
            output_path = MODELLING_DIR / 'individual_model_results.csv'
            all_results.to_csv(output_path, index=False)
            print(f"Saved results table to: {output_path}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the heatmap input table is documenting the exact values used in the master performance heatmap.
            master_heatmap_df = all_results.set_index(['model_set', 'algorithm'])[['accuracy', 'precision', 'recall', 'f1', 'auc']]
            print(master_heatmap_df.to_string())
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the master comparison heatmap is showing the six-model metric profile in one compact view.
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            sns.heatmap(
                master_heatmap_df,
                annot=True,
                fmt=".4f",
                cmap="Greens",
                linewidths=0.8,
                linecolor="white",
                cbar=True,
                ax=ax,
            )
            ax.set_title("Master Performance Heatmap - All Six Individual Models")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Model")

            output_path = IMAGE_DIR / '11_master_comparison_heatmap.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the heatmap insight is turning the full metric grid into a simple message.
            best_overall_auc = all_results.sort_values('auc', ascending=False).iloc[0]
            heatmap_markdown = "\\n".join([
                "### Heatmap Insight",
                "",
                f"- The best overall AUC in the full six-model table is **{best_overall_auc['model']}** at **{best_overall_auc['auc']:.4f}**.",
                "- Darker cells in the heatmap are showing stronger performance.",
                "- This chart is useful because it lets you compare all six models across all five metrics at once.",
                "",
                "**Simple meaning:** The heatmap is giving one quick picture of which model is strongest overall and where each model is weaker.",
            ])
            display(Markdown(heatmap_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the recall comparison table is documenting the exact values used in the recall bar chart.
            recall_plot_df = all_results[['model_set', 'algorithm', 'recall']].copy().sort_values(['algorithm', 'model_set']).reset_index(drop=True)
            print(recall_plot_df.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the recall comparison is showing the practical case-detection difference between Model 1 and Model 2 across algorithms.
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            sns.barplot(
                data=recall_plot_df,
                x='algorithm',
                y='recall',
                hue='model_set',
                palette=[COLORS['model1'], COLORS['model2']],
                ax=ax,
            )
            ax.axhline(0.80, linestyle='--', color=COLORS['neutral'], linewidth=1.2, label='Clinical minimum threshold')
            ax.set_title("Recall Comparison - Model 1 vs Model 2 (All Algorithms)")
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Recall")
            ax.legend(frameon=True)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

            output_path = IMAGE_DIR / '12_recall_comparison_all_models.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the recall insight is explaining the case-finding meaning of the bar chart.
            best_recall_overall = recall_plot_df.sort_values('recall', ascending=False).iloc[0]
            recall_markdown = "\\n".join([
                "### Recall Insight",
                "",
                f"- The highest recall across all six models is **{best_recall_overall['algorithm']}** in **{best_recall_overall['model_set']}** at **{best_recall_overall['recall']:.4f}**.",
                "- Recall matters most when missing a real PCOS case is a serious problem.",
                "- A model with higher recall is catching more of the true PCOS cases.",
                "",
                "**Simple meaning:** This chart is showing which algorithm is best at finding PCOS cases instead of missing them.",
            ])
            display(Markdown(recall_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the AUC comparison table is documenting the exact values used in the AUC bar chart.
            auc_plot_df = all_results[['model_set', 'algorithm', 'auc']].copy().sort_values(['algorithm', 'model_set']).reset_index(drop=True)
            print(auc_plot_df.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the AUC comparison is showing how much discrimination changes between the non-invasive and invasive feature sets.
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            sns.barplot(
                data=auc_plot_df,
                x='algorithm',
                y='auc',
                hue='model_set',
                palette=[COLORS['model1'], COLORS['model2']],
                ax=ax,
            )
            ax.axhline(0.85, linestyle='--', color=COLORS['neutral'], linewidth=1.2, label='AUC reference line')
            ax.set_title("AUC-ROC Comparison - Model 1 vs Model 2 (All Algorithms)")
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("AUC-ROC")
            ax.legend(frameon=True)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)

            output_path = IMAGE_DIR / '13_auc_comparison_all_models.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the AUC insight is explaining the separation quality shown in the bar chart.
            best_auc_overall = auc_plot_df.sort_values('auc', ascending=False).iloc[0]
            auc_markdown = "\\n".join([
                "### AUC Insight",
                "",
                f"- The highest AUC across all six models is **{best_auc_overall['algorithm']}** in **{best_auc_overall['model_set']}** at **{best_auc_overall['auc']:.4f}**.",
                "- A higher AUC means the model is doing a better job of separating PCOS and non-PCOS cases overall.",
                "- This chart is useful because it shows whether the invasive model gains a lot or only a little over the non-invasive model.",
                "",
                "**Simple meaning:** This chart is showing which model is best at overall separation, not just at one decision cutoff.",
            ])
            display(Markdown(auc_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the performance-gap table is documenting the exact AUC difference values used in the gap chart.
            print(gap_df.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the invasive-versus-non-invasive AUC gap is showing the cost of removing invasive features for each algorithm.
            fig, ax = plt.subplots(figsize=(6.6, 4.4))
            sns.barplot(
                data=gap_df,
                y='algorithm',
                x='absolute_gap',
                palette=[COLORS['highlight'], COLORS['highlight'], COLORS['highlight']],
                ax=ax,
            )
            ax.axvline(0, linestyle='--', color=COLORS['neutral'], linewidth=1.2)
            ax.set_title("Performance Gap - Invasive vs Non-Invasive Features (AUC difference)")
            ax.set_xlabel("AUC difference (Model 2 - Model 1)")
            ax.set_ylabel("Algorithm")
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', fontsize=8, padding=3)

            output_path = IMAGE_DIR / '14_model1_vs_model2_performance_gap.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the gap insight is explaining what is gained or lost when invasive features are added.
            largest_gap = gap_df.sort_values('absolute_gap', ascending=False).iloc[0]
            smallest_gap = gap_df.sort_values('absolute_gap', ascending=True).iloc[0]
            gap_markdown = "\\n".join([
                "### Performance Gap Insight",
                "",
                f"- The largest AUC gain from adding invasive features is seen in **{largest_gap['algorithm']}** at **{largest_gap['absolute_gap']:.4f}**.",
                f"- The smallest AUC gain from adding invasive features is seen in **{smallest_gap['algorithm']}** at **{smallest_gap['absolute_gap']:.4f}**.",
                "- A small gap means the non-invasive model is already close to the invasive benchmark.",
                "- A large gap means the invasive markers are adding more extra signal for that algorithm.",
                "",
                "**Simple meaning:** This chart is showing the price paid for removing invasive tests. Smaller bars are better for the screening model because they mean less performance was lost.",
            ])
            display(Markdown(gap_markdown))
            """
        )
    )

    cells.append(
        markdown_cell(
            """
## Key Findings from Individual Model Evaluation

This section is converting the six-model result table into thesis-ready interpretation that can be reused in Chapter 4.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Building the key-findings markdown is turning the metric table into a clear written interpretation.
            best_m1 = all_results.loc[all_results['model_set'] == 'Model 1'].sort_values(['auc', 'recall', 'f1'], ascending=False).iloc[0]
            best_m2 = all_results.loc[all_results['model_set'] == 'Model 2'].sort_values(['auc', 'recall', 'f1'], ascending=False).iloc[0]

            best_algo_gap = gap_df.sort_values('absolute_gap', ascending=False).iloc[0]
            summary_markdown = "\\n".join([
                "### Key Findings",
                "",
                f"**Paragraph 1:** Inside Model 1, the best-performing algorithm is **{best_m1['algorithm']}** with an AUC of **{best_m1['auc']:.4f}**, a recall of **{best_m1['recall']:.4f}**, and an F1 score of **{best_m1['f1']:.4f}**. This means the model is correctly identifying about **{best_m1['recall']*100:.1f}%** of PCOS-positive cases in the held-out test set.",
                "",
                f"**Paragraph 2:** Inside Model 2, the best-performing algorithm is **{best_m2['algorithm']}** with an AUC of **{best_m2['auc']:.4f}**, a recall of **{best_m2['recall']:.4f}**, and an F1 score of **{best_m2['f1']:.4f}**. This is acting as the invasive benchmark for the current stage of the thesis.",
                "",
                f"**Paragraph 3:** The biggest invasive-versus-non-invasive AUC difference is currently seen in **{best_algo_gap['algorithm']}**, where the invasive model exceeds the non-invasive model by **{best_algo_gap['absolute_gap']:.4f}** AUC points, or **{best_algo_gap['percentage_point_gap']:.2f} percentage points**. This is quantifying the price paid for removing invasive tests.",
                "",
                "**Paragraph 4:** These results are setting the baseline for Notebook 09. The strongest individual learner in each model set becomes the main reference point for the later stacking notebook, and the current gaps define the improvement target for later ensemble and optimization steps.",
            ])
            display(Markdown(summary_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Verifying the final output files is checking that every saved artifact exists and is non-empty.
            required_files = [
                MODELS_DIR / 'model1_lr.pkl',
                MODELS_DIR / 'model1_rf.pkl',
                MODELS_DIR / 'model1_xgb.pkl',
                MODELS_DIR / 'model2_lr.pkl',
                MODELS_DIR / 'model2_rf.pkl',
                MODELS_DIR / 'model2_xgb.pkl',
                MODELLING_DIR / 'individual_model_results.csv',
                IMAGE_DIR / '01_model1_lr_confusion_matrix.png',
                IMAGE_DIR / '02_model1_rf_confusion_matrix.png',
                IMAGE_DIR / '03_model1_xgb_confusion_matrix.png',
                IMAGE_DIR / '04_model2_lr_confusion_matrix.png',
                IMAGE_DIR / '05_model2_rf_confusion_matrix.png',
                IMAGE_DIR / '06_model2_xgb_confusion_matrix.png',
                IMAGE_DIR / '07_model1_roc_curves_all_algorithms.png',
                IMAGE_DIR / '08_model2_roc_curves_all_algorithms.png',
                IMAGE_DIR / '09_model1_cv_score_distributions.png',
                IMAGE_DIR / '10_model2_cv_score_distributions.png',
                IMAGE_DIR / '11_master_comparison_heatmap.png',
                IMAGE_DIR / '12_recall_comparison_all_models.png',
                IMAGE_DIR / '13_auc_comparison_all_models.png',
                IMAGE_DIR / '14_model1_vs_model2_performance_gap.png',
            ]

            all_passed = True
            for file_path in required_files:
                exists = file_path.exists()
                non_empty = exists and file_path.stat().st_size > 0
                status = 'PASS' if exists and non_empty else 'FAIL'
                if status == 'FAIL':
                    all_passed = False
                print(f"[{status}] {file_path}")

            print()
            if all_passed:
                print("All output files verified successfully and are non-empty.")
            else:
                print("WARNING: Some output files are missing or empty. Check the cells above.")
            """
        )
    )

    cells.append(
        markdown_cell(
            """
## Notebook Summary and Next Steps

This section is closing the individual-model notebook and pointing forward to the stacking notebook.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Building the closing summary markdown is turning the final outputs into a clean bridge to Notebook 09.
            best_model1_name = best_m1['model']
            best_model2_name = best_m2['model']
            closing_markdown = "\\n".join([
                "### Notebook Summary",
                "",
                "This notebook has trained and saved six individual models:",
                "- `models/model1_lr.pkl`",
                "- `models/model1_rf.pkl`",
                "- `models/model1_xgb.pkl`",
                "- `models/model2_lr.pkl`",
                "- `models/model2_rf.pkl`",
                "- `models/model2_xgb.pkl`",
                "",
                "The full six-model comparison table has been saved to `cleaned_data/modelling_sets/individual_model_results.csv`.",
                "",
                f"The current best Model 1 algorithm is **{best_model1_name}**. The current best Model 2 algorithm is **{best_model2_name}**. The invasive-versus-non-invasive gap measured here is defining the improvement target for the next stage.",
                "",
                "Notebook 09 will build the stacking ensemble for each model set by combining the saved base learners with a meta-learner. It will then test whether stacking improves on the best individual algorithm from this notebook.",
            ])
            display(Markdown(closing_markdown))
            """
        )
    )

    return cells


def main() -> None:
    write_notebook(NOTEBOOK_PATH, build_notebook())


if __name__ == "__main__":
    main()
