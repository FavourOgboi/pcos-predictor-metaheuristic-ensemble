from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_ensemble_stacking.ipynb"


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
    ("`models/model1_stack_lr_meta.pkl`", "Saved Logistic Regression meta-learner for Model 1."),
    ("`models/model1_stack_rf_meta.pkl`", "Saved Random Forest meta-learner for Model 1."),
    ("`models/model1_stack_xgb_meta.pkl`", "Saved XGBoost meta-learner for Model 1."),
    ("`models/model1_oof_predictions.npy`", "Saved Model 1 out-of-fold base-learner probabilities."),
    ("`models/model2_stack_lr_meta.pkl`", "Saved Logistic Regression meta-learner for Model 2."),
    ("`models/model2_stack_rf_meta.pkl`", "Saved Random Forest meta-learner for Model 2."),
    ("`models/model2_stack_xgb_meta.pkl`", "Saved XGBoost meta-learner for Model 2."),
    ("`models/model2_oof_predictions.npy`", "Saved Model 2 out-of-fold base-learner probabilities."),
    ("`cleaned_data/modelling_sets/stacking_results.csv`", "Saved the six stacking-result rows only."),
    ("`cleaned_data/modelling_sets/master_results_all_models.csv`", "Saved the combined individual and stacking results table."),
    ("`images/ensemble/01-15_*.png`", "Saved OOF plots, confusion matrices, ROC plots, comparison charts, and master summary figures."),
]


STACK_MODEL1 = [
    {
        "title": "Logistic Regression",
        "short": "Stack-LR",
        "var": "meta_lr1",
        "result_var": "results_stack_m1_lr",
        "pred_var": "stack_lr1_pred",
        "proba_var": "stack_lr1_proba",
        "cm_var": "cm_stack_m1_lr",
        "cm_df_var": "cm_stack_m1_lr_df",
        "save_name": "model1_stack_lr_meta.pkl",
        "cm_png": "03_model1_stack_lr_confusion_matrix.png",
        "markdown": (
            "Logistic Regression is serving as the first meta-learner for Model 1. "
            "It is learning a simple weighted combination of the three base-learner probabilities. "
            "If this meta-learner wins, it means the base learners mostly need a clean linear weighting rather than a complex second-level rule."
        ),
        "definition": """
            meta_lr1 = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            )
        """,
    },
    {
        "title": "Random Forest",
        "short": "Stack-RF",
        "var": "meta_rf1",
        "result_var": "results_stack_m1_rf",
        "pred_var": "stack_rf1_pred",
        "proba_var": "stack_rf1_proba",
        "cm_var": "cm_stack_m1_rf",
        "cm_df_var": "cm_stack_m1_rf_df",
        "save_name": "model1_stack_rf_meta.pkl",
        "cm_png": "04_model1_stack_rf_confusion_matrix.png",
        "markdown": (
            "Random Forest is serving as the second meta-learner for Model 1. "
            "It is checking whether a non-linear rule can combine the three base-learner probabilities better than a straight weighted average. "
            "If this meta-learner wins, it suggests the base models are making useful predictions in different parts of the patient profile space."
        ),
        "definition": """
            meta_rf1 = RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )
        """,
    },
    {
        "title": "XGBoost",
        "short": "Stack-XGB",
        "var": "meta_xgb1",
        "result_var": "results_stack_m1_xgb",
        "pred_var": "stack_xgb1_pred",
        "proba_var": "stack_xgb1_proba",
        "cm_var": "cm_stack_m1_xgb",
        "cm_df_var": "cm_stack_m1_xgb_df",
        "save_name": "model1_stack_xgb_meta.pkl",
        "cm_png": "05_model1_stack_xgb_confusion_matrix.png",
        "markdown": (
            "XGBoost is serving as the third meta-learner for Model 1. "
            "It is learning a boosted non-linear combination of the three base-learner probabilities. "
            "If this version performs best, it means the best final decision rule is itself boosted and non-linear."
        ),
        "definition": """
            meta_xgb1 = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            )
        """,
    },
]


STACK_MODEL2 = [
    {
        "title": "Logistic Regression",
        "short": "Stack-LR",
        "var": "meta_lr2",
        "result_var": "results_stack_m2_lr",
        "pred_var": "stack_lr2_pred",
        "proba_var": "stack_lr2_proba",
        "cm_var": "cm_stack_m2_lr",
        "cm_df_var": "cm_stack_m2_lr_df",
        "save_name": "model2_stack_lr_meta.pkl",
        "cm_png": "06_model2_stack_lr_confusion_matrix.png",
        "markdown": (
            "Logistic Regression is serving as the first meta-learner for Model 2. "
            "It is checking whether the invasive base-learner probabilities only need a simple linear combination. "
            "This helps show whether the benchmark signal is already very clean."
        ),
        "definition": """
            meta_lr2 = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            )
        """,
    },
    {
        "title": "Random Forest",
        "short": "Stack-RF",
        "var": "meta_rf2",
        "result_var": "results_stack_m2_rf",
        "pred_var": "stack_rf2_pred",
        "proba_var": "stack_rf2_proba",
        "cm_var": "cm_stack_m2_rf",
        "cm_df_var": "cm_stack_m2_rf_df",
        "save_name": "model2_stack_rf_meta.pkl",
        "cm_png": "07_model2_stack_rf_confusion_matrix.png",
        "markdown": (
            "Random Forest is serving as the second meta-learner for Model 2. "
            "It is checking whether the invasive benchmark benefits from a tree-based combination rule at the stacking level. "
            "This matters because the hormonal and ultrasound signals may interact in a non-linear way even after the base models have already learned from them."
        ),
        "definition": """
            meta_rf2 = RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )
        """,
    },
    {
        "title": "XGBoost",
        "short": "Stack-XGB",
        "var": "meta_xgb2",
        "result_var": "results_stack_m2_xgb",
        "pred_var": "stack_xgb2_pred",
        "proba_var": "stack_xgb2_proba",
        "cm_var": "cm_stack_m2_xgb",
        "cm_df_var": "cm_stack_m2_xgb_df",
        "save_name": "model2_stack_xgb_meta.pkl",
        "cm_png": "08_model2_stack_xgb_confusion_matrix.png",
        "markdown": (
            "XGBoost is serving as the third meta-learner for Model 2. "
            "It is testing whether a boosted second-level model can get the most value from the invasive base-learner outputs. "
            "If it performs best, it suggests the strongest final combination rule is still boosted and non-linear even after the first layer."
        ),
        "definition": """
            meta_xgb2 = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            )
        """,
    },
]


def add_meta_block(
    cells: list[dict],
    *,
    model_label: str,
    oof_var: str,
    y_train: str,
    test_preds: str,
    y_test: str,
    stack: dict,
) -> None:
    cells.append(
        markdown_cell(
            f"""
## {model_label}: Meta-Learner - {stack["title"]}

{stack["markdown"]}
"""
        )
    )

    cells.append(
        code_cell(
            f"""
            # Training {model_label} with the {stack["title"]} meta-learner is fitting the second-level model on the out-of-fold base predictions.
            {stack["definition"].strip()}
            {stack["var"]}.fit({oof_var}, {y_train})
            print("{stack['var']} trained successfully.")
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Evaluating the {stack["title"]} meta-learner on the stacked test features is producing the final test-set metrics for this stacking variant.
            {stack["proba_var"]} = {stack["var"]}.predict_proba({test_preds})[:, 1]
            {stack["pred_var"]} = ({stack["proba_var"]} >= 0.5).astype(int)
            {stack["result_var"]} = evaluate_model("{model_label} - {stack['short']}", {stack["var"]}, {test_preds}, {y_test})

            {stack["result_var"]}_df = pd.DataFrame([{stack["result_var"]}]).round(4)
            print({stack["result_var"]}_df.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the simple metric reading is making this stacking result easier to explain aloud.
            metric_markdown = "\\n".join([
                "### Simple Metric Reading",
                "",
                f"- Accuracy: **{{{stack['result_var']}['accuracy']:.4f}}**. This is the share of all test cases this stack model got right.",
                f"- Precision: **{{{stack['result_var']}['precision']:.4f}}**. This shows how often a positive stack prediction was correct.",
                f"- Recall: **{{{stack['result_var']}['recall']:.4f}}**. This shows how many true PCOS cases this stack model caught.",
                f"- F1 score: **{{{stack['result_var']}['f1']:.4f}}**. This is the balance between precision and recall.",
                f"- AUC: **{{{stack['result_var']}['auc']:.4f}}**. This shows how well this stack model separates PCOS and non-PCOS cases overall.",
                "",
                f"**Simple meaning:** {model_label} with {stack['title']} as the meta-learner is catching about **{{{stack['result_var']}['recall'] * 100:.1f}}%** of PCOS-positive cases in the test set.",
            ])
            display(Markdown(metric_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Printing the confusion matrix table is showing the exact counts behind the heatmap for this stacking variant.
            {stack["cm_var"]} = confusion_matrix({y_test}, {stack["pred_var"]})
            {stack["cm_df_var"]} = pd.DataFrame(
                {stack["cm_var"]},
                index=["Actual: No PCOS", "Actual: PCOS"],
                columns=["Predicted: No PCOS", "Predicted: PCOS"],
            )

            print("Confusion Matrix - {model_label} {stack['title']}:")
            print({stack["cm_df_var"]})
            print()
            print(f"True Positives  (caught PCOS cases) : {{{stack['cm_var']}[1, 1]}}")
            print(f"False Negatives (missed PCOS cases) : {{{stack['cm_var']}[1, 0]}}")
            print(f"False Positives (false alarms)      : {{{stack['cm_var']}[0, 1]}}")
            print(f"True Negatives  (correctly cleared) : {{{stack['cm_var']}[0, 0]}}")
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Plotting the confusion matrix is turning the count table into a simple visual summary for this stacking model.
            fig, ax = plt.subplots(figsize=(5.4, 4.4))
            sns.heatmap(
                {stack["cm_df_var"]},
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                linewidths=0.8,
                linecolor="white",
                ax=ax,
            )
            ax.set_title("Confusion Matrix - {model_label} {stack['title']} (Test Set)")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            output_path = IMAGE_DIR / "{stack['cm_png']}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {{output_path}}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Writing the confusion-matrix insight is explaining the error pattern in plain language.
            tp = int({stack["cm_var"]}[1, 1])
            fn = int({stack["cm_var"]}[1, 0])
            fp = int({stack["cm_var"]}[0, 1])
            tn = int({stack["cm_var"]}[0, 0])

            confusion_markdown = "\\n".join([
                "### Confusion Matrix Insight",
                "",
                f"- True positives: **{{tp}}**. These are PCOS cases the stack model caught correctly.",
                f"- False negatives: **{{fn}}**. These are real PCOS cases the stack model still missed.",
                f"- False positives: **{{fp}}**. These are false alarms where the stack model predicted PCOS but the case was negative.",
                f"- True negatives: **{{tn}}**. These are non-PCOS cases the stack model cleared correctly.",
                "",
                f"**Simple meaning:** For {model_label} with {stack['title']} as the meta-learner, the most important number is the false-negative count of **{{fn}}**, because missed PCOS cases matter most in a screening study.",
            ])
            display(Markdown(confusion_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            f"""
            # Saving the fitted meta-learner is keeping this stacking variant available for the optimization notebook.
            model_path = MODELS_DIR / "{stack['save_name']}"
            joblib.dump({stack["var"]}, model_path)
            print(f"Saved model to: {{model_path}}")
            """
        )
    )


def add_stacking_summary_block(
    cells: list[dict],
    *,
    model_label: str,
    model_set: str,
    stack_rows_name: str,
    result_vars: list[str],
    prob_vars: list[str],
    y_test: str,
    prev_filter: str,
    roc_png: str,
    compare_png: str,
) -> None:
    cells.append(
        markdown_cell(
            f"""
## {model_label}: Stacking Results Summary

This section is comparing the three meta-learner options inside {model_label}. The goal is finding the best stacking rule for this model set before building the master comparison table.
"""
        )
    )


def build_notebook() -> list[dict]:
    output_table = markdown_table(["Output File", "Purpose"], OUTPUT_ROWS)
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            f"""
# Ensemble Stacking - Combining Base Learners with a Meta-Learner

## Introduction
This notebook is building the stacking stage of the modelling workflow. The three individual base learners from Notebook 08 are already trained and saved. This notebook is now combining them inside two separate stacking pipelines:

- Model 1 stacking for the **non-invasive clinical model**
- Model 2 stacking for the **invasive benchmark model**

Stacking is using the predictions from the base learners as input features for a second model called the meta-learner. The idea is simple: one base learner may be strong for some cases, while another may be stronger for others. The meta-learner is learning how to combine their outputs in a better final rule.

This notebook is testing three meta-learner choices for each model set:

- Logistic Regression
- Random Forest
- XGBoost

That means this notebook is training and evaluating **six stacking models total**.

Out-of-fold (OOF) predictions are being used to train the meta-learners. This is important because it prevents leakage. Each training row is only receiving a base-learner prediction from models that did **not** train on that row. This keeps the stacking process honest.

No new data splitting, scaling, or SMOTE is being applied here. All preprocessing was already locked in Notebook 07. Notebook 09 is only building stacked models, evaluating them, comparing them against the individual Notebook 08 models, and saving the stacking artifacts for the next optimization step.

## Output Files Produced in This Notebook
{output_table}
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Section 1 - Imports and Configuration

This section is importing the libraries, locating the saved modelling artifacts, and preparing the notebook to load the locked files from Notebook 07 and Notebook 08.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Importing the stacking libraries is preparing the notebook for loading saved models, generating OOF predictions, training meta-learners, and saving figures.
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
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix,
                RocCurveDisplay
            )
            from sklearn.base import clone

            try:
                from xgboost import XGBClassifier
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "xgboost is required for Notebook 09. Install it in your current notebook environment with "
                    "`pip install xgboost` or `conda install -c conda-forge xgboost`, then restart the kernel and run the notebook again."
                ) from exc

            warnings.filterwarnings("ignore")
            sns.set_style("whitegrid")
            plt.rcParams["figure.dpi"] = 150
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

            RANDOM_STATE = 42
            TARGET = "pcos_y_n"

            COLORS = {
                "pcos_pos": "#E63946",
                "pcos_neg": "#457B9D",
                "model1": "#2A9D8F",
                "model2": "#E9C46A",
                "lr": "#264653",
                "rf": "#2A9D8F",
                "xgb": "#E76F51",
                "stack": "#9B2226",
                "improve": "#40916C",
                "decline": "#E63946",
                "neutral": "#2C3E50",
            }

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "cleaned_data").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            if not (PROJECT_ROOT / "cleaned_data").exists():
                raise FileNotFoundError("Project root could not be found from the current working directory.")

            DATA_DIR = PROJECT_ROOT / "cleaned_data" / "modelling_sets"
            MODELS_DIR = PROJECT_ROOT / "models"
            IMAGE_DIR = PROJECT_ROOT / "images" / "ensemble"
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Creating the ensemble image directory is making sure every stacking figure has a valid save location.
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Ensemble image directory ready: {IMAGE_DIR}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Loading the prepared modelling sets, feature contracts, CV object, and saved base learners is pulling in the locked Notebook 07 and Notebook 08 artifacts.
            train1 = pd.read_csv(DATA_DIR / "model1_train.csv")
            test1 = pd.read_csv(DATA_DIR / "model1_test.csv")
            train2 = pd.read_csv(DATA_DIR / "model2_train.csv")
            test2 = pd.read_csv(DATA_DIR / "model2_test.csv")

            with open(DATA_DIR / "feature_names_model1.json", encoding="utf-8") as file_obj:
                MODEL1_FEATURES = json.load(file_obj)
            with open(DATA_DIR / "feature_names_model2.json", encoding="utf-8") as file_obj:
                MODEL2_FEATURES = json.load(file_obj)

            cv = joblib.load(MODELS_DIR / "cv_strategy.pkl")

            lr1 = joblib.load(MODELS_DIR / "model1_lr.pkl")
            rf1 = joblib.load(MODELS_DIR / "model1_rf.pkl")
            xgb1 = joblib.load(MODELS_DIR / "model1_xgb.pkl")
            lr2 = joblib.load(MODELS_DIR / "model2_lr.pkl")
            rf2 = joblib.load(MODELS_DIR / "model2_rf.pkl")
            xgb2 = joblib.load(MODELS_DIR / "model2_xgb.pkl")

            prev_results = pd.read_csv(DATA_DIR / "individual_model_results.csv")

            assert len(MODEL1_FEATURES) == 19, f"Expected 19 Model 1 features, got {len(MODEL1_FEATURES)}"
            assert len(MODEL2_FEATURES) == 15, f"Expected 15 Model 2 features, got {len(MODEL2_FEATURES)}"

            print("Loaded modelling set shapes:")
            print(f"train1: {train1.shape}")
            print(f"test1 : {test1.shape}")
            print(f"train2: {train2.shape}")
            print(f"test2 : {test2.shape}")
            print()

            print(f"Model 1 feature count: {len(MODEL1_FEATURES)}")
            print(MODEL1_FEATURES)
            print()
            print(f"Model 2 feature count: {len(MODEL2_FEATURES)}")
            print(MODEL2_FEATURES)
            print()

            print("Loaded base learners successfully:")
            print("- model1_lr.pkl")
            print("- model1_rf.pkl")
            print("- model1_xgb.pkl")
            print("- model2_lr.pkl")
            print("- model2_rf.pkl")
            print("- model2_xgb.pkl")
            print()

            print(f"Cross-validation object: {cv}")
            print()
            print("Previous individual-model results baseline:")
            print(prev_results.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Separating features and targets is creating the exact training and test arrays used by the stacking pipeline.
            X1_train = train1[MODEL1_FEATURES].values
            y1_train = train1[TARGET].values
            X1_test = test1[MODEL1_FEATURES].values
            y1_test = test1[TARGET].values

            X2_train = train2[MODEL2_FEATURES].values
            y2_train = train2[TARGET].values
            X2_test = test2[MODEL2_FEATURES].values
            y2_test = test2[TARGET].values

            print("Feature matrix shapes:")
            print(f"X1_train: {X1_train.shape}")
            print(f"X1_test : {X1_test.shape}")
            print(f"X2_train: {X2_train.shape}")
            print(f"X2_test : {X2_test.shape}")
            print()

            for label, values in [
                ("y1_train", y1_train),
                ("y1_test", y1_test),
                ("y2_train", y2_train),
                ("y2_test", y2_test),
            ]:
                counts = pd.Series(values).value_counts().sort_index()
                print(label)
                print(counts.to_string())
                print()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Defining the evaluation helper is standardising how the notebook computes hold-out metrics for every stacking model.
            def evaluate_model(name, model, X_test, y_test):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                return {
                    "model": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="binary"),
                    "recall": recall_score(y_test, y_pred, average="binary"),
                    "f1": f1_score(y_test, y_pred, average="binary"),
                    "auc": roc_auc_score(y_test, y_prob),
                }
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Defining the OOF helper is generating leakage-safe base-learner probabilities for the meta-learner training stage.
            def generate_oof_probabilities(loaded_learners, X_train, y_train, splitter):
                oof_matrix = np.zeros((X_train.shape[0], len(loaded_learners)), dtype=float)
                fill_counts = np.zeros(X_train.shape[0], dtype=int)

                for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
                    X_fold_train = X_train[tr_idx]
                    y_fold_train = y_train[tr_idx]
                    X_fold_val = X_train[val_idx]

                    for col_idx, (name, fitted_model) in enumerate(loaded_learners):
                        fold_model = clone(fitted_model)
                        fold_model.fit(X_fold_train, y_fold_train)
                        oof_matrix[val_idx, col_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

                    fill_counts[val_idx] += 1
                    print(f"Fold {fold_idx} / {splitter.n_splits} complete")

                if not np.all(fill_counts == 1):
                    raise ValueError(
                        f"OOF assignment failed. Expected every row to be filled once, got counts: "
                        f"{np.unique(fill_counts, return_counts=True)}"
                    )

                return oof_matrix, fill_counts
            """
        )
    )

    cells.append(
        markdown_cell(
            """
## Section 2 - Out-of-Fold Prediction Generation

This section is building the base-learner probability matrices used to train the meta-learners. The key goal is preventing data leakage.

### Why OOF Predictions Are Required
If the meta-learner is trained on predictions made by base learners that already saw the same rows during fitting, the meta-learner is learning from over-optimistic signals. That is leakage. Out-of-fold predictions solve this problem. Each training row gets a probability from a base learner that was trained on other folds, not on that same row. This means the meta-learner is learning from predictions that were genuinely unseen at the moment they were made.

Another important caution is staying in place here. The training sets already came from Notebook 07 after SMOTE. This means the stacking stage is still using the locked balanced training sets. That is fine for consistency across the project, but it should be remembered when reading the training-side behavior.
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## OOF Generation - Model 1

This section is generating out-of-fold probability predictions for the non-invasive clinical model. The output is a matrix with three columns:

- Logistic Regression probability
- Random Forest probability
- XGBoost probability

These three columns become the training features for the Model 1 meta-learners.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Generating Model 1 OOF probabilities is creating the leakage-safe meta-learner training matrix from the three saved base learners.
            base_learners_1 = [
                ("LR", lr1),
                ("RF", rf1),
                ("XGB", xgb1),
            ]

            oof1, fill_counts1 = generate_oof_probabilities(base_learners_1, X1_train, y1_train, cv)

            assert oof1.shape == (X1_train.shape[0], 3), f"Unexpected Model 1 OOF shape: {oof1.shape}"
            assert np.all(fill_counts1 == 1), "Each Model 1 training row must be filled exactly once."

            print()
            print(f"OOF matrix shape : {oof1.shape}")
            print(f"Fill-count check : {pd.Series(fill_counts1).value_counts().sort_index().to_dict()}")
            print()
            print("OOF matrix - first 5 rows:")
            print(pd.DataFrame(oof1, columns=["LR_prob", "RF_prob", "XGB_prob"]).head())
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Saving the Model 1 OOF matrix is preserving the stacking training inputs for later inspection and reuse.
            model1_oof_path = MODELS_DIR / "model1_oof_predictions.npy"
            np.save(model1_oof_path, oof1)
            print(f"Saved OOF predictions to: {model1_oof_path}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing Model 1 OOF statistics is showing whether the three base learners separate positive and negative labels before the meta-learner is trained.
            oof1_df = pd.DataFrame(oof1, columns=["LR_prob", "RF_prob", "XGB_prob"])
            oof1_df["true_label"] = y1_train

            print("OOF Prediction Statistics - Model 1")
            print(oof1_df[["LR_prob", "RF_prob", "XGB_prob"]].describe().round(4))
            print()
            model1_oof_grouped = oof1_df.groupby("true_label")[["LR_prob", "RF_prob", "XGB_prob"]].mean().round(4)
            print("Mean OOF probability by true label:")
            print(model1_oof_grouped)
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the Model 1 OOF distributions is showing how the base learners score negative and positive cases before stacking.
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
            probability_columns = ["LR_prob", "RF_prob", "XGB_prob"]
            titles = ["Logistic Regression", "Random Forest", "XGBoost"]

            for ax, column, title in zip(axes, probability_columns, titles):
                sns.kdeplot(
                    data=oof1_df[oof1_df["true_label"] == 0],
                    x=column,
                    fill=True,
                    alpha=0.35,
                    color=COLORS["pcos_neg"],
                    label="No PCOS",
                    ax=ax,
                )
                sns.kdeplot(
                    data=oof1_df[oof1_df["true_label"] == 1],
                    x=column,
                    fill=True,
                    alpha=0.35,
                    color=COLORS["pcos_pos"],
                    label="PCOS",
                    ax=ax,
                )
                ax.axvline(0.5, linestyle="--", color="#6B7280", linewidth=1.0)
                ax.set_title(title)
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Density")
                ax.grid(alpha=0.20)

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles[:2], labels[:2], frameon=False)

            fig.suptitle("Model 1 - OOF Prediction Distributions by True PCOS Label", y=1.03)
            output_path = IMAGE_DIR / "01_model1_oof_prediction_distributions.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the Model 1 OOF insight is explaining whether the base learners are giving useful separation before stacking.
            model1_oof_markdown = "\\n".join([
                "### OOF Insight",
                "",
                "- These three panels are showing the out-of-fold probability distributions from the base learners.",
                "- Better separation means the positive cases are getting higher probabilities while the negative cases are staying lower.",
                "- If the two colours overlap less, the meta-learner will usually have an easier job.",
                "",
                "**Simple meaning:** This plot is checking whether the three base models are already giving useful signals that a stack model can learn from.",
            ])
            display(Markdown(model1_oof_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Building the Model 1 test prediction matrix is using the saved full-training base models directly to create meta-features for the hold-out test set.
            test_preds_1 = np.column_stack([
                lr1.predict_proba(X1_test)[:, 1],
                rf1.predict_proba(X1_test)[:, 1],
                xgb1.predict_proba(X1_test)[:, 1],
            ])

            test_preds_1_df = pd.DataFrame(test_preds_1, columns=["LR_prob", "RF_prob", "XGB_prob"])

            print("Model 1 test prediction matrix shape:", test_preds_1.shape)
            print()
            print(test_preds_1_df.head())
            """
        )
    )

    cells.append(
        markdown_cell(
            """
## OOF Generation - Model 2

This section is generating out-of-fold probability predictions for the invasive benchmark model. The same leakage-safe logic is being used here so the meta-learner is trained on genuinely unseen base-learner outputs.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Generating Model 2 OOF probabilities is creating the leakage-safe meta-learner training matrix for the invasive benchmark set.
            base_learners_2 = [
                ("LR", lr2),
                ("RF", rf2),
                ("XGB", xgb2),
            ]

            oof2, fill_counts2 = generate_oof_probabilities(base_learners_2, X2_train, y2_train, cv)

            assert oof2.shape == (X2_train.shape[0], 3), f"Unexpected Model 2 OOF shape: {oof2.shape}"
            assert np.all(fill_counts2 == 1), "Each Model 2 training row must be filled exactly once."

            print()
            print(f"OOF matrix shape : {oof2.shape}")
            print(f"Fill-count check : {pd.Series(fill_counts2).value_counts().sort_index().to_dict()}")
            print()
            print("OOF matrix - first 5 rows:")
            print(pd.DataFrame(oof2, columns=["LR_prob", "RF_prob", "XGB_prob"]).head())
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Saving the Model 2 OOF matrix is preserving the invasive stacking training inputs for later checks and reuse.
            model2_oof_path = MODELS_DIR / "model2_oof_predictions.npy"
            np.save(model2_oof_path, oof2)
            print(f"Saved OOF predictions to: {model2_oof_path}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing Model 2 OOF statistics is showing how strongly the invasive base learners separate the two labels before stacking.
            oof2_df = pd.DataFrame(oof2, columns=["LR_prob", "RF_prob", "XGB_prob"])
            oof2_df["true_label"] = y2_train

            print("OOF Prediction Statistics - Model 2")
            print(oof2_df[["LR_prob", "RF_prob", "XGB_prob"]].describe().round(4))
            print()
            model2_oof_grouped = oof2_df.groupby("true_label")[["LR_prob", "RF_prob", "XGB_prob"]].mean().round(4)
            print("Mean OOF probability by true label:")
            print(model2_oof_grouped)
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the Model 2 OOF distributions is checking how cleanly the invasive base learners separate positive and negative cases before the meta-learner stage.
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
            probability_columns = ["LR_prob", "RF_prob", "XGB_prob"]
            titles = ["Logistic Regression", "Random Forest", "XGBoost"]

            for ax, column, title in zip(axes, probability_columns, titles):
                sns.kdeplot(
                    data=oof2_df[oof2_df["true_label"] == 0],
                    x=column,
                    fill=True,
                    alpha=0.35,
                    color=COLORS["pcos_neg"],
                    label="No PCOS",
                    ax=ax,
                )
                sns.kdeplot(
                    data=oof2_df[oof2_df["true_label"] == 1],
                    x=column,
                    fill=True,
                    alpha=0.35,
                    color=COLORS["pcos_pos"],
                    label="PCOS",
                    ax=ax,
                )
                ax.axvline(0.5, linestyle="--", color="#6B7280", linewidth=1.0)
                ax.set_title(title)
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Density")
                ax.grid(alpha=0.20)

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles[:2], labels[:2], frameon=False)

            fig.suptitle("Model 2 - OOF Prediction Distributions by True PCOS Label", y=1.03)
            output_path = IMAGE_DIR / "02_model2_oof_prediction_distributions.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the Model 2 OOF insight is explaining whether the invasive base learners are giving strong stacking inputs.
            model2_oof_markdown = "\\n".join([
                "### OOF Insight",
                "",
                "- These panels are showing the out-of-fold probability distributions for the invasive base learners.",
                "- If the positive and negative curves separate more clearly here than in Model 1, that suggests the invasive feature set is giving cleaner base-learner signal.",
                "- If the overlap is still large, the meta-learner may still help by combining the three models better than any one model alone.",
                "",
                "**Simple meaning:** This plot is checking how much useful separation the invasive base models are already giving before stacking starts.",
            ])
            display(Markdown(model2_oof_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Building the Model 2 test prediction matrix is using the saved full-training invasive base models directly for the hold-out test set.
            test_preds_2 = np.column_stack([
                lr2.predict_proba(X2_test)[:, 1],
                rf2.predict_proba(X2_test)[:, 1],
                xgb2.predict_proba(X2_test)[:, 1],
            ])

            test_preds_2_df = pd.DataFrame(test_preds_2, columns=["LR_prob", "RF_prob", "XGB_prob"])

            print("Model 2 test prediction matrix shape:", test_preds_2.shape)
            print()
            print(test_preds_2_df.head())
            """
        )
    )

    cells.append(
        markdown_cell(
            """
## Section 3 - Model 1: Three Meta-Learner Variants

This section is training three different meta-learners on the Model 1 out-of-fold matrix. The base learners stay the same. Only the second-level combination rule changes.

The three meta-learner candidates are:

- Logistic Regression
- Random Forest
- XGBoost

The best performer by AUC and recall will be the Model 1 stacking candidate that moves forward to the later optimization notebook.
"""
        )
    )

    for stack in STACK_MODEL1:
        add_meta_block(
            cells,
            model_label="Model 1",
            oof_var="oof1",
            y_train="y1_train",
            test_preds="test_preds_1",
            y_test="y1_test",
            stack=stack,
        )

    add_stacking_summary_block(
        cells,
        model_label="Model 1",
        model_set="Model 1",
        stack_rows_name="m1_stack_results",
        result_vars=["results_stack_m1_lr", "results_stack_m1_rf", "results_stack_m1_xgb"],
        prob_vars=["stack_lr1_proba", "stack_rf1_proba", "stack_xgb1_proba"],
        y_test="y1_test",
        prev_filter="model_set == 'Model 1'",
        roc_png="09_model1_stacking_roc_curves.png",
        compare_png="11_model1_individual_vs_stack_comparison.png",
    )

    cells.append(
        markdown_cell(
            """
## Section 4 - Model 2: Three Meta-Learner Variants

This section is repeating the same stacking design for the invasive benchmark model. The question here is slightly different: which second-level combination rule gets the most value from the hormonal and ultrasound base-learner outputs?
"""
        )
    )

    for stack in STACK_MODEL2:
        add_meta_block(
            cells,
            model_label="Model 2",
            oof_var="oof2",
            y_train="y2_train",
            test_preds="test_preds_2",
            y_test="y2_test",
            stack=stack,
        )

    add_stacking_summary_block(
        cells,
        model_label="Model 2",
        model_set="Model 2",
        stack_rows_name="m2_stack_results",
        result_vars=["results_stack_m2_lr", "results_stack_m2_rf", "results_stack_m2_xgb"],
        prob_vars=["stack_lr2_proba", "stack_rf2_proba", "stack_xgb2_proba"],
        y_test="y2_test",
        prev_filter="model_set == 'Model 2'",
        roc_png="10_model2_stacking_roc_curves.png",
        compare_png="12_model2_individual_vs_stack_comparison.png",
    )

    cells.append(
        markdown_cell(
            """
## Section 5 - Master Stacking Comparison

This section is combining the six individual Notebook 08 results with the six new stacking results. This is giving the full performance picture before the later optimization stage.
"""
        )
    )

    cells.append(
        code_cell(
            """
            # Building the full master table is combining all individual and stacking results into one comparison table.
            individual = prev_results.copy()
            individual["approach"] = "Individual"

            stacking_rows = pd.DataFrame([
                results_stack_m1_lr,
                results_stack_m1_rf,
                results_stack_m1_xgb,
                results_stack_m2_lr,
                results_stack_m2_rf,
                results_stack_m2_xgb,
            ])
            stacking_rows["approach"] = "Stacking"
            stacking_rows["model_set"] = ["Model 1", "Model 1", "Model 1", "Model 2", "Model 2", "Model 2"]
            stacking_rows["algorithm"] = ["Stack-LR", "Stack-RF", "Stack-XGB", "Stack-LR", "Stack-RF", "Stack-XGB"]

            master = pd.concat([individual, stacking_rows], ignore_index=True).round(4)

            print("MASTER RESULTS TABLE - Individual + Stacking (All Models)")
            print("=" * 84)
            print(master.to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Saving the complete master results table is preserving the full comparison for later notebooks and reporting.
            master_results_path = DATA_DIR / "master_results_all_models.csv"
            master.to_csv(master_results_path, index=False)
            print(f"Saved master results to: {master_results_path}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Saving the stacking-only results table is preserving the six new stack-model rows separately.
            stacking_results_path = DATA_DIR / "stacking_results.csv"
            stacking_rows.to_csv(stacking_results_path, index=False)
            print(f"Saved stacking results to: {stacking_results_path}")
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the full heatmap input table is documenting the exact values before plotting the master comparison.
            heatmap_df = master.set_index("model")[["accuracy", "precision", "recall", "f1", "auc"]]
            print(heatmap_df.round(4).to_string())
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Creating the master heatmap is showing the full performance picture before optimization.
            fig, ax = plt.subplots(figsize=(8.4, 6.2))
            sns.heatmap(
                heatmap_df,
                annot=True,
                fmt=".4f",
                cmap="Greens",
                linewidths=0.6,
                linecolor="white",
                cbar_kws={"label": "Metric value"},
                ax=ax,
            )
            ax.axhline(6, color="black", linewidth=1.4)
            ax.set_title("Master Performance Heatmap - Individual Models and Stacking Variants")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Model")

            output_path = IMAGE_DIR / "13_master_stacking_heatmap.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the master heatmap insight is explaining what the full comparison view is showing.
            best_overall_row = master.sort_values("auc", ascending=False).iloc[0]
            heatmap_markdown = "\\n".join([
                "### Master Heatmap Insight",
                "",
                f"- The highest AUC in the full table is coming from **{best_overall_row['model']}** at **{best_overall_row['auc']:.4f}**.",
                "- Rows above the divider are the individual Notebook 08 models.",
                "- Rows below the divider are the new stacking variants from this notebook.",
                "",
                "**Simple meaning:** This heatmap is giving one full view of every model built so far, so it becomes easy to see whether stacking is really moving the project forward.",
            ])
            display(Markdown(heatmap_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the recall-improvement table is showing whether stacking improves the most clinically important metric for each model set.
            best_individual_recall = (
                master[master["approach"] == "Individual"]
                .sort_values(["model_set", "recall"], ascending=[True, False])
                .groupby("model_set")
                .head(1)
                .reset_index(drop=True)
            )
            best_stacking_recall = (
                master[master["approach"] == "Stacking"]
                .sort_values(["model_set", "recall"], ascending=[True, False])
                .groupby("model_set")
                .head(1)
                .reset_index(drop=True)
            )

            recall_compare = pd.DataFrame({
                "model_set": best_individual_recall["model_set"],
                "best_individual_model": best_individual_recall["model"],
                "best_individual_recall": best_individual_recall["recall"].values,
                "best_stacking_model": best_stacking_recall["model"].values,
                "best_stacking_recall": best_stacking_recall["recall"].values,
            })
            recall_compare["absolute_improvement"] = (
                recall_compare["best_stacking_recall"] - recall_compare["best_individual_recall"]
            )
            print(recall_compare.round(4).to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the recall-improvement chart is comparing the best individual recall against the best stacking recall for each model set.
            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            x = np.arange(len(recall_compare))
            width = 0.34

            individual_bars = ax.bar(
                x - width / 2,
                recall_compare["best_individual_recall"],
                width,
                color=COLORS["neutral"],
                label="Best Individual",
            )
            stack_colors = [COLORS["improve"] if value >= 0 else COLORS["decline"] for value in recall_compare["absolute_improvement"]]
            stack_bars = ax.bar(
                x + width / 2,
                recall_compare["best_stacking_recall"],
                width,
                color=stack_colors,
                label="Best Stacking",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(recall_compare["model_set"])
            ax.set_ylabel("Recall")
            ax.set_title("Recall - Best Individual vs Best Stacking Variant")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)

            for bars in [individual_bars, stack_bars]:
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            output_path = IMAGE_DIR / "14_recall_improvement_stacking.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the recall-improvement insight is explaining whether stacking is helping the project catch more PCOS cases.
            recall_markdown = "\\n".join([
                "### Recall Improvement Insight",
                "",
                f"- For Model 1, the recall change is **{recall_compare.loc[recall_compare['model_set'] == 'Model 1', 'absolute_improvement'].values[0]:.4f}**.",
                f"- For Model 2, the recall change is **{recall_compare.loc[recall_compare['model_set'] == 'Model 2', 'absolute_improvement'].values[0]:.4f}**.",
                "- Positive values mean stacking improved recall. Negative values mean the best individual model still caught more PCOS cases.",
                "",
                "**Simple meaning:** This chart is showing whether stacking is helping the study miss fewer real PCOS cases.",
            ])
            display(Markdown(recall_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Printing the AUC-improvement table is showing whether stacking improves the best overall class-separation metric for each model set.
            best_individual_auc = (
                master[master["approach"] == "Individual"]
                .sort_values(["model_set", "auc"], ascending=[True, False])
                .groupby("model_set")
                .head(1)
                .reset_index(drop=True)
            )
            best_stacking_auc = (
                master[master["approach"] == "Stacking"]
                .sort_values(["model_set", "auc"], ascending=[True, False])
                .groupby("model_set")
                .head(1)
                .reset_index(drop=True)
            )

            auc_compare = pd.DataFrame({
                "model_set": best_individual_auc["model_set"],
                "best_individual_model": best_individual_auc["model"].values,
                "best_individual_auc": best_individual_auc["auc"].values,
                "best_stacking_model": best_stacking_auc["model"].values,
                "best_stacking_auc": best_stacking_auc["auc"].values,
            })
            auc_compare["absolute_improvement"] = auc_compare["best_stacking_auc"] - auc_compare["best_individual_auc"]
            print(auc_compare.round(4).to_string(index=False))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Plotting the AUC-improvement chart is comparing the strongest individual and strongest stacking result in each model set.
            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            x = np.arange(len(auc_compare))
            width = 0.34

            individual_bars = ax.bar(
                x - width / 2,
                auc_compare["best_individual_auc"],
                width,
                color=COLORS["neutral"],
                label="Best Individual",
            )
            stack_colors = [COLORS["improve"] if value >= 0 else COLORS["decline"] for value in auc_compare["absolute_improvement"]]
            stack_bars = ax.bar(
                x + width / 2,
                auc_compare["best_stacking_auc"],
                width,
                color=stack_colors,
                label="Best Stacking",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(auc_compare["model_set"])
            ax.set_ylabel("AUC-ROC")
            ax.set_title("AUC - Best Individual vs Best Stacking Variant")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)

            for bars in [individual_bars, stack_bars]:
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            output_path = IMAGE_DIR / "15_auc_improvement_stacking.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {output_path}")
            plt.show()
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Writing the AUC-improvement insight is explaining whether stacking is improving overall class separation.
            auc_markdown = "\\n".join([
                "### AUC Improvement Insight",
                "",
                f"- For Model 1, the AUC change is **{auc_compare.loc[auc_compare['model_set'] == 'Model 1', 'absolute_improvement'].values[0]:.4f}**.",
                f"- For Model 2, the AUC change is **{auc_compare.loc[auc_compare['model_set'] == 'Model 2', 'absolute_improvement'].values[0]:.4f}**.",
                "- Positive values mean stacking improved overall separation. Negative values mean the best individual model still separated the classes better.",
                "",
                "**Simple meaning:** This chart is showing whether stacking is making the final prediction boundary clearer than the best single model.",
            ])
            display(Markdown(auc_markdown))
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Identifying the best stacking configuration per model set is selecting the variants that move forward to the later optimization notebook.
            m1_stacking = master[
                (master["model_set"] == "Model 1") &
                (master["approach"] == "Stacking")
            ]
            m2_stacking = master[
                (master["model_set"] == "Model 2") &
                (master["approach"] == "Stacking")
            ]

            best_m1_stack = m1_stacking.loc[m1_stacking["auc"].idxmax()]
            best_m2_stack = m2_stacking.loc[m2_stacking["auc"].idxmax()]

            print("Best stacking configuration for Model 1:")
            print(best_m1_stack.to_string())
            print()
            print("Best stacking configuration for Model 2:")
            print(best_m2_stack.to_string())
            print()
            print("These configurations will be carried into the optimization notebook.")
            """
        )
    )

    cells.append(
        code_cell(
            '''
            # Writing the key findings section is turning the full stacking comparison into simple thesis-ready result statements.
            best_m1_individual = master[
                (master["model_set"] == "Model 1") &
                (master["approach"] == "Individual")
            ].sort_values("auc", ascending=False).iloc[0]
            best_m2_individual = master[
                (master["model_set"] == "Model 2") &
                (master["approach"] == "Individual")
            ].sort_values("auc", ascending=False).iloc[0]

            m1_gap = best_m1_stack["auc"] - best_m1_individual["auc"]
            m2_gap = best_m2_stack["auc"] - best_m2_individual["auc"]
            cross_model_gap = best_m2_stack["auc"] - best_m1_stack["auc"]
            old_cross_model_gap = best_m2_individual["auc"] - best_m1_individual["auc"]

            findings_markdown = "\\n".join([
                "### Key Findings from Stacking Evaluation",
                "",
                f"For **Model 1**, the best individual model is **{best_m1_individual['model']}** with an AUC of **{best_m1_individual['auc']:.4f}**.",
                f"The best stacking model is **{best_m1_stack['model']}** with an AUC of **{best_m1_stack['auc']:.4f}**.",
                f"This is a change of **{m1_gap:.4f}** AUC points. This is showing whether stacking is helping the non-invasive model combine the base learners better than any single algorithm alone.",
                "",
                f"For **Model 2**, the best individual model is **{best_m2_individual['model']}** with an AUC of **{best_m2_individual['auc']:.4f}**.",
                f"The best stacking model is **{best_m2_stack['model']}** with an AUC of **{best_m2_stack['auc']:.4f}**.",
                f"This is a change of **{m2_gap:.4f}** AUC points. This is showing whether stacking is helping the invasive benchmark use the hormonal and ultrasound signal more effectively.",
                "",
                f"The best meta-learner for Model 1 is **{best_m1_stack['algorithm']}**, while the best meta-learner for Model 2 is **{best_m2_stack['algorithm']}**.",
                "This is revealing which kind of second-level rule is working best for each feature set. If a linear meta-learner wins, the base models only need a simple combination. If a tree-based or boosted meta-learner wins, the best final combination rule is more complex.",
                "",
                f"The AUC gap between the best Model 1 stack and the best Model 2 stack is **{cross_model_gap:.4f}**.",
                f"The earlier best-individual gap was **{old_cross_model_gap:.4f}**.",
                "This is showing whether stacking is closing the non-invasive versus invasive gap or keeping it mostly the same.",
                "",
                "### Simple Summary",
                "",
                "Stacking is now showing whether combining the three base learners is better than trusting the best single learner.",
                "If the gain is small, the individual model was already strong.",
                "If the gain is clear, stacking becomes the better model family to carry into the optimization notebook.",
            ])
            display(Markdown(findings_markdown))
            '''
        )
    )

    cells.append(
        code_cell(
            """
            # Running the final output verification is checking that every required stacking file exists and is not empty.
            required_files = [
                MODELS_DIR / "model1_stack_lr_meta.pkl",
                MODELS_DIR / "model1_stack_rf_meta.pkl",
                MODELS_DIR / "model1_stack_xgb_meta.pkl",
                MODELS_DIR / "model1_oof_predictions.npy",
                MODELS_DIR / "model2_stack_lr_meta.pkl",
                MODELS_DIR / "model2_stack_rf_meta.pkl",
                MODELS_DIR / "model2_stack_xgb_meta.pkl",
                MODELS_DIR / "model2_oof_predictions.npy",
                DATA_DIR / "stacking_results.csv",
                DATA_DIR / "master_results_all_models.csv",
                IMAGE_DIR / "01_model1_oof_prediction_distributions.png",
                IMAGE_DIR / "02_model2_oof_prediction_distributions.png",
                IMAGE_DIR / "03_model1_stack_lr_confusion_matrix.png",
                IMAGE_DIR / "04_model1_stack_rf_confusion_matrix.png",
                IMAGE_DIR / "05_model1_stack_xgb_confusion_matrix.png",
                IMAGE_DIR / "06_model2_stack_lr_confusion_matrix.png",
                IMAGE_DIR / "07_model2_stack_rf_confusion_matrix.png",
                IMAGE_DIR / "08_model2_stack_xgb_confusion_matrix.png",
                IMAGE_DIR / "09_model1_stacking_roc_curves.png",
                IMAGE_DIR / "10_model2_stacking_roc_curves.png",
                IMAGE_DIR / "11_model1_individual_vs_stack_comparison.png",
                IMAGE_DIR / "12_model2_individual_vs_stack_comparison.png",
                IMAGE_DIR / "13_master_stacking_heatmap.png",
                IMAGE_DIR / "14_recall_improvement_stacking.png",
                IMAGE_DIR / "15_auc_improvement_stacking.png",
            ]

            all_passed = True
            for path in required_files:
                exists = path.exists()
                size_ok = exists and path.stat().st_size > 0
                passed = exists and size_ok
                if not passed:
                    all_passed = False
                status = "PASS" if passed else "FAIL"
                size_text = path.stat().st_size if exists else 0
                print(f"[{status}] {path} ({size_text} bytes)")

            print()
            if all_passed:
                print("All 25 stacking output files verified successfully.")
            else:
                print("WARNING: Some stacking output files are missing or empty. Review the cells above.")
            """
        )
    )

    cells.append(
        code_cell(
            '''
            # Writing the closing summary is documenting what this notebook completed and what the next notebook will do.
            closing_markdown = "\\n".join([
                "### Notebook Summary and Next Steps",
                "",
                "This notebook has now trained and saved **six stacking models**:",
                "",
                "- `models/model1_stack_lr_meta.pkl`",
                "- `models/model1_stack_rf_meta.pkl`",
                "- `models/model1_stack_xgb_meta.pkl`",
                "- `models/model2_stack_lr_meta.pkl`",
                "- `models/model2_stack_rf_meta.pkl`",
                "- `models/model2_stack_xgb_meta.pkl`",
                "",
                "It has also saved the out-of-fold matrices, the stacking-only results table, the full master results table, and all ensemble figures.",
                "",
                f"The current best stacking configuration for **Model 1** is **{best_m1_stack['model']}** with an AUC of **{best_m1_stack['auc']:.4f}** and a recall of **{best_m1_stack['recall']:.4f}**.",
                f"The current best stacking configuration for **Model 2** is **{best_m2_stack['model']}** with an AUC of **{best_m2_stack['auc']:.4f}** and a recall of **{best_m2_stack['recall']:.4f}**.",
                "",
                "This notebook is also showing whether stacking is really improving on the best individual models or only matching them. That answer becomes the direct starting point for the next step.",
                "",
                "Notebook 10 will apply metaheuristic optimization to the best stacking configuration for each model set. It will tune the hyperparameters using the selected optimizer workflows, compare convergence behavior, and choose the strongest final thesis model for each feature set.",
            ])
            display(Markdown(closing_markdown))
            '''
        )
    )

    return cells


def main() -> None:
    cells = build_notebook()
    write_notebook(NOTEBOOK_PATH, cells)
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
