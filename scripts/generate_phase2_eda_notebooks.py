from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "EDA"


def normalize_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    if normalized:
        normalized += "\n"
    return normalized.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": normalize_source(text),
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


def question_markdown(number: int, question: str, lead: str | None = None) -> dict:
    lead_text = lead or "This analysis is examining the cleaned dataset through a table-first and plot-second workflow."
    return markdown_cell(
        f"""
        ## Question {number}

        ### {question}

        {lead_text}
        """
    )


def insight_markdown(text: str) -> dict:
    return markdown_cell(
        f"""
        ### Insight

        {text}
        """
    )


def build_clinical_notebook() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            """
            # PCOS Clinical EDA and Phenotype Profiling

            ## Introduction
            This notebook is performing the flagship exploratory data analysis for the cleaned clinical PCOS dataset. The analysis is focusing on the clinical phenotype of PCOS, with special attention to features that may later support accessible and low-burden screening.

            The notebook is using `cleaned_data/PCOS_full_cleaned.csv` as the primary analytical source. The clinical table is containing both routine variables and selected invasive markers, so the notebook is prioritizing non-invasive signals first and then using ovarian and morphological variables only where they are helping the biological interpretation.

            ## Research Context
            The broader MSc study is investigating whether PCOS can be profiled and later predicted using routine clinical, symptom-based, and low-burden indicators. This notebook is therefore asking structured research questions about body composition, menstrual pattern variables, visible symptom clusters, cardiometabolic proxies, and ovarian morphology.

            Every analytical question in this notebook is following the same pattern:
            - a markdown question is introducing the analysis
            - a summary table is being printed before the visual
            - a figure is then being generated and saved to `images/eda/clinical/`
            - an insight cell is then documenting why the pattern matters clinically

            The notebook is not performing model training, heart-dataset comparison, optimization, or explainability work. It is only building the clinical evidence base that will support those later phases.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Reproducibility Setup and Path Configuration

            This section is importing the required libraries, resolving the project root safely, and preparing the figure directory for this clinical EDA workflow.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Importing the libraries is supporting reproducible data handling, plotting, and notebook display.
            from pathlib import Path
            import io

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import display

            # Fixing the random seed is keeping any sampling or jitter behavior stable across reruns.
            np.random.seed(42)

            # Configuring the plotting theme is keeping the visuals readable and consistent across the notebook.
            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.dpi"] = 120
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", 200)

            # Resolving the project root is keeping the notebook runnable from the repository root or the notebook folder.
            def resolve_project_root() -> Path:
                current = Path.cwd().resolve()
                candidates = [current, current.parent, current.parent.parent]
                for candidate in candidates:
                    if (candidate / "cleaned_data").exists() and (candidate / "images").exists():
                        return candidate
                return current

            PROJECT_ROOT = resolve_project_root()
            DATA_PATH = PROJECT_ROOT / "cleaned_data" / "PCOS_full_cleaned.csv"
            IMAGE_DIR = PROJECT_ROOT / "images" / "eda" / "clinical"
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            # Defining the palette is locking the target-color identity for the full PCOS EDA series.
            PCOS_LABEL_ORDER = ["PCOS Negative", "PCOS Positive"]
            PCOS_LABEL_PALETTE = {
                "PCOS Negative": "#3b82f6",
                "PCOS Positive": "#e76f51",
            }
            ACCENT_PALETTE = ["#2a9d8f", "#e9c46a", "#6d597a", "#264653"]

            # Saving figures through a helper is keeping the export path consistent and descriptive.
            def save_figure(fig: plt.Figure, slug: str) -> Path:
                output_path = IMAGE_DIR / slug
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                return output_path

            # Building grouped numeric summaries is standardizing the table-first workflow used throughout the notebook.
            def grouped_numeric_summary(data: pd.DataFrame, feature: str) -> pd.DataFrame:
                summary = (
                    data.groupby("pcos_label")[feature]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .rename(columns={"count": "n"})
                    .round(3)
                    .reset_index()
                )
                return summary

            # Building grouped prevalence summaries is supporting the binary symptom analyses.
            def grouped_binary_prevalence(data: pd.DataFrame, feature: str) -> pd.DataFrame:
                summary = (
                    data.groupby("pcos_label")[feature]
                    .agg(["count", "sum", "mean"])
                    .rename(columns={"count": "n", "sum": "positive_count", "mean": "prevalence"})
                    .reset_index()
                )
                summary["prevalence_pct"] = (summary["prevalence"] * 100).round(1)
                return summary[["pcos_label", "n", "positive_count", "prevalence_pct"]]

            # Plotting numeric distributions through one helper is keeping the comparison style consistent across questions.
            def plot_numeric_by_target(
                data: pd.DataFrame,
                feature: str,
                ylabel: str,
                title: str,
                slug: str,
                kind: str = "box",
            ) -> None:
                fig, ax = plt.subplots(figsize=(8, 5))
                if kind == "violin":
                    sns.violinplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        palette=PCOS_LABEL_PALETTE,
                        cut=0,
                        inner=None,
                        ax=ax,
                    )
                    sns.stripplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        color="#264653",
                        alpha=0.30,
                        size=3,
                        jitter=0.20,
                        ax=ax,
                    )
                elif kind == "kde":
                    for label in PCOS_LABEL_ORDER:
                        subset = data.loc[data["pcos_label"] == label, feature].dropna()
                        sns.kdeplot(
                            subset,
                            fill=True,
                            alpha=0.24,
                            linewidth=2,
                            label=label,
                            color=PCOS_LABEL_PALETTE[label],
                            ax=ax,
                        )
                    ax.legend(frameon=True)
                else:
                    sns.boxplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        palette=PCOS_LABEL_PALETTE,
                        ax=ax,
                    )
                    sns.stripplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        color="#264653",
                        alpha=0.30,
                        size=3,
                        jitter=0.20,
                        ax=ax,
                    )
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                save_figure(fig, slug)
                plt.show()

            # Plotting prevalence bars through one helper is simplifying the repeated binary-feature visuals.
            def plot_binary_prevalence(summary: pd.DataFrame, title: str, slug: str, ylabel: str = "Prevalence (%)") -> None:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(
                    data=summary,
                    x="pcos_label",
                    y="prevalence_pct",
                    order=PCOS_LABEL_ORDER,
                    palette=PCOS_LABEL_PALETTE,
                    ax=ax,
                )
                for patch in ax.patches:
                    height = patch.get_height()
                    ax.annotate(
                        f"{height:.1f}%",
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        xytext=(0, 6),
                        textcoords="offset points",
                    )
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                save_figure(fig, slug)
                plt.show()
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Data Loading and Feature Preparation

            This section is loading the cleaned clinical table, verifying that the expected columns are present, and creating a small set of derived features that will support later EDA questions.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Loading the cleaned clinical dataset is bringing the analysis-ready table into memory for EDA.
            df = pd.read_csv(DATA_PATH)

            # Verifying the expected schema is protecting the notebook from silent upstream changes.
            required_columns = [
                "pcos_y_n",
                "age_yrs",
                "weight_kg",
                "height_cm",
                "bmi",
                "pulse_rate_bpm",
                "respiratory_rate_breaths_min",
                "hb_g_dl",
                "cycle_regularity_code",
                "cycle_length_days",
                "waist_hip_ratio",
                "rbs_mg_dl",
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
                "systolic_bp_mmhg",
                "diastolic_bp_mmhg",
                "follicle_no_left",
                "follicle_no_right",
                "avg_follicle_size_left_mm",
                "avg_follicle_size_right_mm",
                "endometrium_mm",
            ]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing expected columns: {missing_columns}")

            # Standardizing the target and binary fields is keeping the plotting logic explicit and stable.
            binary_columns = [
                "pcos_y_n",
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
            ]
            for column in binary_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

            numeric_columns = [column for column in required_columns if column not in binary_columns]
            for column in numeric_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            # Creating readable label columns is improving the presentation of repeated plots.
            df["pcos_label"] = df["pcos_y_n"].map({0: "PCOS Negative", 1: "PCOS Positive"})
            for column in [
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
            ]:
                df[f"{column}_label"] = df[column].map({0: "No", 1: "Yes"})

            # Creating derived features is supporting the later research questions without modifying the saved cleaned table.
            df["total_follicle_count"] = df["follicle_no_left"] + df["follicle_no_right"]
            df["bmi_category"] = pd.cut(
                df["bmi"],
                bins=[0, 18.5, 25.0, 30.0, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                include_lowest=True,
            )
            df["bmi_overweight_flag"] = (df["bmi"] >= 25).astype(int)
            df["low_exercise_flag"] = (1 - df["regular_exercise_y_n"]).astype(int)
            df["non_invasive_burden_score"] = (
                df["weight_gain_y_n"]
                + df["hair_growth_y_n"]
                + df["skin_darkening_y_n"]
                + df["hair_loss_y_n"]
                + df["pimples_y_n"]
                + df["fast_food_y_n"]
                + df["bmi_overweight_flag"]
                + df["low_exercise_flag"]
            )

            # Displaying a compact preview is confirming that the derived fields are present before the question bank starts.
            display(df.head())
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Dataset Overview and Target Distribution

            This section is establishing the analytical baseline for the notebook before the feature-by-feature profiling begins.
            """
        )
    )

    cells.append(question_markdown(1, "What is the cleaned clinical dataset size, schema, and target balance?"))
    cells.append(
        code_cell(
            """
            # Building the overview tables is summarizing the structure and target balance before plotting.
            overview_q01 = pd.DataFrame(
                {
                    "metric": ["row_count", "column_count", "numeric_column_count", "missing_cells"],
                    "value": [
                        df.shape[0],
                        df.shape[1],
                        int(df.select_dtypes(include=np.number).shape[1]),
                        int(df.isna().sum().sum()),
                    ],
                }
            )
            schema_q01 = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
            target_q01 = (
                df["pcos_label"]
                .value_counts()
                .reindex(PCOS_LABEL_ORDER)
                .rename_axis("pcos_label")
                .reset_index(name="count")
            )
            target_q01["percentage"] = (100 * target_q01["count"] / target_q01["count"].sum()).round(1)

            display(overview_q01)
            display(schema_q01)
            display(target_q01)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the target distribution is showing the class balance that later modeling work will inherit.
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(
                data=target_q01,
                x="pcos_label",
                y="count",
                order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                ax=ax,
            )
            for patch in ax.patches:
                height = patch.get_height()
                ax.annotate(
                    f"{int(height)}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    xytext=(0, 6),
                    textcoords="offset points",
                )
            ax.set_title("Question 1: Cleaned Clinical PCOS Target Distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Participant Count")
            save_figure(fig, "q01_clinical_target_distribution.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The cleaned clinical dataset is containing 541 participants, with 364 PCOS-negative cases and 177 PCOS-positive cases. This class balance is not being extreme, but it is still making false negatives clinically important because each missed positive case represents a participant who may remain without timely screening attention. The target distribution is therefore providing context for every later interpretation in this notebook.
            """
        )
    )

    cells.append(question_markdown(2, "What is the mean and median phenotype profile of PCOS-positive versus PCOS-negative participants across core clinical features?"))
    cells.append(
        code_cell(
            """
            # Building a wide phenotype summary is comparing the central tendency of the core clinical variables by PCOS status.
            core_profile_features = [
                "age_yrs",
                "weight_kg",
                "bmi",
                "waist_hip_ratio",
                "pulse_rate_bpm",
                "respiratory_rate_breaths_min",
                "hb_g_dl",
                "cycle_length_days",
                "systolic_bp_mmhg",
                "diastolic_bp_mmhg",
                "rbs_mg_dl",
                "total_follicle_count",
            ]
            profile_mean_q02 = df.groupby("pcos_label")[core_profile_features].mean().round(3).T
            profile_median_q02 = df.groupby("pcos_label")[core_profile_features].median().round(3).T
            profile_q02 = pd.concat({"mean": profile_mean_q02, "median": profile_median_q02}, axis=1)
            display(profile_q02)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the phenotype heatmap is showing where the group-level clinical profile is shifting most clearly.
            fig, ax = plt.subplots(figsize=(9, 8))
            sns.heatmap(
                profile_mean_q02,
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                linewidths=0.5,
                cbar_kws={"label": "Group Mean"},
                ax=ax,
            )
            ax.set_title("Question 2: Group Mean Clinical Phenotype Profile")
            ax.set_xlabel("")
            ax.set_ylabel("Feature")
            save_figure(fig, "q02_core_phenotype_heatmap.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The phenotype profile is already showing that the clearest upward shifts in the PCOS-positive group are clustering around weight, BMI, symptom-linked burden, and follicle counts, while some general vital signs are moving much less. This pattern is suggesting that body composition, menstrual disruption, and ovarian morphology are carrying more discriminatory signal than routine hemodynamic measures alone.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Non-Invasive Continuous Feature Analysis

            This section is profiling the continuous features that can contribute to low-burden or routine PCOS screening.
            """
        )
    )

    numeric_questions = [
        (3, "Does age differ by PCOS status?", "age_yrs", "Age (years)", "q03_age_vs_pcos_boxplot.png", "This figure is showing whether the age distribution is shifting materially between the two PCOS groups. The small separation already visible in the cleaned cohort is suggesting that age alone is not behaving like a strong standalone discriminator, which matters because it should probably act as a context feature rather than a dominant screening signal."),
        (4, "Does weight differ by PCOS status?", "weight_kg", "Weight (kg)", "q04_weight_vs_pcos_boxplot.png", "The weight distribution is showing a visible upward shift in the PCOS-positive cohort. That pattern is supporting the idea that higher body mass is clustering with PCOS in this sample, which matters because weight is easy to obtain and may strengthen non-invasive screening when combined with more specific symptoms."),
        (5, "Does BMI differ by PCOS status?", "bmi", "Body Mass Index", "q05_bmi_vs_pcos_boxplot.png", "BMI is showing one of the clearer non-invasive shifts in the cleaned clinical cohort, with the PCOS-positive group carrying a higher central tendency. This matters clinically because adiposity is often traveling with insulin resistance and ovulatory dysfunction, making BMI a strong candidate for later routine-feature modeling."),
        (6, "Does waist-hip ratio differ by PCOS status?", "waist_hip_ratio", "Waist-Hip Ratio", "q06_waist_hip_ratio_vs_pcos_boxplot.png", "Waist-hip ratio is showing only a modest separation relative to BMI, which is suggesting that this anthropometric marker may add nuance but not dominate the phenotype story by itself. It is still worth retaining because central fat distribution can signal metabolic stress even when total BMI is only moderately elevated."),
        (7, "Does pulse rate differ by PCOS status?", "pulse_rate_bpm", "Pulse Rate (bpm)", "q07_pulse_rate_vs_pcos_boxplot.png", "Pulse rate is showing little visual separation between the two groups, which is suggesting a weaker direct relationship with PCOS in this cohort. This matters because weak vital-sign movement can help us avoid overvaluing features that are easy to collect but clinically noisy."),
        (8, "Does respiratory rate differ by PCOS status?", "respiratory_rate_breaths_min", "Respiratory Rate (breaths/min)", "q08_respiratory_rate_vs_pcos_boxplot.png", "Respiratory rate is appearing fairly stable across groups, which is suggesting that it is not a major PCOS phenotype marker in this dataset. That weak movement is useful in itself because it helps distinguish broad physiological context variables from more syndrome-specific features."),
        (9, "Does hemoglobin differ by PCOS status?", "hb_g_dl", "Hemoglobin (g/dL)", "q09_hb_vs_pcos_boxplot.png", "Hemoglobin is showing only a light shift between groups, so it is likely acting as a background physiological marker rather than a headline discriminator. This pattern is encouraging a cautious interpretation of blood indices unless they meaningfully improve multivariable performance later."),
        (10, "Does the dataset's recorded cycle-length measure differ by PCOS status?", "cycle_length_days", "Recorded Cycle-Length Measure", "q10_cycle_length_measure_vs_pcos_boxplot.png", "The recorded cycle-length measure is shifting across PCOS status, but it is behaving like a short ordinal clinical record rather than a literal 21-to-35-day cycle field. This still matters because menstrual-pattern recording is clinically relevant, but the variable should be interpreted as a recorded measure rather than as a textbook day count."),
        (11, "Does cycle_regularity_code differ by PCOS status?", "cycle_regularity_code", "Cycle Regularity Code", "q11_cycle_regularity_code_vs_pcos_boxplot.png", "The cycle-regularity code is showing a stronger group shift than many of the routine vital signs, which is reinforcing menstrual irregularity as a core PCOS signal. Because the coding scheme is ordinal and not yet mapped to human labels, it should be handled as a recorded code until a documented mapping is introduced."),
        (12, "Does systolic blood pressure differ by PCOS status?", "systolic_bp_mmhg", "Systolic Blood Pressure (mmHg)", "q12_systolic_bp_vs_pcos_boxplot.png", "Systolic blood pressure is showing only a limited shift between groups in the clinical cohort. This is suggesting that blood pressure may contribute more to cardiometabolic context than to direct PCOS discrimination when used alone."),
        (13, "Does diastolic blood pressure differ by PCOS status?", "diastolic_bp_mmhg", "Diastolic Blood Pressure (mmHg)", "q13_diastolic_bp_vs_pcos_boxplot.png", "Diastolic blood pressure is remaining tightly clustered across the two groups, which is again pointing to a weak standalone relationship with PCOS status in this dataset. This observation will still matter later when the cardiometabolic comparison notebook is examining broader risk context."),
        (14, "Does random blood sugar differ by PCOS status?", "rbs_mg_dl", "Random Blood Sugar (mg/dL)", "q14_rbs_vs_pcos_boxplot.png", "Random blood sugar is showing a slight upward shift and a wider upper tail in the PCOS-positive group. That pattern is clinically relevant because glycemic stress is often co-traveling with PCOS, even when blood sugar alone is not cleanly separating the groups."),
    ]

    for number, question, feature, ylabel, slug, insight in numeric_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped summary is quantifying the distribution of `{feature}` before plotting.
                summary_q{number:02d} = grouped_numeric_summary(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped distribution is showing how `{feature}` is shifting across PCOS status.
                plot_numeric_by_target(
                    data=df,
                    feature="{feature}",
                    ylabel="{ylabel}",
                    title="Question {number}: {question}",
                    slug="{slug}",
                    kind="box",
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(
        markdown_cell(
            """
            ## Binary Symptom and Behavior Feature Analysis

            This section is profiling the self-reported and clinically observed binary features that may be especially valuable for low-cost screening.
            """
        )
    )

    binary_questions = [
        (15, "Does weight gain prevalence differ by PCOS status?", "weight_gain_y_n", "Weight Gain", "q15_weight_gain_prevalence.png", "The weight-gain signal is showing a large prevalence gap between the two groups, with the PCOS-positive cohort carrying a much higher burden. This matters because a strong symptom-prevalence shift in an easy-to-report feature is valuable for accessible screening workflows."),
        (16, "Does hair growth prevalence differ by PCOS status?", "hair_growth_y_n", "Hair Growth", "q16_hair_growth_prevalence.png", "Hair growth is showing a pronounced prevalence difference, which is supporting its relevance as an androgen-linked symptom marker. This is important because visible hyperandrogenic features can contribute strong non-invasive signal when laboratory testing is not immediately available."),
        (17, "Does skin darkening prevalence differ by PCOS status?", "skin_darkening_y_n", "Skin Darkening", "q17_skin_darkening_prevalence.png", "Skin darkening is showing one of the sharpest prevalence differences in the clinical dataset. That pattern is clinically meaningful because acanthosis-like darkening can reflect insulin resistance and metabolic burden, both of which frequently cluster with PCOS."),
        (18, "Does hair loss prevalence differ by PCOS status?", "hair_loss_y_n", "Hair Loss", "q18_hair_loss_prevalence.png", "Hair loss is showing a noticeable but more moderate prevalence gap than some of the stronger symptom variables. This suggests that it may still contribute useful signal, although it is likely to be less specific than hair growth or skin darkening."),
        (19, "Does pimples prevalence differ by PCOS status?", "pimples_y_n", "Pimples", "q19_pimples_prevalence.png", "Pimples are showing a clear upward prevalence shift in the PCOS-positive group. This matters because acne-related features often reflect androgen activity, but the signal may overlap with other visible symptoms and therefore needs redundancy checking later in the notebook."),
        (20, "Does fast-food behavior differ by PCOS status?", "fast_food_y_n", "Fast Food", "q20_fast_food_prevalence.png", "Fast-food behavior is showing a strong prevalence difference, which may be reflecting a broader lifestyle or metabolic risk pattern rather than a syndrome-specific biological marker. This makes the variable interesting, but it should be interpreted carefully because behavior variables can be confounded by social and reporting effects."),
        (21, "Does regular exercise differ by PCOS status?", "regular_exercise_y_n", "Regular Exercise", "q21_regular_exercise_prevalence.png", "Regular exercise is showing only a limited difference between groups in this cohort. That weaker separation suggests it may be more useful as a contextual lifestyle modifier than as a primary screening feature by itself."),
    ]

    for number, question, feature, label, slug, insight in binary_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped prevalence table is quantifying how often `{feature}` is present in each PCOS group.
                summary_q{number:02d} = grouped_binary_prevalence(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped prevalence is showing how `{feature}` is concentrating by PCOS status.
                plot_binary_prevalence(
                    summary=summary_q{number:02d},
                    title="Question {number}: {label} Prevalence by PCOS Status",
                    slug="{slug}",
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(
        markdown_cell(
            """
            ## Reproductive, Cardiometabolic, and Ovarian Structure Questions

            This section is extending the clinical EDA into ovarian morphology and other biologically relevant markers that deepen the phenotype story.
            """
        )
    )

    ovarian_questions = [
        (22, "Do follicle counts on the left ovary differ by PCOS status?", "follicle_no_left", "Left Follicle Count", "q22_follicle_left_vs_pcos_boxplot.png", "Left-ovary follicle counts are showing a marked upward shift in the PCOS-positive group. This is clinically coherent because increased follicle burden is closely tied to the ovarian morphology associated with PCOS."),
        (23, "Do follicle counts on the right ovary differ by PCOS status?", "follicle_no_right", "Right Follicle Count", "q23_follicle_right_vs_pcos_boxplot.png", "Right-ovary follicle counts are also showing a strong upward shift among PCOS-positive participants. Seeing the pattern on both sides is strengthening the interpretation that ovarian morphology is one of the clearest differentiators in the cleaned clinical cohort."),
        (24, "Do average follicle sizes differ by PCOS status?", "avg_follicle_size_right_mm", "Average Right Follicle Size (mm)", "q24_avg_follicle_size_vs_pcos_violin.png", "Average follicle size is showing a more modest group separation than follicle count, which is suggesting that count may be the more informative ovarian measure here. This matters because it helps prioritize which invasive features are truly contributing unique signal."),
        (25, "Does endometrium thickness differ by PCOS status?", "endometrium_mm", "Endometrium Thickness (mm)", "q25_endometrium_vs_pcos_boxplot.png", "Endometrium thickness is showing only a moderate shift between groups, which suggests it may offer supplementary biological context rather than headline discrimination. This is useful for interpretation, but it should likely sit below symptom burden and follicle count in later priority lists."),
    ]

    for number, question, feature, ylabel, slug, insight in ovarian_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped summary is quantifying `{feature}` by PCOS status before plotting.
                summary_q{number:02d} = grouped_numeric_summary(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        plot_kind = "violin" if number == 24 else "box"
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped distribution is showing how `{feature}` is moving across the PCOS classes.
                plot_numeric_by_target(
                    data=df,
                    feature="{feature}",
                    ylabel="{ylabel}",
                    title="Question {number}: {question}",
                    slug="{slug}",
                    kind="{plot_kind}",
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    linkage_questions = [
        (26, "Are follicle counts higher among participants with skin darkening?", "skin_darkening_y_n", "Skin Darkening", "q26_total_follicles_by_skin_darkening.png", "This figure is showing whether visible metabolic-skin changes are clustering with a heavier follicle burden. A higher follicle count among participants with skin darkening would strengthen the link between external symptom expression and deeper ovarian phenotype."),
        (27, "Are follicle counts higher among participants with hair growth?", "hair_growth_y_n", "Hair Growth", "q27_total_follicles_by_hair_growth.png", "This comparison is showing whether androgen-linked hair growth is traveling with higher follicle burden. If the follicle distribution is shifting upward with hair growth, the symptom is reinforcing its role as a clinically meaningful surface marker of a deeper PCOS phenotype."),
        (28, "Are follicle counts higher among participants with weight gain?", "weight_gain_y_n", "Weight Gain", "q28_total_follicles_by_weight_gain.png", "This question is linking metabolic burden and ovarian morphology directly. A heavier follicle burden among participants who report weight gain would support the idea that body-composition stress and ovarian change are clustering in the same high-risk subgroup."),
    ]

    for number, question, feature, feature_label, slug, insight in linkage_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped follicle summary is showing how ovarian burden changes across `{feature}` and PCOS status.
                summary_q{number:02d} = (
                    df.groupby(["pcos_label", "{feature}"])["total_follicle_count"]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .rename(columns={{"count": "n"}})
                    .round(3)
                    .reset_index()
                )
                summary_q{number:02d}["{feature}_label"] = summary_q{number:02d}["{feature}"].map({{0: "No", 1: "Yes"}})
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the follicle burden by symptom status is showing whether the symptom is aligning with ovarian morphology.
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(
                    data=df,
                    x="{feature}_label",
                    y="total_follicle_count",
                    hue="pcos_label",
                    order=["No", "Yes"],
                    hue_order=PCOS_LABEL_ORDER,
                    palette=PCOS_LABEL_PALETTE,
                    ax=ax,
                )
                ax.set_title("Question {number}: Total Follicle Count by {feature_label} and PCOS Status")
                ax.set_xlabel("{feature_label} Reported")
                ax.set_ylabel("Total Follicle Count")
                ax.legend(title="")
                save_figure(fig, "{slug}")
                plt.show()
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(
        markdown_cell(
            """
            ## Joint Shift and Redundancy Analysis

            This section is examining multivariate movement and feature overlap so that later modeling work can prioritize strong but non-redundant signals.
            """
        )
    )

    cells.append(question_markdown(29, "Are BMI and waist-hip ratio jointly shifted in the PCOS-positive cohort?"))
    cells.append(
        code_cell(
            """
            # Building the joint summary is quantifying the anthropometric shift before drawing the two-dimensional scatter.
            summary_q29 = (
                df.groupby("pcos_label")[["bmi", "waist_hip_ratio"]]
                .agg(["mean", "median", "std"])
                .round(3)
            )
            display(summary_q29)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the joint anthropometric space is showing where the two groups are clustering together and apart.
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df,
                x="bmi",
                y="waist_hip_ratio",
                hue="pcos_label",
                hue_order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                alpha=0.75,
                s=70,
                ax=ax,
            )
            ax.set_title("Question 29: BMI and Waist-Hip Ratio by PCOS Status")
            ax.set_xlabel("Body Mass Index")
            ax.set_ylabel("Waist-Hip Ratio")
            ax.legend(title="")
            save_figure(fig, "q29_bmi_waist_hip_joint_scatter.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The anthropometric scatter is showing that BMI is contributing more visible separation than waist-hip ratio in this cohort, even though both are pointing toward adiposity. This matters because it suggests BMI may remain the higher-priority routine feature, while waist-hip ratio may act as a smaller complementary marker rather than a dominant one.
            """
        )
    )

    cells.append(question_markdown(30, "Are blood pressure and random blood sugar jointly shifted in the PCOS-positive cohort?"))
    cells.append(
        code_cell(
            """
            # Building the cardiometabolic summary is quantifying the shared movement of blood pressure and glucose-related variables.
            summary_q30 = (
                df.groupby("pcos_label")[["systolic_bp_mmhg", "diastolic_bp_mmhg", "rbs_mg_dl"]]
                .agg(["mean", "median", "std"])
                .round(3)
            )
            display(summary_q30)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the joint cardiometabolic space is showing whether the PCOS-positive cohort is sitting further along a risk-leaning profile.
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df,
                x="systolic_bp_mmhg",
                y="rbs_mg_dl",
                hue="pcos_label",
                hue_order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                alpha=0.75,
                s=70,
                ax=ax,
            )
            ax.set_title("Question 30: Systolic Blood Pressure and Random Blood Sugar by PCOS Status")
            ax.set_xlabel("Systolic Blood Pressure (mmHg)")
            ax.set_ylabel("Random Blood Sugar (mg/dL)")
            ax.legend(title="")
            save_figure(fig, "q30_bp_rbs_joint_scatter.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The joint cardiometabolic scatter is showing overlap between the two groups, but it is also allowing the wider upper tail of random blood sugar in the PCOS-positive cohort to remain visible. This suggests that blood pressure and glycemic markers may be contributing more as metabolic-context features than as sharp standalone PCOS separators.
            """
        )
    )

    cells.append(question_markdown(31, "Which continuous variables show the strongest Spearman correlation with pcos_y_n?"))
    cells.append(
        code_cell(
            """
            # Building the target-correlation table is ranking the monotonic relationship between each continuous feature and PCOS status.
            continuous_features_q31 = [
                "age_yrs",
                "weight_kg",
                "bmi",
                "waist_hip_ratio",
                "pulse_rate_bpm",
                "respiratory_rate_breaths_min",
                "hb_g_dl",
                "cycle_length_days",
                "cycle_regularity_code",
                "systolic_bp_mmhg",
                "diastolic_bp_mmhg",
                "rbs_mg_dl",
                "follicle_no_left",
                "follicle_no_right",
                "avg_follicle_size_left_mm",
                "avg_follicle_size_right_mm",
                "endometrium_mm",
                "total_follicle_count",
            ]
            summary_q31 = (
                df[continuous_features_q31 + ["pcos_y_n"]]
                .corr(method="spearman")["pcos_y_n"]
                .drop("pcos_y_n")
                .sort_values(key=lambda series: series.abs(), ascending=False)
                .rename("spearman_r")
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            summary_q31["abs_spearman_r"] = summary_q31["spearman_r"].abs().round(3)
            summary_q31["spearman_r"] = summary_q31["spearman_r"].round(3)
            display(summary_q31)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the target-correlation ranking is highlighting which continuous variables deserve the most modeling attention.
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.barplot(
                data=summary_q31,
                y="feature",
                x="spearman_r",
                palette="viridis",
                ax=ax,
            )
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_title("Question 31: Spearman Correlation with PCOS Status")
            ax.set_xlabel("Spearman Correlation")
            ax.set_ylabel("Feature")
            save_figure(fig, "q31_continuous_spearman_with_target.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This ranking is showing that ovarian and adiposity-linked variables are carrying stronger monotonic alignment with PCOS status than general vital signs. That matters because later modeling should be emphasizing features that are both clinically plausible and empirically aligned with the target, rather than keeping weak routine variables simply because they are available.
            """
        )
    )

    cells.append(question_markdown(32, "Which symptoms show the strongest Spearman correlation with each other?"))
    cells.append(
        code_cell(
            """
            # Building the symptom-correlation matrix is quantifying redundancy among the binary symptom and behavior features.
            symptom_features_q32 = [
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
            ]
            summary_q32 = df[symptom_features_q32].corr(method="spearman").round(3)
            display(summary_q32)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the symptom-correlation heatmap is showing where overlapping signal may reduce the need for redundant features.
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                summary_q32,
                annot=True,
                fmt=".2f",
                cmap="mako",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Question 32: Symptom Spearman Correlation Matrix")
            save_figure(fig, "q32_symptom_spearman_heatmap.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The symptom correlation matrix is showing where visible or behavior-linked variables are moving together and potentially carrying overlapping signal. This matters because a later model may not benefit from keeping every correlated symptom if a smaller subset can preserve meaning while reducing redundancy.
            """
        )
    )

    cells.append(question_markdown(33, "Are pimples and skin darkening giving overlapping signal?"))
    cells.append(
        code_cell(
            """
            # Building the overlap table is quantifying how PCOS prevalence changes across combined symptom states.
            summary_q33 = (
                df.groupby(["pimples_y_n", "skin_darkening_y_n"])["pcos_y_n"]
                .agg(["count", "mean"])
                .rename(columns={"count": "n", "mean": "pcos_prevalence"})
                .reset_index()
            )
            summary_q33["pcos_prevalence_pct"] = (summary_q33["pcos_prevalence"] * 100).round(1)
            summary_q33["pimples_label"] = summary_q33["pimples_y_n"].map({0: "No Pimples", 1: "Pimples"})
            summary_q33["skin_darkening_label"] = summary_q33["skin_darkening_y_n"].map({0: "No Skin Darkening", 1: "Skin Darkening"})
            display(summary_q33)

            heatmap_q33 = summary_q33.pivot(
                index="skin_darkening_label",
                columns="pimples_label",
                values="pcos_prevalence_pct",
            )
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the overlap heatmap is showing whether the joint symptom state is concentrating PCOS prevalence.
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                heatmap_q33,
                annot=True,
                fmt=".1f",
                cmap="rocket_r",
                linewidths=0.5,
                cbar_kws={"label": "PCOS Prevalence (%)"},
                ax=ax,
            )
            ax.set_title("Question 33: PCOS Prevalence Across Pimples and Skin Darkening States")
            ax.set_xlabel("")
            ax.set_ylabel("")
            save_figure(fig, "q33_pimples_skin_darkening_overlap_heatmap.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This overlap heatmap is showing whether pimples and skin darkening are mostly repeating the same risk signal or whether their combination is marking a particularly high-burden subgroup. If the joint-positive cell is clearly warmer than the single-symptom cells, the pair may be carrying useful combined information even if each feature is individually correlated.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Derived Feature Analysis

            This section is testing simple engineered views of the data that may become useful during later feature-selection and modeling work.
            """
        )
    )

    cells.append(question_markdown(34, "Does a BMI-category distribution differ by PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the BMI-category table is summarizing how participants are distributing across clinical weight classes.
            summary_q34 = (
                pd.crosstab(df["bmi_category"], df["pcos_label"], normalize="columns")
                .mul(100)
                .round(1)
            )
            display(summary_q34)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the BMI-category distribution is showing whether higher BMI classes are overrepresented among PCOS-positive participants.
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.countplot(
                data=df,
                x="bmi_category",
                hue="pcos_label",
                order=["Underweight", "Normal", "Overweight", "Obese"],
                hue_order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                ax=ax,
            )
            ax.set_title("Question 34: BMI Category Distribution by PCOS Status")
            ax.set_xlabel("BMI Category")
            ax.set_ylabel("Participant Count")
            ax.legend(title="")
            save_figure(fig, "q34_bmi_category_distribution.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The BMI-category view is translating a continuous measure into a screening-friendly clinical lens. If the PCOS-positive group is accumulating more heavily in the overweight and obese categories, that supports keeping BMI both as a raw continuous feature and as a potentially useful engineered categorical marker later on.
            """
        )
    )

    cells.append(question_markdown(35, "Does a simple non-invasive burden score differ by PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the burden-score summary is quantifying how multiple accessible features are clustering within each PCOS group.
            summary_q35 = grouped_numeric_summary(df, "non_invasive_burden_score")
            burden_distribution_q35 = (
                df.groupby(["pcos_label", "non_invasive_burden_score"])
                .size()
                .rename("count")
                .reset_index()
            )
            display(summary_q35)
            display(burden_distribution_q35)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the burden score is showing whether symptom accumulation is concentrating in the PCOS-positive cohort.
            plot_numeric_by_target(
                data=df,
                feature="non_invasive_burden_score",
                ylabel="Non-Invasive Burden Score",
                title="Question 35: Non-Invasive Burden Score by PCOS Status",
                slug="q35_non_invasive_burden_score.png",
                kind="violin",
            )
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The burden score is showing whether multiple accessible symptom and lifestyle signals are stacking together in the PCOS-positive group. This matters because a composite burden feature can sometimes capture the syndrome pattern more efficiently than any single symptom considered in isolation.
            """
        )
    )

    cells.append(question_markdown(36, "What does the 364 versus 177 class split imply for false-negative cost in a screening workflow?"))
    cells.append(
        code_cell(
            """
            # Building the false-negative illustration table is translating class balance into a clinical screening consequence.
            class_counts_q36 = (
                df["pcos_label"]
                .value_counts()
                .reindex(PCOS_LABEL_ORDER)
                .rename_axis("pcos_label")
                .reset_index(name="count")
            )
            total_positive_q36 = int(class_counts_q36.loc[class_counts_q36["pcos_label"] == "PCOS Positive", "count"].iloc[0])
            summary_q36 = pd.DataFrame(
                {
                    "assumed_sensitivity": [0.90, 0.80, 0.70, 0.60],
                    "false_negative_rate": [0.10, 0.20, 0.30, 0.40],
                }
            )
            summary_q36["missed_positive_cases"] = (summary_q36["false_negative_rate"] * total_positive_q36).round().astype(int)
            display(class_counts_q36)
            display(summary_q36)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the missed-case illustration is showing how quickly clinical cost grows when sensitivity drops.
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_q36 = summary_q36.copy()
            plot_q36["false_negative_rate_label"] = (plot_q36["false_negative_rate"] * 100).astype(int).astype(str) + "% FNR"
            sns.barplot(
                data=plot_q36,
                x="false_negative_rate_label",
                y="missed_positive_cases",
                palette=["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
                ax=ax,
            )
            for patch in ax.patches:
                height = patch.get_height()
                ax.annotate(
                    f"{int(height)}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    xytext=(0, 6),
                    textcoords="offset points",
                )
            ax.set_title("Question 36: Illustrative False-Negative Cost in the Clinical Cohort")
            ax.set_xlabel("Assumed False-Negative Rate")
            ax.set_ylabel("Missed PCOS Cases")
            save_figure(fig, "q36_false_negative_cost_illustration.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The class split is reminding us that sensitivity is clinically important even in a moderately imbalanced dataset. With 177 PCOS-positive cases in this cohort, a false-negative rate of 20% would already correspond to about 35 missed cases, which is why later model evaluation should prioritize recall and clinical miss-cost rather than accuracy alone.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Observational Summary

            The clinical EDA is showing that the most persuasive PCOS signals in this cleaned cohort are clustering around BMI, weight, cycle irregularity codes, symptom burden, and follicle counts. The most visible symptom-level differences are appearing in weight gain, hair growth, skin darkening, and pimples, while general vital signs such as pulse rate and respiratory rate are moving much less.

            The cardiometabolic proxies are showing some upward stress in the PCOS-positive cohort, especially through BMI and the upper tail of random blood sugar, but the blood-pressure variables are not separating the groups strongly on their own. This is suggesting that cardiometabolic variables may contribute context rather than headline discrimination inside the PCOS cohort.

            The ovarian variables are showing some of the strongest differences, especially follicle counts. That is clinically coherent, but it also reinforces the importance of keeping the later non-invasive modeling phase separate from the broader clinical interpretation phase.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Feature Selection Insight

            ### Features to Keep Prominently
            - `bmi`
            - `weight_kg`
            - `weight_gain_y_n`
            - `hair_growth_y_n`
            - `skin_darkening_y_n`
            - `pimples_y_n`
            - `cycle_regularity_code`
            - `cycle_length_days` as a recorded measure
            - `rbs_mg_dl`

            ### Features to Treat Carefully
            - `waist_hip_ratio`, because it is appearing weaker than BMI but may still add nuance
            - `fast_food_y_n`, because the signal may be confounded by lifestyle and reporting behavior
            - `regular_exercise_y_n`, because the group separation is limited
            - `hb_g_dl`, because the biological shift is modest

            ### Features That Look Weaker or Noisier as Standalone Signals
            - `pulse_rate_bpm`
            - `respiratory_rate_breaths_min`
            - `systolic_bp_mmhg`
            - `diastolic_bp_mmhg`

            ### Features Worth Engineering Later
            - `non_invasive_burden_score`
            - `bmi_category`
            - symptom interaction features such as pimples plus skin darkening
            - anthropometric interaction views such as BMI plus waist-hip ratio

            ### Important Boundary
            Ovarian morphology variables such as follicle counts are remaining highly informative for interpretation, but they should be kept separate from strictly non-invasive model sets when the later ablation study is being designed.
            """
        )
    )

    return cells


def build_hormonal_notebook() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            """
            # PCOS Hormonal EDA and Invasive-Feature Usefulness

            ## Introduction
            This notebook is performing the exploratory analysis for the cleaned infertility sidecar dataset. The table is functioning as a hormonal subset of the main clinical PCOS cohort rather than as an independent external dataset.

            The notebook is using `cleaned_data/PCOS_infertility_cleaned.csv` and is focusing on the three available hormonal markers: `amh_ng_ml`, `beta_hcg_i_miu_ml`, and `beta_hcg_ii_miu_ml`. The main goal is determining how useful these invasive markers appear for interpretation and later ablation work.

            ## Research Positioning
            This notebook is not performing standalone model training and is not treating the hormonal sidecar as a separate source population. The workflow is instead examining whether these markers are adding clinically meaningful separation beyond the stronger routine and symptom-driven signals already seen in the main clinical notebook.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Reproducibility Setup and Path Configuration

            This section is preparing the notebook environment and the figure-export directory for the hormonal EDA workflow.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Importing the libraries is supporting tabular analysis, plotting, and notebook display.
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import display

            # Fixing the random seed is keeping stochastic behavior stable across reruns.
            np.random.seed(42)

            # Configuring the plotting theme is keeping the visuals aligned with the clinical notebook style.
            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.dpi"] = 120
            pd.set_option("display.max_columns", None)

            # Resolving the project root is keeping the notebook portable across launch locations.
            def resolve_project_root() -> Path:
                current = Path.cwd().resolve()
                candidates = [current, current.parent, current.parent.parent]
                for candidate in candidates:
                    if (candidate / "cleaned_data").exists() and (candidate / "images").exists():
                        return candidate
                return current

            PROJECT_ROOT = resolve_project_root()
            DATA_PATH = PROJECT_ROOT / "cleaned_data" / "PCOS_infertility_cleaned.csv"
            IMAGE_DIR = PROJECT_ROOT / "images" / "eda" / "hormonal"
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            PCOS_LABEL_ORDER = ["PCOS Negative", "PCOS Positive"]
            PCOS_LABEL_PALETTE = {
                "PCOS Negative": "#3b82f6",
                "PCOS Positive": "#e76f51",
            }

            def save_figure(fig: plt.Figure, slug: str) -> Path:
                output_path = IMAGE_DIR / slug
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                return output_path

            def grouped_numeric_summary(data: pd.DataFrame, feature: str) -> pd.DataFrame:
                return (
                    data.groupby("pcos_label")[feature]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .rename(columns={"count": "n"})
                    .round(3)
                    .reset_index()
                )

            def plot_numeric_by_target(
                data: pd.DataFrame,
                feature: str,
                ylabel: str,
                title: str,
                slug: str,
                log_scale: bool = False,
            ) -> None:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(
                    data=data,
                    x="pcos_label",
                    y=feature,
                    order=PCOS_LABEL_ORDER,
                    palette=PCOS_LABEL_PALETTE,
                    ax=ax,
                )
                sns.stripplot(
                    data=data,
                    x="pcos_label",
                    y=feature,
                    order=PCOS_LABEL_ORDER,
                    color="#264653",
                    alpha=0.25,
                    size=3,
                    jitter=0.18,
                    ax=ax,
                )
                if log_scale:
                    ax.set_yscale("log")
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                save_figure(fig, slug)
                plt.show()
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Data Loading and Derived Hormonal Views

            This section is loading the cleaned hormonal sidecar, verifying the expected schema, and creating log-scale versions of the markers so that skewed distributions can be examined more clearly.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Loading the hormonal sidecar dataset is bringing the invasive-marker subset into memory for analysis.
            df = pd.read_csv(DATA_PATH)

            # Verifying the schema is protecting the notebook from upstream naming drift.
            required_columns = [
                "pcos_y_n",
                "beta_hcg_i_miu_ml",
                "beta_hcg_ii_miu_ml",
                "amh_ng_ml",
            ]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing expected columns: {missing_columns}")

            # Casting the marker columns to numeric is keeping the later plots and correlations explicit.
            for column in required_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            # Creating log-scale views is making the heavy right-skew of the hormone markers easier to inspect.
            df["pcos_y_n"] = df["pcos_y_n"].astype(int)
            df["pcos_label"] = df["pcos_y_n"].map({0: "PCOS Negative", 1: "PCOS Positive"})
            df["log_beta_hcg_i"] = np.log10(df["beta_hcg_i_miu_ml"] + 1)
            df["log_beta_hcg_ii"] = np.log10(df["beta_hcg_ii_miu_ml"] + 1)
            df["log_amh"] = np.log10(df["amh_ng_ml"] + 1)

            display(df.head())
            """
        )
    )

    cells.append(question_markdown(1, "What is the sidecar hormonal dataset size, schema, and target balance?"))
    cells.append(
        code_cell(
            """
            # Building the overview tables is showing the sidecar structure before the hormone-by-hormone analysis begins.
            overview_q01 = pd.DataFrame(
                {
                    "metric": ["row_count", "column_count", "missing_cells"],
                    "value": [df.shape[0], df.shape[1], int(df.isna().sum().sum())],
                }
            )
            target_q01 = (
                df["pcos_label"]
                .value_counts()
                .reindex(PCOS_LABEL_ORDER)
                .rename_axis("pcos_label")
                .reset_index(name="count")
            )
            target_q01["percentage"] = (100 * target_q01["count"] / target_q01["count"].sum()).round(1)
            schema_q01 = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})

            display(overview_q01)
            display(schema_q01)
            display(target_q01)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the target distribution is showing that the hormonal file mirrors the labeled portion of the main clinical cohort.
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(
                data=target_q01,
                x="pcos_label",
                y="count",
                order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                ax=ax,
            )
            ax.set_title("Question 1: Hormonal Sidecar Target Distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Participant Count")
            save_figure(fig, "q01_hormonal_target_distribution.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The hormonal table is carrying the same 364 versus 177 class split already seen in the labeled portion of the main clinical cohort. That matching target balance is reinforcing the interpretation that this file is a sidecar subset for hormonal context, not an external validation cohort.
            """
        )
    )

    hormonal_numeric_questions = [
        (2, "How is AMH distributed across PCOS groups?", "amh_ng_ml", "AMH (ng/mL)", "q02_amh_vs_pcos.png", False, "AMH is showing a visibly higher central tendency in the PCOS-positive cohort, which is consistent with its well-known relevance in ovarian reserve and PCOS-related ovarian morphology. This makes AMH one of the strongest invasive candidates for later ablation-style comparison."),
        (3, "How is beta-HCG I distributed across PCOS groups?", "beta_hcg_i_miu_ml", "Beta-hCG I (mIU/mL, log scale)", "q03_beta_hcg_i_vs_pcos.png", True, "Beta-hCG I is showing a very wide and highly skewed spread, with considerable overlap between the two groups. That heavy overlap is already suggesting that beta-hCG may be noisier and less clinically targeted for PCOS discrimination than AMH."),
        (4, "How is beta-HCG II distributed across PCOS groups?", "beta_hcg_ii_miu_ml", "Beta-hCG II (mIU/mL, log scale)", "q04_beta_hcg_ii_vs_pcos.png", True, "Beta-hCG II is also showing substantial skew and wide overlap across the PCOS groups. This pattern is making it look more like a context variable with substantial noise than like a crisp discriminatory feature for later modeling."),
    ]

    for number, question, feature, ylabel, slug, log_scale, insight in hormonal_numeric_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped summary is quantifying `{feature}` before plotting.
                summary_q{number:02d} = grouped_numeric_summary(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped hormone distribution is showing how `{feature}` is separating across PCOS status.
                plot_numeric_by_target(
                    data=df,
                    feature="{feature}",
                    ylabel="{ylabel}",
                    title="Question {number}: {question}",
                    slug="{slug}",
                    log_scale={str(log_scale)},
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(question_markdown(5, "Which hormonal marker shows the strongest group separation by effect size?"))
    cells.append(
        code_cell(
            """
            # Building the effect-size table is ranking which hormonal markers are separating the two groups most strongly.
            effect_rows_q05 = []
            for feature in ["amh_ng_ml", "beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml"]:
                negative = df.loc[df["pcos_y_n"] == 0, feature].dropna()
                positive = df.loc[df["pcos_y_n"] == 1, feature].dropna()
                pooled_std = np.sqrt(((negative.std() ** 2) + (positive.std() ** 2)) / 2)
                smd = (positive.mean() - negative.mean()) / pooled_std if pooled_std else np.nan
                effect_rows_q05.append(
                    {
                        "feature": feature,
                        "mean_negative": round(negative.mean(), 3),
                        "mean_positive": round(positive.mean(), 3),
                        "median_negative": round(negative.median(), 3),
                        "median_positive": round(positive.median(), 3),
                        "standardized_mean_difference": round(float(smd), 3),
                    }
                )
            summary_q05 = pd.DataFrame(effect_rows_q05).sort_values(
                "standardized_mean_difference",
                key=lambda series: series.abs(),
                ascending=False,
            )
            summary_q05["abs_smd"] = summary_q05["standardized_mean_difference"].abs()
            display(summary_q05)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the effect sizes is showing which invasive marker is contributing the clearest group-level separation.
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=summary_q05,
                x="standardized_mean_difference",
                y="feature",
                palette="crest",
                ax=ax,
            )
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_title("Question 5: Standardized Mean Difference by Hormonal Marker")
            ax.set_xlabel("Standardized Mean Difference")
            ax.set_ylabel("Marker")
            save_figure(fig, "q05_hormonal_effect_sizes.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This ranking is helping distinguish the markers that look clinically informative from the ones that are mostly noisy. If AMH is dominating the effect-size plot while beta-hCG features remain small or unstable, that is strengthening the case for keeping AMH and de-prioritizing beta-hCG in later invasive-feature experiments.
            """
        )
    )

    cells.append(question_markdown(6, "How strongly are beta-HCG I and beta-HCG II correlated?"))
    cells.append(
        code_cell(
            """
            # Building the paired-correlation table is quantifying agreement between the two beta-hCG measurements.
            summary_q06 = pd.DataFrame(
                {
                    "metric": ["pearson_r", "spearman_r"],
                    "value": [
                        round(df[["beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml"]].corr(method="pearson").iloc[0, 1], 3),
                        round(df[["beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml"]].corr(method="spearman").iloc[0, 1], 3),
                    ],
                }
            )
            display(summary_q06)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the paired scatter is showing whether the two beta-hCG measures are largely moving together.
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df,
                x="beta_hcg_i_miu_ml",
                y="beta_hcg_ii_miu_ml",
                hue="pcos_label",
                hue_order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                alpha=0.70,
                s=65,
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Question 6: Beta-hCG I versus Beta-hCG II")
            ax.set_xlabel("Beta-hCG I (mIU/mL, log scale)")
            ax.set_ylabel("Beta-hCG II (mIU/mL, log scale)")
            ax.legend(title="")
            save_figure(fig, "q06_beta_hcg_pair_scatter.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This relationship is showing whether the two beta-hCG measurements are largely repeating the same information. If the markers are moving tightly together, later modeling would gain little by keeping both unless they are showing meaningfully different alignment with the target.
            """
        )
    )

    for number, question, x_feature, y_feature, slug, insight in [
        (7, "How strongly is AMH associated with beta-HCG I?", "beta_hcg_i_miu_ml", "amh_ng_ml", "q07_amh_vs_beta_hcg_i_scatter.png", "This scatter is showing whether AMH is traveling together with beta-hCG I or whether the two markers are largely independent. Weak alignment would suggest that AMH is capturing a different hormonal story from the noisier beta-hCG measurements."),
        (8, "How strongly is AMH associated with beta-HCG II?", "beta_hcg_ii_miu_ml", "amh_ng_ml", "q08_amh_vs_beta_hcg_ii_scatter.png", "This comparison is checking whether AMH and beta-hCG II are moving together in any meaningful way. Broad overlap and weak trend would further support the idea that AMH is the more clinically targeted marker for later invasive-feature work."),
    ]:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the pairwise-correlation table is quantifying the relationship before plotting.
                summary_q{number:02d} = pd.DataFrame(
                    {{
                        "metric": ["pearson_r", "spearman_r"],
                        "value": [
                            round(df[["{x_feature}", "{y_feature}"]].corr(method="pearson").iloc[0, 1], 3),
                            round(df[["{x_feature}", "{y_feature}"]].corr(method="spearman").iloc[0, 1], 3),
                        ],
                    }}
                )
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the pairwise scatter is showing whether the two markers are jointly separating the PCOS groups.
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=df,
                    x="{x_feature}",
                    y="{y_feature}",
                    hue="pcos_label",
                    hue_order=PCOS_LABEL_ORDER,
                    palette=PCOS_LABEL_PALETTE,
                    alpha=0.70,
                    s=65,
                    ax=ax,
                )
                ax.set_xscale("log")
                ax.set_title("Question {number}: {question}")
                ax.set_xlabel("{x_feature}")
                ax.set_ylabel("{y_feature}")
                ax.legend(title="")
                save_figure(fig, "{slug}")
                plt.show()
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(question_markdown(9, "Does a multivariate scatter of AMH versus beta-HCG I show meaningful class separation or heavy overlap?"))
    cells.append(
        code_cell(
            """
            # Building the quartile-style summary is preparing a compact view of the joint marker spread.
            summary_q09 = (
                df.groupby("pcos_label")[["amh_ng_ml", "beta_hcg_i_miu_ml"]]
                .quantile([0.25, 0.50, 0.75])
                .round(3)
            )
            display(summary_q09)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the AMH versus beta-hCG I space is showing whether the classes separate or overlap in two dimensions.
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df,
                x="beta_hcg_i_miu_ml",
                y="amh_ng_ml",
                hue="pcos_label",
                hue_order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                alpha=0.70,
                s=70,
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_title("Question 9: AMH versus Beta-hCG I by PCOS Status")
            ax.set_xlabel("Beta-hCG I (mIU/mL, log scale)")
            ax.set_ylabel("AMH (ng/mL)")
            ax.legend(title="")
            save_figure(fig, "q09_amh_beta_hcg_i_multivariate_scatter.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This multivariate view is showing whether AMH can still separate the groups even when beta-hCG is moving unpredictably. If the groups remain layered mainly along the AMH axis rather than the beta-hCG axis, that is strengthening the interpretation that AMH is the more useful invasive marker.
            """
        )
    )

    cells.append(question_markdown(10, "Are the hormone distributions heavily skewed and outlier-prone?"))
    cells.append(
        code_cell(
            """
            # Building the skewness table is quantifying how strongly the markers are leaning to the right.
            summary_q10 = pd.DataFrame(
                {
                    "feature": ["amh_ng_ml", "beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml"],
                    "skewness": [
                        round(df["amh_ng_ml"].skew(), 3),
                        round(df["beta_hcg_i_miu_ml"].skew(), 3),
                        round(df["beta_hcg_ii_miu_ml"].skew(), 3),
                    ],
                    "q95": [
                        round(df["amh_ng_ml"].quantile(0.95), 3),
                        round(df["beta_hcg_i_miu_ml"].quantile(0.95), 3),
                        round(df["beta_hcg_ii_miu_ml"].quantile(0.95), 3),
                    ],
                    "max": [
                        round(df["amh_ng_ml"].max(), 3),
                        round(df["beta_hcg_i_miu_ml"].max(), 3),
                        round(df["beta_hcg_ii_miu_ml"].max(), 3),
                    ],
                }
            )
            display(summary_q10)

            long_q10 = df.melt(
                id_vars="pcos_label",
                value_vars=["log_amh", "log_beta_hcg_i", "log_beta_hcg_ii"],
                var_name="marker",
                value_name="log_value",
            )
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the log-scale hormone densities is showing the extent of skew and tail behavior across markers.
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.violinplot(
                data=long_q10,
                x="marker",
                y="log_value",
                palette=["#2a9d8f", "#e9c46a", "#6d597a"],
                cut=0,
                inner="quartile",
                ax=ax,
            )
            ax.set_title("Question 10: Log-Scaled Hormone Distribution Shapes")
            ax.set_xlabel("Marker")
            ax.set_ylabel("log10(value + 1)")
            save_figure(fig, "q10_hormone_skewness_violin.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The hormone distributions are showing strong right-skew and large upper tails, especially for the beta-hCG variables. This matters because highly skewed markers can destabilize later modeling if they are entered naively, and it also supports careful skepticism when a noisy marker appears biologically less specific.
            """
        )
    )

    cells.append(question_markdown(11, "What do the hormonal correlations with pcos_y_n suggest about usefulness for later invasive-feature augmentation?"))
    cells.append(
        code_cell(
            """
            # Building the target-correlation ranking is summarizing which hormonal markers align most strongly with PCOS status.
            summary_q11 = (
                df[["amh_ng_ml", "beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml", "pcos_y_n"]]
                .corr(method="spearman")["pcos_y_n"]
                .drop("pcos_y_n")
                .sort_values(key=lambda series: series.abs(), ascending=False)
                .rename("spearman_r")
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            summary_q11["spearman_r"] = summary_q11["spearman_r"].round(3)
            display(summary_q11)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the target-correlation ranking is showing which hormonal markers deserve to survive into later ablation experiments.
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(
                data=summary_q11,
                x="spearman_r",
                y="feature",
                palette="flare",
                ax=ax,
            )
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_title("Question 11: Hormonal Spearman Correlation with PCOS Status")
            ax.set_xlabel("Spearman Correlation")
            ax.set_ylabel("Marker")
            save_figure(fig, "q11_hormonal_target_correlation.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            This ranking is summarizing the final usefulness question for the hormonal sidecar. If AMH is clearly outranking both beta-hCG markers, the evidence is supporting AMH as the invasive marker worth preserving for later ablation studies, while beta-hCG should be treated as potentially weak or noisy unless a specific clinical rationale emerges.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Invasive-Feature Usefulness Summary

            The hormonal sidecar is behaving like a focused reference notebook rather than a standalone dataset. The main signal is appearing to come from AMH, which is showing a clearer upward shift in the PCOS-positive group than either beta-hCG measurement.

            The beta-hCG markers are showing heavy skew, wide overlap, and uncertain clinical specificity for PCOS. That does not make them useless, but it does mean they should be treated cautiously and should not be assumed to improve later models simply because they are laboratory measurements.

            The overall interpretation is therefore favoring AMH as the primary invasive marker to preserve for later sensitivity or ablation comparisons, while beta-hCG should remain a secondary exploratory feature unless stronger evidence appears in downstream evaluation.
            """
        )
    )

    return cells


def build_survey_notebook() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            """
            # PCOS Survey EDA and External-Style Non-Invasive Pattern Review

            ## Introduction
            This notebook is performing the exploratory analysis for the cleaned PCOS survey dataset. The survey table is providing a self-reported, non-clinical view of the syndrome and is therefore supporting the later external-style validation story of the MSc project.

            The notebook is using `cleaned_data/PCOS_survey_cleaned.csv`. Because the dataset is self-reported, the analysis is being framed more cautiously than the clinical notebook. The goal is not proving clinical equivalence, but checking whether the main non-invasive pattern still appears in a noisier real-world style dataset.

            ## Research Positioning
            This notebook is not merging the survey table row-wise with the clinical cohort. It is instead examining whether symptom prevalence, menstrual-pattern disruption, and BMI-related burden are still concentrating in survey participants who report PCOS.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Reproducibility Setup and Path Configuration

            This section is preparing the environment, the color palette, and the survey-specific figure-export directory.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Importing the libraries is supporting survey-table analysis, plotting, and notebook display.
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import display

            # Fixing the random seed is keeping reruns reproducible.
            np.random.seed(42)

            # Configuring the plotting theme is keeping the visuals consistent with the clinical and hormonal notebooks.
            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.dpi"] = 120
            pd.set_option("display.max_columns", None)

            # Resolving the project root is keeping the notebook portable across launch locations.
            def resolve_project_root() -> Path:
                current = Path.cwd().resolve()
                candidates = [current, current.parent, current.parent.parent]
                for candidate in candidates:
                    if (candidate / "cleaned_data").exists() and (candidate / "images").exists():
                        return candidate
                return current

            PROJECT_ROOT = resolve_project_root()
            DATA_PATH = PROJECT_ROOT / "cleaned_data" / "PCOS_survey_cleaned.csv"
            IMAGE_DIR = PROJECT_ROOT / "images" / "eda" / "survey"
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            PCOS_LABEL_ORDER = ["PCOS Negative", "PCOS Positive"]
            PCOS_LABEL_PALETTE = {
                "PCOS Negative": "#3b82f6",
                "PCOS Positive": "#e76f51",
            }

            def save_figure(fig: plt.Figure, slug: str) -> Path:
                output_path = IMAGE_DIR / slug
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                return output_path

            def grouped_numeric_summary(data: pd.DataFrame, feature: str) -> pd.DataFrame:
                return (
                    data.groupby("pcos_label")[feature]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .rename(columns={"count": "n"})
                    .round(3)
                    .reset_index()
                )

            def grouped_binary_prevalence(data: pd.DataFrame, feature: str) -> pd.DataFrame:
                summary = (
                    data.groupby("pcos_label")[feature]
                    .agg(["count", "sum", "mean"])
                    .rename(columns={"count": "n", "sum": "positive_count", "mean": "prevalence"})
                    .reset_index()
                )
                summary["prevalence_pct"] = (summary["prevalence"] * 100).round(1)
                return summary[["pcos_label", "n", "positive_count", "prevalence_pct"]]

            def plot_numeric_by_target(
                data: pd.DataFrame,
                feature: str,
                ylabel: str,
                title: str,
                slug: str,
                kind: str = "box",
            ) -> None:
                fig, ax = plt.subplots(figsize=(8, 5))
                if kind == "violin":
                    sns.violinplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        palette=PCOS_LABEL_PALETTE,
                        cut=0,
                        inner=None,
                        ax=ax,
                    )
                    sns.stripplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        color="#264653",
                        alpha=0.30,
                        size=3,
                        jitter=0.20,
                        ax=ax,
                    )
                else:
                    sns.boxplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        palette=PCOS_LABEL_PALETTE,
                        ax=ax,
                    )
                    sns.stripplot(
                        data=data,
                        x="pcos_label",
                        y=feature,
                        order=PCOS_LABEL_ORDER,
                        color="#264653",
                        alpha=0.30,
                        size=3,
                        jitter=0.20,
                        ax=ax,
                    )
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                save_figure(fig, slug)
                plt.show()

            def plot_binary_prevalence(summary: pd.DataFrame, title: str, slug: str) -> None:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(
                    data=summary,
                    x="pcos_label",
                    y="prevalence_pct",
                    order=PCOS_LABEL_ORDER,
                    palette=PCOS_LABEL_PALETTE,
                    ax=ax,
                )
                for patch in ax.patches:
                    height = patch.get_height()
                    ax.annotate(
                        f"{height:.1f}%",
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        xytext=(0, 6),
                        textcoords="offset points",
                    )
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel("Prevalence (%)")
                save_figure(fig, slug)
                plt.show()
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Data Loading and Derived Survey Features

            This section is loading the cleaned survey table, checking the expected schema, and creating simple derived features that summarize non-invasive burden in the self-reported cohort.
            """
        )
    )

    cells.append(
        code_cell(
            """
            # Loading the cleaned survey dataset is bringing the self-reported PCOS table into memory for EDA.
            df = pd.read_csv(DATA_PATH)

            # Verifying the expected schema is protecting the notebook from upstream changes.
            required_columns = [
                "pcos_y_n",
                "age_yrs",
                "weight_kg",
                "height_cm",
                "bmi",
                "cycle_regularity",
                "cycle_length_days",
                "months_between_periods",
                "period_duration_days",
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
                "mood_swings_y_n",
            ]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing expected columns: {missing_columns}")

            # Casting the survey fields to numeric is keeping the summary tables and plots explicit.
            for column in required_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

            binary_columns = [
                "pcos_y_n",
                "cycle_regularity",
                "weight_gain_y_n",
                "hair_growth_y_n",
                "skin_darkening_y_n",
                "hair_loss_y_n",
                "pimples_y_n",
                "fast_food_y_n",
                "regular_exercise_y_n",
                "mood_swings_y_n",
            ]
            for column in binary_columns:
                df[column] = df[column].astype(int)

            # Creating readable labels and a simple burden score is supporting later survey-based interpretation.
            df["pcos_label"] = df["pcos_y_n"].map({0: "PCOS Negative", 1: "PCOS Positive"})
            df["low_exercise_flag"] = (1 - df["regular_exercise_y_n"]).astype(int)
            df["irregular_cycle_flag"] = (1 - df["cycle_regularity"]).astype(int)
            df["non_invasive_burden_score"] = (
                df["weight_gain_y_n"]
                + df["hair_growth_y_n"]
                + df["skin_darkening_y_n"]
                + df["hair_loss_y_n"]
                + df["pimples_y_n"]
                + df["fast_food_y_n"]
                + df["low_exercise_flag"]
                + df["irregular_cycle_flag"]
            )

            display(df.head())
            """
        )
    )

    cells.append(question_markdown(1, "What is the cleaned survey dataset size, schema, and target balance?"))
    cells.append(
        code_cell(
            """
            # Building the overview tables is showing the survey structure before the feature-level questions begin.
            overview_q01 = pd.DataFrame(
                {
                    "metric": ["row_count", "column_count", "missing_cells"],
                    "value": [df.shape[0], df.shape[1], int(df.isna().sum().sum())],
                }
            )
            target_q01 = (
                df["pcos_label"]
                .value_counts()
                .reindex(PCOS_LABEL_ORDER)
                .rename_axis("pcos_label")
                .reset_index(name="count")
            )
            target_q01["percentage"] = (100 * target_q01["count"] / target_q01["count"].sum()).round(1)
            schema_q01 = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
            display(overview_q01)
            display(schema_q01)
            display(target_q01)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the survey target distribution is showing the class balance that later external-style validation will inherit.
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(
                data=target_q01,
                x="pcos_label",
                y="count",
                order=PCOS_LABEL_ORDER,
                palette=PCOS_LABEL_PALETTE,
                ax=ax,
            )
            ax.set_title("Question 1: Survey Target Distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Participant Count")
            save_figure(fig, "q01_survey_target_distribution.png")
            plt.show()
            """
        )
    )
    cells.append(
        insight_markdown(
            """
            The cleaned survey dataset is showing a smaller PCOS-positive group than the clinical cohort, which is expected in a self-reported setting. This balance is still adequate for descriptive EDA, but it is reinforcing the need to interpret later survey validation results with attention to class size and reporting noise.
            """
        )
    )

    numeric_survey_questions = [
        (2, "Does age differ by survey PCOS status?", "age_yrs", "Age (years)", "q02_age_vs_pcos_survey.png", "Age is showing only limited separation across the survey target groups, which is suggesting that age is more likely to behave as a contextual feature than as a dominant self-reported signal."),
        (3, "Does BMI differ by survey PCOS status?", "bmi", "Body Mass Index", "q03_bmi_vs_pcos_survey.png", "BMI is showing a meaningful upward shift in the survey PCOS-positive group, which is directionally consistent with the clinical notebook. This matters because it suggests that adiposity-related signal is surviving even in a noisier self-reported dataset."),
        (5, "Does cycle-length behavior differ by survey PCOS status?", "cycle_length_days", "Estimated Cycle Length (days)", "q05_cycle_length_vs_pcos_survey.png", "The estimated cycle-length feature is showing a strong upward shift in the survey PCOS-positive group. That pattern is clinically important because longer or more disrupted cycles are central to PCOS screening and remain visible even in self-reported data."),
        (6, "Do months between periods differ by survey PCOS status?", "months_between_periods", "Months Between Periods", "q06_months_between_periods_vs_pcos.png", "Months between periods is showing a clear upward shift among survey participants who report PCOS. This reinforces the same menstrual-disruption story seen in the clinical notebook, even though the survey variable is coarser and self-reported."),
        (7, "Does period duration differ by survey PCOS status?", "period_duration_days", "Period Duration (days)", "q07_period_duration_vs_pcos.png", "Period duration is showing a more modest shift than cycle spacing, which suggests that timing irregularity may be more informative than duration alone in this self-reported cohort."),
        (16, "Does a simple non-invasive burden score differ by survey PCOS status?", "non_invasive_burden_score", "Non-Invasive Burden Score", "q16_survey_burden_score.png", "The burden score is showing whether multiple self-reported symptoms and behavior signals are stacking together inside the survey PCOS-positive group. A clear upward shift would support the idea that the non-invasive phenotype remains visible even when the data source is noisier."),
    ]

    cells.append(question_markdown(2, "Does age differ by survey PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the grouped summary is quantifying survey age by PCOS status before plotting.
            summary_q02 = grouped_numeric_summary(df, "age_yrs")
            display(summary_q02)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the survey age distribution is showing whether the target groups are materially age-shifted.
            plot_numeric_by_target(
                data=df,
                feature="age_yrs",
                ylabel="Age (years)",
                title="Question 2: Survey Age by PCOS Status",
                slug="q02_age_vs_pcos_survey.png",
                kind="box",
            )
            """
        )
    )
    cells.append(insight_markdown("Age is showing only limited separation across the survey target groups, which is suggesting that age is more likely to behave as a contextual feature than as a dominant self-reported signal."))

    cells.append(question_markdown(3, "Does BMI differ by survey PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the grouped BMI summary is quantifying the body-composition shift before plotting.
            summary_q03 = grouped_numeric_summary(df, "bmi")
            display(summary_q03)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the survey BMI distribution is showing whether adiposity signal survives in the self-reported cohort.
            plot_numeric_by_target(
                data=df,
                feature="bmi",
                ylabel="Body Mass Index",
                title="Question 3: Survey BMI by PCOS Status",
                slug="q03_bmi_vs_pcos_survey.png",
                kind="box",
            )
            """
        )
    )
    cells.append(insight_markdown("BMI is showing a meaningful upward shift in the survey PCOS-positive group, which is directionally consistent with the clinical notebook. This matters because it suggests that adiposity-related signal is surviving even in a noisier self-reported dataset."))

    cells.append(question_markdown(4, "Does cycle regularity prevalence differ by survey PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the cycle-regularity prevalence table is quantifying how often regular cycles are being reported in each group.
            summary_q04 = grouped_binary_prevalence(df, "cycle_regularity")
            display(summary_q04)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the cycle-regularity prevalence is showing how strongly menstrual disruption is separating the survey groups.
            plot_binary_prevalence(
                summary=summary_q04,
                title="Question 4: Regular Cycle Prevalence by Survey PCOS Status",
                slug="q04_cycle_regularity_prevalence.png",
            )
            """
        )
    )
    cells.append(insight_markdown("Regular-cycle prevalence is dropping sharply in the survey PCOS-positive group, which is providing one of the strongest self-reported signals in the dataset. This is clinically important because menstrual irregularity remains central to low-burden PCOS screening."))

    for number, question, feature, ylabel, slug, insight in numeric_survey_questions[2:5]:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped summary is quantifying `{feature}` before plotting.
                summary_q{number:02d} = grouped_numeric_summary(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped distribution is showing how `{feature}` is shifting across the survey target groups.
                plot_numeric_by_target(
                    data=df,
                    feature="{feature}",
                    ylabel="{ylabel}",
                    title="Question {number}: {question}",
                    slug="{slug}",
                    kind="box",
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    survey_binary_questions = [
        (8, "Does weight gain prevalence differ by survey PCOS status?", "weight_gain_y_n", "Weight Gain", "q08_weight_gain_prevalence_survey.png", "Weight gain is showing a clear prevalence jump in the survey PCOS-positive group, which is helping the self-reported dataset preserve an important metabolic and symptom burden signal."),
        (9, "Does hair growth prevalence differ by survey PCOS status?", "hair_growth_y_n", "Hair Growth", "q09_hair_growth_prevalence_survey.png", "Hair growth is showing a strong prevalence shift in the survey cohort, which is supporting its role as an androgen-linked non-invasive marker even outside the clinical dataset."),
        (10, "Does skin darkening prevalence differ by survey PCOS status?", "skin_darkening_y_n", "Skin Darkening", "q10_skin_darkening_prevalence_survey.png", "Skin darkening is again showing a notable upward prevalence in positive cases, which is directionally compatible with the clinical notebook and supportive of insulin-resistance-linked symptom burden."),
        (11, "Does hair loss prevalence differ by survey PCOS status?", "hair_loss_y_n", "Hair Loss", "q11_hair_loss_prevalence_survey.png", "Hair loss is showing a moderate prevalence difference, suggesting that it remains useful but may not be as dominant as the stronger androgenic symptom features."),
        (12, "Does pimples prevalence differ by survey PCOS status?", "pimples_y_n", "Pimples", "q12_pimples_prevalence_survey.png", "Pimples are showing a noticeable upward prevalence in the positive survey group. This matters because acne-like features are easy to self-report, although they may overlap with other androgen-linked symptoms."),
        (13, "Does fast-food behavior differ by survey PCOS status?", "fast_food_y_n", "Fast Food", "q13_fast_food_prevalence_survey.png", "Fast-food behavior is showing only a modest difference in the survey cohort. This suggests it may contribute contextual lifestyle information but is unlikely to behave like a core screening feature on its own."),
        (14, "Does regular exercise differ by survey PCOS status?", "regular_exercise_y_n", "Regular Exercise", "q14_regular_exercise_prevalence_survey.png", "Regular exercise is showing little separation across the survey groups, which is making it look more like a background behavior variable than a strong direct PCOS signal in this dataset."),
        (15, "Does mood swings prevalence differ by survey PCOS status?", "mood_swings_y_n", "Mood Swings", "q15_mood_swings_prevalence_survey.png", "Mood swings are highly prevalent in both groups but are still showing a stronger concentration in the survey PCOS-positive cohort. This suggests potential relevance, although the high baseline prevalence means the feature may be sensitive but not especially specific."),
    ]

    for number, question, feature, label, slug, insight in survey_binary_questions:
        cells.append(question_markdown(number, question))
        cells.append(
            code_cell(
                f"""
                # Building the grouped prevalence table is quantifying how often `{feature}` is being reported in each survey target group.
                summary_q{number:02d} = grouped_binary_prevalence(df, "{feature}")
                display(summary_q{number:02d})
                """
            )
        )
        cells.append(
            code_cell(
                f"""
                # Plotting the grouped prevalence is showing whether `{feature}` is concentrating in the positive survey cohort.
                plot_binary_prevalence(
                    summary=summary_q{number:02d},
                    title="Question {number}: {label} Prevalence by Survey PCOS Status",
                    slug="{slug}",
                )
                """
            )
        )
        cells.append(insight_markdown(insight))

    cells.append(question_markdown(16, "Does a simple non-invasive burden score differ by survey PCOS status?"))
    cells.append(
        code_cell(
            """
            # Building the burden-score summary is quantifying cumulative survey-side symptom load before plotting.
            summary_q16 = grouped_numeric_summary(df, "non_invasive_burden_score")
            burden_distribution_q16 = (
                df.groupby(["pcos_label", "non_invasive_burden_score"])
                .size()
                .rename("count")
                .reset_index()
            )
            display(summary_q16)
            display(burden_distribution_q16)
            """
        )
    )
    cells.append(
        code_cell(
            """
            # Plotting the burden score is showing whether multiple self-reported risk indicators are stacking together in positive cases.
            plot_numeric_by_target(
                data=df,
                feature="non_invasive_burden_score",
                ylabel="Non-Invasive Burden Score",
                title="Question 16: Survey Non-Invasive Burden Score by PCOS Status",
                slug="q16_survey_burden_score.png",
                kind="violin",
            )
            """
        )
    )
    cells.append(insight_markdown("The burden score is showing whether multiple self-reported symptoms and behavior signals are stacking together inside the survey PCOS-positive group. A clear upward shift would support the idea that the non-invasive phenotype remains visible even when the data source is noisier."))

    cells.append(
        markdown_cell(
            """
            ## Generalization and Data-Quality Summary

            The survey notebook is showing that the major non-invasive PCOS pattern is still visible in self-reported data, especially through BMI, menstrual disruption, weight gain, hair growth, skin darkening, and mood-related burden. These directions are broadly compatible with the stronger clinical signals, even though the survey dataset is noisier and more weakly structured.

            Some features such as fast food and regular exercise are showing much weaker separation than the stronger reproductive and symptom variables. That matters because it suggests the later external-style validation phase should prioritize symptom burden and menstrual features over lifestyle variables that may be more confounded or inconsistently reported.

            The survey table is therefore functioning as a realistic non-clinical stress test for the later non-invasive modeling workflow. It is not replacing the clinical dataset, but it is showing whether the same general pattern survives when measurement precision decreases.
            """
        )
    )

    return cells


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "05a_pcos_clinical_eda.ipynb", build_clinical_notebook())
    write_notebook(NOTEBOOK_DIR / "05b_pcos_hormonal_eda.ipynb", build_hormonal_notebook())
    write_notebook(NOTEBOOK_DIR / "05c_pcos_survey_eda.ipynb", build_survey_notebook())
    print("Generated Phase 2 EDA notebooks in", NOTEBOOK_DIR)


if __name__ == "__main__":
    main()
