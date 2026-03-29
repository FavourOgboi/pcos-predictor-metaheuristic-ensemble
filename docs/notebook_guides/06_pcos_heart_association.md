# PCOS and Cardiovascular Association Analysis

## Notebook Purpose
This notebook studies whether the PCOS cohort shows body-weight, blood-pressure, sugar, and heart-related patterns that connect to cardiovascular risk, while using the heart cohort only as a group-level reference.

This notebook comes after the PCOS-only EDA notebooks and is the main association notebook for the cardiovascular-risk hypothesis.

## Notebook Inputs and Outputs

### Inputs
| File | Exists |
| --- | --- |
| `cleaned_data/PCOS_full_cleaned.csv` | Yes |
| `cleaned_data/heart_cleaned.csv` | Yes |

### Outputs
| File | Exists |
| --- | --- |
| `images/eda/association/age_distribution_ecological_reference.png` | Yes |
| `images/eda/association/chapter_quality_summary_panel.png` | Yes |
| `images/eda/association/effect_size_dot_plot.png` | Yes |
| `images/eda/association/q01_age_trajectory_kde.png` | Yes |
| `images/eda/association/q02_systolic_pressure_with_heart_reference.png` | Yes |
| `images/eda/association/q03_pulse_rate_raincloud.png` | Yes |
| `images/eda/association/q04_bmi_density.png` | Yes |
| `images/eda/association/q05_random_blood_sugar_ecdf.png` | Yes |
| `images/eda/association/q06_haemoglobin_split_violin.png` | Yes |
| `images/eda/association/risk_flag_prevalence_heatmap.png` | Yes |
| `images/eda/association/sensitivity_window_counts_35_45.png` | Yes |
| `images/eda/association/shared_profile_dumbbell_chart.png` | Yes |
| `images/eda/association/shared_profile_radar_chart.png` | Yes |

### Notebook Size
| Item | Value |
| --- | --- |
| Notebook file | `notebooks/EDA/06_pcos_heart_association.ipynb` |
| Markdown cells | 48 |
| Code cells | 37 |
| Total cells | 85 |

## Step-by-Step Walkthrough
This section is following the notebook in cell order. Every code cell is included so you can teach the notebook step by step without guessing what happened.

### Cell 1 - Markdown
**What this cell is doing:** PCOS and Cardiovascular Association Analysis.
**Why this step matters:** This matters because it sets the purpose of the notebook before any code starts.

**Notebook markdown content:**

# PCOS and Cardiovascular Association Analysis

## Objective
This notebook is checking thesis Hypothesis H1: **there is a relationship between PCOS and cardiovascular risk factors**. The analysis has two parts. The first layer looks at heart, blood pressure, and metabolic differences **within the cleaned PCOS cohort itself**. The second part uses the cleaned female heart dataset as a **group-level reference group**, not as a linked or merged cohort.

## Analytical Positioning
The notebook uses:
- `cleaned_data/PCOS_full_cleaned.csv` as the main data source
- `cleaned_data/heart_cleaned.csv` as a heart-risk reference source

The infertility and survey datasets are not used here because this chapter only looks at the clinical PCOS dataset and the heart reference dataset.

## Non-Negotiable Caveats
1. The PCOS dataset and the heart dataset have **different individuals**, so every cross-dataset comparison is being treated as **group-level and non-causal**.
2. The heart reference cohort is already a **small deduplicated female-only sample**, and it is notably older than the PCOS cohort. Age is therefore being treated as the **dominant factor that can distort the comparison** from the start.
3. The cleaned PCOS dataset has **actual blood-pressure columns** (`systolic_bp_mmhg`, `diastolic_bp_mmhg`) as well as `pulse_rate_bpm`, so this notebook is analysing blood pressure directly within the PCOS cohort and is using pulse rate only as a complementary nervous-system proxy.
4. This notebook does not merge rows across datasets, does not run cross-dataset statistical significance tests, and does not claim cause and effect. It is building Chapter 4 evidence through careful within-dataset analysis and careful group-level comparison.

### Cell 2 - Markdown
**What this cell is doing:** Setup and Helper Tools.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Setup and Helper Tools

This section importing the analysis libraries, resolving the project root safely, preparing the figure export directory, and defining reusable helpers for tables, figures, and effect-size calculations.

### Cell 3 - Code
**What this code is doing:** Importing the required libraries is supporting reproducible data analysis, figure export, and statistical testing.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.

```python
# Importing the required libraries is supporting reproducible data analysis, figure export, and statistical testing.
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from scipy import stats

# Fixing the random seed is keeping bootstrap intervals and any jitter stable across reruns.
np.random.seed(42)

# Configuring the visual theme is keeping the notebook figures compact, readable, and chapter-ready.
sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")

# Resolving the project root is keeping the notebook runnable from the repository root or from inside notebooks/EDA.
def resolve_project_root() -> Path:
    current = Path.cwd().resolve()
    candidates = [current, current.parent, current.parent.parent]
    for candidate in candidates:
        if (candidate / "cleaned_data").exists() and (candidate / "images").exists():
            return candidate
    return current

PROJECT_ROOT = resolve_project_root()
PCOS_PATH = PROJECT_ROOT / "cleaned_data" / "PCOS_full_cleaned.csv"
HEART_PATH = PROJECT_ROOT / "cleaned_data" / "heart_cleaned.csv"
IMAGE_DIR = PROJECT_ROOT / "images" / "eda" / "association"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Locking the colour system is keeping the subgroup identity consistent across the notebook.
GROUP_PALETTE = {
    "PCOS Negative": "#3b82f6",
    "PCOS Positive": "#e76f51",
    "Heart Negative": "#7a7a7a",
    "Heart Positive": "#f4a261",
}
PCOS_PALETTE = {
    "PCOS Negative": "#3b82f6",
    "PCOS Positive": "#e76f51",
}

# Saving figures through one helper is keeping filenames, resolution, and whitespace trimming consistent.
def save_figure(fig: plt.Figure, slug: str) -> Path:
    output_path = IMAGE_DIR / slug
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path

# Styling axes through a small helper is reducing repeated plotting code and visual clutter.
def polish_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    sns.despine(ax=ax, trim=True)
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.7)

# Summarising one numeric feature by one grouping variable is supporting the table-first workflow.
def numeric_summary(data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    summary = (
        data.groupby(group_col)[value_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    return summary.round(3)

# Formatting mean and standard deviation together is supporting publication-style descriptive tables.
def mean_sd(series: pd.Series) -> str:
    return f"{series.mean():.2f} ± {series.std():.2f}"

# Building prevalence tables is supporting the engineered risk-flag comparison.
def prevalence(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else np.nan

# Drawing a compact raincloud-style chart is combining density, point spread, and central tendency in one view.
def raincloud_plot(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, order: list[str]) -> None:
    sns.violinplot(
        data=data,
        x=x,
        y=y,
        order=order,
        palette=PCOS_PALETTE,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=0.9,
        ax=ax,
    )
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        width=0.18,
        showcaps=False,
        boxprops={"facecolor": "white", "edgecolor": "#2f2f2f", "alpha": 0.9},
        medianprops={"color": "#2f2f2f", "linewidth": 1.2},
        whiskerprops={"linewidth": 0},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        order=order,
        color="#1f2937",
        size=2.6,
        alpha=0.30,
        jitter=0.18,
        ax=ax,
    )

# Converting a Mann-Whitney U statistic into a rank-biserial effect size is supporting clinically readable reporting.
def rank_biserial_from_u(u_statistic: float, n_group_a: int, n_group_b: int) -> float:
    return (2 * u_statistic) / (n_group_a * n_group_b) - 1

# Bootstrapping the rank-biserial interval is supporting uncertainty reporting around each effect size.
def bootstrap_rank_biserial(
    group_a: pd.Series,
    group_b: pd.Series,
    n_boot: int = 1500,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    a_values = group_a.dropna().to_numpy()
    b_values = group_b.dropna().to_numpy()
    for _ in range(n_boot):
        a_sample = rng.choice(a_values, size=len(a_values), replace=True)
        b_sample = rng.choice(b_values, size=len(b_values), replace=True)
        u_statistic = stats.mannwhitneyu(a_sample, b_sample, alternative="two-sided", method="auto").statistic
        values.append(rank_biserial_from_u(u_statistic, len(a_sample), len(b_sample)))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)

# Labelling the effect-size magnitude is making the statistical section easier to interpret in thesis prose.
def effect_size_label(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value < 0.10:
        return "negligible"
    if absolute_value < 0.30:
        return "small"
    if absolute_value < 0.50:
        return "medium"
    return "large"

# Normalising a metric to the 0-1 range is preparing the cross-cohort profile charts.
def min_max_normalise(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if math.isclose(float(maximum), float(minimum)):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)
```

### Cell 4 - Markdown
**What this cell is doing:** Section 1: Load Data and Check the Age Gap.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Section 1: Load Data and Check the Age Gap

This section loading both cleaned datasets, auditing the available columns, and surfacing the age structure immediately because the age gap is expected to dominate every group-level comparison between different groups of people that follows.

### Cell 5 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**What do these cleaned datasets look like, and what problems must we note before comparing them?**

The following audit is printing the raw shapes, dtypes, missingness counts, subgroup counts, and selected descriptive summaries before any analysis interpretation is being attempted.

### Cell 6 - Code
**What this code is doing:** Loading the cleaned datasets and standardising the working names is keeping the analysis readable without altering the source files.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Loading the cleaned datasets and standardising the working names is keeping the analysis readable without altering the source files.
pcos_df = pd.read_csv(PCOS_PATH).rename(
    columns={
        "age_yrs": "age",
        "pulse_rate_bpm": "pulse_rate",
        "respiratory_rate_breaths_min": "respiratory_rate",
        "hb_g_dl": "haemoglobin",
        "rbs_mg_dl": "random_blood_sugar",
        "systolic_bp_mmhg": "systolic_bp",
        "diastolic_bp_mmhg": "diastolic_bp",
    }
)

heart_df = pd.read_csv(HEART_PATH).rename(columns={"heart_disease_target": "heart_target"})

# Building readable subgroup labels is supporting clearer tables and figures throughout the notebook.
pcos_df["pcos_label"] = pcos_df["pcos_y_n"].map({0: "PCOS Negative", 1: "PCOS Positive"})
heart_df["heart_label"] = heart_df["heart_target"].map({0: "Heart Negative", 1: "Heart Positive"})

# Printing the high-level audit is surfacing completeness and cohort structure before analysis begins.
print("PCOS dataset shape:", pcos_df.shape)
print("Heart dataset shape:", heart_df.shape)
print("\nPCOS missing values:")
display(pcos_df.isnull().sum().to_frame("missing_count").query("missing_count > 0"))
print("\nHeart missing values:")
display(heart_df.isnull().sum().to_frame("missing_count").query("missing_count > 0"))
print("\nPCOS subgroup counts:")
display(pcos_df["pcos_label"].value_counts().rename_axis("subgroup").to_frame("n"))
print("\nHeart subgroup counts:")
display(heart_df["heart_label"].value_counts().rename_axis("subgroup").to_frame("n"))
```

### Cell 7 - Code
**What this code is doing:** Printing the dtype inventory is documenting the analytical schema being used in this notebook.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Printing the dtype inventory is documenting the analytical schema being used in this notebook.
pcos_dtype_table = pcos_df.dtypes.rename("dtype").reset_index().rename(columns={"index": "pcos_column"})
heart_dtype_table = heart_df.dtypes.rename("dtype").reset_index().rename(columns={"index": "heart_column"})

print("PCOS dtype inventory:")
display(pcos_dtype_table)
print("\nHeart dtype inventory:")
display(heart_dtype_table)
```

### Cell 8 - Code
**What this code is doing:** Building the CVD-relevant column inventory is clarifying which features are directly comparable, proxy comparable, or dataset-specific.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Building the CVD-relevant column inventory is clarifying which features are directly comparable, proxy comparable, or dataset-specific.
column_inventory = pd.DataFrame(
    [
        {"PCOS column": "age", "Heart column": "age", "Comparability": "Direct", "Confound risk": "High", "Notes": "Age is the only clean direct demographic bridge."},
        {"PCOS column": "systolic_bp", "Heart column": "resting_bp", "Comparability": "Partial direct", "Confound risk": "High", "Notes": "Both are pressure metrics, but the cohorts differ strongly by age and source."},
        {"PCOS column": "diastolic_bp", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This variable is informative within the PCOS cohort only."},
        {"PCOS column": "pulse_rate", "Heart column": "max_heart_rate", "Comparability": "Weak proxy", "Confound risk": "Very high", "Notes": "Resting pulse and maximum achieved heart rate are not equivalent physiological measures."},
        {"PCOS column": "bmi", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This variable is acting as a cardiometabolic bridge variable within PCOS."},
        {"PCOS column": "random_blood_sugar", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This variable is serving as a metabolic bridge variable within PCOS."},
        {"PCOS column": "haemoglobin", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This variable is supporting an exploratory anaemia-linked pathway analysis."},
        {"PCOS column": "cycle_regularity_code", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This ordered recorded code is representing reproductive disruption rather than a CVD marker."},
        {"PCOS column": "—", "Heart column": "cholesterol", "Comparability": "Heart only", "Confound risk": "N/A", "Notes": "This variable is serving as cardiovascular context rather than a shared measure."},
    ]
)
display(column_inventory)
```

### Cell 9 - Markdown
**What this cell is doing:** Ecological Comparison Caveat.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

### Ecological Comparison Caveat

The column inventory above is showing that this notebook is operating with a **mixed comparability matrix**, not a matched-clinical panel. Age is being compared directly, systolic pressure is being compared cautiously against resting pressure, and several clinically interesting variables are appearing in only one dataset. This means the heart cohort works as an group-level cardiovascular benchmark rather than as a patient-matched validation source.

### Cell 10 - Code
**What this code is doing:** Printing a compact descriptive panel is documenting the scale and spread of the main cardiometabolic variables available in each cohort.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Printing a compact descriptive panel is documenting the scale and spread of the main cardiometabolic variables available in each cohort.
pcos_describe = pcos_df[["age", "bmi", "pulse_rate", "haemoglobin", "random_blood_sugar", "systolic_bp", "diastolic_bp"]].describe().round(3)
heart_describe = heart_df[["age", "resting_bp", "max_heart_rate", "cholesterol"]].describe().round(3)
side_by_side_describe = pd.concat({"PCOS": pcos_describe, "Heart": heart_describe}, axis=1)
display(side_by_side_describe)
```

### Cell 11 - Code
**What this code is doing:** Calculating the age summary early is quantifying the main confound before any cross-cohort interpretation is attempted.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Calculating the age summary early is quantifying the main confound before any cross-cohort interpretation is attempted.
age_summary = pd.DataFrame(
    [
        {"cohort": "PCOS overall", "n": len(pcos_df), "mean_age": pcos_df["age"].mean(), "median_age": pcos_df["age"].median(), "sd_age": pcos_df["age"].std()},
        {"cohort": "PCOS Positive", "n": len(pcos_df.query('pcos_y_n == 1')), "mean_age": pcos_df.query('pcos_y_n == 1')["age"].mean(), "median_age": pcos_df.query('pcos_y_n == 1')["age"].median(), "sd_age": pcos_df.query('pcos_y_n == 1')["age"].std()},
        {"cohort": "PCOS Negative", "n": len(pcos_df.query('pcos_y_n == 0')), "mean_age": pcos_df.query('pcos_y_n == 0')["age"].mean(), "median_age": pcos_df.query('pcos_y_n == 0')["age"].median(), "sd_age": pcos_df.query('pcos_y_n == 0')["age"].std()},
        {"cohort": "Heart overall", "n": len(heart_df), "mean_age": heart_df["age"].mean(), "median_age": heart_df["age"].median(), "sd_age": heart_df["age"].std()},
        {"cohort": "Heart Positive", "n": len(heart_df.query('heart_target == 1')), "mean_age": heart_df.query('heart_target == 1')["age"].mean(), "median_age": heart_df.query('heart_target == 1')["age"].median(), "sd_age": heart_df.query('heart_target == 1')["age"].std()},
        {"cohort": "Heart Negative", "n": len(heart_df.query('heart_target == 0')), "mean_age": heart_df.query('heart_target == 0')["age"].mean(), "median_age": heart_df.query('heart_target == 0')["age"].median(), "sd_age": heart_df.query('heart_target == 0')["age"].std()},
    ]
).round(3)
age_gap_years = heart_df["age"].mean() - pcos_df["age"].mean()
display(age_summary)
print(f"Mean age gap between the full heart cohort and the full PCOS cohort: {age_gap_years:.2f} years")
print("Age-gap warning threshold for this notebook: > 5 years")
```

### Cell 12 - Code
**What this code is doing:** Plotting the age distributions immediately is making the dominant age confound visible before any later ecological comparison is read.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.
**Saved figure or image output in this cell:** `age_distribution_ecological_reference.png`

```python
# Plotting the age distributions immediately is making the dominant age confound visible before any later ecological comparison is read.
age_overlay = pd.concat(
    [
        pcos_df[["age", "pcos_label"]].rename(columns={"pcos_label": "group"}),
        heart_df[["age", "heart_label"]].rename(columns={"heart_label": "group"}),
    ],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
line_styles = {
    "PCOS Negative": "-",
    "PCOS Positive": "--",
    "Heart Negative": "-.",
    "Heart Positive": ":",
}
for group_name in ["PCOS Negative", "PCOS Positive", "Heart Negative", "Heart Positive"]:
    subset = age_overlay.loc[age_overlay["group"] == group_name, "age"]
    sns.kdeplot(
        subset,
        ax=ax,
        label=group_name,
        color=GROUP_PALETTE[group_name],
        linewidth=2.0,
        linestyle=line_styles[group_name],
        fill=False,
        warn_singular=False,
    )

ax.axvline(35, color="#374151", linestyle="--", linewidth=1.1, label="Age 35 reference")
ax.set_title("Age Distribution Across PCOS and Heart Subgroups")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, ncol=2)
save_figure(fig, "age_distribution_ecological_reference.png")
plt.show()
```

### Cell 13 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The age tables and the overlaid density curves above are showing that the heart cohort is materially older than the PCOS cohort, with an average gap of roughly **24 years**. This gap is already far beyond the five-year threshold that would have triggered a major warning, so every cross-dataset comparison in the rest of the notebook must be read as **descriptive context rather than matched-risk evidence**. In practical terms, the heart dataset is helping us understand the later-life cardiovascular reference profile, but it cannot by itself tell us that the younger PCOS-positive group is already expressing the same risk burden at the same level.

### Cell 14 - Markdown
**What this cell is doing:** Section 2: What Can We Compare?.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Section 2: What Can We Compare?

This section turns the cleaned columns into a defensible comparison framework so that direct comparisons, proxy comparisons, and one-dataset-only features are not mixed together uncritically.

### Cell 15 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Which variables can we compare directly, which ones are only rough proxies, and which ones should stay separate?**

The matrix below is defining the comparison logic that will be used for the rest of the notebook.

### Cell 16 - Code
**What this code is doing:** Rebuilding the comparability matrix in a thesis-style layout is making the analytical scope explicit before testing begins.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Rebuilding the comparability matrix in a thesis-style layout is making the analytical scope explicit before testing begins.
comparable_feature_matrix = pd.DataFrame(
    [
        {"PCOS column": "age", "Heart column": "age", "Comparability": "Direct", "Confound risk": "High (age gap)", "Notes": "This is the core shared demographic feature."},
        {"PCOS column": "systolic_bp", "Heart column": "resting_bp", "Comparability": "Partial direct cardiovascular comparison", "Confound risk": "High", "Notes": "Both are pressure measures, but age and cohort structure remain major confounders."},
        {"PCOS column": "diastolic_bp", "Heart column": "—", "Comparability": "PCOS only", "Confound risk": "N/A", "Notes": "This variable is remaining within the PCOS cohort analysis."},
        {"PCOS column": "pulse_rate", "Heart column": "max_heart_rate", "Comparability": "Weak proxy only", "Confound risk": "Very high", "Notes": "These are different physiological constructs and are being interpreted cautiously."},
        {"PCOS column": "bmi", "Heart column": "—", "Comparability": "PCOS only cardiometabolic bridge", "Confound risk": "N/A", "Notes": "This variable is informing the metabolic side of thesis H₁."},
        {"PCOS column": "random_blood_sugar", "Heart column": "—", "Comparability": "PCOS only metabolic bridge", "Confound risk": "N/A", "Notes": "This variable is acting as a partial glucose-risk indicator within PCOS."},
        {"PCOS column": "haemoglobin", "Heart column": "—", "Comparability": "PCOS only exploratory pathway", "Confound risk": "N/A", "Notes": "This variable is enabling an exploratory anaemia-linked cardiovascular discussion."},
        {"PCOS column": "cycle_regularity_code", "Heart column": "—", "Comparability": "PCOS only reproductive signal", "Confound risk": "N/A", "Notes": "This ordered code is remaining a reproductive-disruption feature rather than a shared CVD metric."},
        {"PCOS column": "—", "Heart column": "cholesterol", "Comparability": "Heart only reference", "Confound risk": "N/A", "Notes": "This variable is adding context about the cardiovascular reference cohort."},
    ]
)
display(comparable_feature_matrix)
```

### Cell 17 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The matrix is showing that the notebook has only a **small set of honest bridges** between the two cohorts. Age is the cleanest common feature, systolic-versus-resting pressure is a cautious cardiovascular bridge, and pulse-versus-maximum-heart-rate is only a weak proxy relationship. This matters because the thesis argument becomes stronger when direct and proxy comparisons are kept separate rather than blended into one overconfident claim.

### Cell 18 - Markdown
**What this cell is doing:** Section 3: Main Profile Tables.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Section 3: Main Profile Tables

This section constructing the descriptive baseline tables that will anchor the heart and metabolic interpretation of thesis H1 before any statistical testing is introduced.

### Cell 19 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**How do the PCOS-negative and PCOS-positive groups differ on the main heart and metabolic variables, and how should we place those results beside the heart reference group?**

Two tables are being created here. Table 1A stays strictly within the PCOS cohort, while Table 1B is adding the heart cohort only as group-level reference context.

### Cell 20 - Code
**What this code is doing:** Building Table 1A is establishing the primary within-PCOS descriptive profile for thesis H₁.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Building Table 1A is establishing the primary within-PCOS descriptive profile for thesis H₁.
pcos_table_1a = (
    pcos_df.groupby("pcos_label")
    .apply(
        lambda frame: pd.Series(
            {
                "n": len(frame),
                "Age (mean ± SD)": mean_sd(frame["age"]),
                "BMI (mean ± SD)": mean_sd(frame["bmi"]),
                "Systolic BP (mean ± SD)": mean_sd(frame["systolic_bp"]),
                "Diastolic BP (mean ± SD)": mean_sd(frame["diastolic_bp"]),
                "Pulse rate (mean ± SD)": mean_sd(frame["pulse_rate"]),
                "Random blood sugar (mean ± SD)": mean_sd(frame["random_blood_sugar"]),
                "Haemoglobin (mean ± SD)": mean_sd(frame["haemoglobin"]),
                "Cycle-regularity code (mean ± SD)": mean_sd(frame["cycle_regularity_code"]),
            }
        )
    )
    .reset_index()
    .rename(columns={"pcos_label": "Subgroup"})
)
display(pcos_table_1a)
```

### Cell 21 - Code
**What this code is doing:** Building Table 1B is placing the PCOS results beside the heart reference cohort without implying patient-level alignment.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Building Table 1B is placing the PCOS results beside the heart reference cohort without implying patient-level alignment.
table_1b = pd.DataFrame(
    [
        {
            "Subgroup": "PCOS Negative",
            "n": len(pcos_df.query('pcos_y_n == 0')),
            "Age (mean ± SD)": mean_sd(pcos_df.query('pcos_y_n == 0')["age"]),
            "Pressure metric (mean ± SD)": mean_sd(pcos_df.query('pcos_y_n == 0')["systolic_bp"]),
            "Pressure >130 %": f"{prevalence(pcos_df.query('pcos_y_n == 0')['systolic_bp'] > 130):.1f}",
            "Pulse / max-HR metric": mean_sd(pcos_df.query('pcos_y_n == 0')["pulse_rate"]),
            "BMI (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 0')["bmi"]),
            "Random blood sugar (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 0')["random_blood_sugar"]),
            "Haemoglobin (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 0')["haemoglobin"]),
            "Cholesterol (Heart only)": "reference only",
        },
        {
            "Subgroup": "PCOS Positive",
            "n": len(pcos_df.query('pcos_y_n == 1')),
            "Age (mean ± SD)": mean_sd(pcos_df.query('pcos_y_n == 1')["age"]),
            "Pressure metric (mean ± SD)": mean_sd(pcos_df.query('pcos_y_n == 1')["systolic_bp"]),
            "Pressure >130 %": f"{prevalence(pcos_df.query('pcos_y_n == 1')['systolic_bp'] > 130):.1f}",
            "Pulse / max-HR metric": mean_sd(pcos_df.query('pcos_y_n == 1')["pulse_rate"]),
            "BMI (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 1')["bmi"]),
            "Random blood sugar (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 1')["random_blood_sugar"]),
            "Haemoglobin (PCOS only)": mean_sd(pcos_df.query('pcos_y_n == 1')["haemoglobin"]),
            "Cholesterol (Heart only)": "reference only",
        },
        {
            "Subgroup": "Heart Negative",
            "n": len(heart_df.query('heart_target == 0')),
            "Age (mean ± SD)": mean_sd(heart_df.query('heart_target == 0')["age"]),
            "Pressure metric (mean ± SD)": mean_sd(heart_df.query('heart_target == 0')["resting_bp"]),
            "Pressure >130 %": f"{prevalence(heart_df.query('heart_target == 0')['resting_bp'] > 130):.1f}",
            "Pulse / max-HR metric": mean_sd(heart_df.query('heart_target == 0')["max_heart_rate"]),
            "BMI (PCOS only)": "n/a",
            "Random blood sugar (PCOS only)": "n/a",
            "Haemoglobin (PCOS only)": "n/a",
            "Cholesterol (Heart only)": mean_sd(heart_df.query('heart_target == 0')["cholesterol"]),
        },
        {
            "Subgroup": "Heart Positive",
            "n": len(heart_df.query('heart_target == 1')),
            "Age (mean ± SD)": mean_sd(heart_df.query('heart_target == 1')["age"]),
            "Pressure metric (mean ± SD)": mean_sd(heart_df.query('heart_target == 1')["resting_bp"]),
            "Pressure >130 %": f"{prevalence(heart_df.query('heart_target == 1')['resting_bp'] > 130):.1f}",
            "Pulse / max-HR metric": mean_sd(heart_df.query('heart_target == 1')["max_heart_rate"]),
            "BMI (PCOS only)": "n/a",
            "Random blood sugar (PCOS only)": "n/a",
            "Haemoglobin (PCOS only)": "n/a",
            "Cholesterol (Heart only)": mean_sd(heart_df.query('heart_target == 1')["cholesterol"]),
        },
    ]
)
display(table_1b)
```

### Cell 22 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


Table 1A is showing the **primary thesis evidence base** because it compares PCOS-positive and PCOS-negative women inside the same cleaned cohort. Table 1B is broadening the view by adding the older female heart cohort, but it should be read as background benchmarking rather than direct biological equivalence. The most defensible use of the heart table is to highlight how the PCOS group sits earlier in the age path over time while still showing measurable heart and metabolic burden that can be tracked clinically.

### Cell 23 - Markdown
**What this cell is doing:** Section 4: Compare the Main Distributions.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Section 4: Compare the Main Distributions

This section is checking the core heart and metabolic features that are most relevant to thesis H1. Every question is being answered with a table first, a plot second, and an interpretation third.

### Cell 24 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Are PCOS-positive patients younger than the heart-disease reference group, and what does that age pattern mean?**

This question stays group-level from the start because the PCOS and heart datasets have different individuals.

### Cell 25 - Code
**What this code is doing:** Printing the age summary table is quantifying the location and spread of each subgroup before the density view is read.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Printing the age summary table is quantifying the location and spread of each subgroup before the density view is read.
age_group_summary = pd.concat(
    [
        numeric_summary(pcos_df.rename(columns={"pcos_label": "group"}), "group", "age"),
        numeric_summary(heart_df.rename(columns={"heart_label": "group"}), "group", "age"),
    ],
    ignore_index=True,
)
display(age_group_summary)
```

### Cell 26 - Code
**What this code is doing:** Replotting the age densities in the question section is keeping the ecological comparison visually central.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q01_age_trajectory_kde.png`

```python
# Replotting the age densities in the question section is keeping the ecological comparison visually central.
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for group_name in ["PCOS Negative", "PCOS Positive", "Heart Negative", "Heart Positive"]:
    subset = age_overlay.loc[age_overlay["group"] == group_name, "age"]
    sns.kdeplot(
        subset,
        ax=ax,
        label=group_name,
        color=GROUP_PALETTE[group_name],
        linewidth=2.1,
        linestyle=line_styles[group_name],
        fill=False,
        warn_singular=False,
    )
ax.axvline(35, color="#374151", linestyle="--", linewidth=1.1, label="Age 35 reference")
ax.set_title("Age Trajectory Across PCOS and Heart Subgroups")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, ncol=2)
save_figure(fig, "q01_age_trajectory_kde.png")
plt.show()
```

### Cell 27 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The age densities are showing a clear separation between the younger reproductive-age PCOS cohort and the older cardiovascular reference cohort. This pattern supports the thesis framing that PCOS may be appearing earlier on a longer heart and metabolic path over time, but it does **not** show that the PCOS-positive subgroup is already equivalent to the heart-positive subgroup in absolute cardiovascular burden. The value of this figure is therefore chronological and background: it helps position the PCOS-positive group on a possible earlier-life pathway that would need longitudinal follow-up to confirm.

### Cell 28 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Does systolic blood pressure differ inside the PCOS dataset, and how does it sit beside the older heart reference pressure values?**

The heart cohort is being used as external descriptive reference only. No cross-dataset statistical test is being attempted here.

### Cell 29 - Code
**What this code is doing:** Printing the pressure summaries is keeping the PCOS and heart metrics distinct before the reference lines are drawn.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Printing the pressure summaries is keeping the PCOS and heart metrics distinct before the reference lines are drawn.
pcos_systolic_summary = numeric_summary(pcos_df, "pcos_label", "systolic_bp")
heart_resting_summary = numeric_summary(heart_df, "heart_label", "resting_bp")
print("PCOS systolic blood-pressure summary:")
display(pcos_systolic_summary)
print("\nHeart resting-pressure reference summary:")
display(heart_resting_summary)
```

### Cell 30 - Code
**What this code is doing:** Plotting the within-PCOS systolic distribution with heart reference lines is separating direct cohort evidence from ecological context.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q02_systolic_pressure_with_heart_reference.png`

```python
# Plotting the within-PCOS systolic distribution with heart reference lines is separating direct cohort evidence from ecological context.
fig, ax = plt.subplots(figsize=(6.2, 4.0))
sns.violinplot(
    data=pcos_df,
    x="pcos_label",
    y="systolic_bp",
    order=["PCOS Negative", "PCOS Positive"],
    palette=PCOS_PALETTE,
    inner=None,
    cut=0,
    linewidth=1.0,
    ax=ax,
)
sns.stripplot(
    data=pcos_df,
    x="pcos_label",
    y="systolic_bp",
    order=["PCOS Negative", "PCOS Positive"],
    color="#111827",
    alpha=0.28,
    size=2.4,
    jitter=0.18,
    ax=ax,
)
heart_negative_mean_bp = heart_df.query("heart_target == 0")["resting_bp"].mean()
heart_positive_mean_bp = heart_df.query("heart_target == 1")["resting_bp"].mean()
ax.axhline(heart_negative_mean_bp, color=GROUP_PALETTE["Heart Negative"], linestyle="--", linewidth=1.2, label="Heart Negative mean resting pressure")
ax.axhline(heart_positive_mean_bp, color=GROUP_PALETTE["Heart Positive"], linestyle=":", linewidth=1.4, label="Heart Positive mean resting pressure")
ax.axhline(130, color="#374151", linestyle="-.", linewidth=1.0, label="130 mmHg reference")
ax.set_title("Systolic Pressure in PCOS Groups with Heart Reference Lines")
ax.set_xlabel("")
ax.set_ylabel("Pressure (mmHg)")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, loc="upper right")
save_figure(fig, "q02_systolic_pressure_with_heart_reference.png")
plt.show()
```

### Cell 31 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The systolic-pressure distributions inside the PCOS cohort are appearing heavily overlapped, which suggests that systolic pressure alone is not strongly separating PCOS-positive from PCOS-negative women in this dataset. The heart reference lines are sitting higher, but that pattern is being read cautiously because the heart cohort is much older and is not a matched comparison group. In thesis terms, systolic pressure is still worth monitoring clinically, but it is looking more like a **background cardiovascular context variable** than a strong standalone discriminator of PCOS status.

### Cell 32 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Is pulse rate higher in PCOS-positive women, and does it add useful heart-related context without being confused with blood pressure?**

This analysis is staying strictly within the PCOS cohort because resting pulse and maximum heart rate are not physiologically equivalent enough for direct cross-cohort inference.

### Cell 33 - Code
**What this code is doing:** Printing the pulse summaries is quantifying central tendency and spread before the raincloud plot is read.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Printing the pulse summaries is quantifying central tendency and spread before the raincloud plot is read.
pulse_summary = numeric_summary(pcos_df, "pcos_label", "pulse_rate")
display(pulse_summary)
```

### Cell 34 - Code
**What this code is doing:** Plotting the pulse raincloud is showing density, spread, and threshold context in one compact figure.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.
**Saved figure or image output in this cell:** `q03_pulse_rate_raincloud.png`

```python
# Plotting the pulse raincloud is showing density, spread, and threshold context in one compact figure.
fig, ax = plt.subplots(figsize=(6.2, 4.0))
raincloud_plot(ax, pcos_df, "pcos_label", "pulse_rate", ["PCOS Negative", "PCOS Positive"])
ax.axhline(60, color="#6b7280", linestyle="--", linewidth=1.0, label="60 bpm reference")
ax.axhline(100, color="#9a3412", linestyle="--", linewidth=1.1, label="100 bpm reference")
ax.set_title("Pulse-Rate Distribution by PCOS Status")
ax.set_xlabel("")
ax.set_ylabel("Pulse rate (bpm)")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, loc="upper right")
save_figure(fig, "q03_pulse_rate_raincloud.png")
plt.show()
```

### Cell 35 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The raincloud plot is showing substantial overlap between the two PCOS groups, with any upward shift in pulse rate appearing modest rather than dramatic. That means pulse rate adds more as a **supporting nervous-system context marker** than as a strong diagnostic separator on its own. Clinically, a higher resting pulse can still matter because it may reflect sympathetic or nervous-system strain, but this dataset suggests that the pulse signal should be interpreted alongside metabolic and menstrual-disruption markers rather than in isolation.

### Cell 36 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Does higher BMI act as a bridge between PCOS and later heart risk?**

This analysis is reading BMI as a within-PCOS bridge variable that may connect reproductive endocrinology with later metabolic and cardiovascular burden.

### Cell 37 - Code
**What this code is doing:** Printing the BMI summary table is quantifying the shift before the density plot is interpreted.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Printing the BMI summary table is quantifying the shift before the density plot is interpreted.
bmi_summary = numeric_summary(pcos_df, "pcos_label", "bmi")
display(bmi_summary)
```

### Cell 38 - Code
**What this code is doing:** Plotting the BMI densities is showing how the full distributions move relative to the overweight and obesity thresholds.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q04_bmi_density.png`

```python
# Plotting the BMI densities is showing how the full distributions move relative to the overweight and obesity thresholds.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for label in ["PCOS Negative", "PCOS Positive"]:
    subset = pcos_df.loc[pcos_df["pcos_label"] == label, "bmi"]
    sns.kdeplot(
        subset,
        ax=ax,
        fill=True,
        alpha=0.28,
        linewidth=2.0,
        color=PCOS_PALETTE[label],
        label=label,
        warn_singular=False,
    )
ax.axvline(25, color="#6b7280", linestyle="--", linewidth=1.0, label="BMI 25")
ax.axvline(30, color="#9a3412", linestyle="--", linewidth=1.1, label="BMI 30")
ax.set_title("BMI Distribution by PCOS Status")
ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel("Density")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False)
save_figure(fig, "q04_bmi_density.png")
plt.show()
```

### Cell 39 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The BMI distributions are showing the clearest rightward shift among the continuous within-PCOS variables examined so far, with the PCOS-positive group occupying more of the overweight and upper-tail range. This pattern fits the thesis mechanism that places adiposity and insulin-resistance-related burden along the pathway linking PCOS to later cardiovascular risk. In practical terms, BMI is behaving like one of the most easy to read heart and metabolic markers in the notebook because it is clinically familiar, easy to monitor, and biologically connected to multiple downstream risk processes.

### Cell 40 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Does random blood sugar rise in the PCOS-positive group, and does that strengthen the metabolic-risk part of thesis H1?**

This analysis stays within the PCOS cohort because the heart reference cohort does not contain a directly comparable glucose measure.

### Cell 41 - Code
**What this code is doing:** Printing the random-blood-sugar summary table is quantifying the distribution before the ECDF view is used.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Printing the random-blood-sugar summary table is quantifying the distribution before the ECDF view is used.
rbs_summary = numeric_summary(pcos_df, "pcos_label", "random_blood_sugar")
display(rbs_summary)
```

### Cell 42 - Code
**What this code is doing:** Plotting the ECDF curves is showing cumulative separation without assuming a specific parametric shape.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q05_random_blood_sugar_ecdf.png`

```python
# Plotting the ECDF curves is showing cumulative separation without assuming a specific parametric shape.
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for label in ["PCOS Negative", "PCOS Positive"]:
    subset = pcos_df.loc[pcos_df["pcos_label"] == label, "random_blood_sugar"]
    sns.ecdfplot(
        subset,
        ax=ax,
        stat="proportion",
        linewidth=2.0,
        color=PCOS_PALETTE[label],
        label=label,
    )
ax.axvline(100, color="#6b7280", linestyle="--", linewidth=1.0, label="100 mg/dL reference")
ax.axvline(140, color="#9a3412", linestyle="--", linewidth=1.1, label="140 mg/dL reference")
ax.set_title("Random Blood Sugar by PCOS Status")
ax.set_xlabel("Random blood sugar (mg/dL)")
ax.set_ylabel("Cumulative proportion")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, loc="lower right")
save_figure(fig, "q05_random_blood_sugar_ecdf.png")
plt.show()
```

### Cell 43 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The ECDF curves are showing only a modest separation, which means the glucose-related signal is present but not sharply exclusive to the PCOS-positive group. That is still clinically useful because even a moderate upward shift in random blood sugar can reinforce the interpretation that metabolic dysregulation is clustering around PCOS status. For the thesis, this variable is likely to matter most when it is read together with BMI rather than as a standalone screening feature.

### Cell 44 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Does haemoglobin differ by PCOS status, and does that support an exploratory anaemia-related heart-risk discussion?**

This analysis is being phrased neutrally because the data may or may not support a directional haemoglobin difference.

### Cell 45 - Code
**What this code is doing:** Printing the haemoglobin summary table is documenting the central tendency, spread, and lower-tail context before the split violin is read.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Printing the haemoglobin summary table is documenting the central tendency, spread, and lower-tail context before the split violin is read.
haemoglobin_summary = numeric_summary(pcos_df, "pcos_label", "haemoglobin")
display(haemoglobin_summary)
```

### Cell 46 - Code
**What this code is doing:** Plotting a split violin is comparing the full haemoglobin distributions while marking the adult-women anaemia threshold.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q06_haemoglobin_split_violin.png`

```python
# Plotting a split violin is comparing the full haemoglobin distributions while marking the adult-women anaemia threshold.
haemoglobin_plot_df = pcos_df.copy()
haemoglobin_plot_df["comparison_anchor"] = "Haemoglobin"

fig, ax = plt.subplots(figsize=(5.8, 4.0))
sns.violinplot(
    data=haemoglobin_plot_df,
    x="comparison_anchor",
    y="haemoglobin",
    hue="pcos_label",
    split=True,
    inner=None,
    cut=0,
    linewidth=1.0,
    palette=PCOS_PALETTE,
    ax=ax,
)
ax.axhline(12, color="#9a3412", linestyle="--", linewidth=1.1, label="Hb 12 g/dL reference")
ax.set_title("Haemoglobin Distribution by PCOS Status")
ax.set_xlabel("")
ax.set_ylabel("Haemoglobin (g/dL)")
polish_axis(ax, grid_axis="y")
ax.legend(frameon=False, loc="upper right")
save_figure(fig, "q06_haemoglobin_split_violin.png")
plt.show()
```

### Cell 47 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The split violin is showing heavy overlap between the two haemoglobin distributions, which means haemoglobin acts as an exploratory rather than a decisive PCOS-associated marker in this dataset. That does not make the variable irrelevant, because lower haemoglobin can still increase cardiac workload through compensatory physiology, but the pattern here should be treated cautiously and described as **hypothesis-generating** rather than confirmatory. In the thesis chapter, haemoglobin therefore works best as a supporting pathway discussion rather than as a core screening signal.

### Cell 48 - Markdown
**What this cell is doing:** Section Synthesis.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

### Section Synthesis

The distribution analyses are showing a mixed heart and metabolic picture rather than one single dominant cardiovascular marker. BMI is emerging as the clearest bridge variable, random blood sugar is adding a moderate metabolic layer, and pulse rate plus haemoglobin add useful context without looking strongly good at showing a clear difference on their own. Systolic pressure remains clinically relevant, but in this dataset it is appearing more as part of a broader cardiovascular profile than as a sharp separator of PCOS status.

### Cell 49 - Markdown
**What this cell is doing:** Section 5: Build Simple Risk Flags.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Section 5: Build Simple Risk Flags

This section converting the continuous variables into clinically readable binary flags so that the prevalence of potential risk burden can be compared across the PCOS and heart subgroups.

### Cell 50 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Which simple risk flags separate the PCOS-positive group from the PCOS-negative group, and how do those patterns look beside the heart reference group?**

The cycle flag is being treated explicitly as a heuristic because `cycle_regularity_code` is a recorded ordered code rather than a documented binary irregular-cycle variable.

### Cell 51 - Code
**What this code is doing:** Engineering the binary risk flags is translating the cleaned measurements into clinically interpretable burden indicators.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Engineering the binary risk flags is translating the cleaned measurements into clinically interpretable burden indicators.
pcos_df["older_age"] = pcos_df["age"] > 35
pcos_df["overweight_bmi"] = pcos_df["bmi"] > 25
pcos_df["obese_bmi"] = pcos_df["bmi"] > 30
pcos_df["elevated_pulse"] = pcos_df["pulse_rate"] > 90
pcos_df["tachycardic_pulse"] = pcos_df["pulse_rate"] > 100
pcos_df["high_systolic_bp"] = pcos_df["systolic_bp"] > 130
pcos_df["elevated_rbs"] = pcos_df["random_blood_sugar"] > 100
pcos_df["low_haemoglobin"] = pcos_df["haemoglobin"] < 12
pcos_df["higher_cycle_irregularity_code"] = pcos_df["cycle_regularity_code"] >= 4

heart_df["older_age"] = heart_df["age"] > 35
heart_df["high_resting_bp"] = heart_df["resting_bp"] > 130

risk_flag_prevalence = pd.DataFrame(
    {
        "PCOS Negative": {
            "Older age (>35) %": prevalence(pcos_df.query("pcos_y_n == 0")["older_age"]),
            "Overweight BMI (>25) %": prevalence(pcos_df.query("pcos_y_n == 0")["overweight_bmi"]),
            "Obese BMI (>30) %": prevalence(pcos_df.query("pcos_y_n == 0")["obese_bmi"]),
            "Elevated pulse (>90) %": prevalence(pcos_df.query("pcos_y_n == 0")["elevated_pulse"]),
            "Tachycardic pulse (>100) %": prevalence(pcos_df.query("pcos_y_n == 0")["tachycardic_pulse"]),
            "High systolic BP (>130) %": prevalence(pcos_df.query("pcos_y_n == 0")["high_systolic_bp"]),
            "Elevated RBS (>100) %": prevalence(pcos_df.query("pcos_y_n == 0")["elevated_rbs"]),
            "Low haemoglobin (<12) %": prevalence(pcos_df.query("pcos_y_n == 0")["low_haemoglobin"]),
            "Higher cycle-irregularity code (>=4) %": prevalence(pcos_df.query("pcos_y_n == 0")["higher_cycle_irregularity_code"]),
        },
        "PCOS Positive": {
            "Older age (>35) %": prevalence(pcos_df.query("pcos_y_n == 1")["older_age"]),
            "Overweight BMI (>25) %": prevalence(pcos_df.query("pcos_y_n == 1")["overweight_bmi"]),
            "Obese BMI (>30) %": prevalence(pcos_df.query("pcos_y_n == 1")["obese_bmi"]),
            "Elevated pulse (>90) %": prevalence(pcos_df.query("pcos_y_n == 1")["elevated_pulse"]),
            "Tachycardic pulse (>100) %": prevalence(pcos_df.query("pcos_y_n == 1")["tachycardic_pulse"]),
            "High systolic BP (>130) %": prevalence(pcos_df.query("pcos_y_n == 1")["high_systolic_bp"]),
            "Elevated RBS (>100) %": prevalence(pcos_df.query("pcos_y_n == 1")["elevated_rbs"]),
            "Low haemoglobin (<12) %": prevalence(pcos_df.query("pcos_y_n == 1")["low_haemoglobin"]),
            "Higher cycle-irregularity code (>=4) %": prevalence(pcos_df.query("pcos_y_n == 1")["higher_cycle_irregularity_code"]),
        },
        "Heart Negative": {
            "Older age (>35) %": prevalence(heart_df.query("heart_target == 0")["older_age"]),
            "Overweight BMI (>25) %": np.nan,
            "Obese BMI (>30) %": np.nan,
            "Elevated pulse (>90) %": np.nan,
            "Tachycardic pulse (>100) %": np.nan,
            "High systolic BP (>130) %": np.nan,
            "Elevated RBS (>100) %": np.nan,
            "Low haemoglobin (<12) %": np.nan,
            "Higher cycle-irregularity code (>=4) %": np.nan,
        },
        "Heart Positive": {
            "Older age (>35) %": prevalence(heart_df.query("heart_target == 1")["older_age"]),
            "Overweight BMI (>25) %": np.nan,
            "Obese BMI (>30) %": np.nan,
            "Elevated pulse (>90) %": np.nan,
            "Tachycardic pulse (>100) %": np.nan,
            "High systolic BP (>130) %": np.nan,
            "Elevated RBS (>100) %": np.nan,
            "Low haemoglobin (<12) %": np.nan,
            "Higher cycle-irregularity code (>=4) %": np.nan,
        },
    }
).round(1)
display(risk_flag_prevalence)
```

### Cell 52 - Code
**What this code is doing:** Plotting the prevalence heatmap is making the dominant risk-flag contrasts easier to compare at a glance.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `risk_flag_prevalence_heatmap.png`

```python
# Plotting the prevalence heatmap is making the dominant risk-flag contrasts easier to compare at a glance.
fig, ax = plt.subplots(figsize=(7.4, 4.8))
sns.heatmap(
    risk_flag_prevalence,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "Prevalence (%)"},
    ax=ax,
)
ax.set_title("Risk-Flag Prevalence Across PCOS and Heart Subgroups")
ax.set_xlabel("Subgroup")
ax.set_ylabel("Risk flag")
save_figure(fig, "risk_flag_prevalence_heatmap.png")
plt.show()
```

### Cell 53 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The risk-flag table is helping the heart and metabolic pattern cohere into something clinically easier to read. Inside the PCOS cohort, the clearest differentiators are expected to come from **BMI-related burden, elevated random blood sugar, and the heuristic higher cycle-irregularity code**, while the heart cohort remains dominated by age and pressure-related burden. This strengthens thesis H1 in a nuanced way: the PCOS-positive profile is not simply mirroring clear cardiovascular disease, but it is clustering around a meaningful upstream risk pattern that deserves monitoring.

### Cell 54 - Markdown
**What this cell is doing:** Section 6: Statistical Tests and Effect Sizes.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Section 6: Statistical Tests and Effect Sizes

This section measures the within-cohort contrasts using Mann–Whitney U tests, rank-biserial effect size (how big the difference really is)s, and bootstrapped confidence intervals. The heart dataset is only entering this section for an internal age comparison; no cross-dataset inferential test is being performed.

### Cell 55 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**Which continuous features show useful within-dataset differences, and which ones still look small after effect size (how big the difference really is) and Bonferroni correction (a stricter rule after many tests) are considered together?**

The testing framework looks at effect-size interpretation rather than p-values alone because thesis claims should be based on size as well as statistical evidence.

### Cell 56 - Code
**What this code is doing:** Defining the testing helper is standardising the Mann-Whitney reports across all planned contrasts.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Defining the testing helper is standardising the Mann-Whitney reports across all planned contrasts.
def mannwhitney_rank_biserial_report(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    feature: str,
    group_a_name: str,
    group_b_name: str,
    thesis_relevance: str,
) -> dict:
    group_a = frame_a[feature].dropna()
    group_b = frame_b[feature].dropna()
    u_statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided", method="auto")
    r_value = rank_biserial_from_u(u_statistic, len(group_a), len(group_b))
    ci_lower, ci_upper = bootstrap_rank_biserial(group_a, group_b, n_boot=1500, seed=42)
    return {
        "feature": feature,
        "groups": f"{group_a_name} vs {group_b_name}",
        "n_group_a": len(group_a),
        "n_group_b": len(group_b),
        "median_group_a": group_a.median(),
        "median_group_b": group_b.median(),
        "u_statistic": u_statistic,
        "p_value": p_value,
        "rank_biserial_r": r_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "magnitude": effect_size_label(r_value),
        "thesis_relevance": thesis_relevance,
    }

bonferroni_alpha = 0.05 / 7
nominal_alpha = 0.05
```

### Cell 57 - Code
**What this code is doing:** Running the planned tests is quantifying the strongest within-cohort contrasts relevant to thesis H₁.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Running the planned tests is quantifying the strongest within-cohort contrasts relevant to thesis H₁.
statistical_results = pd.DataFrame(
    [
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "bmi",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying the metabolic bridge component of thesis H₁.",
        ),
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "systolic_bp",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying whether overt pressure differences are already visible within the PCOS cohort.",
        ),
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "pulse_rate",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying the autonomic proxy component of thesis H₁.",
        ),
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "random_blood_sugar",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying glucose-related metabolic burden within the PCOS cohort.",
        ),
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "haemoglobin",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying the exploratory haemoglobin pathway within the PCOS cohort.",
        ),
        mannwhitney_rank_biserial_report(
            pcos_df.query("pcos_y_n == 1"),
            pcos_df.query("pcos_y_n == 0"),
            "age",
            "PCOS Positive",
            "PCOS Negative",
            "This test is quantifying whether age structure differs inside the PCOS cohort itself.",
        ),
        mannwhitney_rank_biserial_report(
            heart_df.query("heart_target == 1"),
            heart_df.query("heart_target == 0"),
            "age",
            "Heart Positive",
            "Heart Negative",
            "This test is describing age differentiation inside the heart reference cohort only.",
        ),
    ]
)

statistical_results["significant_at_0_05"] = (statistical_results["p_value"] < nominal_alpha) & (statistical_results["rank_biserial_r"].abs() >= 0.10)
statistical_results["significant_at_0_0071"] = (statistical_results["p_value"] < bonferroni_alpha) & (statistical_results["rank_biserial_r"].abs() >= 0.10)
statistical_results["rank_biserial_abs"] = statistical_results["rank_biserial_r"].abs()
display(
    statistical_results[
        [
            "feature",
            "groups",
            "median_group_a",
            "median_group_b",
            "u_statistic",
            "p_value",
            "rank_biserial_r",
            "ci_lower",
            "ci_upper",
            "magnitude",
            "significant_at_0_05",
            "significant_at_0_0071",
        ]
    ].round(4)
)
print(f"Bonferroni-adjusted alpha for seven tests: {bonferroni_alpha:.4f}")
```

### Cell 58 - Code
**What this code is doing:** Printing the exact per-test reporting template is making the statistical evidence easy to transfer into thesis prose.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Printing the exact per-test reporting template is making the statistical evidence easy to transfer into thesis prose.
for _, row in statistical_results.iterrows():
    report = f'''
### Test Report — {row["feature"]}

**Feature:** {row["feature"]}  
**Groups:** {row["groups"]}  
**Test:** Mann–Whitney U (non-parametric)  
**U statistic:** {row["u_statistic"]:.3f}  
**p-value:** {row["p_value"]:.4f}  
**Effect size:** rank-biserial r = {row["rank_biserial_r"]:.3f} ({row["magnitude"]})  
**Bootstrapped 95% CI on r:** [{row["ci_lower"]:.3f}, {row["ci_upper"]:.3f}]  
**Nominal significance (α = 0.05):** {row["significant_at_0_05"]}  
**Bonferroni significance (α = 0.0071):** {row["significant_at_0_0071"]}  
**Thesis relevance:** {row["thesis_relevance"]}
    '''
    display(Markdown(report))
```

### Cell 59 - Code
**What this code is doing:** Plotting the effect sizes is summarising which contrasts matter most once uncertainty and multiplicity are considered together.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.
**Saved figure or image output in this cell:** `effect_size_dot_plot.png`

```python
# Plotting the effect sizes is summarising which contrasts matter most once uncertainty and multiplicity are considered together.
effect_plot_df = statistical_results.copy()
effect_plot_df["test_label"] = effect_plot_df["feature"] + " | " + effect_plot_df["groups"]
effect_plot_df["significance_class"] = np.select(
    [
        effect_plot_df["significant_at_0_0071"],
        effect_plot_df["significant_at_0_05"],
    ],
    [
        "Survives Bonferroni",
        "Nominal only",
    ],
    default="Not significant",
)
effect_colour_map = {
    "Survives Bonferroni": "#e76f51",
    "Nominal only": "#e9c46a",
    "Not significant": "#6b7280",
}

fig, ax = plt.subplots(figsize=(7.0, 4.6))
sns.scatterplot(
    data=effect_plot_df.sort_values("rank_biserial_r"),
    x="rank_biserial_r",
    y="test_label",
    hue="significance_class",
    palette=effect_colour_map,
    s=80,
    ax=ax,
)
ax.axvline(0, color="#1f2937", linewidth=1.0)
ax.axvline(0.10, color="#6b7280", linestyle="--", linewidth=1.0)
ax.axvline(-0.10, color="#6b7280", linestyle="--", linewidth=1.0)
ax.set_title("Effect-Size Profile for the Planned Mann–Whitney Tests")
ax.set_xlabel("Rank-biserial r")
ax.set_ylabel("")
polish_axis(ax, grid_axis="x")
ax.legend(frameon=False, loc="lower right")
save_figure(fig, "effect_size_dot_plot.png")
plt.show()
```

### Cell 60 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The statistical section is being designed to prevent overclaiming. A small p-value without a meaningful rank-biserial effect size (how big the difference really is) is not being treated as persuasive evidence, and a result that fails the Bonferroni threshold is being described more cautiously than one that survives multiplicity correction. This is especially important for thesis H1 because the claim becomes most credible when the strongest variables are not only statistically detectable but also large enough to matter clinically.

### Cell 61 - Markdown
**What this cell is doing:** Section 7: Group-Level Profile Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Section 7: Group-Level Profile Summary

**Ecological comparison caveat:** The PCOS cohort and the heart cohort contain different individuals and very different age structures. The profile charts in this section are therefore being used for descriptive benchmarking only.

### Cell 62 - Markdown
**What this cell is doing:** Research Question.
**Why this step matters:** This matters because it tells the reader exactly what the next table and chart are trying to answer.

**Notebook markdown content:**

### Research Question

**How does the PCOS-positive group compare with the heart-positive group when we only use shared or carefully matched profile features?**

The profile charts below are intentionally excluding dataset-specific variables such as haemoglobin, cholesterol, and BMI from the shared-axis radar so that the group-level comparison between different groups of people stays methodologically honest.

### Cell 63 - Code
**What this code is doing:** Building the cross-cohort summary table is separating shared metrics from dataset-specific reference-only metrics.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Building the cross-cohort summary table is separating shared metrics from dataset-specific reference-only metrics.
cross_cohort_profile_table = pd.DataFrame(
    {
        "Metric": [
            "Mean age",
            "Older age (>35) %",
            "Mean pressure metric",
            "Pressure metric >130 %",
            "Mean pulse / max-heart-rate metric",
            "Mean BMI (PCOS only)",
            "Mean random blood sugar (PCOS only)",
            "Mean haemoglobin (PCOS only)",
            "Mean cholesterol (Heart only)",
        ],
        "PCOS Negative": [
            pcos_df.query("pcos_y_n == 0")["age"].mean(),
            prevalence(pcos_df.query("pcos_y_n == 0")["older_age"]),
            pcos_df.query("pcos_y_n == 0")["systolic_bp"].mean(),
            prevalence(pcos_df.query("pcos_y_n == 0")["high_systolic_bp"]),
            pcos_df.query("pcos_y_n == 0")["pulse_rate"].mean(),
            pcos_df.query("pcos_y_n == 0")["bmi"].mean(),
            pcos_df.query("pcos_y_n == 0")["random_blood_sugar"].mean(),
            pcos_df.query("pcos_y_n == 0")["haemoglobin"].mean(),
            np.nan,
        ],
        "PCOS Positive": [
            pcos_df.query("pcos_y_n == 1")["age"].mean(),
            prevalence(pcos_df.query("pcos_y_n == 1")["older_age"]),
            pcos_df.query("pcos_y_n == 1")["systolic_bp"].mean(),
            prevalence(pcos_df.query("pcos_y_n == 1")["high_systolic_bp"]),
            pcos_df.query("pcos_y_n == 1")["pulse_rate"].mean(),
            pcos_df.query("pcos_y_n == 1")["bmi"].mean(),
            pcos_df.query("pcos_y_n == 1")["random_blood_sugar"].mean(),
            pcos_df.query("pcos_y_n == 1")["haemoglobin"].mean(),
            np.nan,
        ],
        "Heart Negative": [
            heart_df.query("heart_target == 0")["age"].mean(),
            prevalence(heart_df.query("heart_target == 0")["older_age"]),
            heart_df.query("heart_target == 0")["resting_bp"].mean(),
            prevalence(heart_df.query("heart_target == 0")["high_resting_bp"]),
            heart_df.query("heart_target == 0")["max_heart_rate"].mean(),
            np.nan,
            np.nan,
            np.nan,
            heart_df.query("heart_target == 0")["cholesterol"].mean(),
        ],
        "Heart Positive": [
            heart_df.query("heart_target == 1")["age"].mean(),
            prevalence(heart_df.query("heart_target == 1")["older_age"]),
            heart_df.query("heart_target == 1")["resting_bp"].mean(),
            prevalence(heart_df.query("heart_target == 1")["high_resting_bp"]),
            heart_df.query("heart_target == 1")["max_heart_rate"].mean(),
            np.nan,
            np.nan,
            np.nan,
            heart_df.query("heart_target == 1")["cholesterol"].mean(),
        ],
    }
).round(2)
display(cross_cohort_profile_table)
```

### Cell 64 - Code
**What this code is doing:** Restricting the shared profile metrics to defensible ecological bridges is preventing overinterpretation in the radar and dumbbell charts.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.

```python
# Restricting the shared profile metrics to defensible ecological bridges is preventing overinterpretation in the radar and dumbbell charts.
shared_profile_source = pd.DataFrame(
    {
        "group": ["PCOS Positive", "Heart Positive"],
        "Mean age": [pcos_df.query("pcos_y_n == 1")["age"].mean(), heart_df.query("heart_target == 1")["age"].mean()],
        "Older age (>35) %": [prevalence(pcos_df.query("pcos_y_n == 1")["older_age"]), prevalence(heart_df.query("heart_target == 1")["older_age"])],
        "Mean pressure metric": [pcos_df.query("pcos_y_n == 1")["systolic_bp"].mean(), heart_df.query("heart_target == 1")["resting_bp"].mean()],
        "Pressure metric >130 %": [prevalence(pcos_df.query("pcos_y_n == 1")["high_systolic_bp"]), prevalence(heart_df.query("heart_target == 1")["high_resting_bp"])],
    }
)

shared_profile_long = shared_profile_source.set_index("group").T
shared_profile_normalised = shared_profile_long.apply(min_max_normalise, axis=1)
display(shared_profile_normalised.round(3))
```

### Cell 65 - Code
**What this code is doing:** Plotting the radar chart is summarising the shared ecological profile on a common 0-1 scale.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.
**Saved figure or image output in this cell:** `shared_profile_radar_chart.png`

```python
# Plotting the radar chart is summarising the shared ecological profile on a common 0-1 scale.
radar_metrics = list(shared_profile_normalised.index)
radar_angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
radar_angles += radar_angles[:1]

fig = plt.figure(figsize=(6.0, 5.2))
ax = plt.subplot(111, polar=True)

for group_name in ["PCOS Positive", "Heart Positive"]:
    values = shared_profile_normalised[group_name].tolist()
    values += values[:1]
    ax.plot(radar_angles, values, linewidth=2.0, label=group_name, color=GROUP_PALETTE[group_name])
    ax.fill(radar_angles, values, alpha=0.18, color=GROUP_PALETTE[group_name])

ax.set_xticks(radar_angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_yticks([0.25, 0.50, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
ax.set_title("Normalised Shared Profile: PCOS Positive vs Heart Positive", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10), frameon=False)
save_figure(fig, "shared_profile_radar_chart.png")
plt.show()
```

### Cell 66 - Code
**What this code is doing:** Plotting the dumbbell chart is making the same shared profile easier to read metric by metric.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.
**Saved figure or image output in this cell:** `shared_profile_dumbbell_chart.png`

```python
# Plotting the dumbbell chart is making the same shared profile easier to read metric by metric.
dumbbell_df = shared_profile_normalised.reset_index().rename(columns={"index": "metric"})

fig, ax = plt.subplots(figsize=(7.0, 4.2))
for _, row in dumbbell_df.iterrows():
    ax.plot(
        [row["PCOS Positive"], row["Heart Positive"]],
        [row["metric"], row["metric"]],
        color="#9ca3af",
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(row["PCOS Positive"], row["metric"], color=GROUP_PALETTE["PCOS Positive"], s=70, zorder=2)
    ax.scatter(row["Heart Positive"], row["metric"], color=GROUP_PALETTE["Heart Positive"], s=70, zorder=2)

ax.set_title("Shared Profile Dumbbell Comparison")
ax.set_xlabel("Normalised value (0-1)")
ax.set_ylabel("")
ax.set_xlim(-0.05, 1.05)
polish_axis(ax, grid_axis="x")
save_figure(fig, "shared_profile_dumbbell_chart.png")
plt.show()
```

### Cell 67 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The shared-profile charts are deliberately narrow because they are being limited to the few variables that can be defended group-levelly across the two cohorts. They are useful for showing directional resemblance or divergence, especially on age and pressure burden, but they are not claiming that the PCOS-positive and heart-positive groups are biologically equivalent. The strongest lesson from this section is that the PCOS-positive subgroup may be showing an upstream heart and metabolic profile, while the heart-positive subgroup is representing a later and older clinical endpoint population.

### Cell 68 - Markdown
**What this cell is doing:** Descriptive-Only Sensitivity Appendix: Ages 35-45.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

### Descriptive-Only Sensitivity Appendix: Ages 35-45

**Ecological comparison caveat:** This mini-section is restricting both cohorts to ages 35-45 only to inspect directionality. It is not being used for inferential claims because the heart reference cohort becomes very small in this window, especially on the heart-negative side.

### Cell 69 - Code
**What this code is doing:** Restricting the cohorts to ages 35-45 is providing a directional sensitivity check around the dominant confound.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Restricting the cohorts to ages 35-45 is providing a directional sensitivity check around the dominant confound.
pcos_35_45 = pcos_df.loc[pcos_df["age"].between(35, 45)].copy()
heart_35_45 = heart_df.loc[heart_df["age"].between(35, 45)].copy()

sensitivity_counts = pd.DataFrame(
    [
        {"subgroup": "PCOS Negative", "n_35_45": len(pcos_35_45.query("pcos_y_n == 0"))},
        {"subgroup": "PCOS Positive", "n_35_45": len(pcos_35_45.query("pcos_y_n == 1"))},
        {"subgroup": "Heart Negative", "n_35_45": len(heart_35_45.query("heart_target == 0"))},
        {"subgroup": "Heart Positive", "n_35_45": len(heart_35_45.query("heart_target == 1"))},
    ]
)

sensitivity_profile = pd.DataFrame(
    [
        {
            "Subgroup": "PCOS Negative",
            "Mean age": pcos_35_45.query("pcos_y_n == 0")["age"].mean(),
            "Mean pressure metric": pcos_35_45.query("pcos_y_n == 0")["systolic_bp"].mean(),
            "Pressure >130 %": prevalence(pcos_35_45.query("pcos_y_n == 0")["systolic_bp"] > 130),
        },
        {
            "Subgroup": "PCOS Positive",
            "Mean age": pcos_35_45.query("pcos_y_n == 1")["age"].mean(),
            "Mean pressure metric": pcos_35_45.query("pcos_y_n == 1")["systolic_bp"].mean(),
            "Pressure >130 %": prevalence(pcos_35_45.query("pcos_y_n == 1")["systolic_bp"] > 130),
        },
        {
            "Subgroup": "Heart Negative",
            "Mean age": heart_35_45.query("heart_target == 0")["age"].mean(),
            "Mean pressure metric": heart_35_45.query("heart_target == 0")["resting_bp"].mean(),
            "Pressure >130 %": prevalence(heart_35_45.query("heart_target == 0")["resting_bp"] > 130),
        },
        {
            "Subgroup": "Heart Positive",
            "Mean age": heart_35_45.query("heart_target == 1")["age"].mean(),
            "Mean pressure metric": heart_35_45.query("heart_target == 1")["resting_bp"].mean(),
            "Pressure >130 %": prevalence(heart_35_45.query("heart_target == 1")["resting_bp"] > 130),
        },
    ]
).round(3)

print("35-45 subgroup counts:")
display(sensitivity_counts)
print("\n35-45 descriptive profile:")
display(sensitivity_profile)
```

### Cell 70 - Code
**What this code is doing:** Plotting the restricted subgroup counts is visualising how sparse the heart reference cohort becomes in the sensitivity window.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `sensitivity_window_counts_35_45.png`

```python
# Plotting the restricted subgroup counts is visualising how sparse the heart reference cohort becomes in the sensitivity window.
fig, ax = plt.subplots(figsize=(6.2, 3.8))
sns.barplot(
    data=sensitivity_counts,
    x="subgroup",
    y="n_35_45",
    palette=[GROUP_PALETTE["PCOS Negative"], GROUP_PALETTE["PCOS Positive"], GROUP_PALETTE["Heart Negative"], GROUP_PALETTE["Heart Positive"]],
    ax=ax,
)
ax.set_title("Available Sample Sizes in the 35-45 Sensitivity Window")
ax.set_xlabel("")
ax.set_ylabel("Number of participants")
ax.tick_params(axis="x", rotation=15)
polish_axis(ax, grid_axis="y")
save_figure(fig, "sensitivity_window_counts_35_45.png")
plt.show()
```

### Cell 71 - Markdown
**What this cell is doing:** Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Interpretation


The sensitivity appendix is useful mainly because it shows how quickly the heart reference sample becomes sparse once the age gap is narrowed. That means the restricted-window view can help us check directionality, but it cannot replace the main caution that the two cohorts are structurally different. For the thesis, this appendix strengthens transparency because it shows that the group-level comparison between different groups of people has been stress-tested rather than accepted at face value.

### Cell 72 - Markdown
**What this cell is doing:** Section 8: How These Results Fit the Thesis Literature.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Section 8: How These Results Fit the Thesis Literature

This section turns the descriptive and statistical findings into four biological pathways that connect the notebook outputs to the literature review and to thesis Hypothesis H1.

### Cell 73 - Code
**What this code is doing:** Building a compact evidence table is collecting the main pathway-relevant quantities in one place before the prose interpretation.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Building a compact evidence table is collecting the main pathway-relevant quantities in one place before the prose interpretation.
pathway_evidence = pd.DataFrame(
    [
        {
            "Pathway": "Insulin-resistance bridge",
            "Evidence from this notebook": "BMI and random blood sugar are being analysed together as metabolic bridge variables.",
            "Key quantitative anchor": f"BMI means: {pcos_df.query('pcos_y_n == 0')['bmi'].mean():.2f} vs {pcos_df.query('pcos_y_n == 1')['bmi'].mean():.2f}; RBS means: {pcos_df.query('pcos_y_n == 0')['random_blood_sugar'].mean():.2f} vs {pcos_df.query('pcos_y_n == 1')['random_blood_sugar'].mean():.2f}",
        },
        {
            "Pathway": "Blood-pressure / vascular-stress bridge",
            "Evidence from this notebook": "Systolic pressure is being analysed directly within PCOS, with pulse rate kept as secondary autonomic context.",
            "Key quantitative anchor": f"Systolic means: {pcos_df.query('pcos_y_n == 0')['systolic_bp'].mean():.2f} vs {pcos_df.query('pcos_y_n == 1')['systolic_bp'].mean():.2f}; pulse means: {pcos_df.query('pcos_y_n == 0')['pulse_rate'].mean():.2f} vs {pcos_df.query('pcos_y_n == 1')['pulse_rate'].mean():.2f}",
        },
        {
            "Pathway": "Exploratory haemoglobin pathway",
            "Evidence from this notebook": "Haemoglobin is being explored as a possible mediator of cardiovascular strain rather than as a pre-specified thesis variable.",
            "Key quantitative anchor": f"Haemoglobin means: {pcos_df.query('pcos_y_n == 0')['haemoglobin'].mean():.2f} vs {pcos_df.query('pcos_y_n == 1')['haemoglobin'].mean():.2f}",
        },
        {
            "Pathway": "Age trajectory hypothesis",
            "Evidence from this notebook": "The age overlay is quantifying the difference between an earlier-life PCOS cohort and a later-life cardiovascular reference cohort.",
            "Key quantitative anchor": f"Mean age: PCOS overall {pcos_df['age'].mean():.2f} vs Heart overall {heart_df['age'].mean():.2f}",
        },
    ]
)
display(pathway_evidence)
```

### Cell 74 - Markdown
**What this cell is doing:** Pathway 1 - Insulin-Resistance Bridge.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Pathway 1 - Insulin-Resistance Bridge

The BMI and random-blood-sugar results from this notebook are being read together as indirect evidence for the insulin-resistance bridge described in the thesis. A rightward shift in BMI, especially when it is accompanied by a modest upward movement in random blood sugar, helps the interpretation that the PCOS-positive subgroup is carrying more metabolic burden than the PCOS-negative subgroup. This reading is aligning with the literature summary in the thesis, including Henney et al. (2024), where obesity and diabetes-related risk are being described as more common in women with PCOS. At the same time, this notebook is not containing fasting insulin, HbA1c, or lipid fractions, so insulin resistance itself is not being directly measured. The pathway is therefore being supported indirectly rather than confirmed biologicalally.

### Cell 75 - Markdown
**What this cell is doing:** Pathway 2 - Blood-Pressure and Vascular-Stress Pathway.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Pathway 2 - Blood-Pressure and Vascular-Stress Pathway

The cleaned clinical dataset lets us this notebook to examine actual systolic and diastolic blood-pressure variables inside the PCOS cohort, which is methodologically stronger than relying on pulse alone. The pulse analysis is still adding value because it is offering complementary nervous-system context, but the vascular interpretation is being anchored first in the pressure metrics. This approach is closer to the thesis literature base, including Wekker et al. (2020), where hypertension-related and metabolic cardiovascular concerns are being discussed alongside PCOS. Even so, the within-PCOS pressure differences may be subtle rather than dramatic, which means the notebook helps a cautious interpretation: vascular stress may be present as part of a broader burden cluster, not necessarily as an isolated high-size signal. The heart cohort remains reference context only and is not converting this observation into causal evidence.

### Cell 76 - Markdown
**What this cell is doing:** Pathway 3 - Exploratory Haemoglobin Pathway.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

### Pathway 3 - Exploratory Haemoglobin Pathway

Haemoglobin is not one of the classic headline markers in most PCOS screening discussions, but the cleaned dataset is making it possible to examine whether a haemoglobin-related cardiovascular pathway deserves exploratory attention. If haemoglobin is lower in part of the PCOS cohort, then compensatory cardiovascular strain becomes biologically plausible because reduced oxygen-carrying capacity can increase cardiac workload. The notebook is therefore treating haemoglobin as an exploratory cardiovascular-adjacent variable rather than as a pre-specified thesis anchor. This is important methodologically because it keeps the chapter honest: the data are enabling the question, but the literature support is thinner than for BMI or vascular pressure. Any claim from this pathway is therefore being framed as suggestive and hypothesis-generating.

### Cell 77 - Markdown
**What this cell is doing:** Pathway 4 - Age Trajectory Hypothesis.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

### Pathway 4 - Age Trajectory Hypothesis

The age comparison helps the thesis argument that PCOS may represent an earlier-life stage on a longer cardiovascular-risk path over time. The PCOS cohort is concentrating in the reproductive-age window, while the heart cohort is representing a much older clinical endpoint population. This is not proving that the PCOS-positive women in this dataset will later develop clear cardiovascular disease, but it is making the path over time hypothesis more intelligible in empirical terms. The chapter can therefore argue that heart and metabolic monitoring in PCOS is clinically sensible even when clear cardiovascular disease is not yet visible. Longitudinal follow-up would still be required to confirm whether that earlier-life burden translates into later cardiovascular outcomes.

### Cell 78 - Markdown
**What this cell is doing:** Section 9: Final Visual Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Section 9: Final Visual Summary

This section condensing the strongest easy to read plots into one multi-panel figure that can support Chapter 4 presentation and later thesis discussion.

### Cell 79 - Code
**What this code is doing:** Building the summary panel is combining the clearest descriptive and inferential visuals into one publication-style figure.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.
**Saved figure or image output in this cell:** `chapter_quality_summary_panel.png`

```python
# Building the summary panel is combining the clearest descriptive and inferential visuals into one publication-style figure.
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
axes = axes.flatten()

# Panel A: age KDE across all subgroups.
for group_name in ["PCOS Negative", "PCOS Positive", "Heart Negative", "Heart Positive"]:
    subset = age_overlay.loc[age_overlay["group"] == group_name, "age"]
    sns.kdeplot(
        subset,
        ax=axes[0],
        label=group_name,
        color=GROUP_PALETTE[group_name],
        linewidth=1.8,
        linestyle=line_styles[group_name],
        fill=False,
        warn_singular=False,
    )
axes[0].axvline(35, color="#374151", linestyle="--", linewidth=1.0)
axes[0].set_title("A. Age KDE")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Density")
polish_axis(axes[0], grid_axis="y")
axes[0].legend(frameon=False, fontsize=7, ncol=2)

# Panel B: pulse raincloud proxy view.
raincloud_plot(axes[1], pcos_df, "pcos_label", "pulse_rate", ["PCOS Negative", "PCOS Positive"])
axes[1].axhline(100, color="#9a3412", linestyle="--", linewidth=1.0)
axes[1].set_title("B. Pulse Rate")
axes[1].set_xlabel("")
axes[1].set_ylabel("Pulse rate")
polish_axis(axes[1], grid_axis="y")

# Panel C: BMI density.
for label in ["PCOS Negative", "PCOS Positive"]:
    subset = pcos_df.loc[pcos_df["pcos_label"] == label, "bmi"]
    sns.kdeplot(
        subset,
        ax=axes[2],
        fill=True,
        alpha=0.24,
        linewidth=1.8,
        color=PCOS_PALETTE[label],
        label=label,
        warn_singular=False,
    )
axes[2].axvline(25, color="#6b7280", linestyle="--", linewidth=1.0)
axes[2].axvline(30, color="#9a3412", linestyle="--", linewidth=1.0)
axes[2].set_title("C. BMI Density")
axes[2].set_xlabel("BMI")
axes[2].set_ylabel("Density")
polish_axis(axes[2], grid_axis="y")

# Panel D: systolic pressure.
sns.violinplot(
    data=pcos_df,
    x="pcos_label",
    y="systolic_bp",
    order=["PCOS Negative", "PCOS Positive"],
    palette=PCOS_PALETTE,
    inner=None,
    cut=0,
    linewidth=1.0,
    ax=axes[3],
)
sns.stripplot(
    data=pcos_df,
    x="pcos_label",
    y="systolic_bp",
    order=["PCOS Negative", "PCOS Positive"],
    color="#111827",
    alpha=0.24,
    size=2.0,
    jitter=0.16,
    ax=axes[3],
)
axes[3].axhline(130, color="#374151", linestyle="--", linewidth=1.0)
axes[3].set_title("D. Systolic Pressure")
axes[3].set_xlabel("")
axes[3].set_ylabel("mmHg")
polish_axis(axes[3], grid_axis="y")

# Panel E: random blood sugar ECDF.
for label in ["PCOS Negative", "PCOS Positive"]:
    subset = pcos_df.loc[pcos_df["pcos_label"] == label, "random_blood_sugar"]
    sns.ecdfplot(
        subset,
        ax=axes[4],
        linewidth=1.8,
        color=PCOS_PALETTE[label],
        label=label,
    )
axes[4].axvline(100, color="#6b7280", linestyle="--", linewidth=1.0)
axes[4].axvline(140, color="#9a3412", linestyle="--", linewidth=1.0)
axes[4].set_title("E. Random Blood Sugar")
axes[4].set_xlabel("mg/dL")
axes[4].set_ylabel("Cumulative proportion")
polish_axis(axes[4], grid_axis="y")

# Panel F: effect-size profile.
sns.scatterplot(
    data=effect_plot_df.sort_values("rank_biserial_r"),
    x="rank_biserial_r",
    y="test_label",
    hue="significance_class",
    palette=effect_colour_map,
    s=55,
    ax=axes[5],
)
axes[5].axvline(0, color="#1f2937", linewidth=1.0)
axes[5].axvline(0.10, color="#6b7280", linestyle="--", linewidth=1.0)
axes[5].axvline(-0.10, color="#6b7280", linestyle="--", linewidth=1.0)
axes[5].set_title("F. Effect Sizes")
axes[5].set_xlabel("Rank-biserial r")
axes[5].set_ylabel("")
polish_axis(axes[5], grid_axis="x")
axes[5].legend(frameon=False, fontsize=7, loc="lower right")

fig.suptitle("Summary Panel: Cardiometabolic Signals in PCOS and Heart Reference Context", y=1.02, fontsize=13)
fig.tight_layout()
save_figure(fig, "chapter_quality_summary_panel.png")
plt.show()
```

### Cell 80 - Markdown
**What this cell is doing:** Notebook explanation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

**Figure caption:** The multi-panel summary is combining the strongest descriptive and inferential views from the notebook. Panels A-E are showing how age structure, pressure, pulse, BMI, and glucose-related burden are being distributed across the relevant cohorts, while Panel F is summarising which within-cohort contrasts remain strongest after effect-size interpretation is added to the p-value framework. The figure is intended for Chapter 4 synthesis rather than as standalone evidence of causality.

### Cell 81 - Markdown
**What this cell is doing:** Final Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Final Summary

This closing section is consolidating the notebook outputs into chapter-ready findings, clinical interpretation, limitations, and a direct answer to thesis Hypothesis H1.

### Cell 82 - Code
**What this code is doing:** Building the key-findings markdown is translating the test table into thesis-ready statements with effect sizes and correction status.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Building the key-findings markdown is translating the test table into thesis-ready statements with effect sizes and correction status.
key_findings = statistical_results.sort_values(
    by=["significant_at_0_0071", "significant_at_0_05", "rank_biserial_abs"],
    ascending=[False, False, False],
).reset_index(drop=True)

finding_lines = []
for finding_number, (_, row) in enumerate(key_findings.iterrows(), start=1):
    if row["significant_at_0_0071"]:
        significance_text = "survived Bonferroni correction"
    elif row["significant_at_0_05"]:
        significance_text = "was nominally significant but did not survive Bonferroni correction"
    else:
        significance_text = "did not meet the study threshold for significance"

    finding_lines.append(
        f"{finding_number}. **{row['feature']}** in **{row['groups']}**: median {row['median_group_a']:.2f} vs {row['median_group_b']:.2f}, "
        f"U = {row['u_statistic']:.2f}, p = {row['p_value']:.4f}, r = {row['rank_biserial_r']:.3f} "
        f"[95% CI {row['ci_lower']:.3f}, {row['ci_upper']:.3f}], and {significance_text}."
    )

display(Markdown("### Key Findings\n\n" + "\n\n".join(finding_lines)))
```

### Cell 83 - Markdown
**What this cell is doing:** Clinical Interpretation.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Clinical Interpretation

The overall clinical message of this notebook is that women with PCOS should not be viewed only through a reproductive-health lens. The within-cohort analyses suggest that heart and metabolic monitoring is clinically relevant, especially when BMI, glucose-related burden, and blood-pressure context are being considered together rather than one marker at a time. For a clinician, this means that a PCOS consultation is also an opportunity to review weight path over time, basic glycaemic burden, and cardiovascular vital signs early rather than waiting for later-life morbidity to appear. This directly supports thesis objective iii on clinical use because the notebook is identifying which measurable signals can realistically be monitored in routine care.

### Cell 84 - Markdown
**What this cell is doing:** Limitations.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Limitations

1. The PCOS cohort and the heart cohort are independent populations, so the cross-dataset sections are not permitting patient-level inference or causality.
2. The cleaned heart dataset is a small female-only reference cohort (`n = 96`), which limits the stability of some group-level patterns.
3. The heart cohort is strongly imbalanced (`72` heart-positive and `24` heart-negative), and this may produce counterintuitive internal reference patterns.
4. The mean age gap between the cohorts is large, making age the dominant factor that can distort the comparison in every group-level comparison between different groups of people.
5. The design is cross-sectional, so the notebook cannot confirm whether the observed heart and metabolic patterns in PCOS translate into later cardiovascular events.
6. The data provenance is secondary and platform-based, so generalisability beyond the group of people may be limited.
7. The PCOS dataset does not contain cholesterol fractions, HbA1c, or lipid panels, which constrains biological cardiovascular interpretation.
8. The cycle-regularity measure is a recorded code rather than a documented binary irregular-cycle variable, so any engineered irregularity flag is heuristic.

### Cell 85 - Markdown
**What this cell is doing:** Conclusion - Three-Part Verdict on Thesis H1.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

### Conclusion - Three-Part Verdict on Thesis H1

**1. What the data supports:** The notebook helps the argument that PCOS-positive women show a meaningful heart and metabolic burden profile within the clinical cohort, especially where BMI, glucose-related measures, and selected cardiovascular-adjacent variables are concerned. The statistical section is identifying which contrasts are large enough to matter and which ones remain modest after correction.

**2. What the data cannot confirm:** The notebook is not proving that PCOS causes cardiovascular disease, nor is it proving that the PCOS-positive subgroup is equivalent to the older heart-positive cohort. The group-level comparison between different groups of peoples are descriptive only, and the age gap is too large to justify stronger cross-cohort claims.

**3. What a stronger follow-up study would need:** A follow-up study would need age-matched cohorts, prospective follow-up, fuller metabolic panels, and repeated cardiovascular assessment over time. That design would allow the thesis argument to move from early heart and metabolic association toward stronger longitudinal cardiovascular-risk inference.


## Results and Findings
### Questions and Results
| Question asked | Saved figure | Main result in simple words |
| --- | --- | --- |
| What do these cleaned datasets look like, and what problems must we note before comparing them? | `age_distribution_ecological_reference.png` | The age tables and the overlaid density curves above are showing that the heart cohort is materially older than the PCOS cohort, with an average gap of roughly **24 years**. This gap is already far beyond the five-year threshold that would have triggered a major warning, so every cross-dataset comparison in the rest of the notebook must be read as **descriptive context rather than matched-risk evidence**. In practical terms, the heart dataset is helping us understand the later-life cardiovascular reference profile, but it cannot by itself tell us that the younger PCOS-positive group is already expressing the same risk burden at the same level. |
| Which variables can we compare directly, which ones are only rough proxies, and which ones should stay separate? | - | The matrix is showing that the notebook has only a **small set of honest bridges** between the two cohorts. Age is the cleanest common feature, systolic-versus-resting pressure is a cautious cardiovascular bridge, and pulse-versus-maximum-heart-rate is only a weak proxy relationship. This matters because the thesis argument becomes stronger when direct and proxy comparisons are kept separate rather than blended into one overconfident claim. |
| How do the PCOS-negative and PCOS-positive groups differ on the main heart and metabolic variables, and how should we place those results beside the heart reference group? | - | Table 1A is showing the **primary thesis evidence base** because it compares PCOS-positive and PCOS-negative women inside the same cleaned cohort. Table 1B is broadening the view by adding the older female heart cohort, but it should be read as background benchmarking rather than direct biological equivalence. The most defensible use of the heart table is to highlight how the PCOS group sits earlier in the age path over time while still showing measurable heart and metabolic burden that can be tracked clinically. |
| Are PCOS-positive patients younger than the heart-disease reference group, and what does that age pattern mean? | `q01_age_trajectory_kde.png` | The age densities are showing a clear separation between the younger reproductive-age PCOS cohort and the older cardiovascular reference cohort. This pattern supports the thesis framing that PCOS may be appearing earlier on a longer heart and metabolic path over time, but it does **not** show that the PCOS-positive subgroup is already equivalent to the heart-positive subgroup in absolute cardiovascular burden. The value of this figure is therefore chronological and background: it helps position the PCOS-positive group on a possible earlier-life pathway that would need longitudinal follow-up to confirm. |
| Does systolic blood pressure differ inside the PCOS dataset, and how does it sit beside the older heart reference pressure values? | `q02_systolic_pressure_with_heart_reference.png` | The systolic-pressure distributions inside the PCOS cohort are appearing heavily overlapped, which suggests that systolic pressure alone is not strongly separating PCOS-positive from PCOS-negative women in this dataset. The heart reference lines are sitting higher, but that pattern is being read cautiously because the heart cohort is much older and is not a matched comparison group. In thesis terms, systolic pressure is still worth monitoring clinically, but it is looking more like a **background cardiovascular context variable** than a strong standalone discriminator of PCOS status. |
| Is pulse rate higher in PCOS-positive women, and does it add useful heart-related context without being confused with blood pressure? | `q03_pulse_rate_raincloud.png` | The raincloud plot is showing substantial overlap between the two PCOS groups, with any upward shift in pulse rate appearing modest rather than dramatic. That means pulse rate adds more as a **supporting nervous-system context marker** than as a strong diagnostic separator on its own. Clinically, a higher resting pulse can still matter because it may reflect sympathetic or nervous-system strain, but this dataset suggests that the pulse signal should be interpreted alongside metabolic and menstrual-disruption markers rather than in isolation. |
| Does higher BMI act as a bridge between PCOS and later heart risk? | `q04_bmi_density.png` | The BMI distributions are showing the clearest rightward shift among the continuous within-PCOS variables examined so far, with the PCOS-positive group occupying more of the overweight and upper-tail range. This pattern fits the thesis mechanism that places adiposity and insulin-resistance-related burden along the pathway linking PCOS to later cardiovascular risk. In practical terms, BMI is behaving like one of the most easy to read heart and metabolic markers in the notebook because it is clinically familiar, easy to monitor, and biologically connected to multiple downstream risk processes. |
| Does random blood sugar rise in the PCOS-positive group, and does that strengthen the metabolic-risk part of thesis H1? | `q05_random_blood_sugar_ecdf.png` | The ECDF curves are showing only a modest separation, which means the glucose-related signal is present but not sharply exclusive to the PCOS-positive group. That is still clinically useful because even a moderate upward shift in random blood sugar can reinforce the interpretation that metabolic dysregulation is clustering around PCOS status. For the thesis, this variable is likely to matter most when it is read together with BMI rather than as a standalone screening feature. |
| Does haemoglobin differ by PCOS status, and does that support an exploratory anaemia-related heart-risk discussion? | `q06_haemoglobin_split_violin.png` | The split violin is showing heavy overlap between the two haemoglobin distributions, which means haemoglobin acts as an exploratory rather than a decisive PCOS-associated marker in this dataset. That does not make the variable irrelevant, because lower haemoglobin can still increase cardiac workload through compensatory physiology, but the pattern here should be treated cautiously and described as **hypothesis-generating** rather than confirmatory. In the thesis chapter, haemoglobin therefore works best as a supporting pathway discussion rather than as a core screening signal. |
| Which simple risk flags separate the PCOS-positive group from the PCOS-negative group, and how do those patterns look beside the heart reference group? | `risk_flag_prevalence_heatmap.png` | The risk-flag table is helping the heart and metabolic pattern cohere into something clinically easier to read. Inside the PCOS cohort, the clearest differentiators are expected to come from **BMI-related burden, elevated random blood sugar, and the heuristic higher cycle-irregularity code**, while the heart cohort remains dominated by age and pressure-related burden. This strengthens thesis H1 in a nuanced way: the PCOS-positive profile is not simply mirroring clear cardiovascular disease, but it is clustering around a meaningful upstream risk pattern that deserves monitoring. |
| Which continuous features show useful within-dataset differences, and which ones still look small after effect size (how big the difference really is) and Bonferroni correction (a stricter rule after many tests) are considered together? | `effect_size_dot_plot.png` | The statistical section is being designed to prevent overclaiming. A small p-value without a meaningful rank-biserial effect size (how big the difference really is) is not being treated as persuasive evidence, and a result that fails the Bonferroni threshold is being described more cautiously than one that survives multiplicity correction. This is especially important for thesis H1 because the claim becomes most credible when the strongest variables are not only statistically detectable but also large enough to matter clinically. |
| How does the PCOS-positive group compare with the heart-positive group when we only use shared or carefully matched profile features? | `shared_profile_radar_chart.png` | The shared-profile charts are deliberately narrow because they are being limited to the few variables that can be defended group-levelly across the two cohorts. They are useful for showing directional resemblance or divergence, especially on age and pressure burden, but they are not claiming that the PCOS-positive and heart-positive groups are biologically equivalent. The strongest lesson from this section is that the PCOS-positive subgroup may be showing an upstream heart and metabolic profile, while the heart-positive subgroup is representing a later and older clinical endpoint population. |


## Important Cautions
- This is an ecological comparison. The PCOS and heart datasets contain different people.
- The age gap between the two cohorts is large and must be treated as a major confound.
- Cross-dataset results are descriptive and non-causal.

## How This Notebook Connects to the Next Notebook
The next notebook turns the cleaned clinical and survey-ready information into modelling sets for the non-invasive and invasive models.
