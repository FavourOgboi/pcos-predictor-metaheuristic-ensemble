# Heart Dataset Cleaning and Initial Exploratory Data Analysis

## Notebook Purpose
This notebook cleans the heart dataset, removes the heavy duplication in the raw file, filters to the female cohort, and performs initial cardiovascular EDA.

This notebook finishes the cleaning phase by preparing the cardiovascular reference cohort used later in the association notebook.

## Notebook Inputs and Outputs

### Inputs
| File | Exists |
| --- | --- |
| `data/heart.csv` | Yes |

### Outputs
| File | Exists |
| --- | --- |
| `cleaned_data/heart_cleaned.csv` | Yes |
| `cleaned_data/heart_cleaning_log.csv` | Yes |
| `cleaned_data/heart_quality_summary.csv` | Yes |
| `images/heart_age_distribution.png` | Yes |
| `images/heart_age_vs_target.png` | Yes |
| `images/heart_bivariate_panel.png` | Yes |
| `images/heart_cholesterol_distribution.png` | Yes |
| `images/heart_cholesterol_vs_target.png` | Yes |
| `images/heart_correlation_matrix.png` | Yes |
| `images/heart_max_heart_rate_distribution.png` | Yes |
| `images/heart_max_heart_rate_vs_target.png` | Yes |
| `images/heart_resting_bp_distribution.png` | Yes |
| `images/heart_resting_bp_vs_target.png` | Yes |
| `images/heart_target_distribution.png` | Yes |

### Notebook Size
| Item | Value |
| --- | --- |
| Notebook file | `notebooks/04_heart_cleaning_eda.ipynb` |
| Markdown cells | 10 |
| Code cells | 8 |
| Total cells | 18 |

## Step-by-Step Walkthrough
This section is following the notebook in cell order. Every code cell is included so you can teach the notebook step by step without guessing what happened.

### Cell 1 - Markdown
**What this cell is doing:** Heart Dataset Cleaning and Initial Exploratory Data Analysis.
**Why this step matters:** This matters because it sets the purpose of the notebook before any code starts.

**Notebook markdown content:**

# Heart Dataset Cleaning and Initial Exploratory Data Analysis

## Objective
This notebook is cleaning, validating, and exploring the heart disease dataset in order to identify cardiovascular risk patterns that can later be compared with PCOS-related features in the broader MSc study. The notebook is focusing on producing a trustworthy cardiovascular reference cohort rather than building any predictive model at this stage.

The analysis is beginning with the raw `heart.csv` dataset and is then restricting attention to a deduplicated female cohort. This choice is supporting later comparison with the PCOS datasets, while still acknowledging that the heart dataset includes women beyond the typical reproductive-age range represented in the PCOS study.

## Research Context
The wider thesis is investigating early, practical, and low-burden screening for PCOS using routine clinical and lifestyle features. Within that context, cardiovascular variables such as age, blood pressure, cholesterol, and heart-rate patterns are important because PCOS is often discussed alongside cardiometabolic risk. This notebook is therefore preparing a cleaned heart-disease comparison dataset that can later be used in the association study without blending it directly into the PCOS training workflow.

## What This Notebook Is Doing
This notebook is:
- loading and auditing the raw heart dataset
- documenting heavy duplication in the source file and removing exact duplicates
- filtering to the female comparison cohort
- standardizing cardiovascular feature names for clarity and consistency
- validating ranges and conditional missing-value handling
- producing initial univariate and bivariate EDA plots
- exporting the cleaned female heart cohort for later comparison with PCOS-related features

## Expected Output
The main cleaned output of this notebook is the deduplicated female cohort saved as:

`./cleaned_data/heart_cleaned.csv`

Supporting outputs are also being produced, including a cleaning log, a compact quality summary, and diagnostic figures saved to `./images/`.

## Notebook Structure
The sections that are following are being organized as a step-by-step cleaning and exploratory workflow:

1. **Reproducibility Setup and Path Configuration**  
   The notebook environment and file paths are being resolved for stable execution.

2. **Data Loading and Initial Inspection**  
   The raw cardiovascular dataset is being inspected for schema, missingness, duplication, and ranges.

3. **Cleaning and Cohort Definition**  
   Exact duplicates are being removed, the female comparison cohort is being selected, and the columns are being standardized.

4. **Validation and Initial EDA**  
   The cleaned cardiovascular cohort is being validated and visualized through univariate, bivariate, and correlation analyses.

5. **Dataset Export and Observational Summary**  
   The cleaned female heart cohort is being saved and the main cardiovascular observations are being summarized for later PCOS comparison.

## Reproducibility Note
All paths in this notebook are being kept relative to the project structure, and every material cleaning decision is being documented so the workflow can be rerun and audited later.

### Cell 2 - Markdown
**What this cell is doing:** Reproducibility Setup and Path Configuration.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Reproducibility Setup and Path Configuration

This section is preparing the notebook environment, importing the required libraries, and resolving the project directories used for cleaned data and figure exports.

### Cell 3 - Code
**What this code is doing:** Importing the libraries that are supporting data cleaning, validation, and visualization.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.

```python
# Importing the libraries that are supporting data cleaning, validation, and visualization.
from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

# Importing seaborn when it is available improves the default plot styling, while a matplotlib fallback keeps the notebook portable.
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='notebook')
except ImportError:
    sns = None
    plt.style.use('seaborn-v0_8-whitegrid')

# Setting display options keeps the cleaned cardiovascular table easier to inspect in wide notebook outputs.
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda value: f'{value:,.3f}')

# Fixing a seed now preserves reproducibility for any later sampling or stochastic checks.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Resolving the project root dynamically allows this notebook to run from the repo root or the notebooks directory.
def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / 'data' / 'heart.csv').exists():
            return candidate
    raise FileNotFoundError('The heart dataset could not be located relative to the current working directory.')

PROJECT_ROOT = resolve_project_root()
DATA_PATH = PROJECT_ROOT / 'data' / 'heart.csv'
CLEANED_DIR = PROJECT_ROOT / 'cleaned_data'
IMAGE_DIR = PROJECT_ROOT / 'images'

# Creating the output directories keeps the export steps idempotent across reruns.
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

display(Markdown(f'**Project root:** `{PROJECT_ROOT}`'))
display(Markdown(f'**Raw heart dataset:** `{DATA_PATH}`'))
display(Markdown(f'**Cleaned data directory:** `{CLEANED_DIR}`'))
display(Markdown(f'**Image directory:** `{IMAGE_DIR}`'))
```

### Cell 4 - Markdown
**What this cell is doing:** Data Loading and Initial Inspection.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

## Data Loading and Initial Inspection

The raw heart dataset is now being loaded and audited for structure, missingness, duplication, and broad distribution ranges before any cleaning decisions are being applied.

### Cell 5 - Code
**What this code is doing:** Loading the raw heart dataset provides the baseline table for the cleaning and cohort-definition workflow.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.

```python
# Loading the raw heart dataset provides the baseline table for the cleaning and cohort-definition workflow.
df_raw = pd.read_csv(DATA_PATH)

# Displaying the first rows provides a quick check of the schema and coded cardiovascular variables.
df_raw.head(10)
```

### Cell 6 - Code
**What this code is doing:** Capturing dataframe info in a text buffer makes the schema report easier to show inside the notebook.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
# Capturing dataframe info in a text buffer makes the schema report easier to show inside the notebook.
info_buffer = io.StringIO()
df_raw.info(buf=info_buffer)

# Building compact inspection tables keeps the initial audit readable and suitable for thesis-style reporting.
raw_shape_df = pd.DataFrame({'metric': ['rows', 'columns'], 'value': [df_raw.shape[0], df_raw.shape[1]]})
raw_missing_summary = (
    pd.DataFrame({'column': df_raw.columns, 'missing_count': df_raw.isna().sum().values, 'missing_percentage': (df_raw.isna().sum().values / len(df_raw)) * 100})
    .sort_values(['missing_count', 'column'], ascending=[False, True])
    .reset_index(drop=True)
)
raw_duplicate_rows = int(df_raw.duplicated().sum())
range_snapshot_df = pd.DataFrame(
    {
        'column': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
        'minimum': [df_raw['age'].min(), df_raw['trestbps'].min(), df_raw['chol'].min(), df_raw['thalach'].min(), df_raw['oldpeak'].min()],
        'maximum': [df_raw['age'].max(), df_raw['trestbps'].max(), df_raw['chol'].max(), df_raw['thalach'].max(), df_raw['oldpeak'].max()],
    }
)

display(Markdown('### Raw Dataset Shape'))
display(raw_shape_df)
display(Markdown('### Raw Data Types'))
print(info_buffer.getvalue())
display(Markdown('### Descriptive Statistics'))
display(df_raw.describe().T)
display(Markdown('### Missing-Value Summary'))
display(raw_missing_summary)
display(Markdown('### Duplicate Summary'))
display(pd.DataFrame({'check': ['Exact duplicate rows'], 'count': [raw_duplicate_rows]}))
display(Markdown('### Range Snapshot for Key Variables'))
display(range_snapshot_df)
```

### Cell 7 - Markdown
**What this cell is doing:** Cleaning and Female Cohort Definition.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Cleaning and Female Cohort Definition

This section is removing the heavy duplication in the raw file, restricting the data to the female comparison cohort, standardizing the schema, and preparing the cleaned cardiovascular dataset used in later analyses.

### Cell 8 - Code
**What this code is doing:** Copying the raw dataframe keeps the import intact so later summaries can compare the raw and cleaned states directly.
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.

```python
# Copying the raw dataframe keeps the import intact so later summaries can compare the raw and cleaned states directly.
df = df_raw.copy()
transformation_log = []

# Recording each material change keeps the heart-cleaning workflow transparent and exportable.
def log_transformation(step: str, detail: str, affected_items: int) -> None:
    transformation_log.append({'step': step, 'detail': detail, 'affected_items': affected_items})

# Converting all columns to numeric is appropriate because this heart dataset is fully coded with numeric fields.
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Removing exact duplicates before cohort filtering ensures that all downstream summaries use unique patient records.
duplicate_rows_removed = int(df.duplicated().sum())
if duplicate_rows_removed:
    df = df.drop_duplicates().copy()
    log_transformation('Removed duplicate rows', 'Dropped exact duplicate records from the raw heart dataset before subgroup filtering.', duplicate_rows_removed)

# Restricting to female records creates the cardiovascular comparison cohort used later alongside the PCOS datasets.
female_rows_before_filter = len(df)
df = df.loc[df['sex'] == 0].copy()
female_rows_selected = len(df)
log_transformation('Filtered female cohort', 'Selected only female records using sex == 0.', female_rows_selected)

# Renaming the coded cardiovascular columns improves readability and makes later cross-dataset comparison easier.
rename_map = {
    'sex': 'sex_code',
    'cp': 'chest_pain_type',
    'trestbps': 'resting_bp',
    'chol': 'cholesterol',
    'fbs': 'fasting_blood_sugar_flag',
    'restecg': 'resting_ecg_result',
    'thalach': 'max_heart_rate',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia_code',
    'target': 'heart_disease_target',
}
rename_log = pd.DataFrame({'original_name': list(rename_map.keys()), 'standardized_name': list(rename_map.values())})
df = df.rename(columns=rename_map)
log_transformation('Standardized column names', 'Renamed coded cardiovascular fields to clearer analysis-friendly names.', len(rename_map))

# Applying conservative sanity ranges helps identify values that are implausible enough to warrant missing-value treatment.
range_rules = {
    'age': (18, 100),
    'resting_bp': (70, 260),
    'cholesterol': (100, 700),
    'max_heart_rate': (50, 230),
    'st_depression': (0, 10),
}
range_issues = []
for column, (lower_bound, upper_bound) in range_rules.items():
    invalid_mask = df[column].notna() & ((df[column] < lower_bound) | (df[column] > upper_bound))
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        range_issues.append({'column': column, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'flagged_rows': invalid_count})
        df.loc[invalid_mask, column] = np.nan
        log_transformation('Applied range sanity checks', f'Set implausible values to missing in {column}.', invalid_count)

# Imputing only when cleaning creates missing values keeps this step conditional, which matches the raw no-missing baseline.
binary_or_coded_columns = ['sex_code', 'chest_pain_type', 'fasting_blood_sugar_flag', 'resting_ecg_result', 'exercise_induced_angina', 'slope', 'num_major_vessels', 'thalassemia_code', 'heart_disease_target']
continuous_columns = [column for column in df.columns if column not in binary_or_coded_columns]
imputation_records = []
for column in binary_or_coded_columns:
    missing_count = int(df[column].isna().sum())
    if missing_count:
        mode_values = df[column].mode(dropna=True)
        if not mode_values.empty:
            fill_value = mode_values.iloc[0]
            df[column] = df[column].fillna(fill_value)
            imputation_records.append({'column': column, 'strategy': 'mode', 'filled_count': missing_count, 'fill_value': fill_value})
            log_transformation('Imputed coded column', f'Filled missing values in {column} using the mode.', missing_count)
for column in continuous_columns:
    missing_count = int(df[column].isna().sum())
    if missing_count:
        median_value = df[column].median(skipna=True)
        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)
            imputation_records.append({'column': column, 'strategy': 'median', 'filled_count': missing_count, 'fill_value': median_value})
            log_transformation('Imputed continuous column', f'Filled missing values in {column} using the median.', missing_count)

# Casting integer-like coded columns back to integer dtype keeps the exported cohort easier to inspect and compare later.
integer_like_columns = ['sex_code', 'chest_pain_type', 'fasting_blood_sugar_flag', 'resting_ecg_result', 'exercise_induced_angina', 'slope', 'num_major_vessels', 'thalassemia_code', 'heart_disease_target']
for column in integer_like_columns:
    series = df[column].dropna().astype(float)
    if not series.empty and np.isclose(series % 1, 0).all():
        df[column] = df[column].round().astype('Int64')

range_issues_df = pd.DataFrame(range_issues)
imputation_log_df = pd.DataFrame(imputation_records)
cohort_summary_df = pd.DataFrame(
    {
        'metric': ['Raw rows', 'Raw columns', 'Duplicate rows removed', 'Unique rows after deduplication', 'Female rows retained'],
        'value': [len(df_raw), df_raw.shape[1], duplicate_rows_removed, female_rows_before_filter, female_rows_selected],
    }
)
target_distribution_df = df['heart_disease_target'].value_counts(dropna=False).rename_axis('heart_disease_target').reset_index(name='count').sort_values('heart_disease_target').reset_index(drop=True)

display(Markdown('### Rename Map'))
display(rename_log)
display(Markdown('### Cohort Summary'))
display(cohort_summary_df)
display(Markdown('### Female Cohort Target Distribution'))
display(target_distribution_df)
display(Markdown('### Conditional Imputation Log'))
display(imputation_log_df if not imputation_log_df.empty else pd.DataFrame({'message': ['No imputation was required after cleaning.']}))
```

### Cell 9 - Markdown
**What this cell is doing:** Validation and Initial Exploratory Data Analysis.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

## Validation and Initial Exploratory Data Analysis

The cleaned female cohort is now being validated and explored through univariate, bivariate, and correlation analyses. These plots are helping to identify the cardiovascular variables that are most relevant for later comparison with PCOS-related metabolic and clinical patterns.

### Cell 10 - Code
**What this code is doing:** Summarizing the remaining missingness confirms whether the cleaned female cohort is ready for later comparison work.
**Why this step matters:** This matters because the dataset has to be brought into memory before any checks or analysis can happen.
**Saved figure or image output in this cell:** `heart_age_distribution.png`, `heart_cholesterol_distribution.png`, `heart_resting_bp_distribution.png`, `heart_max_heart_rate_distribution.png`, `heart_target_distribution.png`

```python
# Summarizing the remaining missingness confirms whether the cleaned female cohort is ready for later comparison work.
remaining_missing_summary = (
    pd.DataFrame({'column': df.columns, 'missing_count': df.isna().sum().values, 'missing_percentage': (df.isna().sum().values / len(df)) * 100})
    .sort_values(['missing_count', 'column'], ascending=[False, True])
    .reset_index(drop=True)
)

# Displaying coded value distributions makes the cleaned categorical fields easier to audit.
coded_field_summary = pd.DataFrame(
    [
        {
            'column': column,
            'unique_non_null_values': ', '.join(map(str, sorted(pd.Series(df[column].dropna().unique()).tolist()))),
        }
        for column in ['sex_code', 'chest_pain_type', 'fasting_blood_sugar_flag', 'resting_ecg_result', 'exercise_induced_angina', 'slope', 'num_major_vessels', 'thalassemia_code', 'heart_disease_target']
    ]
)

# Saving a consistent set of figure files keeps the EDA outputs reusable in later reporting.
saved_figures = []

fig, ax = plt.subplots(figsize=(8, 5))
if sns is not None:
    sns.histplot(df['age'], kde=True, bins=20, ax=ax, color='#2f6690')
else:
    ax.hist(df['age'].dropna(), bins=20, color='#2f6690', edgecolor='white', alpha=0.9)
ax.set_title('Age Distribution in the Female Heart Cohort')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
age_distribution_path = IMAGE_DIR / 'heart_age_distribution.png'
fig.savefig(age_distribution_path, dpi=300, bbox_inches='tight')
saved_figures.append(age_distribution_path.name)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
if sns is not None:
    sns.histplot(df['cholesterol'], kde=True, bins=20, ax=ax, color='#3a7d44')
else:
    ax.hist(df['cholesterol'].dropna(), bins=20, color='#3a7d44', edgecolor='white', alpha=0.9)
ax.set_title('Cholesterol Distribution in the Female Heart Cohort')
ax.set_xlabel('Cholesterol')
ax.set_ylabel('Frequency')
cholesterol_distribution_path = IMAGE_DIR / 'heart_cholesterol_distribution.png'
fig.savefig(cholesterol_distribution_path, dpi=300, bbox_inches='tight')
saved_figures.append(cholesterol_distribution_path.name)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
if sns is not None:
    sns.histplot(df['resting_bp'], kde=True, bins=20, ax=ax, color='#8d6a9f')
else:
    ax.hist(df['resting_bp'].dropna(), bins=20, color='#8d6a9f', edgecolor='white', alpha=0.9)
ax.set_title('Resting Blood Pressure Distribution in the Female Heart Cohort')
ax.set_xlabel('Resting blood pressure')
ax.set_ylabel('Frequency')
resting_bp_distribution_path = IMAGE_DIR / 'heart_resting_bp_distribution.png'
fig.savefig(resting_bp_distribution_path, dpi=300, bbox_inches='tight')
saved_figures.append(resting_bp_distribution_path.name)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
if sns is not None:
    sns.histplot(df['max_heart_rate'], kde=True, bins=20, ax=ax, color='#d17b0f')
else:
    ax.hist(df['max_heart_rate'].dropna(), bins=20, color='#d17b0f', edgecolor='white', alpha=0.9)
ax.set_title('Maximum Heart Rate Distribution in the Female Heart Cohort')
ax.set_xlabel('Maximum heart rate')
ax.set_ylabel('Frequency')
max_heart_rate_distribution_path = IMAGE_DIR / 'heart_max_heart_rate_distribution.png'
fig.savefig(max_heart_rate_distribution_path, dpi=300, bbox_inches='tight')
saved_figures.append(max_heart_rate_distribution_path.name)
plt.show()

target_counts = df['heart_disease_target'].value_counts(dropna=False).sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(target_counts.index.astype(str), target_counts.values, color=['#7f8c8d', '#c1121f'])
ax.set_title('Heart Disease Target Distribution in the Female Cohort')
ax.set_xlabel('heart_disease_target')
ax.set_ylabel('Count')
target_distribution_path = IMAGE_DIR / 'heart_target_distribution.png'
fig.savefig(target_distribution_path, dpi=300, bbox_inches='tight')
saved_figures.append(target_distribution_path.name)
plt.show()

display(Markdown('### Remaining Missingness After Cleaning'))
display(remaining_missing_summary)
display(Markdown('### Coded Field Validation'))
display(coded_field_summary)
```

### Cell 11 - Markdown
**What this cell is doing:** Initial Observations.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Initial Observations

The cleaned female cohort is showing a class imbalance toward positive heart-disease cases, which means later interpretation should remain cautious when comparing target prevalence across datasets. The age, cholesterol, resting blood pressure, and maximum heart-rate distributions are providing the first view of the cardiovascular profile that will later be contrasted with PCOS-related metabolic patterns.

### Cell 12 - Code
**What this code is doing:** Boxplots are helping to show how the main cardiovascular measures differ across heart-disease status in the female cohort.
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `heart_bivariate_panel.png`, `heart_age_vs_target.png`, `heart_cholesterol_vs_target.png`, `heart_resting_bp_vs_target.png`, `heart_max_heart_rate_vs_target.png`

```python
# Boxplots are helping to show how the main cardiovascular measures differ across heart-disease status in the female cohort.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if sns is not None:
    sns.boxplot(x='heart_disease_target', y='age', data=df, ax=axes[0, 0], color='#7fb3d5')
    sns.boxplot(x='heart_disease_target', y='cholesterol', data=df, ax=axes[0, 1], color='#82c4a5')
    sns.boxplot(x='heart_disease_target', y='resting_bp', data=df, ax=axes[1, 0], color='#c39bd3')
    sns.boxplot(x='heart_disease_target', y='max_heart_rate', data=df, ax=axes[1, 1], color='#f0b27a')
else:
    df.boxplot(column='age', by='heart_disease_target', ax=axes[0, 0])
    df.boxplot(column='cholesterol', by='heart_disease_target', ax=axes[0, 1])
    df.boxplot(column='resting_bp', by='heart_disease_target', ax=axes[1, 0])
    df.boxplot(column='max_heart_rate', by='heart_disease_target', ax=axes[1, 1])

axes[0, 0].set_title('Heart Disease vs Age')
axes[0, 1].set_title('Heart Disease vs Cholesterol')
axes[1, 0].set_title('Heart Disease vs Resting Blood Pressure')
axes[1, 1].set_title('Heart Disease vs Maximum Heart Rate')
for ax in axes.flat:
    ax.set_xlabel('heart_disease_target')
plt.suptitle('Bivariate Comparisons by Heart Disease Target', y=1.02)
plt.tight_layout()
bivariate_panel_path = IMAGE_DIR / 'heart_bivariate_panel.png'
plt.savefig(bivariate_panel_path, dpi=300, bbox_inches='tight')
saved_figures.append(bivariate_panel_path.name)
plt.show()

# Saving individual plot files as well keeps the main comparison figures easy to reuse in reports or later notebooks.
for y_column, filename, color in [
    ('age', 'heart_age_vs_target.png', '#7fb3d5'),
    ('cholesterol', 'heart_cholesterol_vs_target.png', '#82c4a5'),
    ('resting_bp', 'heart_resting_bp_vs_target.png', '#c39bd3'),
    ('max_heart_rate', 'heart_max_heart_rate_vs_target.png', '#f0b27a'),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns is not None:
        sns.boxplot(x='heart_disease_target', y=y_column, data=df, ax=ax, color=color)
    else:
        df.boxplot(column=y_column, by='heart_disease_target', ax=ax)
    ax.set_title(f'{y_column} by Heart Disease Target')
    ax.set_xlabel('heart_disease_target')
    figure_path = IMAGE_DIR / filename
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    saved_figures.append(figure_path.name)
    plt.show()
```

### Cell 13 - Markdown
**What this cell is doing:** Comparative Observations.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

### Comparative Observations

These boxplots are helping to assess whether positive heart-disease cases in the female cohort are clustering toward older age, higher cholesterol, and elevated resting blood pressure. Even when the shifts are not extreme for every variable, the comparisons are highlighting which cardiovascular measures may be most informative when the later PCOS association notebook is comparing shared risk patterns across datasets.

### Cell 14 - Code
**What this code is doing:** The correlation matrix is helping to identify which cardiovascular variables move together in the cleaned female cohort.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.
**Saved figure or image output in this cell:** `heart_correlation_matrix.png`

```python
# The correlation matrix is helping to identify which cardiovascular variables move together in the cleaned female cohort.
fig, ax = plt.subplots(figsize=(12, 9))
correlation_matrix = df.corr(numeric_only=True)
if sns is not None:
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
else:
    heatmap = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    fig.colorbar(heatmap, ax=ax)
ax.set_title('Correlation Matrix for the Cleaned Female Heart Cohort')
correlation_matrix_path = IMAGE_DIR / 'heart_correlation_matrix.png'
fig.savefig(correlation_matrix_path, dpi=300, bbox_inches='tight')
saved_figures.append(correlation_matrix_path.name)
plt.show()

correlation_with_target = correlation_matrix['heart_disease_target'].sort_values(ascending=False).reset_index()
correlation_with_target.columns = ['feature', 'correlation_with_heart_disease_target']

display(Markdown('### Correlation with the Heart Disease Target'))
display(correlation_with_target)
display(Markdown('### Saved Diagnostic Figures'))
display(pd.DataFrame({'saved_figure': saved_figures}))
```

### Cell 15 - Markdown
**What this cell is doing:** Correlation Observations.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

### Correlation Observations

The correlation structure is helping to identify which cardiovascular variables are most closely aligned with the heart-disease outcome in this female cohort. These relationships are especially relevant for the later PCOS-versus-cardiovascular comparison because age, cholesterol, resting blood pressure, and heart-rate behavior are among the most plausible shared risk pathways.

### Cell 16 - Markdown
**What this cell is doing:** Dataset Export and Cleaning Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Dataset Export and Cleaning Summary

The cleaned female heart cohort is now being exported together with a compact transformation log and quality summary. This is preserving a reusable comparison dataset for the later association study while keeping the cleaning workflow transparent.

### Cell 17 - Code
**What this code is doing:** Converting the transformation history into a dataframe makes the cleaning record easy to inspect and export.
**Why this step matters:** This matters because later notebooks or the thesis write-up depend on the saved file.

```python
# Converting the transformation history into a dataframe makes the cleaning record easy to inspect and export.
transformation_log_df = pd.DataFrame(transformation_log)

# Building a quality summary table provides a concise before-and-after snapshot of the cleaned cohort.
quality_summary_df = pd.DataFrame(
    {
        'check': [
            'Rows in raw dataset',
            'Columns in raw dataset',
            'Exact duplicate rows removed',
            'Unique rows after deduplication',
            'Female rows retained',
            'Columns in cleaned cohort',
            'Range flags handled',
            'Remaining missing cells after cleaning',
        ],
        'value': [
            len(df_raw),
            df_raw.shape[1],
            duplicate_rows_removed,
            female_rows_before_filter,
            len(df),
            df.shape[1],
            int(range_issues_df['flagged_rows'].sum()) if not range_issues_df.empty else 0,
            int(df.isna().sum().sum()),
        ],
    }
)

# Defining the export paths once keeps the saved outputs consistent with the project layout.
cleaned_dataset_path = CLEANED_DIR / 'heart_cleaned.csv'
transformation_log_path = CLEANED_DIR / 'heart_cleaning_log.csv'
quality_summary_path = CLEANED_DIR / 'heart_quality_summary.csv'

# Exporting the cleaned female cohort preserves the cardiovascular comparison dataset for later notebooks.
df.to_csv(cleaned_dataset_path, index=False)
transformation_log_df.to_csv(transformation_log_path, index=False)
quality_summary_df.to_csv(quality_summary_path, index=False)

display(Markdown(f'**Cleaned female heart cohort saved to:** `{cleaned_dataset_path}`'))
display(Markdown(f'**Transformation log saved to:** `{transformation_log_path}`'))
display(Markdown(f'**Quality summary saved to:** `{quality_summary_path}`'))
display(Markdown('### Cleaning Summary Table'))
display(quality_summary_df)
display(Markdown('### Transformation Log'))
display(transformation_log_df)

# Displaying the first rows of the cleaned cohort confirms the final schema and saved output structure.
df.head(10)
```

### Cell 18 - Markdown
**What this cell is doing:** Observational Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

## Observational Summary

The cleaned female heart cohort is showing that cardiovascular risk is being meaningfully profiled through age, cholesterol levels, resting blood pressure, and maximum heart-rate behavior. The cohort is also appearing imbalanced toward positive heart-disease cases, which should be kept in mind during later interpretation.

These variables are likely to remain central in the later PCOS-versus-cardiovascular comparison because they are capturing major cardiometabolic patterns that may overlap with the broader metabolic and vascular burden often discussed in PCOS research. This notebook is therefore preparing a clean cardiovascular reference point for the downstream association analysis.


## Results and Findings
### Quality Summary
| check | value |
| --- | --- |
| Rows in raw dataset | 1025 |
| Columns in raw dataset | 14 |
| Exact duplicate rows removed | 723 |
| Unique rows after deduplication | 302 |
| Female rows retained | 96 |
| Columns in cleaned cohort | 14 |
| Range flags handled | 0 |
| Remaining missing cells after cleaning | 0 |

### Cleaning Log
| step | detail | affected_items |
| --- | --- | --- |
| Removed duplicate rows | Dropped exact duplicate records from the raw heart dataset before subgroup filtering. | 723 |
| Filtered female cohort | Selected only female records using sex == 0. | 96 |
| Standardized column names | Renamed coded cardiovascular fields to clearer analysis-friendly names. | 12 |


## Important Cautions
- The cleaned heart cohort contains only female records and is not linked to the PCOS patients.
- The notebook is cleaning and descriptive EDA work. It is not used for PCOS model training.
- The heart cohort remains small after deduplication and female-only filtering.

## How This Notebook Connects to the Next Notebook
The next group of notebooks moves into PCOS-focused EDA, starting with the main clinical phenotype patterns.
