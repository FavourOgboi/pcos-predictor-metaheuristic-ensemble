# PCOS Predictor Metaheuristic Thesis Project

## Project Overview
This repository is holding the working notebooks, cleaned datasets, figures, and model-preparation files for an MSc project on early PCOS detection. The current repo already contains the cleaning phase, the EDA phase, the PCOS-heart association notebook, and the feature-engineering notebook.

The project is using four main data branches:

- the cleaned clinical PCOS dataset as the main study table
- the infertility sidecar dataset for hormonal support
- the cleaned survey dataset for self-reported non-invasive pattern checks
- the cleaned female heart dataset as a cardiovascular reference cohort

## Current Project Status
| Notebook | Main role | Current state |
| --- | --- | --- |
| `01_pcos_full_cleaning.ipynb` | Clinical cleaning | Completed and present in repo |
| `02_pcos_infertility_cleaning.ipynb` | Hormonal sidecar cleaning | Completed and present in repo |
| `03_pcos_survey_cleaning.ipynb` | Survey cleaning | Completed and present in repo |
| `04_heart_cleaning_eda.ipynb` | Heart cleaning and initial EDA | Completed and present in repo |
| `05a_pcos_clinical_eda_enhanced.ipynb` | Main clinical EDA | Completed and present in repo |
| `05b_pcos_hormonal_eda_enhanced.ipynb` | Hormonal EDA | Completed and present in repo |
| `05c_pcos_survey_eda_enhanced.ipynb` | Survey EDA | Completed and present in repo |
| `06_pcos_heart_association.ipynb` | PCOS-heart association analysis | Completed and present in repo |
| `07_feature_engineering.ipynb` | Feature engineering and modelling-set preparation | Completed and present in repo |

## How to Read This Project
The easiest order is the same order used in the repo:

1. cleaning notebooks (`01` to `04`)
2. PCOS-only EDA notebooks (`05a`, `05b`, `05c`)
3. PCOS-heart association notebook (`06`)
4. feature engineering and modelling-set preparation (`07`)

## Repository Structure
```text
pcos-predictor-metaheuristic/
|-- data/
|-- cleaned_data/
|   `-- modelling_sets/
|-- images/
|   |-- eda/
|   `-- feature_engineering/
|-- models/
|-- notebooks/
|   `-- EDA/
|-- docs/
|   `-- notebook_guides/
`-- scripts/
```

## Dataset Summary
| Cleaned dataset | Shape | Role |
| --- | --- | --- |
| `cleaned_data/PCOS_full_cleaned.csv` | 541 x 44 | Main clinical PCOS dataset |
| `cleaned_data/PCOS_infertility_cleaned.csv` | 541 x 6 | Hormonal sidecar dataset |
| `cleaned_data/PCOS_survey_cleaned.csv` | 464 x 20 | Self-reported survey dataset |
| `cleaned_data/heart_cleaned.csv` | 96 x 14 | Female-only heart reference cohort |

## Current Important Outputs

### Cleaning Outputs
| File | Exists |
| --- | --- |
| `cleaned_data/PCOS_full_cleaned.csv` | Yes |
| `cleaned_data/PCOS_full_cleaning_log.csv` | Yes |
| `cleaned_data/PCOS_full_quality_summary.csv` | Yes |
| `cleaned_data/PCOS_infertility_cleaned.csv` | Yes |
| `cleaned_data/PCOS_infertility_cleaning_log.csv` | Yes |
| `cleaned_data/PCOS_infertility_quality_summary.csv` | Yes |
| `cleaned_data/PCOS_survey_cleaned.csv` | Yes |
| `cleaned_data/PCOS_survey_cleaning_log.csv` | Yes |
| `cleaned_data/PCOS_survey_quality_summary.csv` | Yes |
| `cleaned_data/heart_cleaned.csv` | Yes |
| `cleaned_data/heart_cleaning_log.csv` | Yes |
| `cleaned_data/heart_quality_summary.csv` | Yes |

### Modelling-Preparation Outputs from Notebook 07
| File | Shape |
| --- | --- |
| `cleaned_data/modelling_sets/model1_train.csv` | 582 x 20 |
| `cleaned_data/modelling_sets/model1_test.csv` | 109 x 20 |
| `cleaned_data/modelling_sets/model2_train.csv` | 582 x 16 |
| `cleaned_data/modelling_sets/model2_test.csv` | 109 x 16 |
| `cleaned_data/modelling_sets/survey_external_validation.csv` | 464 x 20 |

### Saved Model Artifacts
| File | Exists |
| --- | --- |
| `models/scaler_model1.pkl` | Yes |
| `models/scaler_model2.pkl` | Yes |
| `models/cv_strategy.pkl` | Yes |

### Saved Figure Folders
- `images/eda/clinical/`
- `images/eda/hormonal/`
- `images/eda/survey_enhanced/`
- `images/eda/association/`
- `images/feature_engineering/`

Notebook 07 currently has `13` saved feature-engineering figures.

## Documentation Index
The detailed teaching guides live in `docs/notebook_guides/`.

- [01_pcos_full_cleaning.ipynb](docs/notebook_guides/01_pcos_full_cleaning.md)
- [02_pcos_infertility_cleaning.ipynb](docs/notebook_guides/02_pcos_infertility_cleaning.md)
- [03_pcos_survey_cleaning.ipynb](docs/notebook_guides/03_pcos_survey_cleaning.md)
- [04_heart_cleaning_eda.ipynb](docs/notebook_guides/04_heart_cleaning_eda.md)
- [05a_pcos_clinical_eda_enhanced.ipynb](docs/notebook_guides/05a_pcos_clinical_eda_enhanced.md)
- [05b_pcos_hormonal_eda_enhanced.ipynb](docs/notebook_guides/05b_pcos_hormonal_eda_enhanced.md)
- [05c_pcos_survey_eda_enhanced.ipynb](docs/notebook_guides/05c_pcos_survey_eda_enhanced.md)
- [06_pcos_heart_association.ipynb](docs/notebook_guides/06_pcos_heart_association.md)
- [07_feature_engineering.ipynb](docs/notebook_guides/07_feature_engineering.md)

## Important Study Cautions
- The infertility dataset is a sidecar subset, not an external validation dataset.
- The survey dataset is self-reported and noisier than the clinical data.
- The heart dataset is a female-only reference cohort and is not linked to the PCOS patients.
- The PCOS-heart comparison notebook is ecological and non-causal.
- The age gap between the PCOS and heart cohorts is a major confound.
- Notebook 07 uses the survey table only for external-style validation alignment, not for training.

## Practical Use
- Raw source files are in `data/`.
- Cleaned analysis files are in `cleaned_data/`.
- Saved modelling sets are in `cleaned_data/modelling_sets/`.
- Figures are in `images/`.
- Reusable scalers and CV strategy are in `models/`.
- The notebooks are meant to be run locally in order.
