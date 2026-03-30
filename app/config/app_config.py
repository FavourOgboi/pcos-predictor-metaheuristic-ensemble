from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


APP_META = {
    "title": "PCOS Early Detection System",
    "subtitle": "A Metaheuristic-Optimized Ensemble Learning Approach",
    "researcher": "Ojimadu Chinaza Joy",
    "degree": "MSc",
    "institution": "",
    "supervisor": "",
    "year": "2026",
    "version": "1.0.0",
}


COLOR_SCHEME = {
    "primary": "#2176AE",
    "secondary": "#2A9D8F",
    "accent": "#E63946",
    "accent_soft": "#E9C46A",
    "navy": "#0A2342",
    "background": "#F8FAFE",
    "panel": "#EBF4FF",
    "text": "#1A2B4A",
    "muted": "#6B7C93",
}


MODEL_PATHS = {
    "model1_final": REPO_ROOT / "models" / "model1_final_optimized.pkl",
    "model2_final": REPO_ROOT / "models" / "model2_final_optimized.pkl",
    "scaler1": REPO_ROOT / "models" / "scaler_model1.pkl",
    "scaler2": REPO_ROOT / "models" / "scaler_model2.pkl",
    "cv_strategy": REPO_ROOT / "models" / "cv_strategy.pkl",
    "model1_rf": REPO_ROOT / "models" / "model1_rf.pkl",
    "model2_rf": REPO_ROOT / "models" / "model2_rf.pkl",
    "model1_xgb": REPO_ROOT / "models" / "model1_xgb.pkl",
    "model2_xgb": REPO_ROOT / "models" / "model2_xgb.pkl",
}


DATA_PATHS = {
    "pcos_clinical": REPO_ROOT / "cleaned_data" / "PCOS_full_cleaned.csv",
    "pcos_survey": REPO_ROOT / "cleaned_data" / "PCOS_survey_cleaned.csv",
    "heart_cleaned": REPO_ROOT / "cleaned_data" / "heart_cleaned.csv",
    "final_results": REPO_ROOT / "cleaned_data" / "modelling_sets" / "final_model_results.csv",
    "shap_m1": REPO_ROOT / "cleaned_data" / "modelling_sets" / "shap_values_model1.npy",
    "shap_m2": REPO_ROOT / "cleaned_data" / "modelling_sets" / "shap_values_model2.npy",
    "shap_rank_m1": REPO_ROOT / "cleaned_data" / "modelling_sets" / "shap_feature_ranking_model1.csv",
    "shap_rank_m2": REPO_ROOT / "cleaned_data" / "modelling_sets" / "shap_feature_ranking_model2.csv",
    "feat_names_m1": REPO_ROOT / "cleaned_data" / "modelling_sets" / "feature_names_model1.json",
    "feat_names_m2": REPO_ROOT / "cleaned_data" / "modelling_sets" / "feature_names_model2.json",
    "survey_results": REPO_ROOT / "cleaned_data" / "modelling_sets" / "survey_validation_results.csv",
    "opt_results": REPO_ROOT / "cleaned_data" / "modelling_sets" / "optimization_results.csv",
    "m1_best_params": REPO_ROOT / "models" / "model1_best_params.json",
    "m2_best_params": REPO_ROOT / "models" / "model2_best_params.json",
}


IMAGE_PATHS = {
    "shap_m1_bar": REPO_ROOT / "images" / "shap" / "model1" / "03_m1_shap_bar_global.png",
    "shap_m1_beeswarm": REPO_ROOT / "images" / "shap" / "model1" / "04_m1_shap_beeswarm.png",
    "shap_m2_bar": REPO_ROOT / "images" / "shap" / "model2" / "14_m2_shap_bar_global.png",
    "shap_m2_beeswarm": REPO_ROOT / "images" / "shap" / "model2" / "15_m2_shap_beeswarm.png",
    "convergence_m1": REPO_ROOT / "images" / "optimization" / "09_model1_all_convergence_comparison.png",
    "convergence_m2": REPO_ROOT / "images" / "optimization" / "13_model2_all_convergence_comparison.png",
    "master_heatmap": REPO_ROOT / "images" / "optimization" / "19_master_optimization_heatmap.png",
    "researcher_photo": APP_ROOT / "assets" / "researcher_photo.png",
    "logo": APP_ROOT / "assets" / "logo.png",
    "banner": APP_ROOT / "assets" / "pcos_banner.png",
}


FEATURE_NAMES_M1 = [
    "age_yrs",
    "bmi",
    "cycle_regularity_binary",
    "cycle_length_days",
    "weight_gain_y_n",
    "hair_growth_y_n",
    "skin_darkening_y_n",
    "hair_loss_y_n",
    "pimples_y_n",
    "fast_food_y_n",
    "regular_exercise_y_n",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "waist_hip_ratio",
    "rbs_mg_dl",
    "symptom_burden",
    "bmi_category",
    "bp_elevated_flag",
    "lifestyle_risk",
]


FEATURE_NAMES_M2 = [
    "amh_ng_ml",
    "fsh_miu_ml",
    "lh_miu_ml",
    "fsh_lh_ratio",
    "tsh_miu_l",
    "prl_ng_ml",
    "vit_d3_ng_ml",
    "prg_ng_ml",
    "beta_hcg_i_miu_ml",
    "beta_hcg_ii_miu_ml",
    "follicle_no_left",
    "follicle_no_right",
    "avg_follicle_size_left_mm",
    "avg_follicle_size_right_mm",
    "endometrium_mm",
]


MODEL1_LABELS = {
    "age_yrs": "Age (years)",
    "bmi": "BMI",
    "cycle_regularity_binary": "Cycle regularity",
    "cycle_length_days": "Cycle length (days)",
    "weight_gain_y_n": "Weight gain",
    "hair_growth_y_n": "Hair growth",
    "skin_darkening_y_n": "Skin darkening",
    "hair_loss_y_n": "Hair loss",
    "pimples_y_n": "Pimples",
    "fast_food_y_n": "Fast food",
    "regular_exercise_y_n": "Regular exercise",
    "systolic_bp_mmhg": "Systolic BP",
    "diastolic_bp_mmhg": "Diastolic BP",
    "waist_hip_ratio": "Waist-hip ratio",
    "rbs_mg_dl": "Random blood sugar",
    "symptom_burden": "Symptom burden",
    "bmi_category": "BMI category",
    "bp_elevated_flag": "BP elevated flag",
    "lifestyle_risk": "Lifestyle risk",
}


MODEL2_LABELS = {
    "amh_ng_ml": "AMH (ng/mL)",
    "fsh_miu_ml": "FSH (mIU/mL)",
    "lh_miu_ml": "LH (mIU/mL)",
    "fsh_lh_ratio": "FSH/LH ratio",
    "tsh_miu_l": "TSH (mIU/L)",
    "prl_ng_ml": "Prolactin (ng/mL)",
    "vit_d3_ng_ml": "Vitamin D3 (ng/mL)",
    "prg_ng_ml": "Progesterone (ng/mL)",
    "beta_hcg_i_miu_ml": "Beta-HCG I (mIU/mL)",
    "beta_hcg_ii_miu_ml": "Beta-HCG II (mIU/mL)",
    "follicle_no_left": "Follicle count left",
    "follicle_no_right": "Follicle count right",
    "avg_follicle_size_left_mm": "Avg follicle size left (mm)",
    "avg_follicle_size_right_mm": "Avg follicle size right (mm)",
    "endometrium_mm": "Endometrium (mm)",
}


FEATURE_MEANINGS_M1 = {
    "age_yrs": "Age helps place the person within the common reproductive-age PCOS window.",
    "bmi": "BMI reflects body weight pattern and metabolic strain.",
    "cycle_regularity_binary": "Irregular cycles are one of the strongest routine screening signs of PCOS.",
    "cycle_length_days": "Longer cycles often reflect ovulatory disturbance.",
    "weight_gain_y_n": "Unexplained weight gain can reflect insulin resistance or hormonal imbalance.",
    "hair_growth_y_n": "Excess hair growth is a common androgen-related symptom.",
    "skin_darkening_y_n": "Skin darkening can reflect insulin resistance and acanthosis-like changes.",
    "hair_loss_y_n": "Hair thinning can reflect androgen excess.",
    "pimples_y_n": "Acne can reflect androgen-driven skin changes.",
    "fast_food_y_n": "Regular fast food intake raises lifestyle-related metabolic risk.",
    "regular_exercise_y_n": "Regular exercise tends to reduce metabolic risk.",
    "systolic_bp_mmhg": "Higher systolic blood pressure reflects added cardiometabolic strain.",
    "diastolic_bp_mmhg": "Higher diastolic blood pressure reflects vascular strain.",
    "waist_hip_ratio": "Waist-hip ratio reflects central fat distribution.",
    "rbs_mg_dl": "Higher random blood sugar suggests impaired glucose regulation.",
    "symptom_burden": "This combines the main visible symptom count into one quick burden score.",
    "bmi_category": "BMI category turns BMI into a simple weight-status band.",
    "bp_elevated_flag": "This flag marks whether blood pressure has crossed an elevated threshold.",
    "lifestyle_risk": "This combines fast-food habit and exercise pattern into a simple lifestyle score.",
}


FEATURE_MEANINGS_M2 = {
    "amh_ng_ml": "Higher AMH is commonly linked with PCOS and higher follicle burden.",
    "fsh_miu_ml": "FSH helps describe ovarian hormonal balance.",
    "lh_miu_ml": "LH often shifts upward in PCOS-related endocrine patterns.",
    "fsh_lh_ratio": "A lower FSH/LH ratio is often discussed in PCOS literature.",
    "tsh_miu_l": "TSH helps check thyroid-related overlap.",
    "prl_ng_ml": "Prolactin helps separate PCOS-like symptoms from other endocrine causes.",
    "vit_d3_ng_ml": "Vitamin D status is often explored in metabolic and reproductive health.",
    "prg_ng_ml": "Progesterone helps reflect ovulatory function.",
    "beta_hcg_i_miu_ml": "Beta-HCG is included because it is part of the full cleaned invasive feature set.",
    "beta_hcg_ii_miu_ml": "The second Beta-HCG reading supports the benchmark feature contract.",
    "follicle_no_left": "Higher follicle count on ultrasound is part of the Rotterdam picture.",
    "follicle_no_right": "Higher follicle count on ultrasound is part of the Rotterdam picture.",
    "avg_follicle_size_left_mm": "Follicle size describes the ovarian ultrasound pattern.",
    "avg_follicle_size_right_mm": "Follicle size describes the ovarian ultrasound pattern.",
    "endometrium_mm": "Endometrium thickness gives added reproductive context.",
}


NORMAL_REFERENCE_M1 = {
    "age_yrs": (18, 45, "Common reproductive-age screening band"),
    "bmi": (18.5, 24.9, "Normal BMI range"),
    "waist_hip_ratio": (0.6, 0.85, "Lower female central adiposity range"),
    "rbs_mg_dl": (70, 140, "Typical non-emergency random glucose range"),
    "systolic_bp_mmhg": (90, 119, "Normal systolic blood pressure"),
    "diastolic_bp_mmhg": (60, 79, "Normal diastolic blood pressure"),
    "cycle_length_days": (21, 35, "Common menstrual cycle range"),
}


NORMAL_REFERENCE_M2 = {
    "amh_ng_ml": (0.1, 3.5, "Values above 3.5 ng/mL are often associated with PCOS"),
    "fsh_lh_ratio": (1.0, 10.0, "Ratios below 1 are often discussed in PCOS"),
    "follicle_no_left": (0, 11, "Counts of 12 or more can meet the Rotterdam threshold"),
    "follicle_no_right": (0, 11, "Counts of 12 or more can meet the Rotterdam threshold"),
}


DATASET_SUMMARY = [
    {"name": "PCOS Clinical", "n": 541, "features": 44, "note": "Cleaned clinical cohort"},
    {"name": "PCOS Survey", "n": 464, "features": 20, "note": "Self-reported screening cohort"},
    {"name": "Heart Reference", "n": 96, "features": 14, "note": "Female heart reference cohort"},
]


NAV_CARDS = [
    ("main.py", "🏠", "Home", "Study overview, researcher profile, and quick navigation."),
    ("pages/02_noninvasive_screening.py", "🟢", "Non-Invasive Screening", "Live screening with routine clinical and lifestyle inputs."),
    ("pages/03_invasive_benchmark.py", "🔵", "Clinical Benchmark", "Live prediction using hormonal and ultrasound features."),
    ("pages/04_insights_dashboard.py", "📊", "PCOS Insights Dashboard", "Interactive charts from the cleaned PCOS clinical dataset."),
    ("pages/05_heart_pcos_study.py", "❤️", "Heart & PCOS Study", "Group-level cardiometabolic comparison with the heart cohort."),
    ("pages/06_model_performance.py", "📈", "Model Performance", "Individual, stacking, and optimized model results."),
    ("pages/07_recommendations.py", "💡", "Recommendations", "Evidence-based takeaways for practice and research."),
    ("pages/08_disclaimer.py", "⚠️", "Disclaimer", "Appropriate use, limitations, and research-only notice."),
]
