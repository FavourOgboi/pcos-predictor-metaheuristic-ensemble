from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "notebook_guides"
README_PATH = ROOT / "README.md"


@dataclass
class NotebookMeta:
    notebook_path: str
    role: str
    position: str
    inputs: list[str]
    outputs: list[str]
    output_globs: list[str]
    cautions: list[str]
    next_step: str
    quality_summary: str | None = None
    cleaning_log: str | None = None
    extra_result_files: list[str] | None = None


NOTEBOOKS: dict[str, NotebookMeta] = {
    "01_pcos_full_cleaning": NotebookMeta(
        notebook_path="notebooks/01_pcos_full_cleaning.ipynb",
        role="This notebook cleans the main clinical PCOS dataset and creates the analysis-ready table used later in the project.",
        position="This is the first main notebook in the workflow. It creates the core clinical dataset that later EDA, association work, and modelling depend on.",
        inputs=["data/PCOS_data_without_infertility.xlsx"],
        outputs=[
            "cleaned_data/PCOS_full_cleaned.csv",
            "cleaned_data/PCOS_full_cleaning_log.csv",
            "cleaned_data/PCOS_full_quality_summary.csv",
            "images/pcos_full_bmi_distribution.png",
            "images/pcos_full_bmi_recorded_vs_calculated.png",
        ],
        output_globs=[],
        cautions=[
            "This notebook is cleaning only. It does not train any model.",
            "The notebook removes identifier columns and keeps the target column `pcos_y_n` intact.",
            "The saved cleaned clinical CSV is the source of truth for later notebooks.",
        ],
        next_step="The next notebook uses the cleaned hormonal sidecar dataset to preserve the infertility-related markers separately from the main table.",
        quality_summary="cleaned_data/PCOS_full_quality_summary.csv",
        cleaning_log="cleaned_data/PCOS_full_cleaning_log.csv",
    ),
    "02_pcos_infertility_cleaning": NotebookMeta(
        notebook_path="notebooks/02_pcos_infertility_cleaning.ipynb",
        role="This notebook cleans the infertility sidecar dataset and keeps the hormonal subset ready for later invasive-feature interpretation.",
        position="This notebook sits after the main clinical cleaning and before the survey and heart preparation steps.",
        inputs=["data/PCOS_infertility.csv"],
        outputs=[
            "cleaned_data/PCOS_infertility_cleaned.csv",
            "cleaned_data/PCOS_infertility_cleaning_log.csv",
            "cleaned_data/PCOS_infertility_quality_summary.csv",
            "images/infertility_amh_distribution.png",
            "images/infertility_beta_hcg_i_distribution.png",
            "images/infertility_beta_hcg_ii_distribution.png",
            "images/infertility_target_distribution.png",
        ],
        output_globs=[],
        cautions=[
            "This dataset is a sidecar subset of the same PCOS cohort, not an external validation dataset.",
            "It is used to understand invasive hormonal markers, not to create a separate training population.",
            "The notebook uses conservative cleaning because hormonal values can be wide in real data.",
        ],
        next_step="The next notebook prepares the survey data so the project has a non-clinical, self-reported source for later comparison and validation work.",
        quality_summary="cleaned_data/PCOS_infertility_quality_summary.csv",
        cleaning_log="cleaned_data/PCOS_infertility_cleaning_log.csv",
    ),
    "03_pcos_survey_cleaning": NotebookMeta(
        notebook_path="notebooks/03_pcos_survey_cleaning.ipynb",
        role="This notebook cleans and harmonizes the self-reported PCOS survey dataset so it can support non-invasive analysis and external-style validation.",
        position="This notebook follows the main clinical and hormonal cleaning work and prepares the survey branch of the project.",
        inputs=["data/PCOS_survey.csv"],
        outputs=[
            "cleaned_data/PCOS_survey_cleaned.csv",
            "cleaned_data/PCOS_survey_cleaning_log.csv",
            "cleaned_data/PCOS_survey_quality_summary.csv",
            "images/survey_bmi_distribution.png",
            "images/survey_height_distribution.png",
            "images/survey_pcos_target_distribution.png",
        ],
        output_globs=[],
        cautions=[
            "This dataset is self-reported, so it is noisier than the clinical dataset.",
            "Several variables are derived during cleaning, especially height in centimeters, BMI, and cycle features.",
            "The cleaned survey table is useful, but it should not be treated as equal in quality to the clinical data.",
        ],
        next_step="The next notebook prepares the heart dataset as a female-only cardiovascular reference cohort for later descriptive comparison.",
        quality_summary="cleaned_data/PCOS_survey_quality_summary.csv",
        cleaning_log="cleaned_data/PCOS_survey_cleaning_log.csv",
    ),
    "04_heart_cleaning_eda": NotebookMeta(
        notebook_path="notebooks/04_heart_cleaning_eda.ipynb",
        role="This notebook cleans the heart dataset, removes the heavy duplication in the raw file, filters to the female cohort, and performs initial cardiovascular EDA.",
        position="This notebook finishes the cleaning phase by preparing the cardiovascular reference cohort used later in the association notebook.",
        inputs=["data/heart.csv"],
        outputs=[
            "cleaned_data/heart_cleaned.csv",
            "cleaned_data/heart_cleaning_log.csv",
            "cleaned_data/heart_quality_summary.csv",
        ],
        output_globs=["images/heart_*.png"],
        cautions=[
            "The cleaned heart cohort contains only female records and is not linked to the PCOS patients.",
            "The notebook is cleaning and descriptive EDA work. It is not used for PCOS model training.",
            "The heart cohort remains small after deduplication and female-only filtering.",
        ],
        next_step="The next group of notebooks moves into PCOS-focused EDA, starting with the main clinical phenotype patterns.",
        quality_summary="cleaned_data/heart_quality_summary.csv",
        cleaning_log="cleaned_data/heart_cleaning_log.csv",
    ),
    "05a_pcos_clinical_eda_enhanced": NotebookMeta(
        notebook_path="notebooks/EDA/05a_pcos_clinical_eda_enhanced.ipynb",
        role="This notebook is the flagship EDA for the cleaned clinical PCOS dataset. It studies phenotype, symptoms, body measures, ovarian variables, and screening-relevant patterns.",
        position="This is the first deep EDA notebook after cleaning. It tells the main clinical PCOS story for the thesis.",
        inputs=["cleaned_data/PCOS_full_cleaned.csv"],
        outputs=[],
        output_globs=["images/eda/clinical/*.png"],
        cautions=[
            "This notebook is descriptive and exploratory. It does not train a model.",
            "The recorded cycle fields are used as they exist in the cleaned dataset and should not be over-interpreted beyond the source coding.",
            "The class split is imbalanced, so false negatives matter for later screening use.",
        ],
        next_step="The next EDA notebook studies the hormonal sidecar data to understand how invasive markers behave and whether they add distinct signal.",
    ),
    "05b_pcos_hormonal_eda_enhanced": NotebookMeta(
        notebook_path="notebooks/EDA/05b_pcos_hormonal_eda_enhanced.ipynb",
        role="This notebook studies the cleaned hormonal sidecar dataset and focuses on invasive markers such as AMH and beta-HCG.",
        position="This notebook follows the main clinical EDA and narrows the analysis to the smaller hormonal subset.",
        inputs=["cleaned_data/PCOS_infertility_cleaned.csv"],
        outputs=[],
        output_globs=["images/eda/hormonal/*.png"],
        cautions=[
            "This is still the same underlying PCOS cohort sidecar, not a new outside dataset.",
            "AMH is expected to be more informative than beta-HCG, but the notebook keeps beta-HCG in view and judges it carefully.",
            "The notebook supports interpretation and later ablation thinking rather than standalone training.",
        ],
        next_step="The next EDA notebook studies the survey dataset to see whether the non-invasive pattern still appears in self-reported data.",
    ),
    "05c_pcos_survey_eda_enhanced": NotebookMeta(
        notebook_path="notebooks/EDA/05c_pcos_survey_eda_enhanced.ipynb",
        role="This notebook studies the cleaned survey dataset and asks whether non-invasive PCOS patterns still appear in noisier self-reported data.",
        position="This notebook finishes the PCOS-only EDA stage by focusing on external-style, self-reported features.",
        inputs=["cleaned_data/PCOS_survey_cleaned.csv"],
        outputs=[],
        output_globs=["images/eda/survey_enhanced/*.png"],
        cautions=[
            "This dataset is self-reported and should be read as noisier than the clinical data.",
            "The notebook is checking portability of the pattern, not proving that survey data are as strong as clinical data.",
            "Some questions are grouped into multi-panel figures rather than one figure per single question.",
        ],
        next_step="The next notebook compares the cleaned PCOS and heart cohorts at a group level to study the cardiovascular-risk hypothesis.",
    ),
    "06_pcos_heart_association": NotebookMeta(
        notebook_path="notebooks/EDA/06_pcos_heart_association.ipynb",
        role="This notebook studies whether the PCOS cohort shows body-weight, blood-pressure, sugar, and heart-related patterns that connect to cardiovascular risk, while using the heart cohort only as a group-level reference.",
        position="This notebook comes after the PCOS-only EDA notebooks and is the main association notebook for the cardiovascular-risk hypothesis.",
        inputs=["cleaned_data/PCOS_full_cleaned.csv", "cleaned_data/heart_cleaned.csv"],
        outputs=[],
        output_globs=["images/eda/association/*.png"],
        cautions=[
            "This is an ecological comparison. The PCOS and heart datasets contain different people.",
            "The age gap between the two cohorts is large and must be treated as a major confound.",
            "Cross-dataset results are descriptive and non-causal.",
        ],
        next_step="The next notebook turns the cleaned clinical and survey-ready information into modelling sets for the non-invasive and invasive models.",
    ),
    "07_feature_engineering": NotebookMeta(
        notebook_path="notebooks/07_feature_engineering.ipynb",
        role="This notebook prepares the ready-to-train modelling sets for the study. It defines the non-invasive model and the invasive benchmark model from the same clinical cohort.",
        position="This notebook starts the modelling-preparation phase after cleaning and EDA are complete.",
        inputs=["cleaned_data/PCOS_full_cleaned.csv", "cleaned_data/PCOS_survey_cleaned.csv"],
        outputs=[
            "cleaned_data/modelling_sets/model1_train.csv",
            "cleaned_data/modelling_sets/model1_test.csv",
            "cleaned_data/modelling_sets/model2_train.csv",
            "cleaned_data/modelling_sets/model2_test.csv",
            "cleaned_data/modelling_sets/survey_external_validation.csv",
            "cleaned_data/modelling_sets/feature_names_model1.json",
            "cleaned_data/modelling_sets/feature_names_model2.json",
            "models/scaler_model1.pkl",
            "models/scaler_model2.pkl",
            "models/cv_strategy.pkl",
        ],
        output_globs=["images/feature_engineering/*.png"],
        cautions=[
            "Both model contracts use the same clinical train and test patient indices so later comparisons stay fair.",
            "Imputation, scaling, and SMOTE are done on training data only.",
            "The survey table is used only for external-style validation alignment, not for training.",
        ],
        next_step="The next modelling notebook will load these saved files and train the planned algorithms without re-splitting or re-scaling data.",
        extra_result_files=[
            "cleaned_data/modelling_sets/model1_train.csv",
            "cleaned_data/modelling_sets/model1_test.csv",
            "cleaned_data/modelling_sets/model2_train.csv",
            "cleaned_data/modelling_sets/model2_test.csv",
            "cleaned_data/modelling_sets/survey_external_validation.csv",
            "models/scaler_model1.pkl",
            "models/scaler_model2.pkl",
            "models/cv_strategy.pkl",
        ],
    ),
}


PROJECT_SEQUENCE = [
    "01_pcos_full_cleaning",
    "02_pcos_infertility_cleaning",
    "03_pcos_survey_cleaning",
    "04_heart_cleaning_eda",
    "05a_pcos_clinical_eda_enhanced",
    "05b_pcos_hormonal_eda_enhanced",
    "05c_pcos_survey_eda_enhanced",
    "06_pcos_heart_association",
    "07_feature_engineering",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_notebook(path: Path) -> list[dict]:
    return json.loads(read_text(path))["cells"]


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def csv_shape(path: Path) -> tuple[int, int]:
    rows = read_csv_rows(path)
    if not rows:
        return (0, 0)
    return (max(0, len(rows) - 1), len(rows[0]))


def file_exists(path_str: str) -> bool:
    return (ROOT / path_str).exists()


def rel(path: Path | str) -> str:
    p = path if isinstance(path, Path) else ROOT / path
    return p.relative_to(ROOT).as_posix()


def collect_globbed_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(rel(p) for p in ROOT.glob(pattern)))
    return files


def as_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows found._"
    rendered = []
    header = "| " + " | ".join(str(v) for v in rows[0]) + " |"
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    rendered.extend([header, divider])
    for row in rows[1:]:
        rendered.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rendered)


def make_file_table(paths: list[str]) -> str:
    rows = [["File", "Exists"]]
    for path_str in paths:
        rows.append([f"`{path_str}`", "Yes" if file_exists(path_str) else "No"])
    return as_markdown_table(rows)


def make_csv_shape_table(paths: list[str]) -> str:
    rows = [["File", "Shape"]]
    for path_str in paths:
        full = ROOT / path_str
        if full.exists():
            shape = csv_shape(full)
            rows.append([f"`{path_str}`", f"{shape[0]} x {shape[1]}"])
        else:
            rows.append([f"`{path_str}`", "Missing"])
    return as_markdown_table(rows)


def first_heading(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        if line.strip().startswith("#"):
            return line.strip().lstrip("#").strip()
    return ""


def first_comment(code_text: str) -> str:
    for line in code_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            break
    return "This code cell is running the step defined by the section around it."


def clean_md(text: str) -> str:
    return text.replace("\u00a0", " ").strip()


def simple_why_from_text(text: str, cell_type: str) -> str:
    lowered = text.lower()
    if cell_type == "markdown":
        if "introduction" in lowered or lowered.startswith("# "):
            return "This matters because it sets the purpose of the notebook before any code starts."
        if "research question" in lowered or re.search(r"^#+\\s*q\\d+", lowered):
            return "This matters because it tells the reader exactly what the next table and chart are trying to answer."
        if "insight" in lowered or "interpretation" in lowered or "summary" in lowered:
            return "This matters because it turns the output into a result that can be explained clearly."
        if "section" in lowered or "setup" in lowered:
            return "This matters because it marks a new step in the workflow and keeps the notebook easy to follow."
        return "This matters because it explains the logic behind the next part of the notebook."

    if "load" in lowered or "read" in lowered:
        return "This matters because the dataset has to be brought into memory before any checks or analysis can happen."
    if "print" in lowered or "summar" in lowered or "display" in lowered or "audit" in lowered:
        return "This matters because it shows the exact table or check behind the next decision or figure."
    if "plot" in lowered or "figure" in lowered or "chart" in lowered or "visual" in lowered:
        return "This matters because the chart helps compare groups and makes the pattern easier to see."
    if "save" in lowered or "export" in lowered:
        return "This matters because later notebooks or the thesis write-up depend on the saved file."
    if "derive" in lowered or "create" in lowered or "remap" in lowered:
        return "This matters because it creates the feature or structure that later steps depend on."
    if "split" in lowered or "scale" in lowered or "smote" in lowered or "imput" in lowered:
        return "This matters because it prepares the data in a safe way for later modelling."
    return "This matters because it carries out the main step described in this part of the notebook."


def pngs_in_code(code_text: str) -> list[str]:
    found = re.findall(r"([A-Za-z0-9_./-]+\.png)", code_text)
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def collect_question_results(cells: list[dict]) -> list[tuple[str, str, str]]:
    question_results: list[tuple[str, str, str]] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        text = clean_md("".join(cell.get("source", [])))
        heading = first_heading(text)
        question = ""
        if re.match(r"#+\s*Q\d+", text):
            question = heading
        elif "### Research Question" in text or "## Question" in text or "### Question" in text:
            bold = re.search(r"\*\*(.+?)\*\*", text, flags=re.S)
            question = bold.group(1).strip() if bold else heading
        if not question:
            continue

        figure = ""
        result = ""
        for next_idx in range(idx + 1, min(len(cells), idx + 14)):
            next_cell = cells[next_idx]
            next_text = clean_md("".join(next_cell.get("source", [])))
            if next_cell.get("cell_type") == "code" and not figure:
                pngs = pngs_in_code(next_text)
                if pngs:
                    figure = pngs[0]
            if next_cell.get("cell_type") == "markdown":
                if next_text.startswith("### Insight") or next_text.startswith("### Interpretation"):
                    lines = next_text.splitlines()[1:]
                    result = " ".join(line.strip() for line in lines if line.strip())
                    break
        question_results.append((question, figure, result))
    return question_results


def render_question_result_table(cells: list[dict]) -> str:
    pairs = collect_question_results(cells)
    if not pairs:
        return "_This notebook does not use a question-by-question EDA structure._"
    rows = [["Question asked", "Saved figure", "Main result in simple words"]]
    for question, figure, result in pairs:
        rows.append(
            [
                question,
                f"`{figure}`" if figure else "-",
                result if result else "Result text is explained later in the walkthrough.",
            ]
        )
    return as_markdown_table(rows)


def render_code_fence(code: str) -> str:
    return "```python\n" + code.rstrip() + "\n```"


def summarize_cells(cells: list[dict]) -> tuple[int, int]:
    md = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
    code = sum(1 for cell in cells if cell.get("cell_type") == "code")
    return md, code


def render_quality_summary(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = ROOT / path_str
    if not path.exists():
        return "_Quality summary file not found._"
    return as_markdown_table(read_csv_rows(path))


def render_cleaning_log(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = ROOT / path_str
    if not path.exists():
        return "_Cleaning log file not found._"
    return as_markdown_table(read_csv_rows(path))


def render_json_list(path_str: str) -> str:
    path = ROOT / path_str
    if not path.exists():
        return "_File not found._"
    items = json.loads(read_text(path))
    return "\n".join(f"- `{item}`" for item in items)


def render_walkthrough(cells: list[dict]) -> str:
    parts: list[str] = []
    for idx, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type", "unknown").capitalize()
        content = "".join(cell.get("source", []))
        if cell_type == "Markdown":
            heading = first_heading(content)
            summary = heading or "Notebook explanation"
            why = simple_why_from_text(content, "markdown")
            parts.append(f"### Cell {idx} - Markdown")
            parts.append(f"**What this cell is doing:** {summary}.")
            parts.append(f"**Why this step matters:** {why}")
            parts.append("")
            parts.append("**Notebook markdown content:**")
            parts.append("")
            parts.append(clean_md(content))
            parts.append("")
        else:
            summary = first_comment(content)
            why = simple_why_from_text(summary, "code")
            parts.append(f"### Cell {idx} - Code")
            parts.append(f"**What this code is doing:** {summary}")
            parts.append(f"**Why this step matters:** {why}")
            pngs = pngs_in_code(content)
            if pngs:
                parts.append(f"**Saved figure or image output in this cell:** {', '.join(f'`{png}`' for png in pngs)}")
            parts.append("")
            parts.append(render_code_fence(content))
            parts.append("")
    return "\n".join(parts)


def notebook_specific_results(slug: str, meta: NotebookMeta, cells: list[dict]) -> str:
    parts: list[str] = []
    if meta.quality_summary:
        parts.append("### Quality Summary")
        parts.append(render_quality_summary(meta.quality_summary))
        parts.append("")
    if meta.cleaning_log:
        parts.append("### Cleaning Log")
        parts.append(render_cleaning_log(meta.cleaning_log))
        parts.append("")
    if slug.startswith("05") or slug == "06_pcos_heart_association":
        parts.append("### Questions and Results")
        parts.append(render_question_result_table(cells))
        parts.append("")
    if slug == "07_feature_engineering":
        parts.append("### Saved Modelling Set Shapes")
        parts.append(
            make_csv_shape_table(
                [
                    "cleaned_data/modelling_sets/model1_train.csv",
                    "cleaned_data/modelling_sets/model1_test.csv",
                    "cleaned_data/modelling_sets/model2_train.csv",
                    "cleaned_data/modelling_sets/model2_test.csv",
                    "cleaned_data/modelling_sets/survey_external_validation.csv",
                ]
            )
        )
        parts.append("")
        parts.append("### Model 1 Feature Names")
        parts.append(render_json_list("cleaned_data/modelling_sets/feature_names_model1.json"))
        parts.append("")
        parts.append("### Model 2 Feature Names")
        parts.append(render_json_list("cleaned_data/modelling_sets/feature_names_model2.json"))
        parts.append("")
    return "\n".join(parts)


def generate_notebook_guide(slug: str, meta: NotebookMeta) -> str:
    notebook_path = ROOT / meta.notebook_path
    cells = load_notebook(notebook_path)
    md_count, code_count = summarize_cells(cells)
    title = first_heading(clean_md("".join(cells[0].get("source", [])))) or slug

    all_outputs = list(meta.outputs)
    all_outputs.extend(collect_globbed_files(meta.output_globs))
    all_outputs = list(dict.fromkeys(all_outputs))

    lines: list[str] = [
        f"# {title}",
        "",
        "## Notebook Purpose",
        meta.role,
        "",
        meta.position,
        "",
        "## Notebook Inputs and Outputs",
        "",
        "### Inputs",
        make_file_table(meta.inputs),
        "",
        "### Outputs",
        make_file_table(all_outputs),
        "",
        "### Notebook Size",
        as_markdown_table(
            [
                ["Item", "Value"],
                ["Notebook file", f"`{meta.notebook_path}`"],
                ["Markdown cells", str(md_count)],
                ["Code cells", str(code_count)],
                ["Total cells", str(len(cells))],
            ]
        ),
        "",
        "## Step-by-Step Walkthrough",
        "This section is following the notebook in cell order. Every code cell is included so you can teach the notebook step by step without guessing what happened.",
        "",
        render_walkthrough(cells),
        "",
        "## Results and Findings",
        notebook_specific_results(slug, meta, cells),
        "",
        "## Important Cautions",
    ]
    lines.extend(f"- {item}" for item in meta.cautions)
    lines.extend(
        [
            "",
            "## How This Notebook Connects to the Next Notebook",
            meta.next_step,
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def make_dataset_summary_table() -> str:
    rows = [["Cleaned dataset", "Shape", "Role"]]
    dataset_rows = [
        ("cleaned_data/PCOS_full_cleaned.csv", "Main clinical PCOS dataset"),
        ("cleaned_data/PCOS_infertility_cleaned.csv", "Hormonal sidecar dataset"),
        ("cleaned_data/PCOS_survey_cleaned.csv", "Self-reported survey dataset"),
        ("cleaned_data/heart_cleaned.csv", "Female-only heart reference cohort"),
    ]
    for path_str, role in dataset_rows:
        shape = csv_shape(ROOT / path_str)
        rows.append([f"`{path_str}`", f"{shape[0]} x {shape[1]}", role])
    return as_markdown_table(rows)


def make_notebook_status_table() -> str:
    rows = [["Notebook", "Main role", "Current state"]]
    descriptions = {
        "01_pcos_full_cleaning": "Clinical cleaning",
        "02_pcos_infertility_cleaning": "Hormonal sidecar cleaning",
        "03_pcos_survey_cleaning": "Survey cleaning",
        "04_heart_cleaning_eda": "Heart cleaning and initial EDA",
        "05a_pcos_clinical_eda_enhanced": "Main clinical EDA",
        "05b_pcos_hormonal_eda_enhanced": "Hormonal EDA",
        "05c_pcos_survey_eda_enhanced": "Survey EDA",
        "06_pcos_heart_association": "PCOS-heart association analysis",
        "07_feature_engineering": "Feature engineering and modelling-set preparation",
    }
    for slug in PROJECT_SEQUENCE:
        meta = NOTEBOOKS[slug]
        rows.append([f"`{Path(meta.notebook_path).name}`", descriptions[slug], "Completed and present in repo"])
    return as_markdown_table(rows)


def make_doc_index() -> str:
    lines = []
    for slug in PROJECT_SEQUENCE:
        meta = NOTEBOOKS[slug]
        guide_path = f"docs/notebook_guides/{Path(meta.notebook_path).stem}.md"
        lines.append(f"- [{Path(meta.notebook_path).name}]({guide_path})")
    return "\n".join(lines)


def make_root_readme() -> str:
    feature_outputs = collect_globbed_files(["images/feature_engineering/*.png"])
    guide_intro = (
        "This repository is holding the working notebooks, cleaned datasets, figures, and model-preparation files for an MSc project on early PCOS detection. "
        "The current repo already contains the cleaning phase, the EDA phase, the PCOS-heart association notebook, and the feature-engineering notebook."
    )
    modelling_shapes = make_csv_shape_table(
        [
            "cleaned_data/modelling_sets/model1_train.csv",
            "cleaned_data/modelling_sets/model1_test.csv",
            "cleaned_data/modelling_sets/model2_train.csv",
            "cleaned_data/modelling_sets/model2_test.csv",
            "cleaned_data/modelling_sets/survey_external_validation.csv",
        ]
    )
    return "\n".join(
        [
            "# PCOS Predictor Metaheuristic Thesis Project",
            "",
            "## Project Overview",
            guide_intro,
            "",
            "The project is using four main data branches:",
            "",
            "- the cleaned clinical PCOS dataset as the main study table",
            "- the infertility sidecar dataset for hormonal support",
            "- the cleaned survey dataset for self-reported non-invasive pattern checks",
            "- the cleaned female heart dataset as a cardiovascular reference cohort",
            "",
            "## Current Project Status",
            make_notebook_status_table(),
            "",
            "## How to Read This Project",
            "The easiest order is the same order used in the repo:",
            "",
            "1. cleaning notebooks (`01` to `04`)",
            "2. PCOS-only EDA notebooks (`05a`, `05b`, `05c`)",
            "3. PCOS-heart association notebook (`06`)",
            "4. feature engineering and modelling-set preparation (`07`)",
            "",
            "## Repository Structure",
            "```text",
            "pcos-predictor-metaheuristic/",
            "|-- data/",
            "|-- cleaned_data/",
            "|   `-- modelling_sets/",
            "|-- images/",
            "|   |-- eda/",
            "|   `-- feature_engineering/",
            "|-- models/",
            "|-- notebooks/",
            "|   `-- EDA/",
            "|-- docs/",
            "|   `-- notebook_guides/",
            "`-- scripts/",
            "```",
            "",
            "## Dataset Summary",
            make_dataset_summary_table(),
            "",
            "## Current Important Outputs",
            "",
            "### Cleaning Outputs",
            make_file_table(
                [
                    "cleaned_data/PCOS_full_cleaned.csv",
                    "cleaned_data/PCOS_full_cleaning_log.csv",
                    "cleaned_data/PCOS_full_quality_summary.csv",
                    "cleaned_data/PCOS_infertility_cleaned.csv",
                    "cleaned_data/PCOS_infertility_cleaning_log.csv",
                    "cleaned_data/PCOS_infertility_quality_summary.csv",
                    "cleaned_data/PCOS_survey_cleaned.csv",
                    "cleaned_data/PCOS_survey_cleaning_log.csv",
                    "cleaned_data/PCOS_survey_quality_summary.csv",
                    "cleaned_data/heart_cleaned.csv",
                    "cleaned_data/heart_cleaning_log.csv",
                    "cleaned_data/heart_quality_summary.csv",
                ]
            ),
            "",
            "### Modelling-Preparation Outputs from Notebook 07",
            modelling_shapes,
            "",
            "### Saved Model Artifacts",
            make_file_table(
                [
                    "models/scaler_model1.pkl",
                    "models/scaler_model2.pkl",
                    "models/cv_strategy.pkl",
                ]
            ),
            "",
            "### Saved Figure Folders",
            "- `images/eda/clinical/`",
            "- `images/eda/hormonal/`",
            "- `images/eda/survey_enhanced/`",
            "- `images/eda/association/`",
            "- `images/feature_engineering/`",
            "",
            f"Notebook 07 currently has `{len(feature_outputs)}` saved feature-engineering figures.",
            "",
            "## Documentation Index",
            "The detailed teaching guides live in `docs/notebook_guides/`.",
            "",
            make_doc_index(),
            "",
            "## Important Study Cautions",
            "- The infertility dataset is a sidecar subset, not an external validation dataset.",
            "- The survey dataset is self-reported and noisier than the clinical data.",
            "- The heart dataset is a female-only reference cohort and is not linked to the PCOS patients.",
            "- The PCOS-heart comparison notebook is ecological and non-causal.",
            "- The age gap between the PCOS and heart cohorts is a major confound.",
            "- Notebook 07 uses the survey table only for external-style validation alignment, not for training.",
            "",
            "## Practical Use",
            "- Raw source files are in `data/`.",
            "- Cleaned analysis files are in `cleaned_data/`.",
            "- Saved modelling sets are in `cleaned_data/modelling_sets/`.",
            "- Figures are in `images/`.",
            "- Reusable scalers and CV strategy are in `models/`.",
            "- The notebooks are meant to be run locally in order.",
            "",
        ]
    ).rstrip() + "\n"


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for slug in PROJECT_SEQUENCE:
        meta = NOTEBOOKS[slug]
        guide_path = DOCS_DIR / f"{Path(meta.notebook_path).stem}.md"
        guide_path.write_text(generate_notebook_guide(slug, meta), encoding="utf-8")
    README_PATH.write_text(make_root_readme(), encoding="utf-8")


if __name__ == "__main__":
    main()
