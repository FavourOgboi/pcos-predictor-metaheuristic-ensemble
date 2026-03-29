# PCOS Hormonal EDA — Invasive-Feature Usefulness

## Notebook Purpose
This notebook studies the cleaned hormonal sidecar dataset and focuses on invasive markers such as AMH and beta-HCG.

This notebook follows the main clinical EDA and narrows the analysis to the smaller hormonal subset.

## Notebook Inputs and Outputs

### Inputs
| File | Exists |
| --- | --- |
| `cleaned_data/PCOS_infertility_cleaned.csv` | Yes |

### Outputs
| File | Exists |
| --- | --- |
| `images/eda/hormonal/q01_class_balance.png` | Yes |
| `images/eda/hormonal/q02_amh_vs_pcos.png` | Yes |
| `images/eda/hormonal/q03_beta_hcg_i_dist.png` | Yes |
| `images/eda/hormonal/q04_beta_hcg_ii_dist.png` | Yes |
| `images/eda/hormonal/q05_hormonal_correlation_heatmap.png` | Yes |
| `images/eda/hormonal/q06_hcg_redundancy_scatter.png` | Yes |
| `images/eda/hormonal/q07_log_amh_normality.png` | Yes |
| `images/eda/hormonal/q08_mannwhitney_auc.png` | Yes |
| `images/eda/hormonal/q09_amh_boxen.png` | Yes |
| `images/eda/hormonal/q10_amh_vs_hcg_scatter.png` | Yes |
| `images/eda/hormonal/q11_hormonal_target_correlation.png` | Yes |
| `images/eda/hormonal/summary_all_markers.png` | Yes |

### Notebook Size
| Item | Value |
| --- | --- |
| Notebook file | `notebooks/EDA/05b_pcos_hormonal_eda_enhanced.ipynb` |
| Markdown cells | 30 |
| Code cells | 24 |
| Total cells | 54 |

## Step-by-Step Walkthrough
This section is following the notebook in cell order. Every code cell is included so you can teach the notebook step by step without guessing what happened.

### Cell 1 - Markdown
**What this cell is doing:** PCOS Hormonal EDA — Invasive-Feature Usefulness.
**Why this step matters:** This matters because it sets the purpose of the notebook before any code starts.

**Notebook markdown content:**

# PCOS Hormonal EDA — Invasive-Feature Usefulness

## Introduction
This notebook looks at the cleaned infertility side dataset. This table works as a hormonal subset of the main clinical PCOS dataset, not as a separate outside dataset.

The notebook uses `cleaned_data/PCOS_infertility_cleaned.csv` and focuses on three hormone markers: `amh_ng_ml`, `beta_hcg_i_miu_ml`, and `beta_hcg_ii_miu_ml`. The main goal is to see how useful these invasive markers look for explanation and later feature-drop tests.

## Research Positioning
This notebook does not do separate model training and does not treat the hormonal side dataset as a separate group of people. Instead, it checks whether these markers add a clear useful difference beyond the stronger routine and symptom signals already seen in the main clinical notebook.

### Cell 2 - Markdown
**What this cell is doing:** Setup and File Paths.
**Why this step matters:** This matters because it marks a new step in the workflow and keeps the notebook easy to follow.

**Notebook markdown content:**

## Setup and File Paths

### Cell 3 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from IPython.display import display

np.random.seed(42)

# ── Palette & Typography ──────────────────────────────────────────────────────
# Deep midnight background with warm coral / steel blue as accent pair
BG         = "#0d1117"   # near-black canvas
PANEL      = "#161b22"   # slightly lighter panel
GRID_CLR   = "#21262d"   # subtle gridline
TEXT_PRI   = "#e6edf3"   # primary text
TEXT_SEC   = "#8b949e"   # secondary/tick text
ACCENT_A   = "#58a6ff"   # steel blue  → PCOS Negative
ACCENT_B   = "#ff7b72"   # warm coral  → PCOS Positive
ACCENT_C   = "#3fb950"   # sage green  → neutral highlights
ACCENT_D   = "#d2a8ff"   # lavender    → AMH special

PCOS_LABEL_ORDER   = ["PCOS Negative", "PCOS Positive"]
PCOS_LABEL_PALETTE = {"PCOS Negative": ACCENT_A, "PCOS Positive": ACCENT_B}

# ── Global rcParams ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    # Figure
    "figure.facecolor":     BG,
    "figure.dpi":           130,
    "savefig.facecolor":    BG,
    "savefig.dpi":          300,
    # Axes
    "axes.facecolor":       PANEL,
    "axes.edgecolor":       GRID_CLR,
    "axes.labelcolor":      TEXT_PRI,
    "axes.titlecolor":      TEXT_PRI,
    "axes.titlesize":       13,
    "axes.titleweight":     "semibold",
    "axes.labelsize":       10,
    "axes.grid":            True,
    "axes.axisbelow":       True,
    # Grid
    "grid.color":           GRID_CLR,
    "grid.linewidth":       0.7,
    "grid.alpha":           1.0,
    # Ticks
    "xtick.color":          TEXT_SEC,
    "ytick.color":          TEXT_SEC,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    # Spine
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Legend
    "legend.fontsize":      9,
    "legend.facecolor":     PANEL,
    "legend.edgecolor":     GRID_CLR,
    "legend.labelcolor":    TEXT_PRI,
    "legend.framealpha":    1.0,
    # Lines
    "lines.linewidth":      1.8,
    # Font
    "font.family":          ["DejaVu Sans"],
})

pd.set_option("display.max_columns", None)

# ── Gradient colormaps ────────────────────────────────────────────────────────
CMAP_NEG = LinearSegmentedColormap.from_list("neg", ["#0d1117", ACCENT_A])
CMAP_POS = LinearSegmentedColormap.from_list("pos", ["#0d1117", ACCENT_B])
CMAP_DIV = LinearSegmentedColormap.from_list("div", [ACCENT_A, "#0d1117", ACCENT_B])

# ── Project paths ─────────────────────────────────────────────────────────────
def resolve_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, current.parent, current.parent.parent]:
        if (candidate / "cleaned_data").exists() and (candidate / "images").exists():
            return candidate
    return current

PROJECT_ROOT = resolve_project_root()
DATA_PATH    = PROJECT_ROOT / "cleaned_data" / "PCOS_infertility_cleaned.csv"
IMAGE_DIR    = PROJECT_ROOT / "images" / "eda" / "hormonal"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper utilities ──────────────────────────────────────────────────────────
def save_figure(fig: plt.Figure, slug: str) -> Path:
    out = IMAGE_DIR / slug
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return out


def style_ax(ax: plt.Axes, *, grid_axis: str = "y", zero_line: bool = False) -> None:
    """Apply consistent dark-theme polish to an axes."""
    ax.spines["left"].set_color(GRID_CLR)
    ax.spines["bottom"].set_color(GRID_CLR)
    ax.grid(True, axis=grid_axis, color=GRID_CLR, linewidth=0.7, alpha=1.0)
    ax.grid(False, axis=("x" if grid_axis == "y" else "y"))
    if zero_line:
        ax.axvline(0, color=TEXT_SEC, linewidth=1.0, linestyle="--", alpha=0.6)


def add_subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0, 1.02, text, transform=ax.transAxes,
        fontsize=8.5, color=TEXT_SEC, va="bottom",
    )


def grouped_numeric_summary(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    return (
        data.groupby("pcos_label")[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={"count": "n"})
        .round(3)
        .reset_index()
    )


def make_legend(ax: plt.Axes) -> None:
    patches = [
        mpatches.Patch(color=ACCENT_A, label="PCOS Negative"),
        mpatches.Patch(color=ACCENT_B, label="PCOS Positive"),
    ]
    ax.legend(handles=patches, framealpha=1.0, edgecolor=GRID_CLR)


print("✓ Environment ready — dark-theme palette loaded.")
```

### Cell 4 - Markdown
**What this cell is doing:** Load Data and Add Hormone Views.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

## Load Data and Add Hormone Views

### Cell 5 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
df = pd.read_csv(DATA_PATH)

required_columns = ["pcos_y_n", "beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml", "amh_ng_ml"]
missing_columns  = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["pcos_y_n"]       = df["pcos_y_n"].astype(int)
df["pcos_label"]     = df["pcos_y_n"].map({0: "PCOS Negative", 1: "PCOS Positive"})
df["log_beta_hcg_i"] = np.log10(df["beta_hcg_i_miu_ml"]  + 1)
df["log_beta_hcg_ii"]= np.log10(df["beta_hcg_ii_miu_ml"] + 1)
df["log_amh"]        = np.log10(df["amh_ng_ml"]          + 1)

display(df.head())
```

### Cell 6 - Markdown
**What this cell is doing:** Question 1 — Dataset size, schema, and target balance.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 1 — Dataset size, schema, and target balance

### Cell 7 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
overview_q01 = pd.DataFrame({
    "metric": ["row_count", "column_count", "missing_cells"],
    "value":  [df.shape[0], df.shape[1], int(df.isna().sum().sum())],
})
target_q01 = (
    df["pcos_label"].value_counts()
    .reindex(PCOS_LABEL_ORDER)
    .rename_axis("pcos_label")
    .reset_index(name="count")
)
target_q01["percentage"] = (100 * target_q01["count"] / target_q01["count"].sum()).round(1)
schema_q01 = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})

display(overview_q01)
display(schema_q01)
display(target_q01)
```

### Cell 8 - Code
**What this code is doing:** ── Q1 plot: class balance as a horizontal proportional bar ──────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q01_class_balance.png`

```python
# ── Q1 plot: class balance as a horizontal proportional bar ──────────────────
neg_pct = target_q01.loc[target_q01.pcos_label == "PCOS Negative", "percentage"].values[0]
pos_pct = target_q01.loc[target_q01.pcos_label == "PCOS Positive", "percentage"].values[0]
neg_n   = target_q01.loc[target_q01.pcos_label == "PCOS Negative", "count"].values[0]
pos_n   = target_q01.loc[target_q01.pcos_label == "PCOS Positive", "count"].values[0]

fig, ax = plt.subplots(figsize=(8, 1.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.barh([0], [neg_pct], color=ACCENT_A, height=0.55, label="PCOS Negative")
ax.barh([0], [pos_pct], left=[neg_pct], color=ACCENT_B, height=0.55, label="PCOS Positive")

ax.text(neg_pct / 2, 0, f"{neg_pct}%\n(n={neg_n})",
        ha="center", va="center", color=BG, fontsize=10, fontweight="bold")
ax.text(neg_pct + pos_pct / 2, 0, f"{pos_pct}%\n(n={pos_n})",
        ha="center", va="center", color=BG, fontsize=10, fontweight="bold")

ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(False)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.05), ncol=2,
          frameon=False, labelcolor=TEXT_PRI, fontsize=9)
ax.set_title("Class Balance — PCOS Sidecar Dataset (n=541)",
             color=TEXT_PRI, fontsize=12, fontweight="semibold", pad=24)

plt.tight_layout()
save_figure(fig, "q01_class_balance.png")
plt.show()
```

### Cell 9 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The proportional bar shows the same 364 versus 177 class split seen in the main clinical cohort, which is consistent with this file behaving like a hormonal side dataset rather than an independent outside dataset. The imbalance is moderate rather than extreme.

**Simple summary:** This hormonal file has the same target balance as the main clinical data, so it looks like a companion subset of the same cohort.

**Why this matters:** That positioning matters because this notebook should be used for invasive-feature interpretation and later feature-drop work, not for external validation claims.

### Cell 10 - Markdown
**What this cell is doing:** Question 2 — AMH distribution across PCOS groups.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 2 — AMH distribution across PCOS groups

### Cell 11 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
summary_q02 = grouped_numeric_summary(df, "amh_ng_ml")
display(summary_q02)
```

### Cell 12 - Code
**What this code is doing:** ── Q2 plot: AMH violin + strip on dark canvas ────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q02_amh_vs_pcos.png`

```python
# ── Q2 plot: AMH violin + strip on dark canvas ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.6, 1]})

# --- Left: violin + strip ---
ax = axes[0]
parts = ax.violinplot(
    [df.loc[df.pcos_label == g, "amh_ng_ml"].dropna() for g in PCOS_LABEL_ORDER],
    positions=[0, 1],
    widths=0.6,
    showmedians=False,
    showextrema=False,
)
colors = [ACCENT_A, ACCENT_B]
for pc, clr in zip(parts["bodies"], colors):
    pc.set_facecolor(clr)
    pc.set_alpha(0.35)
    pc.set_edgecolor(clr)
    pc.set_linewidth(1.5)

# median lines
for i, (grp, clr) in enumerate(zip(PCOS_LABEL_ORDER, colors)):
    med = df.loc[df.pcos_label == grp, "amh_ng_ml"].median()
    ax.hlines(med, i - 0.18, i + 0.18, color=clr, linewidth=2.5, zorder=5)

# strip
rng = np.random.default_rng(42)
for i, (grp, clr) in enumerate(zip(PCOS_LABEL_ORDER, colors)):
    vals = df.loc[df.pcos_label == grp, "amh_ng_ml"].dropna().values
    jitter = rng.uniform(-0.14, 0.14, len(vals))
    ax.scatter(i + jitter, vals, color=clr, alpha=0.18, s=7, zorder=3)

ax.set_xticks([0, 1])
ax.set_xticklabels(PCOS_LABEL_ORDER, color=TEXT_PRI, fontsize=10)
ax.set_ylabel("AMH (ng/mL)", color=TEXT_PRI)
ax.set_title("AMH Distribution by PCOS Status", pad=10)
add_subtitle(ax, "violin width = density  |  horizontal bar = median")
style_ax(ax)

# --- Right: median comparison bar ---
ax2 = axes[1]
medians = [df.loc[df.pcos_label == g, "amh_ng_ml"].median() for g in PCOS_LABEL_ORDER]
bars = ax2.barh(PCOS_LABEL_ORDER, medians, color=colors, height=0.45)
for bar, val in zip(bars, medians):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", color=TEXT_PRI, fontsize=9)
ax2.set_xlabel("Median AMH (ng/mL)", color=TEXT_PRI)
ax2.set_title("Median Comparison", pad=10)
ax2.tick_params(colors=TEXT_PRI)
style_ax(ax2, grid_axis="x")

fig.suptitle("Q2 — AMH Across PCOS Groups", fontsize=14, y=1.01,
             color=TEXT_PRI, fontweight="bold")
plt.tight_layout()
save_figure(fig, "q02_amh_vs_pcos.png")
plt.show()
```

### Cell 13 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The AMH summary table shows a higher central tendency in the PCOS-positive group, and the violin-plus-strip view shows that the upward shift is present through much of the distribution rather than only at a few extreme values. The positive group also carries a broader upper tail.

**Simple summary:** AMH is generally higher in the PCOS-positive group.

**Why this matters:** Among the invasive hormonal markers in this side dataset dataset, AMH is looking like the clearest candidate for later comparison or feature-drop testing.

### Cell 14 - Markdown
**What this cell is doing:** Question 3 — Beta-HCG I distribution across PCOS groups.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 3 — Beta-HCG I distribution across PCOS groups

### Cell 15 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
summary_q03 = grouped_numeric_summary(df, "beta_hcg_i_miu_ml")
display(summary_q03)
```

### Cell 16 - Code
**What this code is doing:** ── Q3 plot: log-scale histograms overlaid ────────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q03_beta_hcg_i_dist.png`

```python
# ── Q3 plot: log-scale histograms overlaid ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (feat, label, log) in zip(
    axes,
    [("beta_hcg_i_miu_ml", "Beta-HCG I (mIU/mL)  [raw]", False),
     ("log_beta_hcg_i",    "log₁₀(Beta-HCG I + 1)",      True)],
):
    for grp, clr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B]):
        vals = df.loc[df.pcos_label == grp, feat].dropna()
        ax.hist(vals, bins=30, color=clr, alpha=0.40, density=True,
                edgecolor="none", label=grp)
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde   = stats.gaussian_kde(vals)(kde_x)
        ax.plot(kde_x, kde, color=clr, linewidth=1.8)
    ax.set_xlabel(label, color=TEXT_PRI)
    ax.set_ylabel("Density", color=TEXT_PRI)
    style_ax(ax, grid_axis="y")
    make_legend(ax)

axes[0].set_title("Raw Scale — heavy right-skew", pad=8)
axes[1].set_title("Log-Transformed — clearer overlap", pad=8)

fig.suptitle("Q3 — Beta-HCG I Distribution", fontsize=14,
             color=TEXT_PRI, fontweight="bold", y=1.02)
plt.tight_layout()
save_figure(fig, "q03_beta_hcg_i_dist.png")
plt.show()
```

### Cell 17 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The raw Beta-HCG I histogram is strongly right-skewed, and the log-transformed panel still shows large overlap between the two classes. The group pattern is therefore unstable and much weaker than the AMH separation.

**Simple summary:** Beta-HCG I varies a lot, but it does not separate the groups clearly.

**Why this matters:** A laboratory feature is not automatically useful, and this marker already looks too noisy to assume downstream benefit.

### Cell 18 - Markdown
**What this cell is doing:** Question 4 — Beta-HCG II distribution across PCOS groups.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 4 — Beta-HCG II distribution across PCOS groups

### Cell 19 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
summary_q04 = grouped_numeric_summary(df, "beta_hcg_ii_miu_ml")
display(summary_q04)
```

### Cell 20 - Code
**What this code is doing:** ── Q4 plot: log-scale histograms for beta-HCG II ─────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q04_beta_hcg_ii_dist.png`

```python
# ── Q4 plot: log-scale histograms for beta-HCG II ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (feat, label) in zip(
    axes,
    [("beta_hcg_ii_miu_ml", "Beta-HCG II (mIU/mL)  [raw]"),
     ("log_beta_hcg_ii",    "log₁₀(Beta-HCG II + 1)")],
):
    for grp, clr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B]):
        vals = df.loc[df.pcos_label == grp, feat].dropna()
        ax.hist(vals, bins=30, color=clr, alpha=0.40, density=True,
                edgecolor="none", label=grp)
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde   = stats.gaussian_kde(vals)(kde_x)
        ax.plot(kde_x, kde, color=clr, linewidth=1.8)
    ax.set_xlabel(label, color=TEXT_PRI)
    ax.set_ylabel("Density", color=TEXT_PRI)
    style_ax(ax, grid_axis="y")
    make_legend(ax)

axes[0].set_title("Raw Scale", pad=8)
axes[1].set_title("Log-Transformed", pad=8)

fig.suptitle("Q4 — Beta-HCG II Distribution", fontsize=14,
             color=TEXT_PRI, fontweight="bold", y=1.02)
plt.tight_layout()
save_figure(fig, "q04_beta_hcg_ii_dist.png")
plt.show()
```

### Cell 21 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** Beta-HCG II shows the same heavy skew and broad overlap pattern seen for Beta-HCG I, even after log transformation. Any shift by PCOS status is small relative to the amount of shared density.

**Simple summary:** Beta-HCG II behaves a lot like Beta-HCG I and still does not give a clean PCOS split.

**Why this matters:** This increases the likelihood that the two beta-HCG assays are mostly redundant rather than independently informative.

### Cell 22 - Markdown
**What this cell is doing:** Section Synthesis.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Section Synthesis
**Cross-chart reading:** The first four questions are already separating the hormonal story into one clearer marker and two weak ones: AMH shows a coherent upward shift, while both beta-HCG assays remain skewed and overlapping.

**How the findings relate:** The matching class balance with the main clinical cohort supports the view that this notebook is checking a side dataset subset, not a new population, so the interpretation should focus on marker usefulness rather than external generalisation.

**Practical interpretation:** At this stage, AMH looks worth carrying forward, whereas beta-HCG should be treated cautiously until stronger evidence appears.

### Cell 23 - Markdown
**What this cell is doing:** Question 5 — Pairwise correlation among hormonal markers.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 5 — Pairwise correlation among hormonal markers

### Cell 24 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
hormones     = ["amh_ng_ml", "beta_hcg_i_miu_ml", "beta_hcg_ii_miu_ml"]
corr_q05     = df[hormones].corr(method="spearman")
display(corr_q05.round(3))
```

### Cell 25 - Code
**What this code is doing:** ── Q5 plot: annotated heatmap with custom dark diverging colormap ────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q05_hormonal_correlation_heatmap.png`

```python
# ── Q5 plot: annotated heatmap with custom dark diverging colormap ────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

mask = np.triu(np.ones_like(corr_q05, dtype=bool), k=1)
im   = ax.imshow(corr_q05.values, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="auto")

labels = ["AMH", "β-hCG I", "β-hCG II"]
n      = len(labels)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, color=TEXT_PRI, fontsize=10)
ax.set_yticklabels(labels, color=TEXT_PRI, fontsize=10)

for i in range(n):
    for j in range(n):
        val  = corr_q05.values[i, j]
        txt_clr = BG if abs(val) > 0.5 else TEXT_PRI
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                color=txt_clr, fontsize=11, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=TEXT_SEC, labelcolor=TEXT_SEC)
cbar.set_label("Spearman ρ", color=TEXT_SEC, fontsize=9)

ax.grid(False)
ax.set_title("Q5 — Spearman Pairwise Correlation\nHormonal Markers",
             color=TEXT_PRI, fontsize=12, fontweight="semibold")
plt.tight_layout()
save_figure(fig, "q05_hormonal_correlation_heatmap.png")
plt.show()
```

### Cell 26 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The correlation heatmap shows that Beta-HCG I and Beta-HCG II move almost as the same signal, while AMH is comparatively independent of both. The relative intensity therefore points to redundancy between the two beta-HCG measurements rather than complementary information.

**Simple summary:** The two beta-HCG columns are telling almost the same story, but AMH is behaving differently.

**Why this matters:** Highly redundant markers can inflate model complexity without adding much new diagnostic value.

### Cell 27 - Markdown
**What this cell is doing:** Question 6 — Do beta-HCG I and II carry redundant information?.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 6 — Do beta-HCG I and II carry redundant information?

### Cell 28 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
rho_q06, p_q06 = stats.spearmanr(
    df["beta_hcg_i_miu_ml"].dropna(),
    df["beta_hcg_ii_miu_ml"].dropna(),
)
print(f"Spearman ρ (β-hCG I vs II): {rho_q06:.4f}   p = {p_q06:.2e}")
```

### Cell 29 - Code
**What this code is doing:** ── Q6 plot: scatter with 2D density overlay ──────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q06_hcg_redundancy_scatter.png`

```python
# ── Q6 plot: scatter with 2D density overlay ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))

for grp, clr, mkr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B], ["o", "^"]):
    sub = df.loc[df.pcos_label == grp, ["log_beta_hcg_i", "log_beta_hcg_ii"]].dropna()
    ax.scatter(sub["log_beta_hcg_i"], sub["log_beta_hcg_ii"],
               color=clr, alpha=0.28, s=14, marker=mkr, label=grp)

# identity line
lims = [ax.get_xlim(), ax.get_ylim()]
mn, mx = min(l[0] for l in lims), max(l[1] for l in lims)
ax.plot([mn, mx], [mn, mx], color=ACCENT_C, linewidth=1.4,
        linestyle="--", alpha=0.7, label="y = x (identity)")

ax.set_xlabel("log₁₀(Beta-HCG I + 1)", color=TEXT_PRI)
ax.set_ylabel("log₁₀(Beta-HCG II + 1)", color=TEXT_PRI)
ax.set_title(f"Q6 — Beta-HCG I vs II  (ρ = {rho_q06:.3f})",
             color=TEXT_PRI, fontsize=12, fontweight="semibold")
style_ax(ax, grid_axis="both")
ax.legend(framealpha=1.0, edgecolor=GRID_CLR)

plt.tight_layout()
save_figure(fig, "q06_hcg_redundancy_scatter.png")
plt.show()
```

### Cell 30 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The scatter cloud hugs the identity direction, and the tight clustering pattern confirms that the two beta-HCG assays rise and fall together. Class colours overlap heavily, so the relationship is much stronger between the assays than between either assay and PCOS status.

**Simple summary:** The two beta-HCG measures are closely matched to each other, not clearly separated by PCOS.

**Why this matters:** Keeping both markers in a later model would likely duplicate information rather than improve discrimination.

### Cell 31 - Markdown
**What this cell is doing:** Question 7 — AMH log-scale distribution and normality.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 7 — AMH log-scale distribution and normality

### Cell 32 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
summary_q07 = grouped_numeric_summary(df, "log_amh")
display(summary_q07)

for grp in PCOS_LABEL_ORDER:
    vals = df.loc[df.pcos_label == grp, "log_amh"].dropna()
    stat, p = stats.shapiro(vals[:200])  # Shapiro cap at 200
    print(f"{grp:16s} → Shapiro W={stat:.4f}  p={p:.4f}")
```

### Cell 33 - Code
**What this code is doing:** ── Q7 plot: log-AMH KDE + Q-Q panels side by side ───────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q07_log_amh_normality.png`

```python
# ── Q7 plot: log-AMH KDE + Q-Q panels side by side ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# KDE
ax = axes[0]
for grp, clr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B]):
    vals  = df.loc[df.pcos_label == grp, "log_amh"].dropna()
    xr    = np.linspace(vals.min() - 0.1, vals.max() + 0.1, 400)
    kde   = stats.gaussian_kde(vals)(xr)
    ax.fill_between(xr, kde, alpha=0.22, color=clr)
    ax.plot(xr, kde, color=clr, linewidth=2, label=grp)
    ax.axvline(vals.median(), color=clr, linewidth=1.2,
               linestyle="--", alpha=0.8)

ax.set_xlabel("log₁₀(AMH + 1)", color=TEXT_PRI)
ax.set_ylabel("Density", color=TEXT_PRI)
ax.set_title("Log-AMH KDE by Group", pad=8)
add_subtitle(ax, "dashed lines = group medians")
style_ax(ax, grid_axis="y")
make_legend(ax)

# Q-Q
ax2 = axes[1]
for grp, clr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B]):
    vals = df.loc[df.pcos_label == grp, "log_amh"].dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm", fit=True)
    ax2.scatter(osm, osr, color=clr, alpha=0.35, s=8, label=grp)
    fit_x = np.array([min(osm), max(osm)])
    ax2.plot(fit_x, slope * fit_x + intercept, color=clr, linewidth=1.6)

ax2.set_xlabel("Theoretical Quantiles", color=TEXT_PRI)
ax2.set_ylabel("Sample Quantiles", color=TEXT_PRI)
ax2.set_title("Q-Q Plot (Normal Reference)", pad=8)
style_ax(ax2, grid_axis="both")
ax2.legend(framealpha=1.0, edgecolor=GRID_CLR)

fig.suptitle("Q7 — log-AMH Normality Assessment", fontsize=14,
             color=TEXT_PRI, fontweight="bold", y=1.02)
plt.tight_layout()
save_figure(fig, "q07_log_amh_normality.png")
plt.show()
```

### Cell 34 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** Log transformation makes the AMH distribution more symmetric and easier to compare, but the KDE and Q-Q panels still show tail deviations from perfect normality. That means transformation helps interpretation without fully justifying strict based on a fixed shape assumptions.

**Simple summary:** Logging AMH makes the pattern cleaner, but the data still are not perfectly normal.

**Why this matters:** Later statistical testing and modelling should continue to use robust or non-based on a fixed shape thinking rather than assuming textbook Gaussian behaviour.

### Cell 35 - Markdown
**What this cell is doing:** Question 8 — Mann-Whitney U test for each hormonal marker.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 8 — Mann-Whitney U test for each hormonal marker

### Cell 36 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
results_q08 = []
for feat in ["amh_ng_ml", "log_amh", "beta_hcg_i_miu_ml", "log_beta_hcg_i",
             "beta_hcg_ii_miu_ml", "log_beta_hcg_ii"]:
    neg = df.loc[df.pcos_label == "PCOS Negative", feat].dropna()
    pos = df.loc[df.pcos_label == "PCOS Positive",  feat].dropna()
    u, p = stats.mannwhitneyu(neg, pos, alternative="two-sided")
    n1, n2 = len(neg), len(pos)
    auc    = u / (n1 * n2)
    results_q08.append({"feature": feat, "U": u, "p_value": p, "AUC_U": auc})

mw_df = pd.DataFrame(results_q08).round({"U": 0, "p_value": 5, "AUC_U": 4})
display(mw_df)
```

### Cell 37 - Code
**What this code is doing:** ── Q8 plot: Mann-Whitney AUC(U) bar chart ────────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q08_mannwhitney_auc.png`

```python
# ── Q8 plot: Mann-Whitney AUC(U) bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.0))

sorted_mw = mw_df.sort_values("AUC_U", ascending=True)
auc_vals  = sorted_mw["AUC_U"].values

bar_colors = [ACCENT_B if v > 0.5 else ACCENT_A for v in auc_vals]
bars = ax.barh(sorted_mw["feature"], auc_vals, color=bar_colors, height=0.55)

ax.axvline(0.5, color=ACCENT_C, linewidth=1.3, linestyle="--", alpha=0.8,
           label="AUC = 0.5 (no effect)")

for bar, val in zip(bars, auc_vals):
    ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=TEXT_PRI, fontsize=8.5)

ax.set_xlabel("Mann-Whitney AUC(U)", color=TEXT_PRI)
ax.set_title("Q8 — Hormonal Marker Discriminative Power",
             color=TEXT_PRI, fontsize=12, fontweight="semibold")
ax.tick_params(colors=TEXT_PRI)
ax.legend(framealpha=1.0, edgecolor=GRID_CLR)
style_ax(ax, grid_axis="x")

plt.tight_layout()
save_figure(fig, "q08_mannwhitney_auc.png")
plt.show()
```

### Cell 38 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The effect-size chart places AMH clearly above the beta-HCG markers, while the beta-HCG values sit much closer to the no-separation region. This is strong visual evidence that AMH carries the main good at showing a clear difference signal in the side dataset dataset.

**Simple summary:** AMH separates the groups much better than either beta-HCG measurement.

**Why this matters:** If only one invasive hormonal marker is worth looking at, AMH is the most defensible choice from this notebook.

### Cell 39 - Markdown
**What this cell is doing:** Section Synthesis.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Section Synthesis
**Cross-chart reading:** The redundancy, scatter, distribution-shape, and effect-size views are all telling the same story: AMH is distinct, beta-HCG is duplicated and weak, and non-based on a fixed shape handling is appropriate.

**How the findings relate:** This convergence matters because the conclusion is not coming from just one chart type or one statistic.

**Practical interpretation:** The invasive comparison set should stay small and focused, with AMH leading and beta-HCG retained only as a cautious secondary exploratory signal.

### Cell 40 - Markdown
**What this cell is doing:** Question 9 — AMH boxen plot: within-group spread and outlier structure.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 9 — AMH boxen plot: within-group spread and outlier structure

### Cell 41 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
summary_q09 = grouped_numeric_summary(df, "amh_ng_ml")
display(summary_q09)
```

### Cell 42 - Code
**What this code is doing:** ── Q9 plot: enhanced boxenplot ───────────────────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q09_amh_boxen.png`

```python
# ── Q9 plot: enhanced boxenplot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.0))

sns.boxenplot(
    data=df, x="pcos_label", y="amh_ng_ml",
    order=PCOS_LABEL_ORDER,
    palette=PCOS_LABEL_PALETTE,
    linewidth=0.8,
    ax=ax,
)

# Overlay mean markers
for i, (grp, clr) in enumerate(zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B])):
    mean_val = df.loc[df.pcos_label == grp, "amh_ng_ml"].mean()
    ax.scatter(i, mean_val, color="white", s=55, zorder=6,
               edgecolors=clr, linewidth=1.5)

ax.set_xlabel("", color=TEXT_PRI)
ax.set_ylabel("AMH (ng/mL)", color=TEXT_PRI)
ax.set_title("Q9 — AMH Within-Group Spread\n(letter-value plot + mean ○)",
             color=TEXT_PRI, fontsize=12, fontweight="semibold")
ax.tick_params(colors=TEXT_PRI)
style_ax(ax, grid_axis="y")

plt.tight_layout()
save_figure(fig, "q09_amh_boxen.png")
plt.show()
```

### Cell 43 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The boxen plot shows that the PCOS-positive group extends further into the upper AMH quantiles, not just at the centre but deeper into the tail structure. This suggests mixed pattern within the positive group, including a subset with particularly elevated AMH.

**Simple summary:** Some PCOS-positive patients have especially high AMH values, which stretches the upper end of the distribution.

**Why this matters:** That upper-tail structure may be clinically useful and may explain why AMH keeps emerging as the most informative invasive marker.

### Cell 44 - Markdown
**What this cell is doing:** Question 10 — Scatter: AMH vs beta-HCG I coloured by PCOS status.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 10 — Scatter: AMH vs beta-HCG I coloured by PCOS status

### Cell 45 - Code
**What this code is doing:** ── Q10 plot: 2D scatter AMH vs log-β-hCG I ──────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q10_amh_vs_hcg_scatter.png`

```python
# ── Q10 plot: 2D scatter AMH vs log-β-hCG I ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (x, y, xlabel, ylabel) in zip(
    axes,
    [("log_amh", "log_beta_hcg_i",
      "log₁₀(AMH + 1)", "log₁₀(β-hCG I + 1)"),
     ("log_amh", "log_beta_hcg_ii",
      "log₁₀(AMH + 1)", "log₁₀(β-hCG II + 1)")],
):
    for grp, clr, mkr in zip(PCOS_LABEL_ORDER, [ACCENT_A, ACCENT_B], ["o", "^"]):
        sub = df.loc[df.pcos_label == grp, [x, y]].dropna()
        ax.scatter(sub[x], sub[y], color=clr, alpha=0.25,
                   s=12, marker=mkr, label=grp)
    ax.set_xlabel(xlabel, color=TEXT_PRI)
    ax.set_ylabel(ylabel, color=TEXT_PRI)
    style_ax(ax, grid_axis="both")
    ax.legend(framealpha=1.0, edgecolor=GRID_CLR)

axes[0].set_title("AMH vs β-hCG I", pad=8)
axes[1].set_title("AMH vs β-hCG II", pad=8)

fig.suptitle("Q10 — AMH vs Beta-HCG Scatter by PCOS Status",
             fontsize=14, color=TEXT_PRI, fontweight="bold", y=1.02)
plt.tight_layout()
save_figure(fig, "q10_amh_vs_hcg_scatter.png")
plt.show()
```

### Cell 46 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The bivariate scatter shows that higher AMH values in PCOS-positive participants are appearing across a wide range of beta-HCG values. In other words, the AMH signal is not simply a by-product of beta-HCG behaviour.

**Simple summary:** AMH still looks useful even when beta-HCG changes.

**Why this matters:** This strengthens the case that AMH adds genuinely different information rather than repeating what the beta-HCG columns already capture.

### Cell 47 - Markdown
**What this cell is doing:** Question 11 — Spearman correlation of each marker with PCOS status.
**Why this step matters:** This matters because it explains the logic behind the next part of the notebook.

**Notebook markdown content:**

---
## Question 11 — Spearman correlation of each marker with PCOS status

### Cell 48 - Code
**What this code is doing:** This code cell is running the step defined by the section around it.
**Why this step matters:** This matters because it carries out the main step described in this part of the notebook.

```python
target_binary = df["pcos_y_n"]
markers_q11   = ["amh_ng_ml", "log_amh",
                 "beta_hcg_i_miu_ml", "log_beta_hcg_i",
                 "beta_hcg_ii_miu_ml", "log_beta_hcg_ii"]

rows = []
for feat in markers_q11:
    r, p = stats.spearmanr(df[feat].dropna(), target_binary[df[feat].notna()])
    rows.append({"feature": feat, "spearman_r": round(r, 4), "p_value": round(p, 6)})

summary_q11 = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
display(summary_q11)
```

### Cell 49 - Code
**What this code is doing:** ── Q11 plot: Spearman ρ lollipop chart ───────────────────────────────────────
**Why this step matters:** This matters because the chart helps compare groups and makes the pattern easier to see.
**Saved figure or image output in this cell:** `q11_hormonal_target_correlation.png`

```python
# ── Q11 plot: Spearman ρ lollipop chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.0))

sorted_q11 = summary_q11.sort_values("spearman_r", ascending=True)
y_pos      = range(len(sorted_q11))
bar_colors = [ACCENT_B if v > 0 else ACCENT_A for v in sorted_q11["spearman_r"]]

ax.hlines(y_pos, 0, sorted_q11["spearman_r"], color=bar_colors, linewidth=2.2)
ax.scatter(sorted_q11["spearman_r"], y_pos, color=bar_colors, s=60, zorder=5)

ax.axvline(0, color=TEXT_SEC, linewidth=1.0, linestyle="--", alpha=0.6)

for i, (_, row) in enumerate(sorted_q11.iterrows()):
    offset = 0.005 if row.spearman_r >= 0 else -0.005
    ha     = "left" if row.spearman_r >= 0 else "right"
    ax.text(row.spearman_r + offset, i, f"{row.spearman_r:.3f}",
            va="center", ha=ha, color=TEXT_PRI, fontsize=8.5)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(sorted_q11["feature"].tolist(), color=TEXT_PRI, fontsize=9)
ax.set_xlabel("Spearman ρ with PCOS Status", color=TEXT_PRI)
ax.set_title("Q11 — Hormonal Marker Correlation with Target",
             color=TEXT_PRI, fontsize=12, fontweight="semibold")
style_ax(ax, grid_axis="x", zero_line=False)

plt.tight_layout()
save_figure(fig, "q11_hormonal_target_correlation.png")
plt.show()
```

### Cell 50 - Markdown
**What this cell is doing:** Insight.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Insight
**Scientific interpretation:** The lollipop chart ranks AMH highest for monotonic association with PCOS status, while both beta-HCG variables cluster near zero. The consistency between raw and log AMH also suggests that the direction of signal is stable.

**Simple summary:** AMH aligns with the PCOS label much better than the beta-HCG markers do.

**Why this matters:** Target-aligned ranking is one of the clearest ways to decide which invasive feature deserves to survive into later feature selection.

### Cell 51 - Markdown
**What this cell is doing:** Section Synthesis.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

### Section Synthesis
**Cross-chart reading:** Across the distribution, redundancy, effect-size, and target-correlation views, AMH is repeatedly emerging as the only hormonal marker with stable upward separation in the PCOS-positive group.

**How the findings relate:** The beta-HCG assays are behaving more like a redundant pair of weak laboratory measurements than like robust PCOS discriminators, which is exactly why this side dataset should be used for interpretation rather than blindly adding every lab feature to a model.

**Practical interpretation:** If an invasive feature-drop study is run later, AMH should be the lead hormonal candidate, while beta-HCG should remain optional and strongly performance-justified.

### Cell 52 - Markdown
**What this cell is doing:** Summary Dashboard — All Three Markers Side by Side.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

---
## Summary Dashboard — All Three Markers Side by Side

### Cell 53 - Code
**What this code is doing:** ── Summary: 3-panel violin grid for all markers ─────────────────────────────
**Why this step matters:** This matters because it shows the exact table or check behind the next decision or figure.
**Saved figure or image output in this cell:** `summary_all_markers.png`

```python
# ── Summary: 3-panel violin grid for all markers ─────────────────────────────
marker_configs = [
    ("amh_ng_ml",          "AMH (ng/mL)",        ACCENT_D),
    ("log_beta_hcg_i",     "log₁₀(β-hCG I + 1)", ACCENT_A),
    ("log_beta_hcg_ii",    "log₁₀(β-hCG II + 1)",ACCENT_B),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

for ax, (feat, ylabel, accent) in zip(axes, marker_configs):
    data_neg = df.loc[df.pcos_label == "PCOS Negative", feat].dropna()
    data_pos = df.loc[df.pcos_label == "PCOS Positive",  feat].dropna()

    parts = ax.violinplot(
        [data_neg, data_pos], positions=[0, 1],
        widths=0.65, showmedians=False, showextrema=False,
    )
    vcolors = [ACCENT_A, ACCENT_B]
    for pc, clr in zip(parts["bodies"], vcolors):
        pc.set_facecolor(clr); pc.set_alpha(0.35)
        pc.set_edgecolor(clr); pc.set_linewidth(1.5)

    # Median lines
    for i, (vals, clr) in enumerate(zip([data_neg, data_pos], vcolors)):
        ax.hlines(vals.median(), i - 0.16, i + 0.16, color=clr, linewidth=2.5, zorder=5)

    # Jittered dots
    rng = np.random.default_rng(42)
    for i, (vals, clr) in enumerate(zip([data_neg, data_pos], vcolors)):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=clr, alpha=0.18, s=6, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"], color=TEXT_PRI, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_PRI, fontsize=9)
    ax.set_title(feat.replace("_", " ").upper(), pad=8,
                 color=accent, fontsize=10, fontweight="bold")
    style_ax(ax, grid_axis="y")

fig.suptitle("Hormonal Marker Overview — PCOS Negative vs Positive",
             fontsize=14, color=TEXT_PRI, fontweight="bold", y=1.02)
plt.tight_layout()
save_figure(fig, "summary_all_markers.png")
plt.show()
```

### Cell 54 - Markdown
**What this cell is doing:** Invasive-Feature Usefulness Summary.
**Why this step matters:** This matters because it turns the output into a result that can be explained clearly.

**Notebook markdown content:**

---
## Invasive-Feature Usefulness Summary

**Observational summary:** This hormonal side dataset is showing that AMH carries the clearest invasive separation by PCOS status. The beta-HCG markers are highly skewed, overlap heavily between groups, and are strongly redundant with one another, so they do not behave like two independent sources of diagnostic value.

**Simple wrap-up:** In plain language, AMH looks useful, while beta-HCG looks noisy and repetitive.

**Feature usefulness interpretation:**
**Stronger invasive feature:** `amh_ng_ml`
**Use cautiously:** `beta_hcg_i_miu_ml`, `beta_hcg_ii_miu_ml`
**Possible engineering option:** one beta-HCG representative summary only if later evaluation shows added value
**Study positioning:** This file should be treated as a hormonal side dataset of the main clinical cohort, not as an external validation dataset or an independent modelling cohort.


## Results and Findings
### Questions and Results
| Question asked | Saved figure | Main result in simple words |
| --- | --- | --- |
| Question 1 — Dataset size, schema, and target balance | `q01_class_balance.png` | **Scientific interpretation:** The proportional bar shows the same 364 versus 177 class split seen in the main clinical cohort, which is consistent with this file behaving like a hormonal side dataset rather than an independent outside dataset. The imbalance is moderate rather than extreme. **Simple summary:** This hormonal file has the same target balance as the main clinical data, so it looks like a companion subset of the same cohort. **Why this matters:** That positioning matters because this notebook should be used for invasive-feature interpretation and later feature-drop work, not for external validation claims. |
| Question 2 — AMH distribution across PCOS groups | `q02_amh_vs_pcos.png` | **Scientific interpretation:** The AMH summary table shows a higher central tendency in the PCOS-positive group, and the violin-plus-strip view shows that the upward shift is present through much of the distribution rather than only at a few extreme values. The positive group also carries a broader upper tail. **Simple summary:** AMH is generally higher in the PCOS-positive group. **Why this matters:** Among the invasive hormonal markers in this side dataset dataset, AMH is looking like the clearest candidate for later comparison or feature-drop testing. |
| Question 3 — Beta-HCG I distribution across PCOS groups | `q03_beta_hcg_i_dist.png` | **Scientific interpretation:** The raw Beta-HCG I histogram is strongly right-skewed, and the log-transformed panel still shows large overlap between the two classes. The group pattern is therefore unstable and much weaker than the AMH separation. **Simple summary:** Beta-HCG I varies a lot, but it does not separate the groups clearly. **Why this matters:** A laboratory feature is not automatically useful, and this marker already looks too noisy to assume downstream benefit. |
| Question 4 — Beta-HCG II distribution across PCOS groups | `q04_beta_hcg_ii_dist.png` | **Scientific interpretation:** Beta-HCG II shows the same heavy skew and broad overlap pattern seen for Beta-HCG I, even after log transformation. Any shift by PCOS status is small relative to the amount of shared density. **Simple summary:** Beta-HCG II behaves a lot like Beta-HCG I and still does not give a clean PCOS split. **Why this matters:** This increases the likelihood that the two beta-HCG assays are mostly redundant rather than independently informative. |
| Question 5 — Pairwise correlation among hormonal markers | `q05_hormonal_correlation_heatmap.png` | **Scientific interpretation:** The correlation heatmap shows that Beta-HCG I and Beta-HCG II move almost as the same signal, while AMH is comparatively independent of both. The relative intensity therefore points to redundancy between the two beta-HCG measurements rather than complementary information. **Simple summary:** The two beta-HCG columns are telling almost the same story, but AMH is behaving differently. **Why this matters:** Highly redundant markers can inflate model complexity without adding much new diagnostic value. |
| Question 6 — Do beta-HCG I and II carry redundant information? | `q06_hcg_redundancy_scatter.png` | **Scientific interpretation:** The scatter cloud hugs the identity direction, and the tight clustering pattern confirms that the two beta-HCG assays rise and fall together. Class colours overlap heavily, so the relationship is much stronger between the assays than between either assay and PCOS status. **Simple summary:** The two beta-HCG measures are closely matched to each other, not clearly separated by PCOS. **Why this matters:** Keeping both markers in a later model would likely duplicate information rather than improve discrimination. |
| Question 7 — AMH log-scale distribution and normality | `q07_log_amh_normality.png` | **Scientific interpretation:** Log transformation makes the AMH distribution more symmetric and easier to compare, but the KDE and Q-Q panels still show tail deviations from perfect normality. That means transformation helps interpretation without fully justifying strict based on a fixed shape assumptions. **Simple summary:** Logging AMH makes the pattern cleaner, but the data still are not perfectly normal. **Why this matters:** Later statistical testing and modelling should continue to use robust or non-based on a fixed shape thinking rather than assuming textbook Gaussian behaviour. |
| Question 8 — Mann-Whitney U test for each hormonal marker | `q08_mannwhitney_auc.png` | **Scientific interpretation:** The effect-size chart places AMH clearly above the beta-HCG markers, while the beta-HCG values sit much closer to the no-separation region. This is strong visual evidence that AMH carries the main good at showing a clear difference signal in the side dataset dataset. **Simple summary:** AMH separates the groups much better than either beta-HCG measurement. **Why this matters:** If only one invasive hormonal marker is worth looking at, AMH is the most defensible choice from this notebook. |
| Question 9 — AMH boxen plot: within-group spread and outlier structure | `q09_amh_boxen.png` | **Scientific interpretation:** The boxen plot shows that the PCOS-positive group extends further into the upper AMH quantiles, not just at the centre but deeper into the tail structure. This suggests mixed pattern within the positive group, including a subset with particularly elevated AMH. **Simple summary:** Some PCOS-positive patients have especially high AMH values, which stretches the upper end of the distribution. **Why this matters:** That upper-tail structure may be clinically useful and may explain why AMH keeps emerging as the most informative invasive marker. |
| Question 10 — Scatter: AMH vs beta-HCG I coloured by PCOS status | `q10_amh_vs_hcg_scatter.png` | **Scientific interpretation:** The bivariate scatter shows that higher AMH values in PCOS-positive participants are appearing across a wide range of beta-HCG values. In other words, the AMH signal is not simply a by-product of beta-HCG behaviour. **Simple summary:** AMH still looks useful even when beta-HCG changes. **Why this matters:** This strengthens the case that AMH adds genuinely different information rather than repeating what the beta-HCG columns already capture. |
| Question 11 — Spearman correlation of each marker with PCOS status | `q11_hormonal_target_correlation.png` | **Scientific interpretation:** The lollipop chart ranks AMH highest for monotonic association with PCOS status, while both beta-HCG variables cluster near zero. The consistency between raw and log AMH also suggests that the direction of signal is stable. **Simple summary:** AMH aligns with the PCOS label much better than the beta-HCG markers do. **Why this matters:** Target-aligned ranking is one of the clearest ways to decide which invasive feature deserves to survive into later feature selection. |


## Important Cautions
- This is still the same underlying PCOS cohort sidecar, not a new outside dataset.
- AMH is expected to be more informative than beta-HCG, but the notebook keeps beta-HCG in view and judges it carefully.
- The notebook supports interpretation and later ablation thinking rather than standalone training.

## How This Notebook Connects to the Next Notebook
The next EDA notebook studies the survey dataset to see whether the non-invasive pattern still appears in self-reported data.
