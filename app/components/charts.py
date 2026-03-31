import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.config.app_config import COLOR_SCHEME


def apply_plotly_style(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        title_font={"size": 20, "color": COLOR_SCHEME["navy"]},
        paper_bgcolor=COLOR_SCHEME["background"],
        plot_bgcolor="white",
        font={"family": "Inter, sans-serif", "color": COLOR_SCHEME["text"]},
        margin=dict(l=40, r=40, t=70, b=40),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5EEF8")
    fig.update_yaxes(showgrid=True, gridcolor="#E5EEF8")
    return fig


def class_distribution_pie(df: pd.DataFrame, target_col: str = "pcos_y_n") -> go.Figure:
    counts = (
        df[target_col].map({0: "No PCOS", 1: "PCOS"}).value_counts().rename_axis("status").reset_index(name="count")
    )
    fig = px.pie(
        counts,
        names="status",
        values="count",
        color="status",
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
        hole=0.45,
    )
    return apply_plotly_style(fig, "PCOS Class Distribution")


def age_histogram(df: pd.DataFrame, age_col: str = "age_yrs", target_col: str = "pcos_y_n") -> go.Figure:
    plot_df = df.copy()
    plot_df["PCOS status"] = plot_df[target_col].map({0: "No PCOS", 1: "PCOS"})
    fig = px.histogram(
        plot_df,
        x=age_col,
        color="PCOS status",
        nbins=25,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    return apply_plotly_style(fig, "Age Distribution by PCOS Status")


def bmi_boxplot(df: pd.DataFrame, target_col: str = "pcos_y_n") -> go.Figure:
    plot_df = df.copy()
    plot_df["PCOS status"] = plot_df[target_col].map({0: "No PCOS", 1: "PCOS"})
    fig = px.box(
        plot_df,
        x="PCOS status",
        y="bmi",
        color="PCOS status",
        points="outliers",
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    return apply_plotly_style(fig, "BMI by PCOS Status")


def symptom_prevalence_chart(df: pd.DataFrame, symptom_cols: list, target_col: str = "pcos_y_n") -> go.Figure:
    rows = []
    for col in symptom_cols:
        grouped = df.groupby(target_col)[col].mean().mul(100)
        rows.append(
            {
                "feature": col.replace("_y_n", "").replace("_", " ").title(),
                "No PCOS": grouped.get(0, 0.0),
                "PCOS": grouped.get(1, 0.0),
            }
        )
    melted = pd.DataFrame(rows).melt(id_vars="feature", var_name="PCOS status", value_name="prevalence")
    fig = px.bar(
        melted,
        x="feature",
        y="prevalence",
        color="PCOS status",
        barmode="group",
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    fig.update_xaxes(tickangle=-30)
    return apply_plotly_style(fig, "Symptom Prevalence by PCOS Status")


def bp_scatter(df: pd.DataFrame, target_col: str = "pcos_y_n") -> go.Figure:
    plot_df = df.copy()
    plot_df["PCOS status"] = plot_df[target_col].map({0: "No PCOS", 1: "PCOS"})
    fig = px.scatter(
        plot_df,
        x="systolic_bp_mmhg",
        y="diastolic_bp_mmhg",
        color="PCOS status",
        opacity=0.75,
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    fig.add_vline(x=130, line_dash="dash", line_color=COLOR_SCHEME["accent_soft"])
    fig.add_hline(y=80, line_dash="dash", line_color=COLOR_SCHEME["accent_soft"])
    return apply_plotly_style(fig, "Blood Pressure Pattern by PCOS Status")


def cycle_violin(df: pd.DataFrame, target_col: str = "pcos_y_n") -> go.Figure:
    plot_df = df.copy()
    plot_df["PCOS status"] = plot_df[target_col].map({0: "No PCOS", 1: "PCOS"})
    fig = px.violin(
        plot_df,
        x="PCOS status",
        y="cycle_length_days",
        color="PCOS status",
        box=True,
        points="all",
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    fig.add_hline(y=35, line_dash="dash", line_color=COLOR_SCHEME["accent_soft"])
    return apply_plotly_style(fig, "Cycle Length by PCOS Status")


def correlation_heatmap(df: pd.DataFrame, feature_cols: list) -> go.Figure:
    corr = df[feature_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=["#EAF4FE", "#2176AE", "#0A2342"],
        aspect="auto",
    )
    return apply_plotly_style(fig, "Non-Invasive Feature Correlation Heatmap")


def selected_feature_histogram(df: pd.DataFrame, feature: str, target_col: str = "pcos_y_n") -> go.Figure:
    plot_df = df.copy()
    plot_df["PCOS status"] = plot_df[target_col].map({0: "No PCOS", 1: "PCOS"})
    fig = px.histogram(
        plot_df,
        x=feature,
        color="PCOS status",
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"No PCOS": COLOR_SCHEME["primary"], "PCOS": COLOR_SCHEME["accent"]},
    )
    return apply_plotly_style(fig, f"{feature} Distribution by PCOS Status")


def overlapping_histogram(series_map: dict, title: str, x_label: str, threshold_lines: list | None = None) -> go.Figure:
    fig = go.Figure()
    for label, config in series_map.items():
        fig.add_trace(
            go.Histogram(
                x=config["values"],
                name=label,
                opacity=0.6,
                marker_color=config["color"],
                nbinsx=config.get("nbins", 25),
            )
        )
    if threshold_lines:
        for x, label in threshold_lines:
            fig.add_vline(x=x, line_dash="dash", line_color=COLOR_SCHEME["accent_soft"], annotation_text=label)
    fig.update_layout(barmode="overlay", xaxis_title=x_label, yaxis_title="Count")
    return apply_plotly_style(fig, title)


def metric_heatmap(df: pd.DataFrame, row_key: str, title: str) -> go.Figure:
    heat_df = df.set_index(row_key)
    fig = px.imshow(
        heat_df,
        text_auto=".3f",
        color_continuous_scale=["#EAF4FE", "#2A9D8F", "#0A2342"],
        aspect="auto",
    )
    return apply_plotly_style(fig, title)
