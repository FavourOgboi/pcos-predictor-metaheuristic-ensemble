import streamlit as st

from app.config.app_config import APP_META


def _nonempty_lines(*values: str) -> str:
    return "<br>\n                ".join(value for value in values if value)


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFE;
            color: #1A2B4A;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A2342 0%, #1B4F8A 60%, #2176AE 100%);
            color: white;
        }
        [data-testid="stSidebar"] * { color: white !important; }

        [data-testid="stSidebar"] [data-testid^="stBaseButton"] button {
            background: #FFFFFF !important;
            color: #0A2342 !important;
            border: 1px solid #D9E6F3 !important;
            box-shadow: 0 2px 10px rgba(10,35,66,0.08);
        }
        [data-testid="stSidebar"] [data-testid^="stBaseButton"] button * {
            color: #0A2342 !important;
        }
        [data-testid="stSidebar"] [data-testid^="stBaseButton"] button:hover {
            background: #F3F7FB !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] button {
            border-left: 4px solid #2176AE !important;
        }

        .page-header {
            background: linear-gradient(135deg, #0A2342 0%, #1B4F8A 50%, #2176AE 100%);
            padding: 2.5rem 2rem 2rem 2rem;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
            color: white;
        }
        .page-header h1 {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            margin: 0;
            color: white;
        }
        .page-header p {
            font-size: 1rem;
            opacity: 0.85;
            margin: 0.4rem 0 0 0;
            color: #BDD5EA;
        }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.4rem 1.2rem;
            box-shadow: 0 2px 12px rgba(10,35,66,0.08);
            border-left: 4px solid #2176AE;
            margin-bottom: 1rem;
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0A2342;
            line-height: 1;
        }
        .metric-card .metric-label {
            font-size: 0.8rem;
            color: #6B7C93;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.3rem;
        }
        .metric-card .metric-delta {
            font-size: 0.85rem;
            color: #2A9D8F;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        .risk-high {
            background: linear-gradient(135deg, #E63946, #C1121F);
            color: white;
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            box-shadow: 0 4px 20px rgba(230,57,70,0.35);
        }
        .risk-low {
            background: linear-gradient(135deg, #2A9D8F, #1B7268);
            color: white;
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            box-shadow: 0 4px 20px rgba(42,157,143,0.35);
        }
        .risk-medium {
            background: linear-gradient(135deg, #E9C46A, #C9A227);
            color: #1A2B4A;
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            box-shadow: 0 4px 20px rgba(233,196,106,0.35);
        }

        .section-card {
            background: white;
            border-radius: 14px;
            padding: 1.8rem;
            box-shadow: 0 2px 16px rgba(10,35,66,0.07);
            margin-bottom: 1.5rem;
        }

        .info-banner {
            background: linear-gradient(135deg, #EBF4FF, #DBEAFE);
            border-left: 4px solid #2176AE;
            padding: 1rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
            color: #0A2342;
            font-size: 0.92rem;
        }

        .warning-banner {
            background: linear-gradient(135deg, #FFF7ED, #FEF3C7);
            border-left: 4px solid #E9C46A;
            padding: 1rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
            color: #78350F;
            font-size: 0.92rem;
        }

        .nav-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #BDD5EA;
            margin: 1.2rem 0 0.3rem 0;
            padding-left: 0.5rem;
        }

        .placeholder-image {
            background: linear-gradient(135deg, #EBF4FF, #D9EAF8);
            border: 2px dashed #8FB6D8;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            color: #40678C;
        }

        .nav-card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(10,35,66,0.08);
            padding: 1.2rem 1.1rem;
            min-height: 190px;
            border-top: 4px solid #2176AE;
        }

        .timeline-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .timeline-step {
            flex: 1 1 170px;
            background: #F8FAFE;
            border: 1px solid #D9E6F3;
            border-radius: 14px;
            padding: 1rem;
            position: relative;
        }

        .timeline-step::before {
            content: '';
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #2176AE;
            position: absolute;
            top: -7px;
            left: 18px;
            box-shadow: 0 0 0 6px rgba(33,118,174,0.12);
        }

        .risk-reason {
            background: linear-gradient(135deg, #F8FBFF, #EDF5FC);
            border: 1px solid #D5E4F3;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }

        .app-footer {
            text-align: center;
            color: #6B7C93;
            font-size: 0.78rem;
            padding: 2rem 0 1rem 0;
            border-top: 1px solid #E2EBF6;
            margin-top: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    def nav_button(route_key: str, label: str, icon: str, current_route: str) -> str:
        if st.button(
            f"{icon} {label}",
            key=f"nav_{route_key}",
            use_container_width=True,
            type="primary" if current_route == route_key else "secondary",
        ):
            return route_key
        return current_route

    identity_lines = _nonempty_lines(
        APP_META["researcher"],
        APP_META["degree"],
        APP_META["institution"],
    )
    current_route = st.session_state.get("route", "home")
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
                <div style='font-size:2.5rem;'>🔬</div>
                <div style='font-family: Playfair Display, serif; font-size:1.1rem; font-weight:700; color:white; line-height:1.3; margin-top:0.4rem;'>
                    PCOS Detection<br>System
                </div>
                <div style='font-size:0.72rem; color:#BDD5EA; margin-top:0.3rem; letter-spacing:0.06em;'>
                    MSc Research · 2026
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
        current_route = nav_button("home", "Home", "🏠", current_route)
        st.markdown("<div class='nav-label'>Screening Tools</div>", unsafe_allow_html=True)
        current_route = nav_button("noninvasive", "Non-Invasive Screening", "🟢", current_route)
        current_route = nav_button("invasive", "Clinical Benchmark", "🔵", current_route)
        st.markdown("<div class='nav-label'>Research Insights</div>", unsafe_allow_html=True)
        current_route = nav_button("insights", "PCOS Insights Dashboard", "📊", current_route)
        current_route = nav_button("heart", "Heart & PCOS Study", "❤️", current_route)
        st.markdown("<div class='nav-label'>Model Science</div>", unsafe_allow_html=True)
        current_route = nav_button("performance", "Model Performance", "📈", current_route)
        st.markdown("<div class='nav-label'>Information</div>", unsafe_allow_html=True)
        current_route = nav_button("recommendations", "Recommendations", "💡", current_route)
        current_route = nav_button("disclaimer", "Disclaimer", "⚠️", current_route)
        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='font-size:0.7rem; color:#BDD5EA; padding:0.5rem; text-align:center; line-height:1.6;'>
                {identity_lines}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.session_state["route"] = current_route
    return current_route


def render_footer() -> None:
    st.markdown(
        f"""
        <div class='app-footer'>
            {APP_META['title']} · MSc Research {APP_META['year']} ·
            {APP_META['researcher']} ·
            <em>For research and educational purposes only — not a substitute for clinical diagnosis</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str = "", border_color: str = "#2176AE") -> None:
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class='metric-card' style='border-left-color:{border_color};'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_banner(text: str) -> None:
    st.markdown(f"<div class='info-banner'>{text}</div>", unsafe_allow_html=True)


def render_warning_banner(text: str) -> None:
    st.markdown(f"<div class='warning-banner'>{text}</div>", unsafe_allow_html=True)


def render_section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class='section-card'>
            <h3 style='margin-top:0; color:#0A2342;'>{title}</h3>
            <div style='color:#364A63; line-height:1.8;'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
