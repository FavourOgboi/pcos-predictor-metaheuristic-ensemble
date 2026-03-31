from pathlib import Path
import sys
from importlib import import_module

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.styling import apply_global_styles, configure_page, render_footer, render_sidebar


configure_page("PCOS Early Detection System", "🔬")
apply_global_styles()

VIEW_MODULES = {
    "home": "app.views.home",
    "noninvasive": "app.views.noninvasive_screening",
    "invasive": "app.views.invasive_benchmark",
    "insights": "app.views.insights_dashboard",
    "heart": "app.views.heart_pcos_study",
    "performance": "app.views.model_performance",
    "recommendations": "app.views.recommendations",
    "disclaimer": "app.views.disclaimer",
}


def render_view(route: str) -> None:
    module_name = VIEW_MODULES.get(route, VIEW_MODULES["home"])
    module = import_module(module_name)
    module.render()


route = render_sidebar()
try:
    render_view(route)
except Exception as exc:
    st.error(f"Unable to render the selected view: {exc}")
    st.exception(exc)

render_footer()
