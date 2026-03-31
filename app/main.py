from pathlib import Path
import sys
from importlib import import_module
import traceback

import streamlit as st


try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from app.components.styling import apply_global_styles, configure_page, render_footer, render_sidebar
except Exception as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.error(traceback.format_exc())
    st.stop()


try:
    configure_page("PCOS Early Detection System", "🔬")
except Exception as e:
    st.error(f"❌ Failed to configure page: {e}")
    st.error(traceback.format_exc())
    st.stop()

try:
    apply_global_styles()
except Exception as e:
    st.error(f"❌ Failed to apply global styles: {e}")
    st.error(traceback.format_exc())
    st.stop()


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


try:
    route = render_sidebar()
except Exception as e:
    st.error(f"Failed to render sidebar: {e}")
    st.error(traceback.format_exc())
    route = "home"

try:
    render_view(route)
except Exception as exc:
    st.error(f"Unable to render the selected view: {exc}")
    st.exception(exc)

try:
    render_footer()
except Exception as e:
    st.error(f"Failed to render footer: {e}")
    st.error(traceback.format_exc())
