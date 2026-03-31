from pathlib import Path
import sys
from importlib import import_module
import traceback

# Debug: Print when app starts
print("[PCOS DEBUG] main.py starting", flush=True)

import streamlit as st

# Debug: Print after streamlit import
print("[PCOS DEBUG] streamlit imported", flush=True)

# Configure page FIRST - this MUST be the first Streamlit command
try:
    st.set_page_config(
        page_title="PCOS Early Detection System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    print("[PCOS DEBUG] page config set", flush=True)
except Exception as e:
    print(f"[PCOS DEBUG] Failed to set page config: {e}", flush=True)
    raise

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    print("[PCOS DEBUG] About to import styling", flush=True)
    from app.components.styling import apply_global_styles, render_footer, render_sidebar
    print("[PCOS DEBUG] Styling imported successfully", flush=True)
except Exception as e:
    print(f"[PCOS DEBUG] Failed to import required modules: {e}", flush=True)
    st.error(f"❌ Failed to import required modules: {e}")
    st.error(traceback.format_exc())
    st.stop()

try:
    print("[PCOS DEBUG] About to apply global styles", flush=True)
    apply_global_styles()
    print("[PCOS DEBUG] Global styles applied successfully", flush=True)
except Exception as e:
    print(f"[PCOS DEBUG] Failed to apply global styles: {e}", flush=True)
    st.error(f"❌ Failed to apply global styles: {e}")
    st.error(traceback.format_exc())
    st.stop()

print("[PCOS DEBUG] Setting up VIEW_MODULES", flush=True)

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
    print("[PCOS DEBUG] About to render sidebar", flush=True)
    route = render_sidebar()
    print(f"[PCOS DEBUG] Sidebar rendered, route={route}", flush=True)
except Exception as e:
    print(f"[PCOS DEBUG] Error in render_sidebar: {e}", flush=True)
    st.error(f"Failed to render sidebar: {e}")
    st.error(traceback.format_exc())
    route = "home"

try:
    print(f"[PCOS DEBUG] About to render view for route: {route}", flush=True)
    render_view(route)
    print("[PCOS DEBUG] View rendered successfully", flush=True)
except Exception as exc:
    print(f"[PCOS DEBUG] Error in render_view: {exc}", flush=True)
    st.error(f"Unable to render the selected view: {exc}")
    st.exception(exc)

try:
    print("[PCOS DEBUG] About to render footer", flush=True)
    render_footer()
    print("[PCOS DEBUG] Footer rendered successfully", flush=True)
except Exception as e:
    print(f"[PCOS DEBUG] Error in render_footer: {e}", flush=True)
    st.error(f"Failed to render footer: {e}")
    st.error(traceback.format_exc())

print("[PCOS DEBUG] main.py completed successfully", flush=True)
