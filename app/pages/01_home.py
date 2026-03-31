from pathlib import Path
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.home_content import render_home_page
from app.components.styling import apply_global_styles, configure_page, render_footer, render_sidebar


configure_page("Home", "🏠")
apply_global_styles()
render_sidebar()
render_home_page()
st.info("This page mirrors the main home view so the app keeps a clean landing page and a page-based navigation entry.")
render_footer()
