import streamlit as st

from app.components.home_content import render_home_page
from app.components.styling import apply_global_styles, configure_page, render_footer, render_sidebar


configure_page("Home", "🏠")
apply_global_styles()
render_sidebar()
render_home_page()
st.info("This page mirrors the main home view so the app keeps a clean landing page and a page-based navigation entry.")
render_footer()
