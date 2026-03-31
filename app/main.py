import streamlit as st

from app.components.home_content import render_home_page
from app.components.styling import apply_global_styles, configure_page, render_footer, render_sidebar


configure_page("PCOS Early Detection System", "🔬")
apply_global_styles()
render_sidebar()
render_home_page()
render_footer()
