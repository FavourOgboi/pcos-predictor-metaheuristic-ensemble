import streamlit as st

from components.home_content import render_home_page
from components.styling import apply_global_styles, configure_page, render_footer, render_sidebar


configure_page("PCOS Early Detection System", "🔬")
apply_global_styles()
render_sidebar()
render_home_page()
render_footer()
