import streamlit as st

print("[DEBUG] app_minimal.py started", flush=True)

st.set_page_config(page_title="Test", page_icon="🔬")

print("[DEBUG] page config set", flush=True)

st.title("PCOS System - Test")
st.write("If you see this, the app is working!")

print("[DEBUG] rendered content", flush=True)
