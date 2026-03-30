#!/usr/bin/env bash
cd "$(dirname "$0")"
streamlit run main.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base light \
    --theme.primaryColor "#2176AE" \
    --theme.backgroundColor "#F8FAFE" \
    --theme.secondaryBackgroundColor "#EBF4FF" \
    --theme.textColor "#1A2B4A" \
    --theme.font "sans serif"
