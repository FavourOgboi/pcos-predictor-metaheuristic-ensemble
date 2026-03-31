#!/usr/bin/env python
"""Quick diagnostic script to test app imports."""
import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Testing imports...")

try:
    print("1. Importing streamlit...", end=" ")
    import streamlit as st
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("2. Importing styling...", end=" ")
    from app.components.styling import apply_global_styles, render_footer, render_sidebar
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("3. Importing views...", end=" ")
    from app.views import home, noninvasive_screening, invasive_benchmark, insights_dashboard
    from app.views import heart_pcos_study, model_performance, recommendations
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("4. Importing utils...", end=" ")
    from app.components.utils import load_dataframe
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

print("\n✓ All imports successful!\n")

# Try to load a dataframe
try:
    print("5. Testing load_dataframe...", end=" ")
    df = load_dataframe("final_results")
    print(f"✓ Loaded {len(df)} rows")
except Exception as e:
    print(f"⚠ {e}")

print("\nDiagnostic complete!")
