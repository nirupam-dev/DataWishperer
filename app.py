"""
DataWhisper AI — Main Streamlit Application Entry Point.

Two-page architecture:
    Page 1: Home / Landing Page (no sidebar)
    Page 2: Analytics Workspace (sidebar + full functionality)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

# ── CRITICAL: Set BLAS thread limits BEFORE any numpy/pandas import ──────────
# OpenBLAS reads these at library load time. On Streamlit Cloud (1GB RAM),
# multi-threaded BLAS can exhaust memory and crash with:
#   "OpenBLAS error: Memory allocation still failed after 10 retries"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("GOTOBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_MAIN_FREE", "1")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DataWhisper — AI Data Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# DataWhisper AI v1.0.0\n"
            "Natural Language to Data Analytics Assistant.\n\n"
            "Powered by Groq (primary) + Ollama (fallback)."
        ),
    },
)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from frontend.state import init_state
from frontend.theme import inject_custom_css
from frontend.components.home import render_home
from frontend.components.workspace import render_workspace


def main() -> None:
    """Application entry point with two-page routing."""
    # 1. Initialise session state & services (once)
    init_state()

    # 2. Determine current page
    if "_dw_page" not in st.session_state:
        st.session_state["_dw_page"] = "home"

    page = st.session_state["_dw_page"]

    # 3. Inject CSS for the current page
    inject_custom_css(page=page)

    # 4. Render the appropriate page
    if page == "workspace":
        render_workspace()
    else:
        render_home()


if __name__ == "__main__":
    main()
