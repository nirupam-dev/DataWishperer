"""
DataWhisper AI — Main Streamlit Application Entry Point.

Two-page architecture:
    Page 1: Home / Landing Page (no sidebar)
    Page 2: Analytics Workspace (sidebar + full functionality)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

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
