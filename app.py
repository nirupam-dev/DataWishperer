"""
DataWhisperer — Main Streamlit Application Entry Point.

This is the Streamlit entry point that assembles the sidebar, chat, explorer,
and export components into a cohesive single-page application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DataWhisperer — AI Data Analyst",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# DataWhisperer v1.0.0\n"
            "Talk to your CSV with AI.\n\n"
            "Powered by Ollama + LangChain."
        ),
    },
)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from frontend.state import init_state, has_dataset
from frontend.theme import inject_custom_css
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat
from frontend.components.explorer import render_explorer
from frontend.components.export import render_export


def main() -> None:
    """Application entry point."""
    # 1. Initialise session state & services (once)
    init_state()

    # 2. Inject premium CSS
    inject_custom_css()

    # 3. Render Sidebar
    render_sidebar()

    # 4. Main Content Area
    if has_dataset():
        # Show tabs when a dataset is loaded
        chat_tab, explore_tab, export_tab = st.tabs(
            ["💬 Chat", "🔍 Explore", "📥 Export"]
        )

        with chat_tab:
            render_chat()

        with explore_tab:
            render_explorer()

        with export_tab:
            render_export()
    else:
        _render_landing()


def _render_landing() -> None:
    """Render the welcome landing page when no dataset is loaded."""
    st.markdown("")  # spacing

    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="animate-in" style="text-align: center; padding: 2rem 0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            # 🔮 DataWhisperer

            ### Talk to Your Data with AI

            Upload a CSV file and start asking questions in plain English.
            DataWhisperer generates Python code, executes it securely,
            and returns tables, charts, and insights — all powered by
            a local AI model.
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Feature cards
    cols = st.columns(3)

    with cols[0]:
        st.markdown(
            '<div class="metric-card animate-in">'
            '<p class="metric-value" style="font-size: 2rem;">💬</p>'
            '<p class="metric-label">Natural Language</p>'
            '<p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.5rem;">'
            "Ask questions in plain English — no SQL or Python needed."
            "</p></div>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            '<div class="metric-card animate-in">'
            '<p class="metric-value" style="font-size: 2rem;">🔒</p>'
            '<p class="metric-label">Secure Execution</p>'
            '<p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.5rem;">'
            "Generated code runs in a sandboxed subprocess with AST validation."
            "</p></div>",
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            '<div class="metric-card animate-in">'
            '<p class="metric-value" style="font-size: 2rem;">📊</p>'
            '<p class="metric-label">Smart Visualisation</p>'
            '<p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.5rem;">'
            "Automatic chart selection, dark-themed rendering, and explanations."
            "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div style="text-align: center; color: #6B7280; padding: 1rem;">'
        "👈 Upload a CSV file from the sidebar to get started."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
