"""
Streamlit theming and CSS injection.

Provides a premium dark-mode design using custom CSS injected via
st.markdown. Streamlit's native theme is set via .streamlit/config.toml;
this module adds additional polish that Streamlit cannot express natively.
"""

from __future__ import annotations

import streamlit as st

# ── Colour Palette ──────────────────────────────────────────────────────────

PRIMARY = "#6C63FF"       # Vibrant indigo-violet
PRIMARY_LIGHT = "#8B83FF"
ACCENT = "#00D9FF"        # Cyan accent
ACCENT_WARM = "#FF6B6B"   # Coral-red for warnings
SUCCESS = "#00E676"       # Bright green
WARNING = "#FFB300"       # Amber
BG_DARK = "#0E1117"
BG_CARD = "#1A1F2E"
BG_HOVER = "#252B3B"
TEXT_PRIMARY = "#FAFAFA"
TEXT_SECONDARY = "#9CA3AF"
BORDER = "#2D3748"


def inject_custom_css() -> None:
    """Inject premium CSS overrides into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Global ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Smoother scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4A5568; }

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0E1117 100%);
    border-right: 1px solid #1F2937;
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, #6C63FF, #00D9FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.5rem;
}

/* ── Cards ───────────────────────────────────────────────────── */
div.metric-card {
    background: linear-gradient(145deg, #1A1F2E, #161B26);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
}
div.metric-card .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #FAFAFA;
    margin: 0;
}
div.metric-card .metric-label {
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── Chat Bubbles ────────────────────────────────────────────── */
div.chat-user {
    background: linear-gradient(135deg, #6C63FF22, #6C63FF11);
    border-left: 3px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}
div.chat-assistant {
    background: linear-gradient(135deg, #1A1F2E, #161B26);
    border-left: 3px solid #00D9FF;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}

/* ── Code Blocks ─────────────────────────────────────────────── */
div.code-block {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.85rem;
    overflow-x: auto;
}

/* ── Status Badges ───────────────────────────────────────────── */
span.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
span.badge-success { background: #00E67622; color: #00E676; border: 1px solid #00E67644; }
span.badge-warning { background: #FFB30022; color: #FFB300; border: 1px solid #FFB30044; }
span.badge-error { background: #FF6B6B22; color: #FF6B6B; border: 1px solid #FF6B6B44; }
span.badge-info { background: #00D9FF22; color: #00D9FF; border: 1px solid #00D9FF44; }

/* ── Tabs ────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Section Headers ─────────────────────────────────────────── */
div.section-header {
    border-bottom: 2px solid #6C63FF;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
div.section-header h3 {
    color: #FAFAFA;
    font-weight: 600;
    margin: 0;
}

/* ── Empty state ─────────────────────────────────────────────── */
div.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #9CA3AF;
}
div.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
div.empty-state h4 {
    color: #E5E7EB;
    margin-bottom: 0.5rem;
}

/* ── Explanation box ─────────────────────────────────────────── */
div.explanation-box {
    background: linear-gradient(135deg, #00D9FF08, #6C63FF08);
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}

/* ── Animation ───────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
div.animate-in {
    animation: fadeIn 0.4s ease forwards;
}

/* ── Streamlit Button Override ────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #8B83FF) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4) !important;
}
</style>
"""
