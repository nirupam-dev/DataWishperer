"""
Streamlit theming and CSS injection — Cinematic aerospace design system.

Provides a dark navy glassmorphism theme with translucent surfaces,
subtle backdrop blur, restrained blue/purple accents, and professional typography.
"""

from __future__ import annotations

import base64
import streamlit as st
from pathlib import Path


def _get_bg_base64() -> str:
    """Load and encode the background image."""
    bg_path = Path(__file__).parent / "assets" / "space_bg.png"
    if bg_path.exists():
        return base64.b64encode(bg_path.read_bytes()).decode()
    return ""


def inject_custom_css(page: str = "home") -> None:
    """Inject the cinematic aerospace CSS. page='home' or 'workspace'."""
    bg_b64 = _get_bg_base64()
    bg_rule = ""
    if bg_b64:
        bg_rule = f"""
        .stApp {{
            background: url("data:image/png;base64,{bg_b64}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        """

    sidebar_css = ""
    if page == "home":
        sidebar_css = """
        section[data-testid="stSidebar"] { display: none !important; }
        button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        div[data-testid="collapsedControl"] { display: none !important; }
        """

    st.markdown(f"<style>{_BASE_CSS}\n{bg_rule}\n{sidebar_css}</style>", unsafe_allow_html=True)


_BASE_CSS = """
/* ── Global ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

/* Remove default Streamlit backgrounds */
.stApp > header { background: transparent !important; }
.main .block-container {
    background: transparent !important;
    padding-top: 1rem !important;
    max-width: 1400px !important;
}
div[data-testid="stAppViewBlockContainer"] { background: transparent !important; }
div[data-testid="stVerticalBlock"] { background: transparent !important; }
div[data-testid="stHorizontalBlock"] { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(100,116,139,0.4); border-radius: 3px; }

/* ── Typography ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    letter-spacing: 0.5px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 28, 0.92) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

/* ── Glass Panels ────────────────────────────────────────────── */
div.glass-panel {
    background: rgba(10, 15, 30, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 1.25rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
div.glass-panel-dense {
    background: rgba(10, 15, 30, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 1rem;
    backdrop-filter: blur(12px);
}

/* ── Metric Cards (workspace) ────────────────────────────────── */
div.metric-card {
    background: rgba(15, 20, 40, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}
div.metric-card:hover { border-color: rgba(99, 102, 241, 0.35); }
div.metric-card .metric-icon { font-size: 1.2rem; margin-bottom: 0.35rem; color: #818cf8; }
div.metric-card .metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0; }
div.metric-card .metric-label {
    font-size: 0.65rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 0.2rem;
}

/* ── Feature Cards (home) ────────────────────────────────────── */
div.feature-card {
    background: rgba(15, 20, 40, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: left;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s, transform 0.3s;
    height: 100%;
}
div.feature-card:hover { border-color: rgba(99, 102, 241, 0.3); transform: translateY(-2px); }
div.feature-card .fc-icon {
    width: 40px; height: 40px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; margin-bottom: 0.75rem;
    background: rgba(99, 102, 241, 0.12); color: #818cf8;
}
div.feature-card .fc-title { font-size: 0.9rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.4rem; }
div.feature-card .fc-desc { font-size: 0.78rem; color: #94a3b8; line-height: 1.5; }

/* ── Navbar (home) ───────────────────────────────────────────── */
div.dw-navbar {
    background: rgba(8, 12, 28, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 2rem;
}
div.dw-navbar .nav-brand { display: flex; align-items: center; gap: 0.75rem; }
div.dw-navbar .nav-brand .brand-icon {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: white;
}
div.dw-navbar .nav-brand .brand-text {
    font-weight: 700; font-size: 0.85rem; color: #f1f5f9; letter-spacing: 1.2px;
}
div.dw-navbar .nav-brand .brand-sub {
    font-size: 0.55rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;
}
div.dw-navbar .nav-links { display: flex; gap: 2rem; }
div.dw-navbar .nav-links a {
    color: #94a3b8; text-decoration: none; font-size: 0.8rem; font-weight: 500;
    transition: color 0.2s;
}
div.dw-navbar .nav-links a:hover { color: #f1f5f9; }
div.dw-navbar .nav-links a.active { color: #f1f5f9; font-weight: 600; }

/* ── CTA Button ──────────────────────────────────────────────── */
a.cta-btn, div.cta-btn {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    color: white !important; font-weight: 600; font-size: 0.8rem;
    padding: 0.55rem 1.25rem; border-radius: 8px;
    text-decoration: none; transition: opacity 0.2s; cursor: pointer;
    letter-spacing: 0.5px; border: none;
}
a.cta-btn:hover, div.cta-btn:hover { opacity: 0.9; }

/* ── Hero ────────────────────────────────────────────────────── */
div.hero-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px; padding: 0.35rem 1rem;
    font-size: 0.7rem; color: #a5b4fc; font-weight: 500;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 1.5rem;
}
span.gradient-text {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Streamlit Component Overrides ───────────────────────────── */
div.stButton > button {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #e2e8f0 !important; font-weight: 500 !important;
    border-radius: 8px !important; font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important; transition: all 0.2s !important;
    text-transform: none !important; letter-spacing: 0.3px !important;
    min-height: auto !important;
}
div.stButton > button:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
}

/* Primary CTA button style */
div.stButton > button[kind="primary"], div.cta-button > button {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
}

div.stDownloadButton > button {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-size: 0.75rem !important; min-height: auto !important;
}

/* Text Input */
div[data-testid="stTextInput"] input, div.stTextInput input {
    background: rgba(15, 20, 40, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stChatInput"] {
    background: rgba(15, 20, 40, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 10px !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}
div[data-testid="stFileUploader"] section {
    background: rgba(15, 20, 40, 0.5) !important;
    border: 1px dashed rgba(99, 102, 241, 0.25) !important;
    border-radius: 8px !important;
}
div[data-testid="stFileUploader"] button {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 6px !important; color: #e2e8f0 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background: rgba(10, 15, 30, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 8px !important;
}

/* Code blocks */
div[data-testid="stCode"], pre {
    background: rgba(10, 15, 30, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 8px !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: rgba(10, 15, 30, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 8px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.8rem !important; font-weight: 500 !important;
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f1f5f9 !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    background: rgba(15, 20, 40, 0.6) !important;
    border-radius: 8px !important;
}

/* Spinner */
div.stSpinner > div { color: #818cf8 !important; }

/* Caption */
div.stCaption, .stCaption p { color: #64748b !important; }

/* Divider */
hr { border-color: rgba(99, 102, 241, 0.1) !important; }

/* Metric */
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

/* ── Status Badges ───────────────────────────────────────────── */
span.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
}
span.status-online { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.4); }
span.status-standby { background: #f59e0b; }
span.status-offline { background: #ef4444; }

/* ── Workspace Preview (home hero) ───────────────────────────── */
div.workspace-preview {
    background: rgba(10, 15, 30, 0.85);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px; padding: 1.25rem;
    transform: perspective(800px) rotateY(-8deg) rotateX(3deg);
    box-shadow: 0 25px 60px rgba(0,0,0,0.5), 0 0 30px rgba(99,102,241,0.08);
    transition: transform 0.4s;
    backdrop-filter: blur(8px);
}
div.workspace-preview:hover {
    transform: perspective(800px) rotateY(-4deg) rotateX(1deg);
}

/* ── Result Panels (workspace) ───────────────────────────────── */
div.result-panel {
    background: rgba(10, 15, 30, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px; padding: 1rem;
    backdrop-filter: blur(8px); height: 100%;
}
div.result-panel .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
div.result-panel .panel-title {
    font-size: 0.75rem; font-weight: 600; color: #f1f5f9;
    text-transform: uppercase; letter-spacing: 0.8px;
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── Suggested Question Buttons ──────────────────────────────── */
div.suggest-btn button {
    background: rgba(15, 20, 40, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 8px !important; color: #cbd5e1 !important;
    font-size: 0.78rem !important; font-weight: 400 !important;
    text-transform: none !important; letter-spacing: 0 !important;
    padding: 0.6rem 1rem !important; white-space: normal !important;
    text-align: center !important; min-height: auto !important;
}
div.suggest-btn button:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* ── Scroll indicator ────────────────────────────────────────── */
@keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(6px)} }
div.scroll-indicator {
    text-align: center; color: #64748b; font-size: 0.75rem; padding: 1.5rem 0;
}
div.scroll-indicator .chevron { animation: bounce 2s infinite; display: inline-block; }

/* Animation */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(12px); }
    to { opacity:1; transform:translateY(0); }
}
div.animate-in { animation: fadeInUp 0.5s ease forwards; }
"""
