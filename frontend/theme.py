"""
Streamlit theming and CSS injection — Cinematic aerospace design system.

Provides a dark navy glassmorphism theme with translucent surfaces,
subtle backdrop blur, restrained blue/purple accents, and professional typography.

Two background images:
  - home_bg.png   → Home / Landing page (rocket launch, Earth horizon)
  - workspace_bg.png → Workspace / other pages (trajectory arc)
"""

from __future__ import annotations

import base64
import streamlit as st
from pathlib import Path


def _load_b64(filename: str) -> str:
    """Load and base64-encode an asset file."""
    path = Path(__file__).parent / "assets" / filename
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


def inject_custom_css(page: str = "home") -> None:
    """Inject the cinematic aerospace CSS. page='home' or 'workspace'."""

    # Choose background based on page
    if page == "home":
        bg_b64 = _load_b64("home_bg.png")
    else:
        bg_b64 = _load_b64("starts-bg.png")

    bg_rule = ""
    if bg_b64:
        bg_rule = f"""
        .stApp {{
            background: url("data:image/png;base64,{bg_b64}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        """

    sidebar_css = ""
    sidebar_js = ""
    if page == "home":
        sidebar_css = """
        section[data-testid="stSidebar"] { display: none !important; }
        button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        div[data-testid="collapsedControl"] { display: none !important; }
        .stMainBlockContainer { max-width: 1340px !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
        """
    else:
        sidebar_css = """
        section[data-testid="stSidebar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
            transform: none !important;
            margin-left: 0 !important;
            left: 0 !important;
            position: relative !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 260px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: flex !important;
            visibility: visible !important;
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
            margin-left: 0 !important;
            transform: none !important;
        }
        button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        div[data-testid="collapsedControl"] { display: none !important; }
        button[data-testid="stSidebarCollapseButton"] { display: none !important; }
        /* Hide any collapse/expand toggle at the top of the sidebar */
        section[data-testid="stSidebar"] button[kind="header"] { display: none !important; }
        section[data-testid="stSidebar"] > div > div:first-child > button { display: none !important; }
        div[data-testid="stSidebarHeader"] button { display: none !important; }
        div[data-testid="stSidebarHeader"] { display: none !important; }
        .stMainBlockContainer { max-width: 1200px !important; }
        """
        # JavaScript to force-expand sidebar by clicking the collapse control
        # AND by directly setting aria-expanded attribute
        sidebar_js = """
        <script>
        (function() {
            function forceSidebarOpen() {
                // Method 1: Set aria-expanded attribute directly
                var sidebar = document.querySelector('section[data-testid="stSidebar"]');
                if (sidebar) {
                    sidebar.setAttribute('aria-expanded', 'true');
                    sidebar.style.display = 'flex';
                    sidebar.style.visibility = 'visible';
                    sidebar.style.width = '260px';
                    sidebar.style.minWidth = '260px';
                    sidebar.style.maxWidth = '260px';
                    sidebar.style.transform = 'none';
                    sidebar.style.marginLeft = '0';
                    sidebar.style.opacity = '1';
                    sidebar.style.position = 'relative';
                    sidebar.style.left = '0';
                }

                // Method 2: Click the expand button if it exists
                var expandBtn = document.querySelector('button[data-testid="stSidebarCollapsedControl"]');
                if (expandBtn && sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
                    expandBtn.click();
                }

                // Method 3: Also handle the collapsed control div
                var collapsedCtrl = document.querySelector('div[data-testid="collapsedControl"]');
                if (collapsedCtrl) {
                    var btn = collapsedCtrl.querySelector('button');
                    if (btn && sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
                        btn.click();
                    }
                }
            }

            // Run immediately
            forceSidebarOpen();

            // Run after a short delay (Streamlit DOM may not be ready)
            setTimeout(forceSidebarOpen, 100);
            setTimeout(forceSidebarOpen, 300);
            setTimeout(forceSidebarOpen, 600);
            setTimeout(forceSidebarOpen, 1200);

            // Use MutationObserver to keep sidebar open if Streamlit tries to collapse it
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(m) {
                    if (m.attributeName === 'aria-expanded') {
                        var el = m.target;
                        if (el.getAttribute('aria-expanded') === 'false') {
                            el.setAttribute('aria-expanded', 'true');
                            el.style.display = 'flex';
                            el.style.visibility = 'visible';
                            el.style.width = '260px';
                            el.style.minWidth = '260px';
                            el.style.transform = 'none';
                            el.style.marginLeft = '0';
                        }
                    }
                });
            });
            var sb = document.querySelector('section[data-testid="stSidebar"]');
            if (sb) {
                observer.observe(sb, { attributes: true, attributeFilter: ['aria-expanded'] });
            } else {
                // If sidebar not yet in DOM, watch for it
                var bodyObs = new MutationObserver(function(mutations, obs) {
                    var s = document.querySelector('section[data-testid="stSidebar"]');
                    if (s) {
                        forceSidebarOpen();
                        observer.observe(s, { attributes: true, attributeFilter: ['aria-expanded'] });
                        obs.disconnect();
                    }
                });
                bodyObs.observe(document.body, { childList: true, subtree: true });
            }
        })();
        </script>
        """

    st.markdown(
        f"<style>{_BASE_CSS}\n{bg_rule}\n{sidebar_css}</style>",
        unsafe_allow_html=True,
    )
    if sidebar_js:
        st.markdown(sidebar_js, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DESIGN TOKENS (CSS Custom Properties)
# ═══════════════════════════════════════════════════════════════════

_BASE_CSS = """
/* ── Design Tokens ──────────────────────────────────────────── */
:root {
    /* Background */
    --dw-bg-deep: #020817;
    --dw-bg-dark: #0a0f1e;

    /* Glass surfaces */
    --dw-glass-bg: rgba(4, 13, 28, 0.72);
    --dw-glass-bg-heavy: rgba(8, 20, 40, 0.82);
    --dw-glass-bg-light: rgba(10, 18, 35, 0.65);

    /* Borders */
    --dw-border: 1px solid rgba(130, 160, 210, 0.16);
    --dw-border-accent: 1px solid rgba(130, 160, 210, 0.25);
    --dw-border-subtle: 1px solid rgba(100, 130, 180, 0.10);
    --dw-border-color: rgba(130, 160, 210, 0.16);
    --dw-border-color-accent: rgba(130, 160, 210, 0.25);

    /* Text */
    --dw-text-primary: #F5F7FA;
    --dw-text-secondary: #94a3b8;
    --dw-text-muted: #64748b;
    --dw-text-bright: #ffffff;

    /* Accent */
    --dw-accent-blue: #818cf8;
    --dw-accent-violet: #a78bfa;
    --dw-accent-indigo: #6366f1;
    --dw-gradient-primary: linear-gradient(135deg, #6366f1, #7c3aed);

    /* Status */
    --dw-success: #22c55e;
    --dw-warning: #f59e0b;
    --dw-error: #ef4444;

    /* Radius */
    --dw-radius-sm: 8px;
    --dw-radius-md: 10px;
    --dw-radius-lg: 12px;
    --dw-radius-xl: 14px;

    /* Shadows */
    --dw-shadow-sm: 0 2px 8px rgba(0,0,0,0.25);
    --dw-shadow-md: 0 8px 24px rgba(0,0,0,0.35);
    --dw-shadow-lg: 0 20px 60px rgba(0,0,0,0.5);

    /* Blur */
    --dw-blur-sm: 8px;
    --dw-blur-md: 12px;
    --dw-blur-lg: 18px;

    /* Spacing */
    --dw-navbar-height: 56px;
    --dw-sidebar-width: 260px;
    --dw-content-max-width: 1200px;

    /* Typography */
    --dw-font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --dw-font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}

/* ── Global ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: var(--dw-font-family) !important;
    color: var(--dw-text-primary) !important;
}

/* Remove default Streamlit backgrounds */
.stApp > header { background: transparent !important; }
header[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

.main .block-container,
.stMainBlockContainer {
    background: transparent !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
div[data-testid="stAppViewBlockContainer"] { background: transparent !important; }
div[data-testid="stVerticalBlock"] { background: transparent !important; }
div[data-testid="stHorizontalBlock"] { background: transparent !important; }
div[data-testid="stAppViewContainer"] { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
::-webkit-scrollbar-thumb { background: rgba(100,116,139,0.35); border-radius: 3px; }

/* ── Typography ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    color: var(--dw-text-primary) !important;
    letter-spacing: 0.3px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(6, 10, 24, 0.92) !important;
    border-right: 1px solid rgba(130, 160, 210, 0.12) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
    padding-top: 1rem !important;
}

/* ── Glass Panels ────────────────────────────────────────────── */
div.glass-panel {
    background: var(--dw-glass-bg);
    border: var(--dw-border);
    border-radius: var(--dw-radius-md);
    padding: 1.25rem;
    backdrop-filter: blur(var(--dw-blur-md));
    -webkit-backdrop-filter: blur(var(--dw-blur-md));
}

/* ── Metric Cards (workspace) ────────────────────────────────── */
div.metric-card {
    background: rgba(8, 15, 32, 0.72);
    border: 1px solid rgba(130, 160, 210, 0.14);
    border-radius: var(--dw-radius-lg);
    padding: 0.85rem 0.75rem;
    text-align: center;
    backdrop-filter: blur(var(--dw-blur-sm));
    transition: border-color 0.2s ease;
}
div.metric-card:hover { border-color: rgba(130, 160, 210, 0.3); }
div.metric-card .metric-icon { font-size: 1.15rem; margin-bottom: 0.25rem; color: var(--dw-accent-blue); }
div.metric-card .metric-label {
    font-size: 0.6rem; color: var(--dw-text-secondary); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 0.15rem; font-weight: 600;
}
div.metric-card .metric-value { font-size: 1.35rem; font-weight: 700; color: var(--dw-text-bright); margin: 0.1rem 0; }
div.metric-card .metric-sub {
    font-size: 0.58rem; color: var(--dw-text-muted); margin-top: 0.1rem;
}

/* ── Feature Cards (home) ────────────────────────────────────── */
div.feature-card {
    background: rgba(10, 16, 32, 0.6);
    border: 1px solid rgba(130, 160, 210, 0.1);
    border-radius: var(--dw-radius-lg);
    padding: 1.25rem 1.1rem;
    text-align: left;
    backdrop-filter: blur(var(--dw-blur-sm));
    transition: border-color 0.3s, transform 0.3s;
    height: 155px;
}
div.feature-card:hover { border-color: rgba(130, 160, 210, 0.28); transform: translateY(-2px); }
div.feature-card .fc-icon {
    width: 38px; height: 38px; border-radius: var(--dw-radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; margin-bottom: 0.65rem;
}
div.feature-card .fc-title {
    font-size: 0.82rem; font-weight: 600; color: var(--dw-text-primary); margin-bottom: 0.35rem;
}
div.feature-card .fc-desc {
    font-size: 0.72rem; color: var(--dw-text-secondary); line-height: 1.55;
}

/* ── Navbar (home) ───────────────────────────────────────────── */
div.dw-navbar {
    background: rgba(6, 10, 24, 0.7);
    border: 1px solid rgba(130, 160, 210, 0.12);
    border-radius: var(--dw-radius-xl);
    padding: 0.5rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    backdrop-filter: blur(var(--dw-blur-lg));
    -webkit-backdrop-filter: blur(var(--dw-blur-lg));
    margin-bottom: 1.5rem;
    height: var(--dw-navbar-height);
}

/* ── CTA Button ──────────────────────────────────────────────── */
a.cta-btn, div.cta-btn {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--dw-gradient-primary);
    color: white !important; font-weight: 600; font-size: 0.78rem;
    padding: 0.5rem 1.15rem; border-radius: var(--dw-radius-sm);
    text-decoration: none; transition: opacity 0.2s; cursor: pointer;
    letter-spacing: 0.3px; border: none;
}
a.cta-btn:hover, div.cta-btn:hover { opacity: 0.88; }

/* ── Hero ────────────────────────────────────────────────────── */
div.hero-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 20px; padding: 0.3rem 0.9rem;
    font-size: 0.63rem; color: #a5b4fc; font-weight: 500;
    letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 1.2rem;
}
span.gradient-text {
    background: linear-gradient(135deg, var(--dw-accent-blue), var(--dw-accent-violet));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Streamlit Component Overrides ───────────────────────────── */
div.stButton > button {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(130, 160, 210, 0.2) !important;
    color: var(--dw-text-primary) !important; font-weight: 500 !important;
    border-radius: var(--dw-radius-sm) !important; font-size: 0.78rem !important;
    padding: 0.45rem 0.9rem !important; transition: all 0.2s !important;
    text-transform: none !important; letter-spacing: 0.2px !important;
    min-height: auto !important; font-family: var(--dw-font-family) !important;
}
div.stButton > button:hover {
    background: rgba(99, 102, 241, 0.22) !important;
    border-color: rgba(130, 160, 210, 0.35) !important;
}

/* Primary CTA button style */
div.stButton > button[kind="primary"] {
    background: var(--dw-gradient-primary) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
}

div.stDownloadButton > button {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(130, 160, 210, 0.18) !important;
    color: var(--dw-text-primary) !important; border-radius: var(--dw-radius-sm) !important;
    font-size: 0.72rem !important; min-height: auto !important;
}

/* Text Input */
div[data-testid="stTextInput"] input, div.stTextInput input {
    background: rgba(10, 18, 35, 0.6) !important;
    border: 1px solid rgba(130, 160, 210, 0.18) !important;
    border-radius: var(--dw-radius-sm) !important; color: var(--dw-text-primary) !important;
    font-size: 1rem !important; font-family: var(--dw-font-family) !important;
}

/* Chat Input */
div[data-testid="stChatInput"] {
    background: rgba(10, 18, 35, 0.65) !important;
    border: 1px solid rgba(130, 160, 210, 0.2) !important;
    border-radius: var(--dw-radius-md) !important;
}
div[data-testid="stChatInput"] textarea {
    color: var(--dw-text-primary) !important;
    font-family: var(--dw-font-family) !important;
    font-size: 1rem !important;
}
/* Remove dark background behind chat input bar */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"],
.stChatFloatingInputContainer,
div.stChatInputContainer {
    background: transparent !important;
    background-color: transparent !important;
}
div[data-testid="stBottom"]::before {
    background: transparent !important;
    display: none !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}
div[data-testid="stFileUploader"] section {
    background: rgba(10, 18, 35, 0.45) !important;
    border: 1px dashed rgba(130, 160, 210, 0.2) !important;
    border-radius: var(--dw-radius-sm) !important;
}
div[data-testid="stFileUploader"] button {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(130, 160, 210, 0.2) !important;
    border-radius: 6px !important; color: var(--dw-text-primary) !important;
}
div[data-testid="stFileUploader"] small {
    color: var(--dw-text-muted) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background: rgba(8, 14, 28, 0.55) !important;
    border: 1px solid rgba(130, 160, 210, 0.08) !important;
    border-radius: var(--dw-radius-sm) !important;
}

/* Code blocks */
div[data-testid="stCode"], pre, code {
    background: rgba(6, 12, 24, 0.8) !important;
    border: 1px solid rgba(130, 160, 210, 0.08) !important;
    border-radius: var(--dw-radius-sm) !important;
    font-family: var(--dw-font-mono) !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: rgba(10, 15, 30, 0.45) !important;
    border: 1px solid rgba(130, 160, 210, 0.08) !important;
    border-radius: var(--dw-radius-sm) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.78rem !important; font-weight: 500 !important;
    color: var(--dw-text-secondary) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--dw-text-primary) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    background: rgba(10, 18, 35, 0.55) !important;
    border-radius: var(--dw-radius-sm) !important;
}

/* Spinner */
div.stSpinner > div { color: var(--dw-accent-blue) !important; }

/* Caption */
div.stCaption, .stCaption p { color: var(--dw-text-muted) !important; }

/* Divider */
hr { border-color: rgba(130, 160, 210, 0.08) !important; }

/* Metric */
div[data-testid="stMetric"] label { color: var(--dw-text-secondary) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--dw-text-primary) !important; }

/* ── Status Badges ───────────────────────────────────────────── */
span.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
    vertical-align: middle;
}
span.status-online { background: var(--dw-success); box-shadow: 0 0 6px rgba(34,197,94,0.4); }
span.status-standby { background: var(--dw-warning); }
span.status-offline { background: var(--dw-error); }

/* ── Result Panels (workspace) ───────────────────────────────── */
div.result-panel {
    background: rgba(8, 14, 28, 0.72);
    border: 1px solid rgba(130, 160, 210, 0.12);
    border-radius: var(--dw-radius-md); padding: 0.9rem;
    backdrop-filter: blur(var(--dw-blur-sm));
}
div.result-panel .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(130, 160, 210, 0.06);
}
div.result-panel .panel-title {
    font-size: 0.72rem; font-weight: 600; color: var(--dw-text-primary);
    text-transform: uppercase; letter-spacing: 0.8px;
    display: flex; align-items: center; gap: 0.4rem;
}
div.result-panel .panel-action {
    font-size: 0.65rem; color: var(--dw-text-secondary); cursor: pointer;
    display: flex; align-items: center; gap: 0.3rem;
    background: rgba(99,102,241,0.08); padding: 0.25rem 0.6rem;
    border-radius: 6px; border: 1px solid rgba(130,160,210,0.1);
    transition: all 0.2s;
}
div.result-panel .panel-action:hover {
    background: rgba(99,102,241,0.15); border-color: rgba(130,160,210,0.2);
}

/* ── Suggested Question Buttons ──────────────────────────────── */
div.suggest-btn button {
    background: rgba(10, 18, 35, 0.55) !important;
    border: 1px solid rgba(130, 160, 210, 0.18) !important;
    border-radius: var(--dw-radius-sm) !important; color: #cbd5e1 !important;
    font-size: 0.72rem !important; font-weight: 400 !important;
    text-transform: none !important; letter-spacing: 0 !important;
    padding: 0.55rem 0.8rem !important; white-space: normal !important;
    text-align: center !important; min-height: auto !important;
    font-family: var(--dw-font-family) !important;
}
div.suggest-btn button:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(130, 160, 210, 0.35) !important;
}

/* ── Scroll indicator ────────────────────────────────────────── */
@keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(6px)} }
div.scroll-indicator {
    text-align: center; color: var(--dw-text-muted); font-size: 0.72rem; padding: 1rem 0 0.5rem;
}
div.scroll-indicator .chevron { animation: bounce 2s infinite; display: inline-block; }

/* Animation */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(10px); }
    to { opacity:1; transform:translateY(0); }
}
div.animate-in { animation: fadeInUp 0.45s ease forwards; }

/* ── Footer ──────────────────────────────────────────────────── */
div.dw-footer {
    text-align: center; padding: 1rem 0 0.5rem;
    font-size: 0.65rem; color: var(--dw-text-muted);
    border-top: 1px solid rgba(130, 160, 210, 0.06);
    margin-top: 1.5rem;
}

/* ── Workspace Preview (home hero) ───────────────────────────── */
div.workspace-preview-container {
    perspective: 1200px;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 0.5rem;
}
div.workspace-preview-container img {
    width: 100%;
    max-width: 540px;
    border-radius: var(--dw-radius-lg);
    border: 1px solid rgba(130, 160, 210, 0.12);
    transform: rotateY(-10deg) rotateX(4deg) scale(1.0);
    box-shadow: 0 35px 70px rgba(0,0,0,0.55),
                0 0 40px rgba(99,102,241,0.06),
                inset 0 0 0 1px rgba(99,102,241,0.06);
    transition: transform 0.5s ease;
}
div.workspace-preview-container:hover img {
    transform: rotateY(-5deg) rotateX(2deg) scale(1.01);
}
"""
