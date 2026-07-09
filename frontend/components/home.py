"""
Home / Landing Page — Pixel-accurate reproduction of Reference Image 1.

Uses the user-provided space/rocket background image and renders
the navbar, hero section with workspace preview, feature cards,
and scroll indicator.

NO SIDEBAR on this page.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def render_home() -> None:
    """Render the complete home landing page."""
    _render_navbar()
    _render_hero()
    _render_features()
    _render_scroll_indicator()


# ═══════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════

def _render_navbar() -> None:
    """Translucent glass navbar matching reference Image 1."""
    st.markdown("""
    <div style="
        background: rgba(6, 10, 24, 0.7);
        border: 1px solid rgba(130, 160, 210, 0.12);
        border-radius: 14px;
        padding: 0 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        margin-bottom: 1.8rem; font-family: 'Inter', sans-serif;
        height: 56px;
    ">
        <div style="display: flex; align-items: center; gap: 0.65rem;">
            <div style="width: 36px; height: 36px; border-radius: 50%;
                         background: linear-gradient(135deg, #6366f1, #8b5cf6);
                         display: flex; align-items: center; justify-content: center;
                         font-size: 0.95rem; color: white; font-weight: 700;">✦</div>
            <div>
                <div style="font-weight: 700; font-size: 0.8rem; color: #f1f5f9;
                             letter-spacing: 1.5px;">DATAWHISPERER AI</div>
                <div style="font-size: 0.5rem; color: #64748b; text-transform: uppercase;
                             letter-spacing: 0.8px;">TALK TO YOUR CSV WITH AI</div>
            </div>
        </div>
        <div style="display: flex; gap: 2.2rem; align-items: center;">
            <a href="#" style="color: #f1f5f9; text-decoration: none; font-size: 0.78rem;
                               font-weight: 600; border-bottom: 2px solid #818cf8;
                               padding-bottom: 2px;">Home</a>
            <a href="#how" style="color: #94a3b8; text-decoration: none; font-size: 0.78rem;
                                  font-weight: 500;">How It Works</a>
            <a href="#features" style="color: #94a3b8; text-decoration: none; font-size: 0.78rem;
                                       font-weight: 500;">Features</a>
            <a href="#about" style="color: #94a3b8; text-decoration: none; font-size: 0.78rem;
                                    font-weight: 500;">About</a>
        </div>
        <div style="min-width: 170px; text-align: right;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit CTA button aligned to right of navbar
    _, _, _, _, cta_col = st.columns([4, 1, 1, 1, 1.3])
    with cta_col:
        st.markdown('<div style="margin-top: -4.5rem;">', unsafe_allow_html=True)
        if st.button("✦ Launch DataWhisperer", key="nav_cta_btn", type="primary"):
            st.session_state["_dw_page"] = "workspace"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════

def _render_hero() -> None:
    """Two-column hero: left text + CTA, right floating workspace preview."""
    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="padding-top: 1rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.45rem;
                         background: rgba(99, 102, 241, 0.1);
                         border: 1px solid rgba(99, 102, 241, 0.22);
                         border-radius: 20px; padding: 0.28rem 0.85rem;
                         font-size: 0.62rem; color: #a5b4fc; font-weight: 500;
                         letter-spacing: 1.2px; text-transform: uppercase;
                         margin-bottom: 1rem;">✦ AI-POWERED DATA ANALYTICS</div>
            <h1 style="font-size: 2.8rem; font-weight: 700; line-height: 1.1;
                        margin: 0 0 1.1rem 0; color: #ffffff;
                        text-transform: none !important;
                        letter-spacing: -0.5px !important;">
                Turn Your CSV Data<br>
                Into <span style="background: linear-gradient(135deg, #818cf8, #a78bfa);
                                   -webkit-background-clip: text;
                                   -webkit-text-fill-color: transparent;
                                   background-clip: text;
                                   font-style: italic;">Powerful Insights</span>
            </h1>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.7;
                       margin-bottom: 1.5rem; max-width: 440px;">
                DataWhisper AI lets you talk to your CSV files using natural language.
                Ask questions, get SQL, explore results, and visualize insights —
                all in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✦  Launch DataWhisperer  →", key="hero_launch_btn",
                      type="primary", use_container_width=False):
            st.session_state["_dw_page"] = "workspace"
            st.rerun()

        st.markdown("""
        <p style="color: #64748b; font-size: 0.68rem; margin-top: 0.6rem;">
            🔒 Secure. Private. Your data stays with you.
        </p>
        """, unsafe_allow_html=True)

    with col_right:
        _render_workspace_preview()


def _render_workspace_preview() -> None:
    """Workspace preview as a tilted image with perspective transform."""
    preview_path = Path(__file__).parent.parent / "assets" / "workspace_preview.png"
    if preview_path.exists():
        img_b64 = base64.b64encode(preview_path.read_bytes()).decode()
        st.markdown(f"""
        <div class="workspace-preview-container">
            <img src="data:image/png;base64,{img_b64}"
                 alt="DataWhisper AI Workspace Preview" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Preview image not available.")


# ═══════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════

def _render_features() -> None:
    """Five feature cards matching reference Image 1."""
    st.markdown("""
    <div style="text-align: center; padding: 1.8rem 0 1.2rem 0;" id="features">
        <h3 style="font-size: 1.05rem; color: #f1f5f9; text-transform: none !important;
                    font-weight: 600; letter-spacing: 0 !important; margin: 0;">
            Everything you need to analyze your data
        </h3>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        ("💬", "#6366f1", "Natural Language Queries",
         "Ask questions in plain English and get insights instantly."),
        ("&lt;/&gt;", "#8b5cf6", "AI Generated SQL",
         "See the SQL behind the answers with full transparency."),
        ("📋", "#22c55e", "Smart Results",
         "Get clean, structured results you can explore and export."),
        ("📊", "#f43f5e", "Beautiful Visualizations",
         "Auto-generate charts that help you see the bigger picture."),
        ("🛡️", "#06b6d4", "Privacy First",
         "Your data is never shared. 100% secure and private."),
    ]

    cols = st.columns(5, gap="small")
    for idx, (icon, color, title, desc) in enumerate(cards):
        with cols[idx]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="fc-icon" style="background: {color}18; color: {color};">{icon}</div>
                <div class="fc-title">{title}</div>
                <div class="fc-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SCROLL INDICATOR
# ═══════════════════════════════════════════════════════════════════

def _render_scroll_indicator() -> None:
    """Minimal scroll-to-explore indicator."""
    st.markdown("""
    <div class="scroll-indicator">
        <div>Scroll to explore</div>
        <div class="chevron" style="margin-top: 0.2rem; font-size: 1.1rem;">⌄</div>
    </div>
    """, unsafe_allow_html=True)
