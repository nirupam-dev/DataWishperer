"""
Home / Landing Page — Pixel-perfect reproduction of Reference Image 1.

Uses the user-provided space/rocket background image and renders
the navbar, hero section with workspace preview, feature cards,
and scroll indicator.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


def render_home() -> None:
    """Render the complete home landing page."""
    _render_navbar()
    _render_hero()
    _render_features()
    _render_scroll_indicator()


def _render_navbar() -> None:
    """Translucent glass navbar matching reference."""
    st.markdown("""
    <div style="
        background: rgba(8, 12, 28, 0.65);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 14px;
        padding: 0.55rem 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        margin-bottom: 1.8rem; font-family: 'Inter', sans-serif;
    ">
        <div style="display: flex; align-items: center; gap: 0.7rem;">
            <div style="width: 38px; height: 38px; border-radius: 50%;
                         background: linear-gradient(135deg, #6366f1, #8b5cf6);
                         display: flex; align-items: center; justify-content: center;
                         font-size: 1rem; color: white; font-weight: 700;">✦</div>
            <div>
                <div style="font-weight: 700; font-size: 0.82rem; color: #f1f5f9;
                             letter-spacing: 1.5px;">DATAWHISPERER AI</div>
                <div style="font-size: 0.52rem; color: #64748b; text-transform: uppercase;
                             letter-spacing: 1px;">TALK TO YOUR CSV WITH AI</div>
            </div>
        </div>
        <div style="display: flex; gap: 2.2rem; align-items: center;">
            <a href="#" style="color: #f1f5f9; text-decoration: none; font-size: 0.8rem;
                               font-weight: 600; border-bottom: 2px solid #818cf8;
                               padding-bottom: 2px;">Home</a>
            <a href="#how" style="color: #94a3b8; text-decoration: none; font-size: 0.8rem;
                                  font-weight: 500;">How It Works</a>
            <a href="#features" style="color: #94a3b8; text-decoration: none; font-size: 0.8rem;
                                       font-weight: 500;">Features</a>
            <a href="#pricing" style="color: #94a3b8; text-decoration: none; font-size: 0.8rem;
                                      font-weight: 500;">Pricing</a>
            <a href="#about" style="color: #94a3b8; text-decoration: none; font-size: 0.8rem;
                                    font-weight: 500;">About</a>
        </div>
        <div id="nav-cta-placeholder"></div>
    </div>
    """, unsafe_allow_html=True)

    # Overlay CTA button near navbar right edge
    _, _, _, _, cta_col = st.columns([4, 1, 1, 1, 1.2])
    with cta_col:
        st.markdown('<div style="margin-top: -4.2rem;">', unsafe_allow_html=True)
        if st.button("✦ Launch DataWhisperer", key="nav_cta_btn", type="primary"):
            st.session_state["_dw_page"] = "workspace"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def _render_hero() -> None:
    """Two-column hero: left text + CTA, right floating workspace preview."""
    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="padding-top: 1.5rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.45rem;
                         background: rgba(99, 102, 241, 0.1);
                         border: 1px solid rgba(99, 102, 241, 0.22);
                         border-radius: 20px; padding: 0.3rem 0.9rem;
                         font-size: 0.65rem; color: #a5b4fc; font-weight: 500;
                         letter-spacing: 1.2px; text-transform: uppercase;
                         margin-bottom: 1.2rem;">✦ AI-POWERED DATA ANALYTICS</div>
            <h1 style="font-size: 3rem; font-weight: 700; line-height: 1.12;
                        margin: 0 0 1.2rem 0; color: #ffffff;
                        text-transform: none !important;
                        letter-spacing: -0.5px !important;">
                Turn Your CSV Data<br>
                Into <span style="background: linear-gradient(135deg, #818cf8, #a78bfa);
                                   -webkit-background-clip: text;
                                   -webkit-text-fill-color: transparent;
                                   background-clip: text;
                                   font-style: italic;">Powerful Insights</span>
            </h1>
            <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.75;
                       margin-bottom: 1.6rem; max-width: 460px;">
                DataWhisperer AI lets you talk to your CSV files using natural language.
                Ask questions, get SQL, explore results, and visualize insights —
                all in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Launch DataWhisperer  →", key="hero_launch_btn",
                      type="primary", use_container_width=False):
            st.session_state["_dw_page"] = "workspace"
            st.rerun()

        st.markdown("""
        <p style="color: #64748b; font-size: 0.7rem; margin-top: 0.7rem;">
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
        <div style="padding: 0; perspective: 1200px; margin-top: -0.5rem;">
            <img src="data:image/png;base64,{img_b64}"
                 style="width: 100%; max-width: 560px; border-radius: 12px;
                        border: 1px solid rgba(99,102,241,0.15);
                        transform: rotateY(-12deg) rotateX(5deg) scale(1.02);
                        box-shadow: 0 40px 80px rgba(0,0,0,0.6),
                                    0 0 50px rgba(99,102,241,0.08),
                                    inset 0 0 0 1px rgba(99,102,241,0.08);
                        transition: transform 0.6s ease;"
                 alt="DataWhisper AI Workspace Preview" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Preview image not available.")


def _render_features() -> None:
    """Five feature cards matching reference."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1.3rem 0;" id="features">
        <h3 style="font-size: 1.1rem; color: #f1f5f9; text-transform: none !important;
                    font-weight: 600; letter-spacing: 0 !important;">
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
            <div style="background: rgba(15, 20, 40, 0.55);
                         border: 1px solid rgba(99, 102, 241, 0.1);
                         border-radius: 12px; padding: 1.2rem 1rem;
                         backdrop-filter: blur(8px); height: 160px;">
                <div style="width: 38px; height: 38px; border-radius: 10px;
                             background: {color}22;
                             display: flex; align-items: center; justify-content: center;
                             font-size: 1rem; margin-bottom: 0.7rem;
                             color: {color};">{icon}</div>
                <div style="font-size: 0.82rem; font-weight: 600; color: #f1f5f9;
                             margin-bottom: 0.35rem;">{title}</div>
                <div style="font-size: 0.72rem; color: #94a3b8;
                             line-height: 1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_scroll_indicator() -> None:
    """Minimal scroll-to-explore indicator."""
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.72rem;
                 padding: 1.2rem 0 0.5rem 0;">
        <div>Scroll to explore</div>
        <div style="margin-top: 0.2rem; font-size: 1.1rem;
                     animation: bounce 2s infinite;">⌄</div>
    </div>
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(6px); }
    }
    </style>
    """, unsafe_allow_html=True)
