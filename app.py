"""
Professional Disease Prediction Chatbot - Main Application
Advanced UI/UX with Complete Feature Set
"""

import streamlit as st
try:
    from streamlit_option_menu import option_menu
except ImportError:
    def option_menu(menu_title, options, icons=None, menu_icon=None, default_index=0, styles=None):
        """Fallback sidebar menu when streamlit-option-menu is not installed."""
        if menu_title:
            st.markdown(f"#### {menu_title}")

        selected = options[default_index]
        for option in options:
            button_type = "primary" if option == selected else "secondary"
            if st.button(option, key=f"fallback_nav_{option}", use_container_width=True, type=button_type):
                selected = option
        return selected
import os
import pickle
import json
import html
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px

from advanced_chatbot import (
    ChatbotConfig,
    AuthenticationManager,
    AdvancedModelManager,
    ConversationEngine,
    SessionStateManager,
    UIComponents,
    format_symptom_name,
    check_urgent_symptoms
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=ChatbotConfig.APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional responsive styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    :root {
        --primary-color: #0066CC;
        --success-color: #00CC66;
        --warning-color: #FF6600;
        --danger-color: #FF3333;
        --light-bg: #f0f2f6;
        --card-bg: #ffffff;
        --border-radius: 0.5rem;
        --button-radius: 0.5rem;
    }
    
    /* Base Responsive Typography */
    html {
        font-size: 16px;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background-color: #fafbfc;
        color: #1f2937;
    }

    /* Main Container - CONSISTENT ACROSS ALL DEVICES */
    .main {
        padding: 2rem 3rem;
        max-width: 100%;
        overflow-x: hidden;
    }

    /* Consistent on all screens - no breakpoint changes */
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }

    /* Metrics - SAME ON ALL DEVICES */
    .stMetric {
        background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 0.5rem !important;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 16px rgba(0, 102, 204, 0.1);
        transform: translateY(-2px);
    }

    /* Cards - FIXED Radius */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem !important;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
    }

    @media (max-width: 767px) {
        .prediction-card {
            padding: 1rem;
            margin: 0.75rem 0;
            border-radius: 1rem !important;
        }
    }

    @media (min-width: 768px) and (max-width: 1024px) {
        .prediction-card {
            border-radius: 1rem !important;
        }
    }

    @media (min-width: 1025px) {
        .prediction-card {
            border-radius: 1rem !important;
        }
    }

    .prediction-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.35);
    }

    /* Symptom Chips - SAME ON ALL DEVICES */
    .symptom-chip {
        display: inline-block;
        background-color: #e8f4f8;
        color: var(--primary-color);
        padding: 0.5rem 0.75rem;
        border-radius: 1.5rem !important;
        margin: 0.25rem 0.25rem 0.25rem 0;
        border: 2px solid var(--primary-color);
        font-size: 0.875rem;
        transition: all 0.2s ease;
        flex-wrap: wrap;
    }

    .symptom-chip:hover {
        background-color: var(--primary-color);
        color: white;
        cursor: pointer;
    }

    /* Alert Boxes - FIXED Radius */
    .urgent-alert {
        background-color: #FFE5E5;
        border-left: 5px solid var(--danger-color);
        color: #8B0000;
        padding: 1rem;
        border-radius: 0.5rem !important;
        margin: 1rem 0;
        font-size: 0.95rem;
        animation: slideIn 0.3s ease;
    }

    .success-alert {
        background-color: #E5F9E5;
        border-left: 5px solid var(--success-color);
        color: #006600;
        padding: 1rem;
        border-radius: 0.5rem !important;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #E5F2FF;
        border-left: 5px solid var(--primary-color);
        color: #003399;
        padding: 1rem;
        border-radius: 0.5rem !important;
        margin: 1rem 0;
    }

    @media (max-width: 480px) {
        .urgent-alert,
        .success-alert,
        .info-box {
            padding: 0.75rem;
            font-size: 0.9rem;
            border-radius: 0.5rem !important;
        }
    }

    @media (min-width: 481px) and (max-width: 1024px) {
        .urgent-alert,
        .success-alert,
        .info-box {
            border-radius: 0.5rem !important;
        }
    }

    @media (min-width: 1025px) {
        .urgent-alert,
        .success-alert,
        .info-box {
            border-radius: 0.5rem !important;
        }
    }

    /* Chat Messages - Mobile Optimized */
    .chat-message {
        padding: 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        word-wrap: break-word;
        animation: slideIn 0.2s ease;
    }
    
    .chat-message.user {
        background-color: #e8f4f8;
        border-left: 5px solid var(--primary-color);
        margin-left: 0;
    }
    
    .chat-message.assistant {
        background-color: #f0f2f6;
        border-left: 5px solid #667eea;
        margin-right: 0;
    }

    @media (max-width: 480px) {
        .chat-message {
            padding: 0.75rem;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
    }

    /* Stats Container - Responsive Grid */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    @media (max-width: 1024px) {
        .stats-container {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
        }
    }

    @media (max-width: 768px) {
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
    }

    @media (max-width: 480px) {
        .stats-container {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
    }

    /* Stat Cards - SAME ON ALL DEVICES */
    .stat-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem !important;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
        transform: translateY(-2px);
    }

    /* Buttons - SAME ON ALL DEVICES */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        border-radius: 0.5rem !important;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
        min-height: 45px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
    }

    /* Input Fields - Responsive & FIXED Radius */
    .stTextInput > div > input,
    .stSelectbox > div > select,
    .stTextArea > div > textarea {
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 0.5rem !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > input:focus,
    .stSelectbox > div > select:focus,
    .stTextArea > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
        border-radius: 0.5rem !important;
    }

    @media (max-width: 480px) {
        .stTextInput > div > input,
        .stSelectbox > div > select,
        .stTextArea > div > textarea {
            font-size: 16px !important;
            padding: 0.6rem !important;
            border-radius: 0.5rem !important;
        }
        
        .stTextInput > div > input:focus,
        .stSelectbox > div > select:focus,
        .stTextArea > div > textarea:focus {
            border-radius: 0.5rem !important;
        }
    }

    @media (min-width: 481px) {
        .stTextInput > div > input,
        .stSelectbox > div > select,
        .stTextArea > div > textarea {
            border-radius: 0.5rem !important;
        }
    }

    /* Expanders - FIXED Radius */
    .streamlit-expanderHeader {
        padding: 0.75rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .streamlit-expander {
        border: 2px solid #e0e0e0 !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }

    @media (max-width: 480px) {
        .streamlit-expander {
            border-radius: 0.5rem !important;
        }
    }

    @media (min-width: 481px) {
        .streamlit-expander {
            border-radius: 0.5rem !important;
        }
    }

    /* Columns Layout - Mobile Responsive */
    .element-container {
        width: 100%;
    }

    /* Charts - Responsive */
    .plotly-graph-div {
        width: 100% !important;
        height: auto !important;
        min-height: 300px;
    }

    @media (max-width: 768px) {
        .plotly-graph-div {
            min-height: 250px;
        }
    }

    @media (max-width: 480px) {
        .plotly-graph-div {
            min-height: 200px;
        }
    }

    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }

    @media (max-width: 480px) {
        hr {
            margin: 1rem 0;
        }
    }

    /* Sidebar & Navigation - Mobile Column Stack */
    [data-testid="stSidebar"] {
        display: flex;
        flex-direction: column;
    }

    /* Navigation Menu Container */
    .nav-menu {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    /* Navigation Items - Stack in Column */
    [data-testid="stSidebar"] .streamlit-optionMenu {
        width: 100% !important;
        display: flex;
        flex-direction: column !important;
    }

    [data-testid="stSidebar"] [role="navigation"] {
        display: flex;
        flex-direction: column !important;
        width: 100%;
    }

    /* Sidebar Responsive */
    @media (max-width: 1024px) {
        [data-testid="stSidebar"] {
            width: 280px !important;
            flex-direction: column;
        }
    }

    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
            position: relative !important;
            flex-direction: column;
        }
        
        [data-testid="stSidebar"] .streamlit-optionMenu {
            display: flex !important;
            flex-direction: column !important;
        }
    }

    @media (max-width: 480px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
            flex-direction: column;
        }
    }

    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #0066CC;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #004999;
    }

    /* Print Styles */
    @media print {
        .stButton,
        .stDownloadButton {
            display: none;
        }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Modern responsive polish layered after the legacy theme above.
st.markdown("""
<style>
    :root {
        --primary-color: #0f766e;
        --primary-soft: #dff7f3;
        --secondary-color: #2563eb;
        --accent-color: #f59e0b;
        --success-color: #16a34a;
        --warning-color: #d97706;
        --danger-color: #dc2626;
        --ink: #111827;
        --muted: #667085;
        --subtle: #f7fafc;
        --surface: #ffffff;
        --surface-soft: #eef6f5;
        --line: #d9e5e2;
        --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        --radius: 0.5rem;
    }

    html,
    body,
    [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
        letter-spacing: 0;
    }

    body {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 28rem),
            linear-gradient(180deg, #f8fbfb 0%, #eef4f3 100%);
    }

    .block-container {
        max-width: 1180px;
        padding: 2rem 2.25rem 4rem;
    }

    h1, h2, h3 {
        letter-spacing: 0 !important;
        color: var(--ink);
    }

    h1 {
        font-size: 2.25rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        font-size: 1.6rem !important;
        line-height: 1.25 !important;
    }

    h3 {
        font-size: 1.1rem !important;
        line-height: 1.35 !important;
        margin-top: 1.25rem !important;
    }

    .app-page-header {
        width: 100%;
        padding: 1.4rem 0 1.15rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--line);
    }

    .page-eyebrow {
        color: var(--primary-color);
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }

    .page-heading {
        margin: 0;
        color: var(--ink);
        font-size: 2.15rem;
        line-height: 1.16;
        font-weight: 800;
    }

    .page-lede {
        max-width: 760px;
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
    }

    .login-brand {
        text-align: center;
        margin: 0 auto 1.2rem;
        max-width: 560px;
    }

    .login-mark {
        width: 4rem;
        height: 4rem;
        margin: 0 auto 1rem;
        display: grid;
        place-items: center;
        border-radius: var(--radius);
        background: var(--primary-soft);
        color: var(--primary-color);
        font-size: 2rem;
        border: 1px solid rgba(15, 118, 110, 0.18);
    }

    .login-brand h1 {
        margin: 0;
        font-size: 2rem !important;
    }

    .login-brand p {
        margin: 0.65rem auto 0;
        color: var(--muted);
        line-height: 1.6;
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1.25rem;
        box-shadow: var(--shadow);
    }

    .trust-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .trust-pill,
    .quick-card,
    .feature-card,
    .ux-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: var(--radius);
    }

    .trust-pill {
        padding: 0.75rem;
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        font-weight: 700;
    }

    .section-label {
        margin: 1.4rem 0 0.75rem;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
    }

    .card-grid,
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 0.75rem 0 1rem;
    }

    .feature-grid.two-col {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .quick-card,
    .feature-card,
    .ux-card {
        padding: 1rem;
        min-height: 100%;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }

    .quick-card h3,
    .feature-card h3,
    .ux-card h3 {
        margin: 0 0 0.45rem !important;
        font-size: 1rem !important;
    }

    .quick-card p,
    .feature-card p,
    .ux-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid var(--line) !important;
        border-left: 4px solid var(--primary-color) !important;
        border-radius: var(--radius) !important;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05) !important;
        min-height: 105px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        color: var(--ink) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 700 !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: var(--radius) !important;
        border: 1px solid rgba(15, 118, 110, 0.22) !important;
        background: #ffffff !important;
        color: var(--ink) !important;
        min-height: 2.85rem;
        font-weight: 800 !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(15, 118, 110, 0.14);
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
        color: #ffffff !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] {
        border-radius: var(--radius) !important;
    }

    .chat-message {
        display: block;
        border-radius: var(--radius) !important;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    .chat-message.user {
        background: #ffffff;
        border-left: 4px solid var(--secondary-color);
    }

    .chat-message.assistant {
        background: var(--surface-soft);
        border-left: 4px solid var(--primary-color);
    }

    .chat-role {
        color: var(--ink);
        font-weight: 800;
    }

    .chat-content {
        color: #344054;
        margin-top: 0.45rem;
        line-height: 1.6;
        overflow-wrap: anywhere;
    }

    .symptom-chip {
        background: var(--primary-soft) !important;
        border: 1px solid rgba(15, 118, 110, 0.28) !important;
        color: var(--primary-color) !important;
        border-radius: 999px !important;
        font-weight: 700;
    }

    .urgent-alert {
        border-left-color: var(--danger-color) !important;
        background: #fff1f2 !important;
        color: #7f1d1d !important;
    }

    [data-testid="stSidebar"] {
        background: #f7fbfa !important;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] h3 {
        color: var(--ink);
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: var(--radius) !important;
        background: rgba(255, 255, 255, 0.86);
    }

    .element-container img,
    .plotly-graph-div {
        max-width: 100%;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1.35rem 1rem 3rem;
        }

        .page-heading,
        .login-brand h1 {
            font-size: 1.75rem !important;
        }

        .card-grid,
        .feature-grid,
        .feature-grid.two-col,
        .trust-row {
            grid-template-columns: 1fr;
        }

        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        .stMetric {
            min-height: auto;
        }
    }

    @media (max-width: 520px) {
        .block-container {
            padding: 1rem 0.75rem 2.5rem;
        }

        .app-page-header {
            padding-top: 0.75rem;
        }

        h1 {
            font-size: 1.7rem !important;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        .page-heading {
            font-size: 1.55rem !important;
        }

        .page-lede {
            font-size: 0.95rem;
        }

        div[data-testid="stForm"] {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


NAV_OPTIONS = ["Dashboard", "Chat", "Symptoms", "Analytics", "History", "About", "Logout"]
LEGACY_PAGE_NAMES = {
    "Chatbot": "Chat",
    "Symptom Analyzer": "Symptoms",
}


def route_to(page: str) -> None:
    """Set the next page while accepting older internal page names."""
    st.session_state.app_page = LEGACY_PAGE_NAMES.get(page, page)


def current_page_index() -> int:
    """Return the selected sidebar index from session state."""
    current_page = LEGACY_PAGE_NAMES.get(st.session_state.get("app_page", "Dashboard"), "Dashboard")
    if current_page not in NAV_OPTIONS:
        current_page = "Dashboard"
    st.session_state.app_page = current_page
    return NAV_OPTIONS.index(current_page)


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <section class="app-page-header">
            <div class="page-eyebrow">{html.escape(eyebrow)}</div>
            <h1 class="page-heading">{html.escape(title)}</h1>
            <p class="page-lede">{html.escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_card_grid(cards, columns: int = 3) -> None:
    grid_class = "feature-grid two-col" if columns == 2 else "feature-grid"
    cards_html = "".join(
        f"""
        <article class="feature-card">
            <h3>{html.escape(card["title"])}</h3>
            <p>{html.escape(card["body"])}</p>
        </article>
        """
        for card in cards
    )
    st.markdown(f"<div class='{grid_class}'>{cards_html}</div>", unsafe_allow_html=True)


def render_trust_row(items) -> None:
    pills = "".join(f"<div class='trust-pill'>{html.escape(item)}</div>" for item in items)
    st.markdown(f"<div class='trust-row'>{pills}</div>", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION
# ============================================================================

SessionStateManager.init_session()


# ============================================================================
# AUTHENTICATION PAGES
# ============================================================================

def page_login():
    """Login page."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-brand">
            <div class="login-mark">+</div>
            <h1>Disease Prediction Chatbot</h1>
            <p>Private symptom screening with conversational guidance, model confidence, and saved history for every user.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Your unique username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Your secure password"
            )
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                login_btn = st.form_submit_button(
                    "Sign In",
                    use_container_width=True,
                    help="Sign in to your account"
                )
            with btn_col2:
                signup_btn = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                    help="Create a new account"
                )
            
            if login_btn:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    users = AuthenticationManager.load_users()
                    if username in users:
                        if AuthenticationManager.verify_password(password, users[username]["password_hash"]):
                            st.session_state.logged_in = True
                            st.session_state.current_user = username
                            route_to("Dashboard")
                            st.success("Login successful. Redirecting...")
                            st.rerun()
                        else:
                            st.error("Invalid password. Please try again.")
                    else:
                        st.error("Username not found. Please create an account.")
            
            if signup_btn:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif len(username) < 3:
                    st.error("Username must be at least 3 characters long.")
                else:
                    users = AuthenticationManager.load_users()
                    if username in users:
                        st.error("Username already exists. Please choose another.")
                    else:
                        users[username] = {
                            "password_hash": AuthenticationManager.hash_password(password),
                            "created_at": datetime.now().isoformat(),
                            "email": "",
                            "preferences": {}
                        }
                        AuthenticationManager.save_users(users)
                        st.success("Account created. You can sign in now.")
        
        render_trust_row(["Local storage", "No tracking", "PBKDF2-SHA256"])


def page_logout():
    """Logout functionality"""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.conversation_engine = ConversationEngine()
    route_to("Dashboard")
    st.success("Logged out successfully.")
    st.rerun()


# ============================================================================
# MAIN PAGES
# ============================================================================

def page_dashboard():
    """Dashboard with quick stats and overview."""
    render_page_header(
        "Workspace",
        f"Welcome back, {st.session_state.current_user}",
        "Track conversation progress, review selected symptoms, and jump into the next step quickly.",
    )

    col1, col2 = st.columns(2)
    with col1:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("Messages", len(st.session_state.conversation_engine.conversation_history))
        with subcol2:
            st.metric("Symptoms", len(st.session_state.selected_symptoms))
    
    with col2:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("Predictions", 1 if st.session_state.current_prediction else 0)
        with subcol2:
            st.metric("Status", "Active")
    
    st.markdown("<div class='section-label'>Quick Actions</div>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("Open Chat", use_container_width=True, key="quick_chat", help="Start chatting about symptoms"):
            route_to("Chat")
            st.rerun()
    
    with action_col2:
        if st.button("Analyze Symptoms", use_container_width=True, key="quick_symptoms", help="Analyze your symptoms"):
            route_to("Symptoms")
            st.rerun()
    
    with action_col3:
        if st.button("View History", use_container_width=True, key="quick_history", help="View chat history"):
            route_to("History")
            st.rerun()
    
    st.markdown("<div class='section-label'>System Snapshot</div>", unsafe_allow_html=True)
    render_card_grid(
        [
            {
                "title": "Model",
                "body": "Ensemble screening with Random Forest and Gradient Boosting across 130+ symptoms.",
            },
            {
                "title": "Safety",
                "body": "The output is a screening aid only. Urgent or severe symptoms need professional care.",
            },
            {
                "title": "Privacy",
                "body": "Accounts, histories, and preferences stay in local project files.",
            },
        ]
    )

    st.markdown("<div class='section-label'>Core Tools</div>", unsafe_allow_html=True)
    render_card_grid(
        [
            {"title": "Chat", "body": "Discuss symptoms naturally and keep useful context for analysis."},
            {"title": "Symptoms", "body": "Select symptoms by category and generate a confidence-based prediction."},
            {"title": "Analytics", "body": "Review confidence, alternatives, and charted model output after prediction."},
        ]
    )


def page_chatbot():
    """Advanced chatbot interface."""
    render_page_header(
        "Conversation",
        "Health Chat",
        "Describe symptoms in plain language and keep the conversation context connected to the analyzer.",
    )
    
    # Sidebar with conversation controls (collapsible on mobile)
    with st.sidebar:
        st.markdown("### Controls")
        
        if st.button("Clear Chat", use_container_width=True, help="Clear conversation"):
            st.session_state.conversation_engine = ConversationEngine()
            st.session_state.selected_symptoms = []
            st.session_state.current_prediction = None
            st.success("Cleared.")
            st.rerun()
        
        st.markdown("---")
        
        # Current Context
        st.markdown("### Context")
        
        symptom_count = len(st.session_state.selected_symptoms)
        st.metric("Symptoms", symptom_count)
        
        if st.session_state.selected_symptoms:
            with st.expander("Detected Symptoms", expanded=True):
                for symptom in st.session_state.selected_symptoms:
                    st.write(f"• {format_symptom_name(symptom)}")
        
        if st.session_state.current_prediction:
            st.markdown("---")
            st.metric("Last Prediction", st.session_state.current_prediction['predicted_disease'])
            st.write(f"Confidence: {st.session_state.current_prediction['confidence']:.1f}%")
    
    # Chat display area
    st.markdown("<div class='section-label'>Conversation</div>", unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation_engine.get_history():
            role = html.escape(message["role"].title())
            content = html.escape(message["content"])
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user'>
                    <div class="chat-role">You</div>
                    <div class="chat-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant'>
                    <div class="chat-role">{role}</div>
                    <div class="chat-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-label'>Message</div>", unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Your message",
        placeholder="Describe symptoms or ask questions...",
        height=80,
        key=f"chat_input_{st.session_state.chat_input_key}",
        label_visibility="collapsed"
    )
    
    # Responsive button layout
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        send_btn = st.button("Send", use_container_width=True, help="Send message")
    with btn_col2:
        symptom_btn = st.button("Symptoms", use_container_width=True, help="Add symptom")
    
    if send_btn and user_input:
        # Process user input
        st.session_state.conversation_engine.add_message("user", user_input)
        
        # Detect intent and symptoms
        intent = st.session_state.conversation_engine.detect_intent(user_input)
        
        # Load model data
        try:
            with open(ChatbotConfig.MODEL_FILE, "rb") as f:
                model_data = pickle.load(f)
                feature_names = model_data.get("feature_names", [])
        except:
            feature_names = []
        
        detected_new = st.session_state.conversation_engine.extract_symptoms(user_input, feature_names)
        st.session_state.selected_symptoms.extend(detected_new)
        st.session_state.selected_symptoms = list(set(st.session_state.selected_symptoms))
        
        # Check for urgent symptoms
        if check_urgent_symptoms(st.session_state.selected_symptoms):
            response = "🚨 **URGENT**: Please seek immediate medical attention or call emergency services!"
            intent = "urgent"
        else:
            response = st.session_state.conversation_engine.generate_contextual_response(
                intent, st.session_state.selected_symptoms
            )
        
        st.session_state.conversation_engine.add_message("assistant", response)
        SessionStateManager.save_chat_history(st.session_state.current_user)
        st.session_state.chat_input_key += 1
        st.rerun()
    
    if symptom_btn:
        route_to("Symptoms")
        st.rerun()


def page_symptom_analyzer():
    """Advanced symptom analyzer."""
    render_page_header(
        "Analyzer",
        "Symptom Selection",
        "Choose symptoms by category, then run the model to see confidence, alternatives, and next-step guidance.",
    )
    
    # Load model
    try:
        with open(ChatbotConfig.MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
    except:
        st.error("Model not loaded. Please train the model first.")
        return
    
    feature_names = model_data.get("feature_names", [])
    label_encoder = model_data.get("label_encoder")
    
    # Symptom categories - Mobile optimized
    st.markdown("<div class='section-label'>Categories</div>", unsafe_allow_html=True)
    
    categorized_symptoms = UIComponents.render_symptom_categories(feature_names)
    
    selected = []
    
    for category, symptoms in categorized_symptoms.items():
        if symptoms:
            # Mobile-friendly expanders
            with st.expander(f"{category} ({len(symptoms)})", expanded=False):
                for symptom in symptoms:
                    if st.checkbox(format_symptom_name(symptom), key=symptom):
                        selected.append(symptom)
    
    st.markdown("<div class='section-label'>Selected Symptoms</div>", unsafe_allow_html=True)
    if selected or st.session_state.selected_symptoms:
        all_selected = list(set(selected + st.session_state.selected_symptoms))
        st.session_state.selected_symptoms = all_selected
        
        # Display as chips
        symptom_text = "".join([
            f'<span class="symptom-chip">{format_symptom_name(s)} ✓</span>'
            for s in all_selected
        ])
        st.markdown(symptom_text, unsafe_allow_html=True)
        
        st.info(f"Total selected: **{len(all_selected)}** symptoms")
    else:
        st.warning("No symptoms selected. Select at least one.")
    
    st.markdown("<div class='section-label'>Prediction</div>", unsafe_allow_html=True)
    
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        predict_btn = st.button("Run Prediction", use_container_width=True, key="predict_btn", help="Run AI analysis")
    with btn_col2:
        clear_btn = st.button("Clear Selection", use_container_width=True, key="clear_btn", help="Clear selection")
    
    if clear_btn:
        st.session_state.selected_symptoms = []
        st.session_state.current_prediction = None
        for symptom in feature_names:
            if symptom in st.session_state:
                st.session_state[symptom] = False
        st.rerun()
    
    if predict_btn:
        if not st.session_state.selected_symptoms:
            st.error("Select at least one symptom first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    model_manager = AdvancedModelManager()
                    model_manager.model = model_data.get("model")
                    model_manager.label_encoder = label_encoder
                    model_manager.feature_names = feature_names
                    model_manager.feature_importance = model_data.get("feature_importance", {})
                    
                    prediction = model_manager.predict_with_confidence(st.session_state.selected_symptoms)
                    st.session_state.current_prediction = prediction
                    st.success("Analysis complete.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display prediction results - Responsive
    if st.session_state.current_prediction:
        st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)
        
        prediction = st.session_state.current_prediction
        
        # Check urgency
        if prediction["requires_urgent_care"]:
            st.markdown("""
            <div class='urgent-alert'>
            <strong>Alert:</strong> Multiple symptoms detected. Consider consulting a healthcare professional.
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge
        UIComponents.render_confidence_gauge(prediction["confidence"], prediction["predicted_disease"])
        
        # Prediction breakdown
        st.markdown("<div class='section-label'>Analysis</div>", unsafe_allow_html=True)
        UIComponents.render_prediction_breakdown(prediction)
        
        st.markdown("<div class='section-label'>Other Possibilities</div>", unsafe_allow_html=True)
        UIComponents.render_alternatives_chart(prediction["alternatives"])
        
        if prediction["symptom_importance"]:
            st.markdown("<div class='section-label'>Symptom Impact</div>", unsafe_allow_html=True)
            UIComponents.render_symptom_importance(prediction["symptom_importance"])
        
        st.markdown("<div class='section-label'>Next Steps</div>", unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns([1, 1])
        
        with rec_col1:
            st.markdown("""
            **Next Steps:**
            1. Save or note this result
            2. Contact your healthcare provider
            3. Schedule an appointment
            4. Research the condition with trusted sources
            5. Follow professional medical advice
            """)
        
        with rec_col2:
            st.markdown("""
            **Important Notes:**
            - This is screening only
            - Not a professional diagnosis
            - Always seek expert opinion
            - Use for reference only
            - Emergency? Call 911
            """)


def page_analytics():
    """Analytics dashboard."""
    render_page_header(
        "Insights",
        "Analytics",
        "Review model confidence, alternative predictions, and symptom coverage after running an analysis.",
    )
    
    if not st.session_state.current_prediction:
        st.info("Make a prediction first to see analytics.")
        return
    
    prediction = st.session_state.current_prediction
    
    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.metric("Confidence", f"{prediction['confidence']:.1f}%")
    with met_col2:
        st.metric("Symptoms", prediction.get('symptom_count', 0))
    with met_col3:
        st.metric("Alternatives", len(prediction.get('alternatives', [])))
    
    st.markdown("<div class='section-label'>Predictions</div>", unsafe_allow_html=True)
    
    prob_data = [{
        "Disease": prediction["predicted_disease"],
        "Probability": prediction["confidence"]
    }]
    
    for alternative in prediction.get("alternatives", []):
        prob_data.append({
            "Disease": alternative.get("disease", "Unknown"),
            "Probability": alternative.get("confidence", 0)
        })
    
    if prob_data:
        df_prob = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
        fig = px.bar(
            df_prob,
            x="Probability",
            y="Disease",
            orientation="h",
            title="Model Probabilities",
            color="Probability",
            color_continuous_scale=["#dff7f3", "#0f766e"],
        )
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='section-label'>Summary</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"""
        **Analyzed:**
        • {prediction['symptom_count']} symptoms
        • {(prediction['symptom_count'] / 130) * 100:.0f}% coverage
        """)
    
    with col2:
        st.success(f"""
        **Status:**
        • Confidence: {prediction['confidence']:.1f}%
        • Algorithm: Ensemble
        """)


def page_history():
    """View conversation history."""
    render_page_header(
        "Timeline",
        "Conversation History",
        "Review previous messages from the current session.",
    )
    
    history = st.session_state.conversation_engine.get_history()
    
    if not history:
        st.info("No history yet. Start chatting.")
        return
    
    st.markdown(f"<div class='section-label'>{len(history)} Messages</div>", unsafe_allow_html=True)
    
    for i, message in enumerate(history, 1):
        role = html.escape(message["role"].title())
        with st.expander(f"{role} - Message {i}"):
            st.write(message["content"])


def page_about():
    """About page."""
    render_page_header(
        "System",
        "About This App",
        "A local healthcare screening assistant combining conversational symptom capture with ensemble machine learning.",
    )

    render_card_grid(
        [
            {
                "title": "Machine Learning",
                "body": "Random Forest and Gradient Boosting combine predictions with confidence scores and alternatives.",
            },
            {
                "title": "Conversation",
                "body": "The chatbot detects intent, extracts symptoms, and carries context into the analyzer.",
            },
            {
                "title": "Privacy",
                "body": "User accounts and chat history are saved locally inside the project folder.",
            },
        ]
    )

    st.markdown("<div class='section-label'>Important Notice</div>", unsafe_allow_html=True)
    st.warning(
        "This app is a screening aid, not medical advice. For severe symptoms or emergencies, contact emergency services or a qualified healthcare professional."
    )

    st.markdown("<div class='section-label'>Technology</div>", unsafe_allow_html=True)
    render_card_grid(
        [
            {"title": "Model", "body": "scikit-learn ensemble trained on symptom and disease data."},
            {"title": "Interface", "body": "Streamlit with responsive custom styling and Plotly charts."},
            {"title": "Security", "body": "PBKDF2-SHA256 password hashing for local user accounts."},
        ]
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    
    if not st.session_state.logged_in:
        page_login()
        return
    
    with st.sidebar:
        st.markdown(f"### {st.session_state.current_user}")
        
        selected = option_menu(
            menu_title="Menu",
            options=NAV_OPTIONS,
            icons=["speedometer2", "chat-dots", "search", "graph-up", "clock-history", "info-circle", "box-arrow-right"],
            menu_icon="list",
            default_index=current_page_index(),
            styles={
                "container": {
                    "padding": "0.25rem 0 !important",
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "#0f766e",
                    "font-size": "18px"
                },
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "3px 0px",
                    "border-radius": "8px",
                    "font-weight": "600",
                },
                "nav-link-selected": {
                    "background-color": "#0f766e",
                    "color": "white",
                    "border-radius": "8px"
                },
            }
        )
        route_to(selected)
        
        st.markdown("---")
        st.caption("v2.0 Professional Edition")
        
        if selected == "Logout":
            page_logout()
            return
    
    # Route to correct page
    page_map = {
        "Dashboard": page_dashboard,
        "Chat": page_chatbot,
        "Symptoms": page_symptom_analyzer,
        "Analytics": page_analytics,
        "History": page_history,
        "About": page_about,
    }
    
    if selected in page_map:
        page_map[selected]()


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
