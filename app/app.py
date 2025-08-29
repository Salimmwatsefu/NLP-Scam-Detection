import streamlit as st
from components import (
    render_css, render_hero_section, render_footer,
    render_message_scanner_tab, render_safety_education_tab,
    render_batch_analysis_tab, render_dataset_explorer_tab
)
from utils import load_models

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="SMS Scam Shield | Advanced Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Render CSS
render_css()

# Render Hero Section
render_hero_section()

# Load models at startup
lr_model, xgb_model, tfidf_vectorizer = load_models()
if lr_model is not None:
    st.success("✅ **AI Models Loaded Successfully** - Ready to protect you from scams!")
else:
    st.error("❌ **Model Loading Failed** - Please ensure model files exist in the correct directories and try again.")

# Enhanced tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Message Scanner", "🎓 Safety Education", "📊 Batch Analysis", "📂 Dataset Explorer"])

with tab1:
    render_message_scanner_tab(lr_model, xgb_model, tfidf_vectorizer)

with tab2:
    render_safety_education_tab()

with tab3:
    render_batch_analysis_tab(lr_model, xgb_model, tfidf_vectorizer)

with tab4:
    render_dataset_explorer_tab()