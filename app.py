"""
BigFeat AutoML - Streamlit Application
Advanced Feature Engineering and Automated Machine Learning System
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from src.styles import CUSTOM_CSS
from src.ui_components import render_header, render_sidebar
from src.pages import (
    render_home_page,
    render_data_preprocessing_page,
    render_data_page,
    render_feature_engineering_page,
    render_model_training_page,
    render_results_page
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="BigFeat AutoML System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main application"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    
    if 'fe_model' not in st.session_state:
        st.session_state.fe_model = None
    
    if 'X_train_engineered' not in st.session_state:
        st.session_state.X_train_engineered = None
    
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    
    if 'best_score' not in st.session_state:
        st.session_state.best_score = None
    
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    
    render_header()
    render_sidebar()
    
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'data_preprocessing':
        render_data_preprocessing_page()
    elif st.session_state.page == 'data':
        render_data_page()
    elif st.session_state.page == 'feature_engineering':
        render_feature_engineering_page()
    elif st.session_state.page == 'model_training':
        render_model_training_page()
    elif st.session_state.page == 'results':
        render_results_page()

if __name__ == "__main__":
    main()
