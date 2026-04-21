"""
UI Components for BigFeat AutoML
"""

import streamlit as st


def render_header():
    """Render application header"""
    st.markdown("""
        <div class="main-header">
            <h1>🚀 BigFeat AutoML System</h1>
            <p>Advanced Feature Engineering & Automated Machine Learning</p>
            <p style="font-size: 1rem; margin-top: 0.8rem; opacity: 0.9;">
                Scalable Feature Engineering | Interpretable Models | High Performance
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; color: white;">
                <h1 style="font-size: 4rem; margin: 0;">🚀</h1>
                <h2 style="margin: 1rem 0; color: #9b59b6; font-weight: 800;">BigFeat AutoML</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Navigation")
        
        if st.button("Problem Statement", key="nav_home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("Data Preprocessing", key="nav_preprocessing", use_container_width=True):
            st.session_state.page = 'data_preprocessing'
            st.rerun()
        
        if st.button("Data Explorer", key="nav_data", use_container_width=True):
            st.session_state.page = 'data'
            st.rerun()
        
        if st.button("Feature Engineering", key="nav_fe", use_container_width=True):
            st.session_state.page = 'feature_engineering'
            st.rerun()
        
        if st.button("Model Training", key="nav_model", use_container_width=True):
            st.session_state.page = 'model_training'
            st.rerun()
        
        if st.button("Results & Analytics", key="nav_results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🎯 System Status")
        
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            if st.session_state.X_train is not None:
                st.metric("Train Samples", st.session_state.X_train.shape[0])
                st.metric("Features", st.session_state.X_train.shape[1])
        else:
            st.warning("⚠️ No Data Loaded")
        
        if st.session_state.X_train_engineered is not None:
            st.success("✅ Features Engineered")
            st.metric("Engineered Features", st.session_state.X_train_engineered.shape[1])
        
        if st.session_state.best_model is not None:
            st.success("✅ Model Trained")
            if st.session_state.best_score:
                st.metric("CV F1 Score", f"{st.session_state.best_score:.4f}")
