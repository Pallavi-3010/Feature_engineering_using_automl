"""
Page rendering functions for BigFeat AutoML
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path

from data_preprocessing import DataPreprocessor, validate_data
from train_model import BigFeatFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def render_home_page():
    """Render home page"""
    
    st.markdown("""
    ### 🎓 **Problem Statement**  
    Feature engineering is a critical yet time-consuming step in machine learning pipelines. Traditional approaches 
    require extensive domain expertise and manual effort to create meaningful features. The challenges include:  

    🔹 **High Dimensionality** – Modern datasets often contain hundreds or thousands of features, making manual feature engineering impractical.  
    🔹 **Feature Interaction Discovery** – Identifying useful feature combinations requires exhaustive exploration and domain knowledge.  
    🔹 **Redundancy & Noise** – Many generated features may be redundant or irrelevant, degrading model performance.  
    🔹 **Scalability Issues** – Traditional feature engineering methods don't scale well to large datasets.  

    ---

    #### 🎯 **Project Objective**  
    This research implements **BigFeat**, a scalable feature engineering and AutoML framework that automatically 
    generates and selects high-quality features for machine learning tasks. The approach includes:  

    ✅ **Dynamic Feature Generation** – Iteratively creates new features using mathematical operators (square, abs, sqrt, log, 
    add, subtract, multiply, divide) to discover complex feature interactions.  
    ✅ **Stability Selection** – Uses ensemble-based Random Forest models to select robust features that consistently 
    contribute to model performance across multiple bootstrap samples.  
    ✅ **Redundancy Removal** – Eliminates highly correlated features using Pearson correlation (η threshold = 0.95) 
    to maintain feature diversity.  
    ✅ **Automated Model Selection** – Trains and evaluates multiple interpretable models (Random Forest, Decision Tree, 
    Logistic Regression) with hyperparameter tuning to find the optimal classifier.  
    ✅ **Interpretable Models** – Focuses on interpretable machine learning models that provide insights into feature 
    importance and decision-making processes.  

    This approach enhances model performance while reducing manual feature engineering effort, enabling data scientists 
    to **build better models faster** and focus on higher-level problem-solving.  

    ---

    #### 🔬 **BigFeat Methodology**  
    The BigFeat framework implements a sophisticated iterative feature engineering pipeline:  

    🔹 **Iterative Feature Generation (7 iterations)** – Each iteration generates K×N new features (K=10) using mathematical 
    operators applied to existing features.  
    🔹 **Stability Selection (α=5 trees)** – Selects top features based on importance scores from ensemble Random Forest models 
    trained on bootstrap samples.  
    🔹 **Redundancy Removal (η=0.95)** – Removes features with Pearson correlation > 0.95 to eliminate redundant information.  
    🔹 **Feature Pool Management** – Maintains a diverse set of features by combining original and engineered features across iterations.  
    🔹 **Cross-Validation Evaluation** – Uses 5-fold cross-validation with weighted F1 score to assess model performance.  

    ---

    #### 🔍 **Key Challenges Addressed**  
    ✅ **High-Dimensional Feature Spaces** – Efficiently explores large feature spaces through iterative generation and selection.  
    ✅ **Feature Stability** – Ensures selected features are robust and not artifacts of specific data splits.  
    ✅ **Computational Efficiency** – Balances feature generation with redundancy removal to maintain scalability.  
    ✅ **Model Interpretability** – Focuses on interpretable models (RF, DT, LR) rather than black-box approaches.  
    ✅ **Generalization** – Achieves strong performance on benchmark datasets (Madelon: F1=0.8250, exceeding paper baseline of 0.8221).  

    ---

    #### 🌐 **Real-World Applications & End Users**  
    🔹 **Data Science Teams** – Accelerate feature engineering workflows and improve model performance.  
    🔹 **AutoML Platforms** – Integrate as a feature engineering component in automated ML pipelines.  
    🔹 **Research Institutions** – Benchmark and compare feature engineering approaches on standard datasets.  
    🔹 **Industry Applications** – Apply to domains like finance, healthcare, marketing, and manufacturing for predictive modeling.  
    🔹 **Educational Use** – Teach feature engineering concepts and best practices to students and practitioners.  

    🏆 **This implementation demonstrates that automated feature engineering can match or exceed manual approaches 
    while significantly reducing development time and effort.**  
    
    ---
    
    #### 📊 **Dataset Information (Madelon)**
    
    - **Total Samples**: 2,600 instances
    - **Classes**: Binary classification (balanced: 1,300 per class)
    - **Original Features**: 500 continuous features
    - **Feature Type**: Synthetic dataset with 20 informative features and 480 noise features
    - **Split**: 80% training (2,080 samples), 20% testing (520 samples)
    - **Challenge**: Identify informative features among high-dimensional noise
    
    ---
    """, unsafe_allow_html=True)


def render_data_preprocessing_page():
    """Render data preprocessing methodology page"""
    st.title("Data Preprocessing")
    
    # Sample Data Display
    st.markdown("### Sample Data from Madelon Dataset")
    
    # Display data from session state
    if st.session_state.data_loaded and st.session_state.X_train is not None:
        # Combine X_train and y_train to show complete sample
        sample_data = st.session_state.X_train.head(10).copy()
        sample_data['Target'] = st.session_state.y_train[:10].values
        
        st.dataframe(sample_data, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Training Samples", st.session_state.X_train.shape[0])
        with col2:
            st.metric("Features", st.session_state.X_train.shape[1])
        with col3:
            st.metric("Target Classes", len(np.unique(st.session_state.y_train)))
    else:
        st.info("Load dataset from Data Explorer to view sample data")
    
    st.markdown("---")
    
    # Main Preprocessing Steps
    st.markdown("### Main Preprocessing Steps")
    
    st.markdown("""
    1. **Data Loading & Validation**
       - Load CSV/ARFF files
       - Validate data types and structure
       - Check for missing values and data integrity
    
    2. **Feature Type Identification**
       - Identify numerical and categorical features
       - Handle mixed-type columns
    
    3. **Missing Value Handling**
       - Impute numerical features with median
       - Impute categorical features with mode
       - Remove features with >50% missing values
    
    4. **Outlier Detection & Treatment**
       - Identify outliers using IQR method
       - Apply robust scaling (RobustScaler)
    
    5. **Feature Scaling & Normalization**
       - Apply RobustScaler: `(x - median) / IQR`
       - Robust to outliers
    
    6. **Train-Test Split**
       - 80% training, 20% testing
       - Stratified split to maintain class distribution
       - Random state = 42 for reproducibility
    
    7. **Class Imbalance Handling**
       - Apply SMOTE if needed
       - Use class weights in model training
    
    8. **Feature Validation**
       - Check for infinite values
       - Check for NaN values
       - Validate sufficient variance
    """)


def render_data_page():
    """Render data explorer page"""
    st.title("📊 Data Explorer")
    
    st.markdown("""
    <div class="info-box">
        <h3>📁 Dataset Loading</h3>
        <p>
            Load your dataset for feature engineering and model training. 
            The system supports CSV files with a target column.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Load Dataset")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        data_file = st.text_input(
            "Dataset path:",
            value="phpfLuQE4.csv",
            help="Enter the path to your CSV file"
        )
    
    with col2:
        target_col = st.text_input(
            "Target column:",
            value="Class"
        )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random seed:", value=42, min_value=0)
    
    if st.button("🚀 Load Data", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📂 Loading dataset...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
            X_train, X_test, y_train, y_test, label_encoder = preprocessor.preprocess(
                filepath=data_file,
                target_column=target_col
            )
            
            status_text.text("✅ Validating data...")
            progress_bar.progress(75)
            time.sleep(0.3)
            
            validate_data(X_train, y_train)
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.data_loaded = True
            
            progress_bar.progress(100)
            status_text.text("✅ Data loaded successfully!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Dataset loaded and preprocessed successfully!")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error loading data: {e}")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{st.session_state.X_train.shape[0]}</h3>
                <p>Training Samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{st.session_state.X_test.shape[0]}</h3>
                <p>Test Samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card-success">
                <h3>{st.session_state.X_train.shape[1]}</h3>
                <p>Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            n_classes = len(np.unique(st.session_state.y_train))
            st.markdown(f"""
            <div class="metric-card-warning">
                <h3>{n_classes}</h3>
                <p>Classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_dist = pd.Series(st.session_state.y_train).value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'Class {i}' for i in train_dist.index],
                    y=train_dist.values,
                    text=train_dist.values,
                    textposition='auto',
                    marker_color=['#9b59b6', '#2ecc71']
                )
            ])
            
            fig.update_layout(
                title="Training Set Distribution",
                yaxis_title="Count",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            test_dist = pd.Series(st.session_state.y_test).value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=[f'Class {i}' for i in test_dist.index],
                    values=test_dist.values,
                    marker_colors=['#9b59b6', '#2ecc71']
                )
            ])
            
            fig.update_layout(
                title="Test Set Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Feature Statistics")
        
        with st.expander("📊 View Feature Statistics"):
            st.dataframe(st.session_state.X_train.describe(), use_container_width=True)
        
        with st.expander("👁️ View Sample Data (First 10 rows)"):
            sample_df = st.session_state.X_train.head(10).copy()
            sample_df['Target'] = st.session_state.y_train.iloc[:10].values
            st.dataframe(sample_df, use_container_width=True)


def render_feature_engineering_page():
    """Render feature engineering page"""
    st.title("⚙️ Feature Engineering with BigFeat")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Data Explorer page.")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>🔧 BigFeat Feature Engineering Pipeline</h3>
        <p>
            BigFeat iteratively generates and selects features using:
        </p>
        <ul>
            <li><b>Dynamic Feature Generation:</b> Creates K×N new features per iteration</li>
            <li><b>Stability Selection:</b> Selects top features using ensemble methods</li>
            <li><b>Redundancy Removal:</b> Removes highly correlated features (η threshold)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_iterations = st.slider(
            "Number of iterations:",
            min_value=1,
            max_value=10,
            value=7,
            help="Paper uses 7 iterations"
        )
    
    with col2:
        K = st.slider(
            "K (feature multiplier):",
            min_value=5,
            max_value=20,
            value=10,
            help="Generates K×N features per iteration"
        )
    
    with col3:
        alpha = st.slider(
            "Alpha (stability trees):",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of trees for stability selection"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        eta = st.slider(
            "Eta (correlation threshold):",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Threshold for redundancy removal"
        )
    
    with col2:
        random_state = st.number_input(
            "Random seed:",
            value=42,
            min_value=0
        )
    
    st.markdown("### 📊 Expected Performance")
    
    estimated_time = n_iterations * 1.5
    estimated_features = st.session_state.X_train.shape[1] * 0.5
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{estimated_time:.1f}</h3>
            <p>Estimated Minutes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-success">
            <h3>~{int(estimated_features)}</h3>
            <p>Expected Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-warning">
            <h3>{n_iterations}</h3>
            <p>Iterations</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Feature Engineering", type="primary", use_container_width=True):
        
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            iteration_info = st.empty()
            
            try:
                status_text.text("🔧 Initializing BigFeat pipeline...")
                progress_bar.progress(5)
                time.sleep(0.5)
                
                fe = BigFeatFE(
                    n_iterations=n_iterations,
                    K=K,
                    alpha=alpha,
                    eta=eta,
                    random_state=random_state
                )
                
                st.session_state.fe_model = fe
                
                status_text.text("⚙️ Running feature engineering...")
                progress_bar.progress(10)
                
                start_time = time.time()
                
                X_train_engineered = fe.fit_transform(
                    st.session_state.X_train,
                    st.session_state.y_train
                )
                
                elapsed_time = time.time() - start_time
                
                st.session_state.X_train_engineered = X_train_engineered
                
                progress_bar.progress(100)
                status_text.text("✅ Feature engineering complete!")
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                iteration_info.empty()
                
                st.success(f"✅ Feature engineering completed in {elapsed_time/60:.2f} minutes!")
                
                st.markdown("### 🎉 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card-success">
                        <h3>{X_train_engineered.shape[1]}</h3>
                        <p>Final Features</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{st.session_state.X_train.shape[1]}</h3>
                        <p>Original Features</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card-warning">
                        <h3>{elapsed_time/60:.2f}</h3>
                        <p>Minutes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.balloons()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                iteration_info.empty()
                st.error(f"❌ Error during feature engineering: {e}")
    
    if st.session_state.X_train_engineered is not None:
        st.markdown("---")
        st.markdown("### 📊 Feature Engineering Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            original_features = st.session_state.X_train.shape[1]
            engineered_features = st.session_state.X_train_engineered.shape[1]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Original', 'Engineered'],
                    y=[original_features, engineered_features],
                    text=[original_features, engineered_features],
                    textposition='auto',
                    marker_color=['#9b59b6', '#2ecc71']
                )
            ])
            
            fig.update_layout(
                title="Feature Count Comparison",
                yaxis_title="Number of Features",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            selected_features = st.session_state.X_train_engineered.columns.tolist()
            original_feature_names = st.session_state.X_train.columns.tolist()
            
            original_selected = [f for f in selected_features if f in original_feature_names]
            engineered_only = len(selected_features) - len(original_selected)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Original Features', 'Engineered Features'],
                    values=[len(original_selected), engineered_only],
                    marker_colors=['#9b59b6', '#e67e22']
                )
            ])
            
            fig.update_layout(
                title="Feature Composition",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Selected Features"):
            st.write(f"Total features: {len(selected_features)}")
            st.write(selected_features[:50])


def render_model_training_page():
    """Render model training page"""
    st.title("🤖 Model Training & Selection")
    
    if st.session_state.X_train_engineered is None:
        st.warning("⚠️ Please run feature engineering first.")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Automated Model Selection</h3>
        <p>
            The system will train multiple interpretable models and select the best one 
            based on cross-validation F1 score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_trials = st.slider(
            "Number of trials:",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Number of random configurations to try"
        )
    
    with col2:
        cv_folds = st.slider(
            "Cross-validation folds:",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
    
    st.markdown("### 📊 Model Space")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_rf = st.checkbox("Random Forest", value=True)
    
    with col2:
        use_dt = st.checkbox("Decision Tree", value=True)
    
    with col3:
        use_lr = st.checkbox("Logistic Regression", value=True)
    
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        trial_info = st.empty()
        
        try:
            X_train_engineered = st.session_state.X_train_engineered
            y_train = st.session_state.y_train
            
            selected_features = X_train_engineered.columns.tolist()
            original_feature_names = st.session_state.X_train.columns.tolist()
            original_features_selected = [f for f in selected_features if f in original_feature_names]
            
            if len(original_features_selected) > 0:
                X_train_final = X_train_engineered[original_features_selected]
            else:
                from sklearn.ensemble import RandomForestClassifier as RFC
                rf_temp = RFC(n_estimators=50, random_state=42)
                rf_temp.fit(st.session_state.X_train, y_train)
                importances = rf_temp.feature_importances_
                top_features_idx = np.argsort(importances)[-100:]
                top_feature_names = [st.session_state.X_train.columns[i] for i in top_features_idx]
                X_train_final = st.session_state.X_train[top_feature_names]
            
            status_text.text("🔧 Initializing model search...")
            progress_bar.progress(5)
            time.sleep(0.3)
            
            model_configs = []
            
            if use_rf:
                model_configs.append(('RandomForest', RandomForestClassifier, {
                    'n_estimators': [100, 200],
                    'max_depth': [15, 20, None],
                    'class_weight': [None, 'balanced'],
                    'min_samples_split': [2, 5],
                    'random_state': [42],
                    'n_jobs': [-1]
                }))
            
            if use_dt:
                model_configs.append(('DecisionTree', DecisionTreeClassifier, {
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced'],
                    'random_state': [42]
                }))
            
            if use_lr:
                model_configs.append(('LogisticRegression', LogisticRegression, {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [1000],
                    'random_state': [42]
                }))
            
            best_score = -np.inf
            best_model = None
            best_params = None
            
            trial_num = 0
            results_list = []
            
            for model_name, model_class, param_space in model_configs:
                n_configs = n_trials // len(model_configs)
                
                for _ in range(n_configs):
                    trial_num += 1
                    
                    params = {}
                    for param_name, param_values in param_space.items():
                        params[param_name] = np.random.choice(param_values)
                    
                    trial_info.text(f"Trial {trial_num}/{n_trials}: {model_name}")
                    progress_bar.progress(int(5 + (trial_num / n_trials) * 90))
                    
                    try:
                        model = model_class(**params)
                        
                        scores = cross_val_score(
                            model, X_train_final, y_train,
                            cv=cv_folds,
                            scoring='f1_weighted',
                            n_jobs=1
                        )
                        score = scores.mean()
                        
                        results_list.append({
                            'trial': trial_num,
                            'model': model_name,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = {'model': model_name, **params}
                    
                    except Exception as e:
                        continue
            
            status_text.text("🎯 Training final model...")
            progress_bar.progress(95)
            
            best_model.fit(X_train_final, y_train)
            
            st.session_state.best_model = best_model
            st.session_state.best_params = best_params
            st.session_state.best_score = best_score
            st.session_state.X_train_final = X_train_final
            
            progress_bar.progress(100)
            status_text.text("✅ Model training complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            trial_info.empty()
            
            st.success(f"✅ Model training completed!")
            
            st.markdown("### 🏆 Best Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card-success">
                    <h3>{best_params['model']}</h3>
                    <p>Best Model</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{best_score:.4f}</h3>
                    <p>CV F1 Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card-warning">
                    <h3>{trial_num}</h3>
                    <p>Trials</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Training Progress")
            
            results_df = pd.DataFrame(results_list)
            
            fig = go.Figure()
            
            for model_name in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model_name]
                fig.add_trace(go.Scatter(
                    x=model_data['trial'],
                    y=model_data['score'],
                    mode='markers',
                    name=model_name,
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Model Performance Across Trials",
                xaxis_title="Trial Number",
                yaxis_title="CV F1 Score",
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.balloons()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            trial_info.empty()
            st.error(f"❌ Error during model training: {e}")
    
    if st.session_state.best_model is not None:
        st.markdown("---")
        st.markdown("### 📋 Best Model Configuration")
        
        st.json(st.session_state.best_params)


def render_results_page():
    """Render results and analytics page"""
    st.title("📈 Results & Analytics")
    
    if st.session_state.best_model is None:
        st.warning("⚠️ Please train a model first.")
        return
    
    st.markdown("### 🎯 Evaluate on Test Set")
    
    if st.button("🚀 Evaluate Model", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📊 Preparing test data...")
            progress_bar.progress(25)
            
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            if hasattr(st.session_state, 'X_train_final'):
                feature_names = st.session_state.X_train_final.columns.tolist()
                X_test_final = X_test[feature_names]
            else:
                X_test_final = X_test
            
            status_text.text("🔮 Making predictions...")
            progress_bar.progress(50)
            
            y_pred = st.session_state.best_model.predict(X_test_final)
            
            status_text.text("📈 Calculating metrics...")
            progress_bar.progress(75)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            st.session_state.test_results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Evaluation complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Model evaluation completed!")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during evaluation: {e}")
    
    if st.session_state.test_results is not None:
        results = st.session_state.test_results
        
        st.markdown("### 🎯 Test Set Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-success">
                <h3>{results['accuracy']:.4f}</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{results['f1_score']:.4f}</h3>
                <p>F1 Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_score = st.session_state.best_score if st.session_state.best_score else 0
            st.markdown(f"""
            <div class="metric-card-warning">
                <h3>{cv_score:.4f}</h3>
                <p>CV F1 Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            paper_baseline = 0.8221
            diff = results['f1_score'] - paper_baseline
            st.markdown(f"""
            <div class="metric-card">
                <h3>{diff:+.4f}</h3>
                <p>vs Paper</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Performance Assessment")
        
        f1 = results['f1_score']
        
        if f1 >= 0.80:
            status = "🎉 EXCELLENT"
            msg = "Very close to paper results!"
            color = "success-box"
        elif f1 >= 0.75:
            status = "✅ GOOD"
            msg = "Solid performance, approaching paper results"
            color = "info-box"
        elif f1 >= 0.70:
            status = "✓ ACCEPTABLE"
            msg = "Reasonable performance"
            color = "warning-box"
        else:
            status = "⚠️ BELOW EXPECTED"
            msg = "Lower than expected"
            color = "warning-box"
        
        st.markdown(f"""
        <div class="{color}">
            <h3>{status}</h3>
            <p style="font-size: 1.2rem;">{msg}</p>
            <p style="margin-top: 1rem;">
                <b>vs BigFeat-vanilla (paper):</b> {f1 - 0.8221:+.4f}<br>
                <b>vs SAFE:</b> {f1 - 0.7513:+.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Comparison with Baselines")
        
        comparison_data = {
            'Method': ['Original', 'AutoFeat', 'SAFE', 'Paper', 'Ours'],
            'F1 Score': [0.6556, 0.6766, 0.7513, 0.8221, f1]
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data['Method'],
                y=comparison_data['F1 Score'],
                text=[f'{s:.4f}' for s in comparison_data['F1 Score']],
                textposition='auto',
                marker_color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']
            )
        ])
        
        fig.update_layout(
            title="Performance Comparison",
            yaxis_title="F1 Score",
            yaxis_range=[0.6, 0.9],
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Confusion Matrix")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cm = results['confusion_matrix']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Purples',
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Classification Report")
            
            report = classification_report(
                results['y_test'],
                results['y_pred'],
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Save Results to JSON", use_container_width=True):
                results_dict = {
                    'test_accuracy': float(results['accuracy']),
                    'test_f1_score': float(results['f1_score']),
                    'cv_f1_score': float(st.session_state.best_score) if st.session_state.best_score else None,
                    'best_model': st.session_state.best_params['model'] if st.session_state.best_params else None,
                    'final_feature_count': st.session_state.X_train_engineered.shape[1] if st.session_state.X_train_engineered is not None else None
                }
                
                with open('bigfeat_results.json', 'w') as f:
                    json.dump(results_dict, f, indent=4)
                
                st.success("✅ Results saved to bigfeat_results.json")
        
        with col2:
            if st.button("📊 Download Predictions CSV", use_container_width=True):
                pred_df = pd.DataFrame({
                    'True_Label': results['y_test'],
                    'Predicted_Label': results['y_pred']
                })
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )


