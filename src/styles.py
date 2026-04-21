"""
Styles for BigFeat AutoML Streamlit App
"""

CUSTOM_CSS = """
<style>
    /* Main theme colors - Purple/Green/Orange */
    :root {
        --primary-color: #9b59b6;
        --secondary-color: #27ae60;
        --accent-color: #e67e22;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --dark-bg: #2c3e50;
        --light-bg: #ecf0f1;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(142, 68, 173, 0.4);
        border: 3px solid #9b59b6;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(155, 89, 182, 0.8);
        letter-spacing: 2px;
    }
    
    .main-header p {
        margin: 0.8rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 500;
    }
    
    /* Metric cards - Different colors */
    .metric-card {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(142, 68, 173, 0.3);
        margin: 0.5rem 0;
        border: 2px solid #9b59b6;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(142, 68, 173, 0.5);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 15px rgba(155, 89, 182, 0.6);
    }
    
    .metric-card p {
        margin: 0.8rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
    }
    
    /* Success metric card */
    .metric-card-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border: 2px solid #2ecc71;
    }
    
    /* Warning metric card */
    .metric-card-warning {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        border: 2px solid #f39c12;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(142, 68, 173, 0.5);
        border: 3px solid #9b59b6;
    }
    
    .prediction-card h2 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0 0 25px rgba(155, 89, 182, 0.7);
    }
    
    /* Alert badges */
    .alert-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .alert-excellent {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }
    
    .alert-good {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.1) 0%, rgba(142, 68, 173, 0.15) 100%);
        border-left: 5px solid #9b59b6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .info-box h3 {
        color: #8e44ad;
        margin-top: 0;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(39, 174, 96, 0.15) 100%);
        border-left: 5px solid #2ecc71;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(230, 126, 34, 0.15) 100%);
        border-left: 5px solid #f39c12;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        color: white;
        border: 2px solid #9b59b6;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(155, 89, 182, 0.5);
        border: 2px solid #2ecc71;
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #9b59b6 0%, #2ecc71 100%);
    }
</style>
"""
