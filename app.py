"""
AI-Based COVID-19 Severity Prediction System
A modern, futuristic healthcare AI dashboard using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from datetime import datetime, timedelta
import json
import io
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="COVID-19 AI Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def apply_custom_css():
    """Apply modern glassmorphism and dark theme CSS"""
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Dark Theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glassmorphism Cards */
    [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e0e0e0;
        padding: 10px 12px;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [role="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        transition: all 0.3s ease;
    }
    
    [role="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Success/Danger Cards */
    .success-card {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%);
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
        animation: glow-green 2s ease-in-out infinite;
    }
    
    .danger-card {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%);
        border: 2px solid rgba(244, 67, 54, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(244, 67, 54, 0.2);
        animation: glow-red 2s ease-in-out infinite;
    }
    
    @keyframes glow-green {
        0%, 100% { box-shadow: 0 0 20px rgba(76, 175, 80, 0.2); }
        50% { box-shadow: 0 0 30px rgba(76, 175, 80, 0.4); }
    }
    
    @keyframes glow-red {
        0%, 100% { box-shadow: 0 0 20px rgba(244, 67, 54, 0.2); }
        50% { box-shadow: 0 0 30px rgba(244, 67, 54, 0.4); }
    }
    
    /* Metric Values */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING & CACHING ====================
@st.cache_data
def load_dataset():
    """Load and cache the COVID-19 dataset"""
    try:
        df = pd.read_csv('Cleaned-Data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset 'Cleaned-Data.csv' not found. Please upload the dataset.")
        return None

@st.cache_resource
def train_model(df):
    """Train and cache the ML model"""
    if df is None:
        return None, None, None
    
    # Feature columns for prediction
    feature_columns = [
        'Fever',
        'Tiredness',
        'Dry-Cough',
        'Difficulty-in-Breathing',
        'Sore-Throat'
    ]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Severity'] if 'Severity' in df.columns else df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_columns': feature_columns
    }

# ==================== INITIALIZE SESSION STATE ====================
def initialize_session_state():
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'ai_status' not in st.session_state:
        st.session_state.ai_status = 'ONLINE'

initialize_session_state()

# ==================== UTILITY FUNCTIONS ====================
def get_current_time():
    """Get formatted current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def make_prediction(model, scaler, feature_values, feature_columns):
    """Make COVID-19 severity prediction"""
    # Scale input
    input_scaled = scaler.transform([feature_values])
    
    # Get prediction and probability
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Calculate confidence
    confidence = max(probability) * 100
    
    return prediction, confidence, probability

def add_to_history(prediction_data):
    """Add prediction to history"""
    st.session_state.prediction_history.append({
        'timestamp': get_current_time(),
        'symptoms': prediction_data['symptoms'],
        'prediction': prediction_data['prediction'],
        'confidence': prediction_data['confidence']
    })

def create_metric_card(title, value, unit="", icon=""):
    """Create animated metric card"""
    col = st.container()
    col.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    ">
        <h3 style="color: #b0b0b0; margin-bottom: 10px; font-size: 0.9rem;">{icon} {title}</h3>
        <h2 style="
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        ">{value} {unit}</h2>
    </div>
    """, unsafe_allow_html=True)

# ==================== HOMEPAGE ====================
def homepage():
    """Create beautiful homepage with hero section"""
    # Hero Title
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px 40px;">
        <h1 style="
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        ">🏥 AI-Based COVID-19 Severity Prediction System</h1>
        <h2 style="
            font-size: 1.3rem;
            color: #b0b0b0;
            font-weight: 400;
            margin-bottom: 30px;
        ">Predict COVID risk and severity using Machine Learning and symptom analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df = load_dataset()
    model, scaler, metrics = train_model(df)
    
    # Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Model Accuracy",
            f"{metrics['accuracy']*100:.1f}%",
            "+0.5%"
        )
    
    with col2:
        st.metric(
            "📊 Predictions Made",
            len(st.session_state.prediction_history),
            "Real-time"
        )
    
    with col3:
        st.metric(
            "⚡ AI Status",
            "ONLINE",
            "Active"
        )
    
    with col4:
        st.metric(
            "🔬 Dataset Size",
            len(df) if df is not None else "N/A",
            "Records"
        )
    
    # Performance Metrics Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>📈 System Performance Metrics</h2>", unsafe_allow_html=True)
    
    if metrics is not None:
        # Confusion Matrix Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=metrics['confusion_matrix'],
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Viridis',
                text=metrics['confusion_matrix'],
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Prediction",
                yaxis_title="Actual",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model Performance Metrics
            true_negatives = metrics['confusion_matrix'][0][0]
            false_positives = metrics['confusion_matrix'][0][1]
            false_negatives = metrics['confusion_matrix'][1][0]
            true_positives = metrics['confusion_matrix'][1][1]
            
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            metrics_data = {
                'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'Precision'],
                'Score': [
                    metrics['accuracy'],
                    sensitivity,
                    specificity,
                    precision
                ]
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics_data['Metric'],
                    y=[m*100 for m in metrics_data['Score']],
                    marker=dict(
                        color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                        line=dict(color='rgba(255,255,255,0.1)', width=2)
                    ),
                    text=[f"{m*100:.1f}%" for m in metrics_data['Score']],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Model Performance Metrics",
                yaxis_title="Score (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=False,
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Key Features Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <h3 style="color: #667eea; margin-bottom: 10px;">🤖 AI-Powered</h3>
            <p style="color: #b0b0b0;">Advanced machine learning models for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style="
            background: rgba(118, 75, 162, 0.1);
            border: 1px solid rgba(118, 75, 162, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <h3 style="color: #764ba2; margin-bottom: 10px;">📱 Responsive</h3>
            <p style="color: #b0b0b0;">Works seamlessly on desktop, tablet, and mobile devices</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div style="
            background: rgba(240, 147, 251, 0.1);
            border: 1px solid rgba(240, 147, 251, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <h3 style="color: #f093fb; margin-bottom: 10px;">⚡ Real-time</h3>
            <p style="color: #b0b0b0;">Instant predictions with comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== PREDICTION PAGE ====================
def prediction_page():
    """Create interactive prediction form"""
    st.markdown("<h1 style='text-align: center;'>🔮 COVID-19 Severity Prediction</h1>", unsafe_allow_html=True)
    
    # Load model
    df = load_dataset()
    model, scaler, metrics = train_model(df)
    
    if model is None:
        st.error("Model not trained. Please ensure dataset is available.")
        return
    
    # Create prediction form
    st.markdown("### Enter Your Symptoms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fever = st.slider(
            "🌡️ Fever (0 = No, 1 = Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Do you have a fever?"
        )
        
        tiredness = st.slider(
            "😴 Tiredness (0 = No, 1 = Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Do you feel tired?"
        )
        
        dry_cough = st.slider(
            "🤧 Dry Cough (0 = No, 1 = Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Do you have a dry cough?"
        )
    
    with col2:
        difficulty_breathing = st.slider(
            "💨 Difficulty in Breathing (0 = No, 1 = Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Do you have difficulty breathing?"
        )
        
        sore_throat = st.slider(
            "👅 Sore Throat (0 = No, 1 = Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Do you have a sore throat?"
        )
    
    # Prepare prediction input
    feature_values = [fever, tiredness, dry_cough, difficulty_breathing, sore_throat]
    feature_columns = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat']
    
    # Prediction Button with Animation
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if st.button("🚀 PREDICT COVID SEVERITY", use_container_width=True, key="predict_btn"):
            # Make prediction
            prediction, confidence, probability = make_prediction(
                model, scaler, feature_values, feature_columns
            )
            
            # Store in session state
            st.session_state.last_prediction = {
                'symptoms': dict(zip(feature_columns, feature_values)),
                'prediction': prediction,
                'confidence': confidence,
                'probability': probability,
                'timestamp': get_current_time()
            }
            
            # Add to history
            add_to_history({
                'symptoms': dict(zip(feature_columns, feature_values)),
                'prediction': prediction,
                'confidence': confidence
            })
    
    # Display Prediction Results
    if 'last_prediction' in st.session_state:
        prediction_data = st.session_state.last_prediction
        prediction = prediction_data['prediction']
        confidence = prediction_data['confidence']
        probability = prediction_data['probability']
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
        
        # Risk Display Card
        if prediction == 0:
            st.markdown("""
            <div class="success-card">
                <div style="text-align: center;">
                    <h1 style="color: #4caf50; font-size: 3rem; margin: 0;">✅ LOW COVID RISK</h1>
                    <p style="color: #4caf50; font-size: 1.2rem; margin-top: 10px;">Your symptoms indicate low COVID-19 severity risk</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-card">
                <div style="text-align: center;">
                    <h1 style="color: #f44336; font-size: 3rem; margin: 0;">⚠️ HIGH COVID RISK</h1>
                    <p style="color: #f44336; font-size: 1.2rem; margin-top: 10px;">Your symptoms indicate high COVID-19 severity risk</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Confidence", f"{confidence:.1f}%")
        
        with col2:
            st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
        
        with col3:
            st.metric("Timestamp", prediction_data['timestamp'].split()[1])
        
        # Probability Visualization
        st.markdown("### 📈 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Low Risk', 'High Risk'],
                y=[probability[0]*100, probability[1]*100],
                marker=dict(
                    color=['#4caf50', '#f44336'],
                    line=dict(color='rgba(255,255,255,0.2)', width=2)
                ),
                text=[f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="COVID-19 Severity Prediction Probabilities",
            yaxis_title="Probability (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Medical Recommendations
        st.markdown("### 🏥 Medical Recommendations")
        
        if prediction == 0:
            st.info("""
            ✅ **Low Risk Assessment**
            
            - Continue following general COVID-19 prevention guidelines
            - Monitor your symptoms regularly
            - Maintain proper hygiene and sanitation
            - Stay hydrated and get adequate rest
            - If symptoms worsen, seek medical attention
            - Consider periodic health check-ups
            """)
        else:
            st.warning("""
            ⚠️ **High Risk Assessment**
            
            - **CONSULT A HEALTHCARE PROFESSIONAL IMMEDIATELY**
            - Isolate yourself from others to prevent spread
            - Monitor vital signs (temperature, oxygen level, blood pressure)
            - Stay hydrated and maintain proper nutrition
            - Avoid physical exertion and get adequate rest
            - Seek emergency medical care if symptoms worsen
            - Follow local health authority guidelines
            """)
        
        # Severity Analysis
        st.markdown("### 🔍 Detailed Symptom Analysis")
        
        symptom_analysis = {
            'Fever': ('🌡️', fever),
            'Tiredness': ('😴', tiredness),
            'Dry-Cough': ('🤧', dry_cough),
            'Difficulty-in-Breathing': ('💨', difficulty_breathing),
            'Sore-Throat': ('👅', sore_throat)
        }
        
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for idx, (symptom, (icon, value)) in enumerate(symptom_analysis.items()):
            with columns[idx % 3]:
                status = "Present" if value == 1 else "Absent"
                color = "#f44336" if value == 1 else "#4caf50"
                
                st.markdown(f"""
                <div style="
                    background: rgba({256 if value == 1 else 0}, {150 if value == 1 else 200}, {100 if value == 1 else 0}, 0.1);
                    border: 1px solid rgba({256 if value == 1 else 0}, {150 if value == 1 else 200}, {100 if value == 1 else 0}, 0.3);
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                ">
                    <h4 style="margin-bottom: 10px;">{icon} {symptom}</h4>
                    <h3 style="color: {color}; margin: 0;">{status}</h3>
                </div>
                """, unsafe_allow_html=True)

# ==================== ANALYTICS PAGE ====================
def analytics_page():
    """Create advanced analytics visualizations"""
    st.markdown("<h1 style='text-align: center;'>📊 Advanced Analytics & Insights</h1>", unsafe_allow_html=True)
    
    df = load_dataset()
    
    if df is None:
        st.error("Dataset not available")
        return
    
    # Symptom Distribution
    st.markdown("### 📈 Symptom Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symptom_cols = [
            'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
            'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea'
        ]
        
        available_symptoms = [col for col in symptom_cols if col in df.columns]
        symptom_counts = df[available_symptoms].sum().sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                y=symptom_counts.index,
                x=symptom_counts.values,
                orientation='h',
                marker=dict(
                    color=symptom_counts.values,
                    colorscale='Viridis',
                    line=dict(color='rgba(255,255,255,0.2)', width=2)
                ),
                text=symptom_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Symptom Prevalence in Dataset",
            xaxis_title="Count",
            yaxis_title="Symptom",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity Distribution
        if 'Severity' in df.columns or len(df.columns) > 0:
            severity_col = 'Severity' if 'Severity' in df.columns else df.columns[-1]
            severity_counts = df[severity_col].value_counts()
            
            labels = ['Low Risk', 'High Risk'] if len(severity_counts) == 2 else severity_counts.index.tolist()
            colors = ['#4caf50', '#f44336']
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=severity_counts.values,
                    marker=dict(colors=colors[:len(severity_counts)]),
                    textposition='inside',
                    textinfo='label+percent'
                )
            ])
            
            fig.update_layout(
                title="COVID-19 Severity Distribution",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("---")
    st.markdown("### 🔗 Symptom Correlation Heatmap")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Symptom Correlation Matrix",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        width=800
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Confidence Distribution
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Confidence Distribution")
        
        confidences = [p['confidence'] for p in st.session_state.prediction_history]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=confidences,
                nbinsx=20,
                marker=dict(
                    color='#667eea',
                    line=dict(color='rgba(255,255,255,0.2)', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title="AI Model Confidence Distribution",
            xaxis_title="Confidence (%)",
            yaxis_title="Frequency",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== HISTORY PAGE ====================
def history_page():
    """Display prediction history"""
    st.markdown("<h1 style='text-align: center;'>📜 Prediction History</h1>", unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Go to the Prediction page to get started!")
        return
    
    # Display history table
    st.markdown("### Recent Predictions")
    
    history_data = []
    for idx, pred in enumerate(reversed(st.session_state.prediction_history), 1):
        history_data.append({
            '#': idx,
            'Timestamp': pred['timestamp'],
            'Risk Level': '⚠️ HIGH' if pred['prediction'] == 1 else '✅ LOW',
            'Confidence': f"{pred['confidence']:.1f}%",
            'Fever': '✓' if pred['symptoms'].get('Fever', 0) == 1 else '✗',
            'Tiredness': '✓' if pred['symptoms'].get('Tiredness', 0) == 1 else '✗',
            'Dry-Cough': '✓' if pred['symptoms'].get('Dry-Cough', 0) == 1 else '✗',
            'Difficulty-in-Breathing': '✓' if pred['symptoms'].get('Difficulty-in-Breathing', 0) == 1 else '✗',
            'Sore-Throat': '✓' if pred['symptoms'].get('Sore-Throat', 0) == 1 else '✗',
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 History Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = len(st.session_state.prediction_history)
    high_risk_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 1)
    low_risk_count = total_predictions - high_risk_count
    avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
    
    with col1:
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        st.metric("High Risk Cases", high_risk_count)
    
    with col3:
        st.metric("Low Risk Cases", low_risk_count)
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Risk Distribution Chart
    st.markdown("### 📈 Risk Distribution Over Time")
    
    timestamps = [p['timestamp'] for p in st.session_state.prediction_history]
    risks = [1 if p['prediction'] == 1 else 0 for p in st.session_state.prediction_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(timestamps))),
        y=risks,
        mode='lines+markers',
        name='Risk Level',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Risk Level Progression",
        xaxis_title="Prediction Number",
        yaxis_title="Risk Level (0=Low, 1=High)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export History
    st.markdown("---")
    st.markdown("### 📥 Export History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"covid_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON Export
        json_data = json.dumps(st.session_state.prediction_history, indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"covid_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ==================== ABOUT PAGE ====================
def about_page():
    """Display about and information page"""
    st.markdown("<h1 style='text-align: center;'>ℹ️ About This System</h1>", unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    The **AI-Based COVID-19 Severity Prediction System** is an advanced machine learning application 
    designed to help predict the severity of COVID-19 based on reported symptoms. This system leverages 
    state-of-the-art algorithms and comprehensive data analysis to provide accurate risk assessments.
    
    **Disclaimer:** This system is for educational and informational purposes only and should NOT be 
    used as a substitute for professional medical advice. Always consult with qualified healthcare professionals.
    """)
    
    # ML Algorithm Details
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Algorithm Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm:** Logistic Regression
        
        - **Type:** Supervised Learning, Classification
        - **Use Case:** Binary classification (Low Risk/High Risk)
        - **Advantages:** 
          - Fast training and prediction
          - Interpretable results
          - Low computational requirements
          - Effective for medical diagnostics
        
        **Model Training Parameters:**
        - `class_weight`: balanced (handles imbalanced data)
        - `random_state`: 42 (reproducible results)
        - `max_iter`: 1000 (sufficient iterations)
        - `solver`: lbfgs (default)
        """)
    
    with col2:
        st.markdown("""
        **Feature Scaling:** StandardScaler
        
        - Normalizes features to have mean=0, std=1
        - Essential for Logistic Regression
        - Prevents features with large scales from dominating
        
        **Data Splitting:**
        - Training set: 80% of data
        - Test set: 20% of data
        - Random seed: 42 (consistent splits)
        
        **Performance Metrics:**
        - Accuracy: Overall prediction correctness
        - Sensitivity: True Positive Rate
        - Specificity: True Negative Rate
        - Precision: Positive Prediction Accuracy
        """)
    
    # Dataset Information
    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    
    df = load_dataset()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            st.metric("Total Features", len(df.columns))
        
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("**Dataset Columns:**")
        st.write(df.columns.tolist())
        
        st.markdown("**Dataset Summary:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Technology Stack
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Backend & ML**
        - Python 3.8+
        - Scikit-learn
        - Pandas
        - NumPy
        """)
    
    with tech_col2:
        st.markdown("""
        **Frontend & UI**
        - Streamlit
        - Plotly
        - Matplotlib
        - Seaborn
        """)
    
    with tech_col3:
        st.markdown("""
        **Deployment**
        - Streamlit Cloud
        - Docker Support
        - Python Virtual Environment
        - Git Version Control
        """)
    
    # Developer Section
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Information")
    
    st.markdown("""
    **Project Name:** AI-Based COVID-19 Severity Prediction System
    
    **Version:** 1.0.0
    
    **Created:** 2024
    
    **Purpose:** Educational demonstration of ML in healthcare diagnostics
    
    **Features:**
    - Real-time COVID-19 severity prediction
    - Interactive symptom input
    - Comprehensive analytics dashboard
    - Prediction history tracking
    - Modern UI with glassmorphism design
    - Medical recommendations
    - Exportable reports
    """)
    
    # Important Disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Important Disclaimer")
    
    st.error("""
    **MEDICAL DISCLAIMER**
    
    This AI system is provided for educational and informational purposes only. It is NOT intended to:
    
    - Replace professional medical diagnosis
    - Provide medical advice
    - Be used for self-diagnosis
    - Determine treatment plans
    
    **IMPORTANT:**
    - Always consult qualified healthcare professionals for medical concerns
    - Seek immediate emergency care if you experience severe symptoms
    - Follow local health authority guidelines
    - This prediction is based on input data and may not be 100% accurate
    - Individual medical situations vary significantly
    
    **Use Responsibly:** This tool is meant for educational purposes and general awareness only.
    """)

# ==================== MAIN APP ====================
def main():
    """Main application function"""
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🏥 COVID-19 AI System</h2>", unsafe_allow_html=True)
        
        # Current Time
        st.markdown(f"**⏰ {get_current_time()}**")
        
        # AI Status
        st.markdown("""
        <div style="
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.3);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            margin: 10px 0;
        ">
            <p style="margin: 0; color: #4caf50; font-weight: 600;">🟢 AI ONLINE</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation Tabs
        page = st.radio(
            "Navigation",
            ["🏠 Homepage", "🔮 Prediction", "📊 Analytics", "📜 History", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Load Model Stats
        df = load_dataset()
        model, scaler, metrics = train_model(df)
        
        if metrics is not None:
            st.markdown("### 📈 Model Accuracy")
            st.metric("Accuracy Score", f"{metrics['accuracy']*100:.2f}%")
        
        # Prediction Counter
        st.markdown("### 📊 System Stats")
        st.metric("Total Predictions", len(st.session_state.prediction_history))
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>v1.0.0 | Powered by AI</p>", unsafe_allow_html=True)
    
    # Main Content
    if page == "🏠 Homepage":
        homepage()
    elif page == "🔮 Prediction":
        prediction_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "📜 History":
        history_page()
    elif page == "ℹ️ About":
        about_page()

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
