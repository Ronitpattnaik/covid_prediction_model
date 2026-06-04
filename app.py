"""
================================================================================
    AI-BASED COVID-19 SEVERITY PREDICTION SYSTEM
    Modern Medical Dashboard using Streamlit & Machine Learning
================================================================================
Author: AI Development Team
Date: 2024
Description: Enterprise-grade COVID-19 severity prediction with advanced ML
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from datetime import datetime
import json
import time

warnings.filterwarnings('ignore')

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="COVID-19 Severity Prediction AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS & STYLING
# ================================================================================

def load_custom_css():
    """Load custom CSS for modern glassmorphism UI"""
    custom_css = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, rgba(15, 12, 41, 0.95), rgba(48, 43, 99, 0.95));
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Success Card - Green Glow */
        .success-card {
            background: rgba(52, 211, 153, 0.1);
            backdrop-filter: blur(10px);
            border: 2px solid #34d399;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 0 30px rgba(52, 211, 153, 0.5);
            animation: glow-green 2s ease-in-out infinite;
        }
        
        /* Danger Card - Red Glow */
        .danger-card {
            background: rgba(239, 68, 68, 0.1);
            backdrop-filter: blur(10px);
            border: 2px solid #ef4444;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
            animation: glow-red 2s ease-in-out infinite;
        }
        
        /* Glowing Animations */
        @keyframes glow-green {
            0%, 100% {
                box-shadow: 0 0 20px rgba(52, 211, 153, 0.4),
                            0 0 30px rgba(52, 211, 153, 0.3);
            }
            50% {
                box-shadow: 0 0 40px rgba(52, 211, 153, 0.6),
                            0 0 50px rgba(52, 211, 153, 0.4);
            }
        }
        
        @keyframes glow-red {
            0%, 100% {
                box-shadow: 0 0 20px rgba(239, 68, 68, 0.4),
                            0 0 30px rgba(239, 68, 68, 0.3);
            }
            50% {
                box-shadow: 0 0 40px rgba(239, 68, 68, 0.6),
                            0 0 50px rgba(239, 68, 68, 0.4);
            }
        }
        
        /* Stat Card Animation */
        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
        }
        
        .stat-card {
            animation: float 3s ease-in-out infinite;
        }
        
        /* Gradient Button */
        .gradient-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .gradient-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        }
        
        /* Sidebar Styling */
        .sidebar-content {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        
        /* Title Gradient */
        .gradient-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Loading Animation */
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .pulse {
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        /* Status Indicator */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        .status-online {
            background-color: #34d399;
        }
        
        .status-offline {
            background-color: #f97316;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

load_custom_css()

# ================================================================================
# SESSION STATE INITIALIZATION
# ================================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    if 'df' not in st.session_state:
        st.session_state.df = None

initialize_session_state()

# ================================================================================
# DATA LOADING & PREPROCESSING
# ================================================================================

@st.cache_resource
def load_data():
    """
    Load and preprocess the COVID-19 dataset
    - Handles missing values
    - Encodes categorical variables
    - Cleans non-numeric data
    """
    try:
        df = pd.read_csv('Cleaned-Data.csv')
        
        # Remove rows with non-numeric values in critical columns
        feature_columns = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat']
        
        # Convert features to numeric, coercing errors to NaN
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Find severity column (case-insensitive)
        severity_col = None
        for col in df.columns:
            if 'severity' in col.lower():
                severity_col = col
                break
        
        if severity_col is None:
            # If no severity column, use last column as target
            severity_col = df.columns[-1]
        
        # Convert severity to numeric if it's categorical
        if df[severity_col].dtype == 'object':
            # Map string values to numeric
            severity_mapping = {}
            unique_values = df[severity_col].unique()
            for i, val in enumerate(unique_values):
                if pd.notna(val):
                    severity_mapping[str(val).lower()] = i
            
            df[severity_col] = df[severity_col].astype(str).str.lower().map(severity_mapping)
        
        # Remove non-numeric severity values
        df[severity_col] = pd.to_numeric(df[severity_col], errors='coerce')
        
        # Drop rows with NaN values in critical columns
        df = df.dropna(subset=feature_columns + [severity_col])
        
        # Fill any remaining NaN values with median
        for col in feature_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df, feature_columns, severity_col
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# ================================================================================
# MACHINE LEARNING MODEL
# ================================================================================

@st.cache_resource
def train_model(df, feature_columns, severity_col):
    """
    Train Logistic Regression model with balanced class weights
    Returns: model, scaler, metrics dictionary
    """
    try:
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[severity_col].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Ensure all values are numeric
        X = X.astype(float)
        y = y.astype(float)

        y = (y > 0).astype(int)
       
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with balanced class weights
        model =  RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        report = classification_report(
            y_test, y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return model, scaler, metrics
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# ================================================================================
# PREDICTION FUNCTION
# ================================================================================

def make_prediction(model, scaler, feature_values, feature_columns):
    """
    Make prediction for given symptoms
    Returns: prediction (0/1), confidence percentage
    """
    try:
        # Create input array
        input_data = np.array([feature_values]).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability
        probability = model.predict_proba(input_scaled)[0]
        
        # Confidence is the maximum probability
        confidence = max(probability) * 100
        
        return int(prediction), confidence, probability
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# ================================================================================
# UI COMPONENTS
# ================================================================================

def create_stat_card(title, value, icon, color="blue"):
    """Create animated stat card"""
    color_map = {
        "blue": "#667eea",
        "green": "#34d399",
        "red": "#ef4444",
        "purple": "#a855f7"
    }
    
    card_html = f"""
    <div class="glass-card stat-card" style="border-left: 4px solid {color_map.get(color, '#667eea')};">
        <div style="font-size: 24px; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 14px; opacity: 0.8; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 28px; font-weight: bold; background: linear-gradient(135deg, #667eea, #764ba2); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>
    </div>
    """
    return card_html

def display_hero_section():
    """Display beautiful hero section on homepage"""
    hero_html = """
    <div style="text-align: center; margin: 40px 0; padding: 40px; 
                background: rgba(255, 255, 255, 0.05); border-radius: 20px;">
        <h1 style="font-size: 48px; margin-bottom: 20px;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏥 AI-Based COVID-19 Severity Prediction System
            </span>
        </h1>
        <p style="font-size: 20px; opacity: 0.9; margin-bottom: 30px;">
            Predict COVID risk and severity using Machine Learning and symptom analysis
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div style="background: rgba(52, 211, 153, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(52, 211, 153, 0.3);">
                <div style="font-size: 28px; margin-bottom: 10px;">🤖</div>
                <div style="font-size: 18px; font-weight: bold;">AI-Powered</div>
            </div>
            <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="font-size: 28px; margin-bottom: 10px;">⚡</div>
                <div style="font-size: 18px; font-weight: bold;">Real-Time</div>
            </div>
            <div style="background: rgba(168, 85, 247, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(168, 85, 247, 0.3);">
                <div style="font-size: 28px; margin-bottom: 10px;">📊</div>
                <div style="font-size: 18px; font-weight: bold;">Analytics</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

def display_risk_card(is_high_risk, confidence):
    """Display risk result card with glow effect"""
    if is_high_risk:
        card_html = f"""
        <div class="danger-card">
            <h2 style="color: #ef4444; margin-bottom: 20px; font-size: 32px;">⚠️ HIGH COVID RISK</h2>
            <p style="font-size: 18px; margin-bottom: 10px;">Confidence: <strong>{confidence:.1f}%</strong></p>
            <p style="opacity: 0.8; font-size: 14px;">⚕️ Please consult with a healthcare professional for further evaluation</p>
        </div>
        """
    else:
        card_html = f"""
        <div class="success-card">
            <h2 style="color: #34d399; margin-bottom: 20px; font-size: 32px;">✅ LOW COVID RISK</h2>
            <p style="font-size: 18px; margin-bottom: 10px;">Confidence: <strong>{confidence:.1f}%</strong></p>
            <p style="opacity: 0.8; font-size: 14px;">🎉 Continue following preventive measures</p>
        </div>
        """
    
    st.markdown(card_html, unsafe_allow_html=True)

# ================================================================================
# PAGE: HOME
# ================================================================================

def page_home():
    """Home page with statistics and system overview"""
    
    display_hero_section()
    
    st.markdown("---")
    
    # Load data for statistics
    df, feature_columns, severity_col = load_data()
    
    if df is not None and severity_col is not None:
        # Calculate statistics
        total_records = len(df)
        high_risk_count = int((df[severity_col] > df[severity_col].median()).sum())
        low_risk_count = total_records - high_risk_count
        
        # Create statistics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_stat_card(
                "Total Cases", 
                f"{total_records:,}", 
                "📊",
                "blue"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_stat_card(
                "High Risk", 
                f"{high_risk_count:,}", 
                "⚠️",
                "red"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_stat_card(
                "Low Risk", 
                f"{low_risk_count:,}", 
                "✅",
                "green"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_stat_card(
                "Risk Rate", 
                f"{(high_risk_count/total_records*100):.1f}%", 
                "📈",
                "purple"
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model performance metrics
        model, scaler, metrics = train_model(df, feature_columns, severity_col)
        
        if model is not None and metrics is not None:
            st.subheader("🤖 AI System Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%", "✅")
            
            with col2:
                st.metric("Status", "Online", "✅", delta_color="off")
            
            with col3:
                st.metric("Training Samples", f"{len(df)*0.8:.0f}", "✅")
            
            # Display confusion matrix
            st.markdown("---")
            st.subheader("📊 Model Evaluation Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix Heatmap
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    metrics['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=True,
                    ax=ax
                )
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
            
            with col2:
                # Classification Report
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)

# ================================================================================
# PAGE: PREDICTION
# ================================================================================

def page_prediction():
    """Interactive prediction page"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔮 COVID-19 Risk Prediction
        </h1>
        <p style="opacity: 0.8; font-size: 16px;">Enter your symptoms for AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df, feature_columns, severity_col = load_data()
    
    if df is None:
        st.error("Unable to load dataset. Please check the file path.")
        return
    
    model, scaler, metrics = train_model(df, feature_columns, severity_col)
    
    if model is None:
        st.error("Unable to train model. Please check your data.")
        return
    
    # Create prediction form
    st.markdown("""
    <div class="glass-card" style="padding: 30px; margin-bottom: 30px;">
        <h3 style="margin-bottom: 20px;">📋 Symptom Assessment</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    symptoms = {}
    
    with col1:
        st.markdown("### Symptom Severity (0-3)")
        symptoms['Fever'] = st.slider(
            "🌡️ Fever",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe"
        )
        
        symptoms['Tiredness'] = st.slider(
            "😴 Tiredness",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe"
        )
        
        symptoms['Dry-Cough'] = st.slider(
            "🫁 Dry Cough",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe"
        )
    
    with col2:
        symptoms['Difficulty-in-Breathing'] = st.slider(
            "💨 Difficulty in Breathing",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe"
        )
        
        symptoms['Sore-Throat'] = st.slider(
            "🗣️ Sore Throat",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button(
            "🚀 PREDICT",
            use_container_width=True,
            key="predict_button"
        )
    
    if predict_button:
        with st.spinner("🔄 AI is analyzing symptoms..."):
            time.sleep(1)  # Simulate processing
            
            # Prepare input
            feature_values = [symptoms[col] for col in feature_columns]
            
            # Make prediction
            prediction, confidence, probability = make_prediction(
                model, scaler, feature_values, feature_columns
            )
            
            if prediction is not None:
                # Determine risk level
                is_high_risk = prediction == 1
                
                # Display risk card
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                display_risk_card(is_high_risk, confidence)
                
                # Show probability distribution
                st.markdown("---")
                st.markdown("### 📈 Confidence Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Low Risk', 'High Risk'],
                            y=[probability[0]*100, probability[1]*100],
                            marker=dict(
                                color=['#34d399', '#ef4444']
                            ),
                            text=[f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"],
                            textposition='outside',
                            showlegend=False
                        )
                    ])
                    
                    fig.update_layout(
                        title="Probability Distribution",
                        xaxis_title="Risk Category",
                        yaxis_title="Probability (%)",
                        template="plotly_dark",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Gauge chart
                    fig = go.Figure(data=[
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Confidence Score"},
                            delta={'reference': 80},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#ef4444"},
                                    {'range': [50, 100], 'color': "#34d399"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        )
                    ])
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Medical recommendations
                st.markdown("---")
                st.markdown("### 🏥 Medical Recommendations")
                
                if is_high_risk:
                    recommendations = [
                        "🚨 Schedule an appointment with a healthcare provider immediately",
                        "🏥 Consider getting tested for COVID-19",
                        "🏠 Stay home and avoid close contact with others",
                        "💊 Follow doctor's advice on treatment",
                        "📞 Contact emergency services if symptoms worsen"
                    ]
                else:
                    recommendations = [
                        "✅ Continue monitoring your health",
                        "🛡️ Maintain preventive measures (masks, hand hygiene)",
                        "📊 Track your symptoms regularly",
                        "🏃 Stay active and maintain a healthy lifestyle",
                        "📱 Seek medical attention if symptoms develop"
                    ]
                
                for rec in recommendations:
                    st.info(rec)
                
                # Save to history
                prediction_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symptoms': symptoms,
                    'prediction': "High Risk" if is_high_risk else "Low Risk",
                    'confidence': confidence,
                    'probability': probability.tolist()
                }
                
                st.session_state.prediction_history.append(prediction_entry)
                
                # Download report
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate report
                    report_text = f"""
                    COVID-19 SEVERITY PREDICTION REPORT
                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    
                    SYMPTOMS ASSESSMENT:
                    {'='*50}
                    """
                    
                    for symptom, value in symptoms.items():
                        report_text += f"\n{symptom}: {value}/3"
                    
                    report_text += f"""
                    
                    PREDICTION RESULTS:
                    {'='*50}
                    Risk Level: {'HIGH RISK' if is_high_risk else 'LOW RISK'}
                    Confidence: {confidence:.2f}%
                    
                    PROBABILITY DISTRIBUTION:
                    {'='*50}
                    Low Risk Probability: {probability[0]*100:.2f}%
                    High Risk Probability: {probability[1]*100:.2f}%
                    
                    DISCLAIMER:
                    This prediction is for informational purposes only and should not be 
                    used as a substitute for professional medical advice. Always consult 
                    with a healthcare provider for accurate diagnosis and treatment.
                    """
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"covid_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

# ================================================================================
# PAGE: ANALYTICS
# ================================================================================

def page_analytics():
    """Advanced analytics and visualizations"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📊 Advanced Analytics
        </h1>
        <p style="opacity: 0.8; font-size: 16px;">Explore data insights and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, feature_columns, severity_col = load_data()
    
    if df is None:
        st.error("Unable to load dataset.")
        return
    
    # Symptom distribution
    st.markdown("### 📈 Symptom Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of symptoms
        fig = go.Figure()
        
        for col in feature_columns:
            fig.add_trace(go.Histogram(
                x=df[col],
                name=col,
                opacity=0.75
            ))
        
        fig.update_layout(
            title="Symptoms Distribution",
            xaxis_title="Severity Level",
            yaxis_title="Frequency",
            barmode='overlay',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = go.Figure()
        
        for col in feature_columns:
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                showlegend=True
            ))
        
        fig.update_layout(
            title="Symptoms Range Analysis",
            yaxis_title="Severity Level",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("---")
    st.markdown("### 🔗 Correlation Analysis")
    
    correlation_matrix = df[feature_columns + [severity_col]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        text=correlation_matrix.values.round(2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_dark",
        height=500,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Severity comparison
    st.markdown("---")
    st.markdown("### ⚠️ Severity Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for risk distribution
        risk_counts = (df[severity_col] > df[severity_col].median()).value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Low Risk', 'High Risk'],
                values=[risk_counts[False], risk_counts[True]],
                marker=dict(colors=['#34d399', '#ef4444']),
                hole=0.4
            )
        ])
        
        fig.update_layout(
            title="COVID Risk Distribution",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average symptoms by risk level
        df['risk_level'] = df[severity_col] > df[severity_col].median()
        
        avg_symptoms = df.groupby('risk_level')[feature_columns].mean()
        
        fig = go.Figure()
        
        for idx, risk in enumerate([False, True]):
            fig.add_trace(go.Bar(
                name='High Risk' if risk else 'Low Risk',
                x=feature_columns,
                y=avg_symptoms.loc[risk],
                marker=dict(color='#ef4444' if risk else '#34d399')
            ))
        
        fig.update_layout(
            title="Average Symptoms by Risk Level",
            xaxis_title="Symptoms",
            yaxis_title="Average Severity",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("---")
    st.markdown("### 📋 Statistical Summary")
    
    st.dataframe(
        df[feature_columns + [severity_col]].describe().round(2),
        use_container_width=True
    )

# ================================================================================
# PAGE: HISTORY
# ================================================================================

def page_history():
    """Prediction history page"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📜 Prediction History
        </h1>
        <p style="opacity: 0.8; font-size: 16px;">Your previous predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("📭 No prediction history yet. Make a prediction to get started!")
        return
    
    # Display history
    st.markdown("### 📊 Recent Predictions")
    
    # Create history dataframe
    history_data = []
    
    for entry in st.session_state.prediction_history:
        history_data.append({
            'Timestamp': entry['timestamp'],
            'Risk Level': entry['prediction'],
            'Confidence': f"{entry['confidence']:.1f}%",
            'Fever': entry['symptoms']['Fever'],
            'Tiredness': entry['symptoms']['Tiredness'],
            'Dry-Cough': entry['symptoms']['Dry-Cough'],
            'Difficulty-in-Breathing': entry['symptoms']['Difficulty-in-Breathing'],
            'Sore-Throat': entry['symptoms']['Sore-Throat']
        })
    
    history_df = pd.DataFrame(history_data[::-1])  # Reverse for newest first
    
    st.dataframe(history_df, use_container_width=True)
    
    # Confidence trend chart
    if len(st.session_state.prediction_history) > 1:
        st.markdown("---")
        st.markdown("### 📈 Confidence Trend")
        
        trend_data = {
            'Timestamp': [entry['timestamp'] for entry in st.session_state.prediction_history],
            'Confidence': [entry['confidence'] for entry in st.session_state.prediction_history]
        }
        
        fig = go.Figure(data=[
            go.Scatter(
                x=trend_data['Timestamp'],
                y=trend_data['Confidence'],
                mode='lines+markers',
                marker=dict(color='#667eea', size=10),
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence (%)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export history
    st.markdown("---")
    st.markdown("### 📥 Export History")
    
    if st.button("Download History as JSON"):
        json_data = json.dumps(st.session_state.prediction_history, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    if st.button("Download History as CSV"):
        csv_data = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Clear history
    if st.button("🗑️ Clear History", help="Clear all prediction history"):
        st.session_state.prediction_history = []
        st.success("✅ History cleared!")
        st.rerun()

# ================================================================================
# PAGE: ABOUT
# ================================================================================

def page_about():
    """About and information page"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ℹ️ About This System
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("### 📋 Project Overview")
    
    overview_text = """
    The **AI-Based COVID-19 Severity Prediction System** is an advanced machine learning 
    application designed to assess the risk and severity of COVID-19 based on reported symptoms. 
    
    This system uses state-of-the-art algorithms to provide real-time predictions that can help 
    users understand their potential health risk and take appropriate medical action.
    """
    
    st.info(overview_text)
    
    # ML Algorithm Details
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Algorithm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm: Logistic Regression**
        - Type: Supervised Learning
        - Model: Binary Classification
        - Solver: LBFGS
        - Max Iterations: 1000
        - Class Weight: Balanced
        """)
    
    with col2:
        st.markdown("""
        **Feature Scaling: StandardScaler**
        - Normalizes features to have zero mean and unit variance
        - Essential for Logistic Regression
        - Applied to both training and prediction data
        """)
    
    # Dataset Information
    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    
    df, feature_columns, severity_col = load_data()
    
    if df is not None:
        dataset_info = f"""
        **Dataset Name:** Cleaned-Data.csv
        
        **Total Records:** {len(df):,}
        
        **Input Features ({len(feature_columns)}):**
        """
        
        st.markdown(dataset_info)
        
        for i, feature in enumerate(feature_columns, 1):
            st.write(f"  {i}. {feature}")
        
        st.markdown(f"**Target Variable:** {severity_col}")
        st.markdown(f"**Data Range:** 0-3 (None, Mild, Moderate, Severe)")
    
    # Technology Stack
    st.markdown("---")
    st.markdown("### 💻 Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Backend**
        - Python 3.x
        - Pandas
        - NumPy
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn
        - StandardScaler
        - LogisticRegression
        """)
    
    with col3:
        st.markdown("""
        **Frontend & Visualization**
        - Streamlit
        - Plotly
        - Matplotlib
        - Seaborn
        """)
    
    # Developer Section
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Section")
    
    st.markdown("""
    **Developed By:** AI Development Team
    
    **Version:** 1.0.0
    
    **Last Updated:** 2024
    
    **Contact:** [Your Contact Information]
    """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("### ⚖️ Disclaimer")
    
    disclaimer_text = """
    ⚠️ **IMPORTANT DISCLAIMER**
    
    This AI-Based COVID-19 Severity Prediction System is intended for **informational purposes only** 
    and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    - The predictions generated by this system are based on machine learning models trained on 
      historical data and may not be 100% accurate.
    
    - Always consult with a qualified healthcare professional for medical advice.
    
    - In case of emergency, please seek immediate medical attention from your healthcare provider 
      or call emergency services.
    
    - The developers and creators of this system assume no responsibility for any adverse outcomes 
      resulting from the use of this application.
    
    - By using this system, you agree to the above disclaimer and acknowledge that you understand 
      these limitations.
    """
    
    st.warning(disclaimer_text)
    
    # Features List
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    features = [
        "🎯 Real-time COVID-19 risk assessment",
        "🤖 Advanced machine learning predictions",
        "📊 Comprehensive analytics and visualizations",
        "📜 Prediction history tracking",
        "📥 Download reports and history",
        "🌙 Modern dark theme with glassmorphism UI",
        "⚡ Fast and responsive interface",
        "🔒 Client-side processing (no data stored)"
    ]
    
    for feature in features:
        st.write(f"✅ {feature}")

# ================================================================================
# SIDEBAR NAVIGATION
# ================================================================================

def create_sidebar():
    """Create sidebar navigation"""
    
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏥 COVID-19 AI
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        st.markdown("### 📱 Navigation")
        
        pages = {
            "🏠 Home": "home",
            "🔮 Prediction": "prediction",
            "📊 Analytics": "analytics",
            "📜 History": "history",
            "ℹ️ About": "about"
        }
        
        page = st.radio(
            "Select Page:",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("### 🤖 Model Status")
        
        df, feature_columns, severity_col = load_data()
        
        if df is not None:
            model, scaler, metrics = train_model(df, feature_columns, severity_col)
            
            if model is not None and metrics is not None:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("Status:")
                
                with col2:
                    st.markdown("""
                    <span class="status-indicator status-online"></span>
                    <span style="opacity: 0.8;">Online</span>
                    """, unsafe_allow_html=True)
                
                st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                st.metric("Dataset Size", f"{len(df):,} records")
        
        st.markdown("---")
        
        # System Time
        st.markdown("### 🕐 System Information")
        
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        st.write(f"**Time:** {current_time}")
        st.write(f"**Date:** {current_date}")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; opacity: 0.6; font-size: 12px; margin-top: 30px;">
            <p>🏥 AI-Based COVID-19 Severity Prediction System</p>
            <p>v1.0.0 | © 2024</p>
        </div>
        """, unsafe_allow_html=True)
        
        return pages.get(page, "home")

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main application function"""
    
    # Create sidebar and get selected page
    selected_page = create_sidebar()
    
    # Route to selected page
    if selected_page == "home":
        page_home()
    elif selected_page == "prediction":
        page_prediction()
    elif selected_page == "analytics":
        page_analytics()
    elif selected_page == "history":
        page_history()
    elif selected_page == "about":
        page_about()

# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()
