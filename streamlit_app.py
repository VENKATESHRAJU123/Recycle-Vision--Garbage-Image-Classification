"""
Garbage Classification - Streamlit Web Application
Main entry point for the multi-page app
"""

import streamlit as st
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))

# Page configuration
st.set_page_config(
    page_title="Garbage Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">♻️ Smart Garbage Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Waste Sorting for a Sustainable Future</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=150)
    st.markdown("## Navigation")
    st.info("👈 Select a page from the sidebar to get started!")
    
    st.markdown("---")
    
    st.markdown("### About This Project")
    st.markdown("""
    This application uses **Deep Learning** to automatically classify waste into different categories:
    
    - 📦 Cardboard
    - 🥤 Glass
    - 🔩 Metal
    - 📄 Paper
    - 🛍️ Plastic
    - 🗑️ Trash
    
    **Technology Stack:**
    - TensorFlow/Keras
    - Streamlit
    - Python
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Accuracy</h3>
        <h2 style="color: #2ecc71;">~85-95%</h2>
        <p>High accuracy classification</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Speed</h3>
        <h2 style="color: #3498db;">< 1 sec</h2>
        <p>Real-time predictions</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Categories</h3>
        <h2 style="color: #e74c3c;">6 Types</h2>
        <p>Comprehensive coverage</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Features section
st.markdown("## 🌟 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔍 Image Classification
    - Upload waste images
    - Get instant predictions
    - View confidence scores
    - See top-3 predictions
    
    ### 📊 Model Performance
    - View accuracy metrics
    - Analyze confusion matrix
    - Per-class performance
    - Training history charts
    """)

with col2:
    st.markdown("""
    ### 📈 Analytics Dashboard
    - Real-time statistics
    - Prediction history
    - Usage analytics
    - Performance trends
    
    ### 💡 Smart Insights
    - Recycling recommendations
    - Environmental impact
    - Educational resources
    - Tips for waste reduction
    """)

st.markdown("---")

# How it works
st.markdown("## 🚀 How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center;">
        <h1>📸</h1>
        <h4>1. Upload Image</h4>
        <p>Take or upload a photo of waste item</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h1>🤖</h1>
        <h4>2. AI Analysis</h4>
        <p>Deep learning model analyzes the image</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center;">
        <h1>✅</h1>
        <h4>3. Classification</h4>
        <p>Get category and confidence score</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center;">
        <h1>♻️</h1>
        <h4>4. Recycle</h4>
        <p>Dispose waste in correct bin</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Call to action
st.markdown("## 🎯 Get Started")
st.info("👈 Navigate to **🔍 Classify Image** from the sidebar to start classifying waste!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>Garbage Classification System v1.0 | Powered by TensorFlow & Streamlit</p>
    <p>🌍 Building a sustainable future, one classification at a time</p>
</div>
""", unsafe_allow_html=True)
