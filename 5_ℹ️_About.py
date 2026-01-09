"""
About Page - Project information and documentation
"""

import streamlit as st
import sys
import os

# Page config
st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

# Title
st.title("ℹ️ About This Project")

st.markdown("---")

# Project overview
st.markdown("## 🎯 Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Garbage Classification Using Deep Learning
    
    This project uses **deep learning** and **computer vision** to automatically classify waste 
    into different categories, helping to automate and improve recycling processes.
    
    **Key Features:**
    - 🤖 AI-powered image classification
    - 📊 Real-time performance analytics
    - ♻️ Recycling recommendations
    - 📈 Environmental impact tracking
    - 🌐 User-friendly web interface
    
    **Use Cases:**
    - Smart recycling bins
    - Municipal waste management
    - Educational tools
    - Environmental monitoring
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=250)

st.markdown("---")

# Technology stack
st.markdown("## 🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🧠 Machine Learning
    - **TensorFlow / Keras**
    - **Transfer Learning**
    - **MobileNetV2 / EfficientNet**
    - **Data Augmentation**
    """)

with col2:
    st.markdown("""
    ### 💻 Development
    - **Python 3.8+**
    - **NumPy / Pandas**
    - **Scikit-learn**
    - **OpenCV**
    """)

with col3:
    st.markdown("""
    ### 🌐 Web Application
    - **Streamlit**
    - **Matplotlib / Seaborn**
    - **Plotly**
    - **PIL (Pillow)**
    """)

st.markdown("---")

# Model architecture
st.markdown("## 🏗️ Model Architecture")

st.markdown("""
The garbage classification system uses **Transfer Learning** with pre-trained models:

1. **Base Model:** MobileNetV2 (trained on ImageNet)
2. **Custom Layers:** Dense layers for garbage classification
3. **Output:** 6 categories (Cardboard, Glass, Metal, Paper, Plastic, Trash)

**Training Process:**
- **Dataset:** ~2,500 labeled images
- **Augmentation:** Rotation, zoom, shift, flip
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, Precision, Recall, F1-Score
""")

# Architecture diagram (text-based)
st.code("""
Input Image (224x224x3)
        ↓
MobileNetV2 Base Model
        ↓
Global Average Pooling
        ↓
Dense Layer (512 neurons) + ReLU + Dropout
        ↓
Dense Layer (256 neurons) + ReLU + Dropout
        ↓
Output Layer (6 classes) + Softmax
        ↓
Predicted Category + Confidence Score
""", language="text")

st.markdown("---")

# Dataset
st.markdown("## 📊 Dataset")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📁 Categories
    
    1. **Cardboard** 📦
       - Boxes, packaging, corrugated cardboard
    
    2. **Glass** 🥤
       - Bottles, jars, glass containers
    
    3. **Metal** 🔩
       - Cans, aluminum, steel containers
    
    4. **Paper** 📄
       - Documents, newspapers, magazines
    
    5. **Plastic** 🛍️
       - Bottles, containers, plastic bags
    
    6. **Trash** 🗑️
       - Non-recyclable general waste
    """)

with col2:
    st.markdown("""
    ### 📈 Dataset Statistics
    
    - **Total Images:** ~2,500
    - **Training Set:** 70% (~1,750 images)
    - **Validation Set:** 15% (~375 images)
    - **Test Set:** 15% (~375 images)
    
    **Data Sources:**
    - TrashNet Dataset
    - Kaggle Garbage Classification
    - Custom collected images
    
    **Preprocessing:**
    - Resize to 224×224 pixels
    - Normalize pixel values (0-1)
    - Data augmentation applied
    """)

st.markdown("---")

# Performance metrics
st.markdown("## 📈 Model Performance")

# Load metadata if available
metadata_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'models', 'metadata', 'best_model_metadata.json'
)

if os.path.exists(metadata_path):
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metrics = metadata.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
    with col4:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
else:
    st.info("Train a model to see performance metrics here.")

st.markdown("---")

# How it works
st.markdown("## 🔄 How It Works")

steps = [
    {
        'icon': '📸',
        'title': '1. Image Upload',
        'description': 'User uploads or captures an image of waste'
    },
    {
        'icon': '🔍',
        'title': '2. Preprocessing',
        'description': 'Image is resized and normalized'
    },
    {
        'icon': '🧠',
        'title': '3. AI Analysis',
        'description': 'Deep learning model analyzes the image'
    },
    {
        'icon': '📊',
        'title': '4. Classification',
        'description': 'Model predicts category and confidence score'
    },
    {
        'icon': '♻️',
        'title': '5. Recommendation',
        'description': 'Provides recycling guidelines'
    }
]

cols = st.columns(len(steps))

for col, step in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h1>{step['icon']}</h1>
            <h4>{step['title']}</h4>
            <p style="color: #666;">{step['description']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Future improvements
st.markdown("## 🚀 Future Improvements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📱 Features
    - [ ] Mobile app version
    - [ ] Real-time camera integration
    - [ ] Multi-object detection
    - [ ] Batch image processing
    - [ ] User accounts and history
    - [ ] API endpoints for integration
    """)

with col2:
    st.markdown("""
    ### 🧠 Model Enhancements
    - [ ] Expand to more categories
    - [ ] Improve accuracy with more data
    - [ ] Faster inference (model optimization)
    - [ ] Handle partial/occluded objects
    - [ ] Multi-language support
    - [ ] Regional waste guidelines
    """)

st.markdown("---")

# Environmental impact
st.markdown("## 🌍 Environmental Impact")

st.info("""
**Why This Matters:**

- ♻️ **Proper recycling reduces landfill waste by up to 75%**
- 🌳 **Recycling 1 ton of paper saves 17 trees**
- 💧 **Recycling saves water and energy**
- 🌎 **Reduces greenhouse gas emissions**
- 🔄 **Promotes circular economy**

By accurately classifying waste, we can:
- Increase recycling rates
- Reduce contamination in recycling streams
- Educate people about proper waste disposal
- Track environmental impact
""")

st.markdown("---")

# Team and contact
st.markdown("## 👥 Project Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📧 Contact
    
    **Project Repository:**  
    [GitHub - Garbage Classification](https://github.com/yourusername/garbage-classification)
    
    **Documentation:**  
    See `docs/` folder in the project repository
    
    **Issues & Support:**  
    [GitHub Issues](https://github.com/yourusername/garbage-classification/issues)
    """)

with col2:
    st.markdown("""
    ### 📚 Resources
    
    **Dataset Sources:**
    - [TrashNet Dataset](https://github.com/garythung/trashnet)
    - [Kaggle Waste Classification](https://www.kaggle.com/datasets)
    
    **Technologies:**
    - [TensorFlow](https://www.tensorflow.org/)
    - [Streamlit](https://streamlit.io/)
    - [MobileNetV2](https://arxiv.org/abs/1801.04381)
    """)

st.markdown("---")

# License and credits
st.markdown("## 📜 License & Credits")

st.markdown("""
### License
This project is licensed under the **MIT License**.

### Credits
- **Deep Learning Framework:** TensorFlow / Keras
- **Web Framework:** Streamlit
- **Dataset:** TrashNet, Kaggle Community
- **Icons:** Flaticon, Font Awesome

### Acknowledgments
Special thanks to the open-source community and all contributors to the datasets and libraries used in this project.
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>♻️ Garbage Classification System v1.0</p>
    <p>Built with ❤️ using TensorFlow and Streamlit</p>
    <p><em>Making the world greener, one classification at a time</em> 🌍</p>
</div>
""", unsafe_allow_html=True)
