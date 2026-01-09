"""
Image Classification Page - Upload and classify garbage images
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src'))

# Import from project_config (NOT config!)
from project_config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, MODELS_DIR

# Page config
st.set_page_config(page_title="Classify Waste", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2ecc71;
        background-color: #e8f8f5;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the best trained model"""
    model_path = os.path.join(MODELS_DIR, 'best_model.h5')
    metadata_path = os.path.join(MODELS_DIR, 'metadata', 'best_model_metadata.json')
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        model = keras.models.load_model(model_path, compile=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    
    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
    
    return model, metadata

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Resize
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to array
    img_array = np.array(img)
    
    # Ensure RGB (in case of RGBA or grayscale)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image):
    """Make prediction on image"""
    # Preprocess
    img_array = preprocess_image(image)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    results = []
    for idx in top_3_idx:
        results.append({
            'class': CLASS_NAMES[idx],
            'confidence': float(predictions[0][idx])
        })
    
    return results

def get_recycling_info(waste_class):
    """Get recycling information for waste class"""
    info = {
        'cardboard': {
            'emoji': '📦',
            'bin': 'Recycling Bin (Blue)',
            'tips': [
                'Flatten boxes to save space',
                'Remove tape and labels',
                'Keep dry and clean',
                'No grease-stained cardboard'
            ],
            'impact': 'Recycling 1 ton of cardboard saves 17 trees and 7,000 gallons of water'
        },
        'glass': {
            'emoji': '🥤',
            'bin': 'Glass Recycling Bin',
            'tips': [
                'Rinse containers before recycling',
                'Remove caps and lids',
                'Can be recycled infinitely',
                'Separate by color if required'
            ],
            'impact': 'Glass can be recycled endlessly without loss of quality'
        },
        'metal': {
            'emoji': '🔩',
            'bin': 'Metal Recycling Bin',
            'tips': [
                'Rinse food containers',
                'Crush cans to save space',
                'Aluminum cans are highly recyclable',
                'Steel cans are also recyclable'
            ],
            'impact': 'Recycling aluminum saves 95% of the energy needed to make new aluminum'
        },
        'paper': {
            'emoji': '📄',
            'bin': 'Paper Recycling Bin',
            'tips': [
                'Keep paper dry and clean',
                'Remove plastic windows from envelopes',
                'Shred sensitive documents',
                'No tissue or paper towels'
            ],
            'impact': 'Recycling 1 ton of paper saves 17 trees and 380 gallons of oil'
        },
        'plastic': {
            'emoji': '🛍️',
            'bin': 'Plastic Recycling Bin',
            'tips': [
                'Check recycling number (1-7)',
                'Rinse containers',
                'Remove caps and labels',
                'Not all plastics are recyclable'
            ],
            'impact': 'Only 9% of all plastic ever made has been recycled'
        },
        'trash': {
            'emoji': '🗑️',
            'bin': 'General Waste Bin (Black)',
            'tips': [
                'This item cannot be recycled',
                'Consider reducing similar waste',
                'Look for reusable alternatives',
                'Proper disposal prevents pollution'
            ],
            'impact': 'Reducing waste is better than recycling - choose reusable items when possible'
        }
    }
    
    return info.get(waste_class, info['trash'])

# Main app
st.title("🔍 Garbage Classification")
st.markdown("Upload an image of waste to classify it and get recycling recommendations")

# Load model
model, metadata = load_model()

if model is None:
    st.error("❌ Model not found!")
    st.info("""
    **Please train a model first:**
    
    Run one of these notebooks:
    - `notebooks/04_quick_training.ipynb` (Fast - 15-30 min)
    - `notebooks/05_transfer_learning_models.ipynb` (Full - 2-4 hours)
    """)
    st.stop()

# Display model info
with st.expander("ℹ️ Model Information"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", metadata.get('best_model', 'Unknown'))
    with col2:
        accuracy = metadata.get('metrics', {}).get('accuracy', metadata.get('test_accuracy', 0))
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col3:
        st.metric("Categories", len(CLASS_NAMES))
    with col4:
        st.metric("Image Size", f"{IMG_WIDTH}x{IMG_HEIGHT}")

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of the waste item"
)

st.markdown("---")

# Process uploaded image
if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
        # Image info
        st.info(f"""
        **Image Details:**
        - Size: {image.size[0]} x {image.size[1]} pixels
        - Format: {image.format}
        - Mode: {image.mode}
        """)
    
    with col2:
        st.markdown("### 🎯 Classification Results")
        
        # Classify button
        if st.button("🚀 Classify Image", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing image..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Make prediction
                try:
                    results = predict_image(model, image)
                    
                    # Display top prediction
                    top_prediction = results[0]
                    confidence_percent = top_prediction['confidence'] * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #2ecc71; margin: 0;">
                            {get_recycling_info(top_prediction['class'])['emoji']} 
                            {top_prediction['class'].upper()}
                        </h2>
                        <p style="margin: 0.5rem 0;">Confidence: <strong>{confidence_percent:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.markdown(f"""
                    <div class="confidence-bar" style="width: {confidence_percent}%;">
                        {confidence_percent:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Top 3 predictions
                    st.markdown("### 📊 Top 3 Predictions")
                    for i, result in enumerate(results, 1):
                        conf_percent = result['confidence'] * 100
                        st.markdown(f"""
                        **{i}. {result['class'].title()}** - {conf_percent:.2f}%
                        """)
                        st.progress(result['confidence'])
                    
                    st.markdown("---")
                    
                    # Recycling information
                    info = get_recycling_info(top_prediction['class'])
                    
                    st.markdown(f"### ♻️ Recycling Guidelines")
                    
                    st.success(f"**Dispose in:** {info['bin']}")
                    
                    st.markdown("**Tips:**")
                    for tip in info['tips']:
                        st.markdown(f"- {tip}")
                    
                    st.info(f"**Environmental Impact:** {info['impact']}")
                    
                    # Download prediction
                    st.markdown("---")
                    prediction_data = {
                        'prediction': top_prediction['class'],
                        'confidence': confidence_percent,
                        'all_predictions': results
                    }
                    
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(prediction_data, indent=2),
                        file_name=f"prediction_{top_prediction['class']}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.info("Please try a different image or check the model file.")

# Instructions
else:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("### 📝 Tips for Best Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Image Quality:**
        - Use clear, well-lit photos
        - Avoid blurry images
        - Center the waste item
        - Remove background clutter
        """)
    
    with col2:
        st.markdown("""
        **Supported Items:**
        - Cardboard boxes
        - Glass bottles/jars
        - Metal cans
        - Paper products
        - Plastic containers
        - General trash
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🌍 Help protect the environment by recycling properly</p>
    <p>Every item classified correctly makes a difference!</p>
</div>
""", unsafe_allow_html=True)
