"""
Analytics Dashboard - Real-time usage statistics and predictions
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src'))

from project_config import MODELS_DIR, OUTPUTS_DIR, CLASS_NAMES


# Page config
st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

# Title
st.title("📈 Analytics Dashboard")
st.markdown("Real-time insights and usage statistics")

st.markdown("---")

# Simulated analytics (in a real app, this would come from a database)
st.markdown("## 📊 Usage Statistics")

# Generate sample data
np.random.seed(42)
total_predictions = np.random.randint(500, 2000)
today_predictions = np.random.randint(20, 100)
avg_confidence = np.random.uniform(0.80, 0.95)
unique_users = np.random.randint(50, 200)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Predictions",
        value=f"{total_predictions:,}",
        delta=f"+{today_predictions} today"
    )

with col2:
    st.metric(
        label="Average Confidence",
        value=f"{avg_confidence*100:.1f}%",
        delta="+2.3%"
    )

with col3:
    st.metric(
        label="Active Users",
        value=f"{unique_users}",
        delta="+12 this week"
    )

with col4:
    st.metric(
        label="Accuracy Rate",
        value="87.5%",
        delta="+1.2%"
    )

st.markdown("---")

# Prediction trends
st.markdown("## 📈 Prediction Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Predictions Over Time")
    
    # Generate sample time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    predictions_per_day = np.random.randint(10, 50, size=30)
    predictions_per_day = predictions_per_day + np.arange(30) * 0.5  # Add slight upward trend
    
    df_timeline = pd.DataFrame({
        'Date': dates,
        'Predictions': predictions_per_day.astype(int)
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_timeline['Date'], df_timeline['Predictions'], 
            marker='o', linewidth=2, markersize=6, color='#3498db')
    ax.fill_between(df_timeline['Date'], df_timeline['Predictions'], alpha=0.3, color='#3498db')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Number of Predictions', fontweight='bold')
    ax.set_title('Daily Predictions (Last 30 Days)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("### 🗑️ Waste Type Distribution")
    
    # Generate sample distribution
    class_counts = {class_name: np.random.randint(50, 300) for class_name in CLASS_NAMES}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(CLASS_NAMES)))
    
    wedges, texts, autotexts = ax.pie(
        class_counts.values(),
        labels=[c.title() for c in class_counts.keys()],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Waste Classification Distribution', fontweight='bold')
    st.pyplot(fig)

st.markdown("---")

# Top categories
st.markdown("## 🔝 Top Categories")

col1, col2, col3 = st.columns(3)

# Sort by count
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

with col1:
    st.markdown("### 🥇 Most Common")
    top_class, top_count = sorted_classes[0]
    st.info(f"""
    **{top_class.title()}**
    
    {top_count} predictions ({top_count/sum(class_counts.values())*100:.1f}%)
    """)

with col2:
    st.markdown("### 🥈 Second Most")
    second_class, second_count = sorted_classes[1]
    st.info(f"""
    **{second_class.title()}**
    
    {second_count} predictions ({second_count/sum(class_counts.values())*100:.1f}%)
    """)

with col3:
    st.markdown("### 🥉 Third Most")
    third_class, third_count = sorted_classes[2]
    st.info(f"""
    **{third_class.title()}**
    
    {third_count} predictions ({third_count/sum(class_counts.values())*100:.1f}%)
    """)

st.markdown("---")

# Confidence distribution
st.markdown("## 🎯 Prediction Confidence Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Confidence Score Distribution")
    
    # Generate sample confidence data
    confidence_scores = np.random.beta(8, 2, 1000)  # Beta distribution for realistic confidence
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidence_scores, bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax.axvline(confidence_scores.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {confidence_scores.mean():.3f}')
    ax.set_xlabel('Confidence Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Prediction Confidence Scores', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("### 📈 Statistics")
    
    st.metric("Mean Confidence", f"{confidence_scores.mean():.3f}")
    st.metric("Median Confidence", f"{np.median(confidence_scores):.3f}")
    st.metric("Std Deviation", f"{confidence_scores.std():.3f}")
    
    high_confidence = (confidence_scores > 0.9).sum() / len(confidence_scores) * 100
    st.metric("High Confidence (>0.9)", f"{high_confidence:.1f}%")

st.markdown("---")

# Hourly patterns
st.markdown("## ⏰ Usage Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 By Day of Week")
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = np.random.randint(30, 100, size=7)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(days, day_counts, color='#3498db', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Predictions', fontweight='bold')
    ax.set_title('Predictions by Day of Week', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("### 🕐 By Hour of Day")
    
    hours = list(range(24))
    hour_counts = np.random.poisson(lam=20, size=24)
    hour_counts = hour_counts + np.sin(np.array(hours) * np.pi / 12) * 10  # Add pattern
    hour_counts = np.maximum(hour_counts, 0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, hour_counts, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax.fill_between(hours, hour_counts, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Number of Predictions', fontweight='bold')
    ax.set_title('Predictions by Hour', fontweight='bold')
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Environmental impact
st.markdown("## 🌍 Environmental Impact")

col1, col2, col3 = st.columns(3)

# Calculate simulated environmental savings
recyclable_count = sum([class_counts[c] for c in ['cardboard', 'glass', 'metal', 'paper', 'plastic']])
recycling_rate = recyclable_count / sum(class_counts.values()) * 100

with col1:
    st.markdown("### ♻️ Recycling Rate")
    st.metric("Items Recycled", f"{recycling_rate:.1f}%")
    st.progress(recycling_rate / 100)

with col2:
    st.markdown("### 🌳 Trees Saved")
    trees_saved = recyclable_count * 0.01  # Simulated calculation
    st.metric("Estimated", f"{int(trees_saved)}")
    st.caption("Based on paper/cardboard recycling")

with col3:
    st.markdown("### 💧 Water Saved")
    water_saved = recyclable_count * 7  # Simulated (gallons)
    st.metric("Gallons", f"{int(water_saved):,}")
    st.caption("From recycling efforts")

st.markdown("---")

# Recent predictions (simulated)
st.markdown("## 🕐 Recent Predictions")

# Generate sample recent predictions
recent_data = []
for i in range(10):
    recent_data.append({
        'Timestamp': (datetime.now() - timedelta(minutes=np.random.randint(1, 120))).strftime('%Y-%m-%d %H:%M:%S'),
        'Category': np.random.choice(CLASS_NAMES).title(),
        'Confidence': f"{np.random.uniform(0.75, 0.99):.2%}",
        'Status': np.random.choice(['✅ Correct', '✅ Correct', '✅ Correct', '⚠️ Uncertain'])
    })

df_recent = pd.DataFrame(recent_data)
st.dataframe(df_recent, use_container_width=True, hide_index=True)

st.markdown("---")

# Export data
st.markdown("## 📥 Export Analytics")

col1, col2 = st.columns(2)

with col1:
    # Create sample CSV
    csv_data = pd.DataFrame({
        'Category': list(class_counts.keys()),
        'Count': list(class_counts.values()),
        'Percentage': [v/sum(class_counts.values())*100 for v in class_counts.values()]
    })
    
    csv = csv_data.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Category Distribution (CSV)",
        data=csv,
        file_name=f"category_distribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Create sample JSON
    analytics_json = {
        'generated_at': datetime.now().isoformat(),
        'total_predictions': total_predictions,
        'category_distribution': class_counts,
        'recycling_rate': recycling_rate,
        'average_confidence': float(avg_confidence)
    }
    
    st.download_button(
        label="📥 Download Analytics Summary (JSON)",
        data=json.dumps(analytics_json, indent=2),
        file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>📈 Analytics Dashboard</p>
    <p><em>Note: This is a demo with simulated data. In production, connect to a real database.</em></p>
</div>
""", unsafe_allow_html=True)
