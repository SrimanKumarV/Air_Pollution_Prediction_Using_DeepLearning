import sys
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import altair as alt
import datetime
import shutil
import os

st.set_page_config(page_title="Pollution Vision", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Glassmorphism & UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glassmorphism for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: 600;
    }
    
    .stButton>button {
        border-radius: 20px;
        border: None;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍃 AI‑Driven Air Pollution Prediction")
st.markdown("Upload a leaf image to estimate particulate matter density.")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
REGRESSOR_PATH = BASE_DIR / "models" / "regressor.keras"
CLASSIFIER_PATH = BASE_DIR / "models" / "classifier.keras"
HISTORY_FILE = BASE_DIR / "history.csv"
DATA_DIR = BASE_DIR / "data" / "processed"

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# --- Model Loading ---
@st.cache_resource
def load_model_from_disk(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def load_models():
    try:
        import tensorflow as tf
    except ImportError:
        st.error("TensorFlow is not installed.")
        return None, None
        
    classifier, regressor = None, None
    if CLASSIFIER_PATH.exists():
        classifier = load_model_from_disk(str(CLASSIFIER_PATH))
    if REGRESSOR_PATH.exists():
        regressor = load_model_from_disk(str(REGRESSOR_PATH))
    return classifier, regressor

classifier, regressor = load_models()

# --- Sidebar: History & Downloads ---
with st.sidebar:
    st.header("⚙️ Controls & Data")
    
    st.subheader("Dataset Download")
    st.write("Download the augmented dataset used for training.")
    if st.button("Prepare Dataset Zip"):
        with st.spinner("Zipping data..."):
            shutil.make_archive("dataset_augmented", 'zip', str(DATA_DIR))
            st.success("Ready!")
            with open("dataset_augmented.zip", "rb") as fp:
                st.download_button(
                    label="Download Zip",
                    data=fp,
                    file_name="dataset_augmented.zip",
                    mime="application/zip"
                )

# --- History Tracking ---
def save_history(filename, prediction):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": now, "Filename": filename, "PM_Density": prediction}])
    if HISTORY_FILE.exists():
        new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, index=False)

# --- Main App ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded Leaf", use_container_width=True)

with col2:
    if uploaded:
        if regressor is None or classifier is None:
            st.info("Prediction is unavailable until both models are trained and placed in models/.")
        else:
            with st.spinner("Analyzing..."):
                x = np.expand_dims(np.array(img), axis=0)
                
                # First Classify
                is_dusty_prob = classifier.predict(x)[0][0]
                
                st.write(f"*Classifier probability of being dusty: {is_dusty_prob:.4f}*")
                
                if is_dusty_prob < 0.5:
                    st.success("Air Quality: Healthy (Clean Leaf)")
                    st.metric("Estimated Particulate Matter Density", "0.00")
                    save_history(uploaded.name, 0.0)
                else:
                    # Then Regress
                    pred = regressor.predict(x)[0][0]
                    save_history(uploaded.name, pred)
                    
                    st.metric("Estimated Particulate Matter Density", f"{pred:.2f}")
                    if pred < 30:
                        st.success("Air Quality: Healthy")
                    elif pred < 70:
                        st.warning("Air Quality: Moderate")
                    else:
                        st.error("Air Quality: Severe")

# --- Graphs & Analytics ---
st.markdown("---")
st.header("📈 Historical Analytics")

if HISTORY_FILE.exists():
    df = pd.read_csv(HISTORY_FILE)
    if not df.empty:
        # Create an Altair Chart
        base = alt.Chart(df).encode(
            x=alt.X('Timestamp:T', title='Time')
        )
        
        line = base.mark_line(point=True, color='#667eea').encode(
            y=alt.Y('PM_Density:Q', title='PM Density'),
            tooltip=['Timestamp', 'Filename', 'PM_Density']
        ).interactive()
        
        # Add Threshold Lines
        healthy_line = alt.Chart(pd.DataFrame({'y': [30], 'label': ['Healthy (<30)']})).mark_rule(color='green', strokeDash=[5, 5]).encode(y='y:Q')
        severe_line = alt.Chart(pd.DataFrame({'y': [70], 'label': ['Severe (>=70)']})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
        
        chart = (line + healthy_line + severe_line).properties(width='container', height=400)
        
        st.altair_chart(chart, use_container_width=True)
        
        with st.expander("View Raw History Data"):
            st.dataframe(df)
else:
    st.info("No history available yet. Upload an image to generate data.")