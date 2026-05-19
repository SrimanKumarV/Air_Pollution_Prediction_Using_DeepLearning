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

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
REGRESSOR_PATH = BASE_DIR / "models" / "regressor.keras"
CLASSIFIER_PATH = BASE_DIR / "models" / "classifier.keras"
HISTORY_FILE = BASE_DIR / "history.csv"
DATA_DIR = BASE_DIR / "data" / "processed"
CLASSIFIER_HIST_PATH = BASE_DIR / "models" / "classifier_history.csv"
REGRESSOR_HIST_PATH = BASE_DIR / "models" / "regressor_history.csv"

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

def save_history(filename, prediction):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": now, "Filename": filename, "PM_Density": prediction}])
    if HISTORY_FILE.exists():
        new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, index=False)

# --- Zero-Shot Leaf Detection (OOD) ---
@st.cache_resource
def load_imagenet_model():
    import tensorflow as tf
    return tf.keras.applications.MobileNetV2(weights='imagenet')

def is_image_a_leaf(img):
    import tensorflow as tf
    model = load_imagenet_model()
    # Preprocess for ImageNet MobileNetV2
    img_resized = img.resize((224, 224))
    x = np.expand_dims(np.array(img_resized), axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    preds = model.predict(x)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=10)[0]
    
    # Botanical keywords to look for in top 10 ImageNet predictions
    botanical_keywords = ['leaf', 'plant', 'pot', 'flower', 'tree', 'fruit', 'vegetable', 'daisy', 'rose', 'mushroom', 'fern', 'greenhouse', 'strawberry', 'lemon', 'orange', 'fig', 'pineapple', 'banana', 'apple', 'broccoli', 'cabbage', 'cucumber', 'zucchini', 'corn', 'acorn', 'bell_pepper', 'head_cabbage', 'cardoon', 'artichoke']
    
    for _, label, _ in decoded:
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in botanical_keywords):
            return True
            
    return False

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Models", "Training Metrics", "About Us", "Technical Stack"])

st.sidebar.markdown("---")
st.sidebar.subheader("Download Resources")

# Initialize session state for zip files to prevent nested button bug
if 'dataset_zipped' not in st.session_state:
    st.session_state.dataset_zipped = False
if 'models_zipped' not in st.session_state:
    st.session_state.models_zipped = False

# Dataset Download
st.sidebar.write("Download augmented dataset:")
if st.sidebar.button("Prepare Dataset Zip"):
    with st.spinner("Zipping data..."):
        shutil.make_archive("dataset_augmented", 'zip', str(DATA_DIR))
        st.session_state.dataset_zipped = True

if st.session_state.dataset_zipped:
    with open("dataset_augmented.zip", "rb") as fp:
        st.sidebar.download_button(
            label="Download Dataset Zip",
            data=fp,
            file_name="dataset_augmented.zip",
            mime="application/zip"
        )

# Models Download
st.sidebar.write("Download trained models:")
if st.sidebar.button("Prepare Models Zip"):
    with st.spinner("Zipping models..."):
        shutil.make_archive("trained_models", 'zip', str(BASE_DIR / "models"))
        st.session_state.models_zipped = True

if st.session_state.models_zipped:
    with open("trained_models.zip", "rb") as fp:
        st.sidebar.download_button(
            label="Download Models Zip",
            data=fp,
            file_name="trained_models.zip",
            mime="application/zip"
        )

# ==========================================
# PAGE: HOME
# ==========================================
if page == "Home":
    st.title("🍃 AI‑Driven Air Pollution Prediction")
    st.markdown("Upload a leaf image or take a picture using your camera to estimate particulate matter density.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])
        
        uploaded = None
        camera_img = None
        img_source = None
        
        with tab1:
            uploaded = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
        with tab2:
            camera_img = st.camera_input("Take a picture of a leaf")
            
        final_img_file = camera_img if camera_img is not None else uploaded
        
        if final_img_file:
            img = Image.open(final_img_file).convert("RGB").resize((224, 224))
            st.image(img, caption="Image to Analyze", use_container_width=True)
            filename = final_img_file.name if hasattr(final_img_file, 'name') else 'camera_capture.jpg'

    with col2:
        if final_img_file:
            if regressor is None or classifier is None:
                st.info("Prediction is unavailable until both models are trained and placed in models/.")
            else:
                with st.spinner("Analyzing..."):
                    if not is_image_a_leaf(img):
                        st.error("❌ Out of Distribution Error: This image does not appear to be a leaf or plant. Please upload a valid botanical image for environmental analysis.")
                    else:
                        x = np.expand_dims(np.array(img), axis=0)
                        
                        # First Classify
                        is_dusty_prob = classifier.predict(x)[0][0]
                        
                        st.write(f"*Classifier probability of being dusty: {is_dusty_prob:.4f}*")
                        
                        if is_dusty_prob < 0.5:
                            st.success("Classification: Clean Leaf Detected")
                        else:
                            st.warning("Classification: Dust Patterns Detected")
                            
                        # Always Regress to show continuous density
                        pred = regressor.predict(x)[0][0]
                        # Ensure prediction doesn't go below zero due to linear activation
                        pred = max(0.0, float(pred))
                        save_history(filename, pred)
                        
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
            base = alt.Chart(df).encode(
                x=alt.X('Timestamp:T', title='Time')
            )
            
            line = base.mark_line(point=True, color='#667eea').encode(
                y=alt.Y('PM_Density:Q', title='PM Density'),
                tooltip=['Timestamp', 'Filename', 'PM_Density']
            ).interactive()
            
            healthy_line = alt.Chart(pd.DataFrame({'y': [30], 'label': ['Healthy (<30)']})).mark_rule(color='green', strokeDash=[5, 5]).encode(y='y:Q')
            severe_line = alt.Chart(pd.DataFrame({'y': [70], 'label': ['Severe (>=70)']})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
            
            chart = (line + healthy_line + severe_line).properties(width='container', height=400)
            
            st.altair_chart(chart, use_container_width=True)
            
            with st.expander("View Raw History Data"):
                st.dataframe(df)
    else:
        st.info("No history available yet. Analyze an image to generate data.")

# ==========================================
# PAGE: MODELS
# ==========================================
elif page == "Models":
    st.title("🧠 Model Architecture")
    st.markdown("""
    ### 1. OpenCV Pre-Processing (The Filter)
    Before deep learning is applied, our dataset generation uses an HSV (Hue, Saturation, Value) color space mask to mathematically manipulate and isolate green leaf properties, simulating and analyzing the exact color profile of urban dust.
    
    ### 2. MobileNetV2 (The Deep CNN Backbone)
    Instead of building a simple network from scratch, we use **MobileNetV2**. It acts as our highly efficient deep neural network "brain", replacing manual feature extraction. It uses "Inverted Residual Blocks" (skip connections) to pass visual information through dozens of mathematical filters, shrinking high-resolution leaf images into a dense list of mathematical features.
    
    ### 3. Two-Stage Inference Pipeline
    We use a Two-Stage Pipeline for robust prediction:
    * **Stage 1 (Classifier)**: Uses Binary Cross-Entropy to mathematically classify the image as 'Clean' or 'Dusty'.
    * **Stage 2 (Regressor)**: Only if the leaf is dusty, Stage 2 uses a Linear Regression model specifically optimized on the Mean Absolute Error (MAE) of dust density.
    """)

# ==========================================
# PAGE: TRAINING METRICS
# ==========================================
elif page == "Training Metrics":
    st.title("📊 Training Metrics Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classifier Training History")
        if CLASSIFIER_HIST_PATH.exists():
            clf_df = pd.read_csv(CLASSIFIER_HIST_PATH)
            clf_df['epoch'] = clf_df.index + 1
            
            # Melt the dataframe for Altair
            clf_melted = clf_df.melt(id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy', 'loss', 'val_loss'], 
                                     var_name='Metric', value_name='Value')
            
            # Accuracy Chart
            acc_chart = alt.Chart(clf_melted[clf_melted['Metric'].str.contains('accuracy')]).mark_line(point=True).encode(
                x='epoch:O',
                y='Value:Q',
                color='Metric:N',
                tooltip=['epoch', 'Metric', 'Value']
            ).properties(title='Classifier Accuracy vs Epochs', height=300)
            st.altair_chart(acc_chart, use_container_width=True)
            
            # Loss Chart
            loss_chart = alt.Chart(clf_melted[clf_melted['Metric'].str.contains('loss')]).mark_line(point=True).encode(
                x='epoch:O',
                y='Value:Q',
                color='Metric:N',
                tooltip=['epoch', 'Metric', 'Value']
            ).properties(title='Classifier Loss vs Epochs', height=300)
            st.altair_chart(loss_chart, use_container_width=True)
        else:
            st.info("Classifier history not found. Train the model to generate metrics.")
            
    with col2:
        st.subheader("Regressor Training History")
        if REGRESSOR_HIST_PATH.exists():
            reg_df = pd.read_csv(REGRESSOR_HIST_PATH)
            reg_df['epoch'] = reg_df.index + 1
            
            reg_melted = reg_df.melt(id_vars=['epoch'], 
                                     value_vars=['loss', 'val_loss', 'root_mean_squared_error', 'val_root_mean_squared_error'], 
                                     var_name='Metric', value_name='Value')
            
            # RMSE Chart
            rmse_chart = alt.Chart(reg_melted[reg_melted['Metric'].str.contains('error')]).mark_line(point=True).encode(
                x='epoch:O',
                y='Value:Q',
                color='Metric:N',
                tooltip=['epoch', 'Metric', 'Value']
            ).properties(title='Regressor RMSE vs Epochs', height=300)
            st.altair_chart(rmse_chart, use_container_width=True)
            
            # Loss Chart
            r_loss_chart = alt.Chart(reg_melted[reg_melted['Metric'].str.contains('loss')]).mark_line(point=True).encode(
                x='epoch:O',
                y='Value:Q',
                color='Metric:N',
                tooltip=['epoch', 'Metric', 'Value']
            ).properties(title='Regressor MAE (Loss) vs Epochs', height=300)
            st.altair_chart(r_loss_chart, use_container_width=True)
        else:
            st.info("Regressor history not found. Train the model to generate metrics.")

# ==========================================
# PAGE: ABOUT US
# ==========================================
elif page == "About Us":
    st.title("👥 About Us")
    st.markdown("""
    Welcome to **Pollution Vision**!
    
    This project was designed to tackle the growing environmental concern of particulate matter pollution. By leveraging the natural world—specifically, the leaves of trees—as bio-indicators, we can estimate localized air pollution levels simply by taking a picture of a leaf.
    
    Our goal is to make environmental monitoring accessible, fast, and driven by state-of-the-art Artificial Intelligence.
    """)

# ==========================================
# PAGE: TECHNICAL STACK
# ==========================================
elif page == "Technical Stack":
    st.title("💻 Technical Stack Used")
    st.markdown("""
    Here are the core technologies powering Pollution Vision:
    
    - **Frontend**: [Streamlit](https://streamlit.io/) - For building the interactive, responsive, and dynamic web application.
    - **Deep Learning Framework**: [TensorFlow / Keras](https://www.tensorflow.org/) - For building, training, and running the inference of our MobileNetV2 models.
    - **Pre-Trained Architecture**: **MobileNetV2** - For rapid feature extraction via Transfer Learning.
    - **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - For handling dataset labels, numerical arrays, and training histories.
    - **Computer Vision**: [OpenCV](https://opencv.org/) (cv2) - For manipulating images in the HSV color space and synthetically generating the dusty leaf dataset.
    - **Data Visualization**: [Altair](https://altair-viz.github.io/) - For rendering the beautiful, interactive charts for both training metrics and historical predictions.
    """)