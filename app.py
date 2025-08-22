import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import json
from PIL import Image

# === SETTINGS ===
MODEL_DIR = 'models'
CONFIDENCE_THRESHOLD = 0.7

@st.cache_resource
def load_selected_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        st.error(f" Failed to load model: {e}")
        st.stop()

def load_metadata(model_path):
    """Load matching metadata JSON for the model."""
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    # Remove "_best" if present to match metadata naming
    if base_name.endswith("_best"):
        base_name = base_name.replace("_best", "")
    meta_file = os.path.join(MODEL_DIR, f"{base_name}_metadata.json")
    if not os.path.exists(meta_file):
        st.error(f"Metadata file not found: {meta_file}")
        st.stop()
    with open(meta_file, "r") as f:
        return json.load(f)

def get_preprocess_fn(preprocess_name, model_name=None):
    """Return preprocessing function from metadata."""
    if preprocess_name == "rescale_0_1":
        return lambda x: x / 255.0

    if preprocess_name == "preprocess_input" and model_name:
        model_map = {
            "VGG16": tf.keras.applications.vgg16,
            "ResNet50": tf.keras.applications.resnet50,
            "MobileNet": tf.keras.applications.mobilenet,
            "InceptionV3": tf.keras.applications.inception_v3,
            "EfficientNetB0": tf.keras.applications.efficientnet
        }
        if model_name in model_map:
            return model_map[model_name].preprocess_input

    # If stored as module.function
    if "." in preprocess_name:
        parts = preprocess_name.split(".")
        module = getattr(tf.keras.applications, parts[0])
        return getattr(module, parts[1])

    raise ValueError(f"Unknown preprocess function: {preprocess_name}")

def preprocess_img(uploaded_file, img_size, preprocess_fn):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_fn(img_array)
    return np.expand_dims(img_array, axis=0)

def plot_confidence_bar(prob_dict):
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color='skyblue')
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Scores")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    st.pyplot(fig)

# === Streamlit UI ===
st.title("🐟 Fish Species Classifier")
st.markdown("Upload a fish image and select the model to classify the species.")

# Model selection
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
selected_model = st.selectbox("🔍 Choose a model", model_files)

# Load model + metadata
model_path = os.path.join(MODEL_DIR, selected_model)
metadata = load_metadata(model_path)
model = load_selected_model(model_path)
st.success(f"✅ Loaded model: `{selected_model}`")

# Extract settings from metadata
img_size = tuple(metadata["img_size"])
class_indices = metadata["class_indices"]
idx_to_class = {v: k for k, v in class_indices.items()}
preprocess_fn = get_preprocess_fn(metadata["preprocess"], metadata["model_name"])

# Upload image
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    img_array = preprocess_img(uploaded_file, img_size, preprocess_fn)
    start_time = time.perf_counter()
    prediction = model.predict(img_array)[0]
    end_time = time.perf_counter()
    inference_time = end_time - start_time

    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Model is not confident. This may not be a fish image.")
    else:
        st.success(f"🎯 Prediction: **{idx_to_class[predicted_idx]}** ({confidence*100:.2f}% confidence)")

    st.subheader("🏆 Top 3 Predictions")
    top_indices = prediction.argsort()[-3:][::-1]
    for i in top_indices:
        st.markdown(f"- **{idx_to_class[i]}**: {prediction[i]*100:.2f}%")

    st.subheader("📊 All Class Confidence Scores")
    prob_dict = {idx_to_class[i]: float(prediction[i]) for i in range(len(prediction))}
    plot_confidence_bar(prob_dict)

    st.info(f"⏱️ Inference Time: {inference_time:.3f} seconds")