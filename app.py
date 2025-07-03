import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from tensorflow.keras.applications import efficientnet, resnet50, vgg16, inception_v3, mobilenet

# === SETTINGS ===
MODEL_DIR = 'models'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

# === Model-specific preprocessing ===
PREPROCESS_MAP = {
    'efficientnetb0.h5': efficientnet.preprocess_input,
    'resnet50.h5': resnet50.preprocess_input,
    'vgg16.h5': vgg16.preprocess_input,
    'inceptionv3.h5': inception_v3.preprocess_input,
    'mobilenet.h5': mobilenet.preprocess_input
}

@st.cache_resource
def load_selected_model(path):
    return load_model(path)

def preprocess_img(uploaded_file, model_name):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    preprocess_fn = PREPROCESS_MAP.get(model_name.lower())
    if preprocess_fn:
        img_array = preprocess_fn(img_array)
    else:
        img_array = img_array / 255.0

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
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', ha='center', va='bottom')
    st.pyplot(fig)

# === Streamlit UI ===
st.title("🐟 Fish Species Classifier")
st.markdown("Upload a fish image and select the model to classify the species.")

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
selected_model = st.selectbox("🔍 Choose a model", model_files)

# Load model (cached)
model_path = os.path.join(MODEL_DIR, selected_model)
model = load_selected_model(model_path)
st.success(f"✅ Loaded model: `{selected_model}`")

# Load class names from test set
DATA_DIR = 'data/test'
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# Upload image
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    img_array = preprocess_img(uploaded_file, selected_model)
    start_time = time.perf_counter()
    prediction = model.predict(img_array)[0]
    end_time = time.perf_counter()
    inference_time = end_time - start_time

    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Model is not confident. This may not be a fish image.")
    else:
        st.success(f"🎯 Prediction: **{class_names[predicted_idx]}** ({confidence*100:.2f}% confidence)")

    st.subheader("🏆 Top 3 Predictions")
    top_indices = prediction.argsort()[-3:][::-1]
    for i in top_indices:
        if i < len(class_names):
            st.markdown(f"- **{class_names[i]}**: {prediction[i]*100:.2f}%")

    st.subheader("📊 All Class Confidence Scores")
    prob_dict = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    plot_confidence_bar(prob_dict)

    st.info(f"⏱️ Inference Time: {inference_time:.3f} seconds")