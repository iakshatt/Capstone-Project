"""
Streamlit App — CIFAR-10 CNN Image Classifier
================================================
Upload an image and the trained CNN will predict which of the 10
CIFAR-10 classes it belongs to.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🖼️",
    layout="centered"
)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model('cnn_cifar10_model.keras')

model = load_model()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload an image and this CNN (trained from scratch on CIFAR-10) will "
    "predict which of 10 categories it belongs to: "
    f"*{', '.join(CLASS_NAMES)}*."
)

st.info(
    "⚠️ This model was trained on 32x32 low-resolution images from 10 "
    "specific categories. It works best on simple, centered photos of "
    "those categories — not general-purpose images."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess: resize to 32x32 and normalize, matching training pipeline
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.1f}%**")

    # Show full probability breakdown
    st.subheader("Class Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(10)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(sorted_probs)

else:
    st.write("👆 Upload an image to get started.")

st.markdown("---")
st.caption(
    "Built with TensorFlow/Keras · Trained on CIFAR-10 · "
    "[View source on GitHub](https://github.com/iakshatt/Capstone-Project)"
<<<<<<< HEAD
)
=======
)
>>>>>>> 86e0382d8838e8d724c7b59de93cda0d55fbba90
