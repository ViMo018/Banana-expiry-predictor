import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import layers, models

st.set_page_config(page_title="Banana Quality Analyzer", page_icon="🍌")

st.markdown("""
    <style>
    .reportview-container { background: #fdfaf0; }
    .status-box { padding: 20px; border-radius: 10px; color: white; font-weight: bold; text-align: center; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_banana_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.load_weights('banana_weights.weights.h5')
    return model


st.title("🍌 Banana Shelf-Life & Quality")
uploaded_file = st.file_uploader("Upload a photo of your banana", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=400)

    if st.button("Analyze Quality"):
        model = load_banana_model()

        size = (224, 224)
        processed_img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        img_array = np.array(processed_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)


        prediction = model.predict(img_array)
        days = float(prediction[0][0])
        days = max(0, days)  # Prevent negative numbers

        st.divider()

        # Define Parameters
        if days >= 7:
            quality = "EXCELLENT"
            color = "#2ecc71"  # Green
            description = "Unripe / Very Fresh"
            advice = "Best for long-term storage. Keep at room temperature."
        elif 4 <= days < 7:
            quality = "GOOD"
            color = "#f1c40f"  # Yellow
            description = "Perfectly Ripe"
            advice = "Consume within a few days. High vitamin content."
        elif 2 <= days < 4:
            quality = "FAIR"
            color = "#e67e22"  # Orange
            description = "Overripe / Spotted"
            advice = "Very sweet. Best for immediate eating or smoothies."
        else:
            quality = "POOR"
            color = "#e74c3c"  # Red
            description = "Near Spoilage / Rotten"
            advice = "Not recommended for raw eating. Use for banana bread or compost."

        col1, col2 = st.columns(2)
        col1.metric("Estimated Days Left", f"{days:.1f}")
        col2.metric("Quality Grade", quality)

        st.markdown(f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">GRADE: {quality}</h2>
                <p style="color:white; margin:0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"**Pro Tip:** {advice}")

        st.write("Freshness Meter:")
        st.progress(min(days / 10.0, 1.0))