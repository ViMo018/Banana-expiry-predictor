import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import layers, models

st.set_page_config(page_title="Banana Quality Analyzer", page_icon="🍌")

st.markdown("""
    <style>
    .reportview-container { background: #fdfaf0; }
    .stButton>button { 
        width: 100%; 
        border-radius: 20px; 
        background-color: #f4d03f; 
        color: black; 
        font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_banana_model():
    base_structure = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_structure.trainable = False
    
    banana_ai = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        base_structure,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    try:
        banana_ai.load_weights('banana_weights.weights.h5')
    except Exception:
        st.error("I couldn't find the 'banana_weights.weights.h5' file. Please make sure it's in your project folder.")
    
    return banana_ai

st.title("🍌 Banana Shelf-Life & Quality")
uploaded_file = st.file_uploader("Show me a photo of your banana", type=["jpg", "jpeg", "png"])

if uploaded_file:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, width=400)

    if st.button("Analyze Quality"):
        predictor = load_banana_model()

        standard_size = (224, 224)
        ready_image = ImageOps.fit(raw_image, standard_size, Image.Resampling.LANCZOS)
        
        pixel_data = np.array(ready_image)
        brightness_level = pixel_data.mean()

        normalized_data = pixel_data.astype('float32') / 255.0
        final_input = np.expand_dims(normalized_data, axis=0)

        ai_guess = predictor.predict(final_input)
        days_left = max(0, float(ai_guess[0][0]))

        if brightness_level > 190 and days_left < 3:
            days_left = 7.2  
        elif brightness_level < 90 and days_left > 2:
            days_left = 0.5

        st.divider()

        if days_left >= 7:
            status, hue, note = "EXCELLENT", "#2ecc71", "Unripe / Very Fresh"
            tip = "These will last a while! Keep them at room temperature."
        elif 4 <= days_left < 7:
            status, hue, note = "GOOD", "#f1c40f", "Perfectly Ripe"
            tip = "They are at their peak right now. Enjoy!"
        elif 2 <= days_left < 4:
            status, hue, note = "FAIR", "#e67e22", "Overripe / Spotted"
            tip = "Very sweet! If you don't eat them now, they'll be great for smoothies."
        else:
            status, hue, note = "POOR", "#e74c3c", "Near Spoilage / Rotten"
            tip = "Not great for eating raw, but perfect for banana bread!"

        left_col, right_col = st.columns(2)
        left_col.metric("Time Remaining", f"{days_left:.1f} Days")
        right_col.metric("Current Grade", status)

        st.markdown(f"""
            <div style="background-color:{hue}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">GRADE: {status}</h2>
                <p style="color:white; margin:0;">{note}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"**Our Advice:** {tip}")
        st.write("Freshness Progress:")
        st.progress(min(days_left / 10.0, 1.0))
