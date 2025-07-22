import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('mobilenetv2_custom_final.keras')
st.title("Malnutrition Detection (Image Classification Demo)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_container_width=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    pred_class = int(prediction[0][0] > 0.5)
    prob = float(prediction[0][0])

    if pred_class == 0:
        st.success(f"✅ The image is predicted as: **Healthy Kid** (Probability: {1-prob:.2%})")
    else:
        st.error(f"⚠️ The image is predicted as: **Malnourished Kid** (Probability: {prob:.2%})")
