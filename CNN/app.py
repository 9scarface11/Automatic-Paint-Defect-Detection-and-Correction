# app.py

import streamlit as st
from PIL import Image

from src.inference import predict_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Defect Detection Demo",
    layout="centered"
)

st.title("Computer Vision Defect Detection")
st.write("Upload an image to detect defects.")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Input Image", width="stretch")



    if st.button("Run Detection"):
        with st.spinner("Running inference..."):
            result = predict_image(image)

        st.subheader("Prediction Result")
        st.write(f"Prediction: **{result['prediction']}**")
        #st.write(f"Confidence: **{result['confidence']:.4f}**")
