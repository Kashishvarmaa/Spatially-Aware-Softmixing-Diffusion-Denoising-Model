import streamlit as st
from PIL import Image
import tempfile
import os

from main_moe import moe_predict

st.set_page_config(page_title="Smart Denoiser MoE", layout="centered")
st.title("🧠📸 Mixture-of-Experts Denoiser + Diffusion Enhancer")

st.markdown("""
This app auto-detects whether the uploaded image is a **medical CT scan** or a **raindrop-corrupted natural image**,
applies the correct expert **U-Net denoiser**, and enhances it using a **diffusion-based super-resolution model**.
""")

uploaded_file = st.file_uploader("Upload an image (CT or raindrop-affected)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.subheader("Step 1: Original Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Step 2: Classifying the image type")
    with st.spinner("Classifying using gating model..."):
        prediction_label, denoised_pil, enhanced_pil = moe_predict(image, return_steps=True)
        label_name = "Medical (CT)" if prediction_label == 0 else "Raindrop (RGB)"
    st.success(f"Classified as: {label_name}")

    st.subheader("Step 3: Expert Denoising")
    st.image(denoised_pil, caption="Denoised Output", use_column_width=True)

    st.subheader("Step 4: Diffusion-based Enhancement")
    st.image(enhanced_pil, caption="Final Enhanced Output", use_column_width=True)

    # Download
    st.subheader("Download Final Result")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        enhanced_pil.save(tmp_file.name)
        st.download_button(label="📥 Download Enhanced Image", data=open(tmp_file.name, "rb"), file_name="enhanced_output.png")
        os.unlink(tmp_file.name)
else:
    st.info("👆 Upload an image to begin.")