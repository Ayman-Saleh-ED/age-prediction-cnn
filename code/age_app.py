import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Age Prediction from a Face")
st.write("Take a photo or upload one. Use a close-up of a single face.")

if "model" not in st.session_state:
    st.session_state.model = load_model(os.path.join(os.path.dirname(__file__), "age_model.keras"))

tab_camera, tab_upload = st.tabs(["Camera", "Upload"])

with tab_camera:
    cam_file = st.camera_input("Take a photo")

with tab_upload:
    up_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

file = cam_file if cam_file is not None else up_file

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your photo", width=250)

    small = img.resize((64, 64))
    x = np.array(small, dtype="float32")[None, ...]
    age = max(float(st.session_state.model.predict(x)[0][0]), 0)

    total_months = round(age * 12)
    years = total_months // 12
    months = total_months % 12

    if months == 0:
        st.subheader(f"Predicted age: {years} years")
    else:
        st.subheader(f"Predicted age: {years} years & {months} months")

    st.caption("Typical error of this model is about 7 years, so treat it as a rough estimate.")