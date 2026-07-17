import streamlit as st
from src.diffusion.generator import generate_image, load_model

#diffusion_model = load_model()

with st.sidebar:
    left, right = st.columns([1, 6])
    with left:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.image("public/logo.png", width=100)
    with right:
        st.title("TensorRoom")

    if st.button("Generate Image"):
        #generate_image(diffusion_model)
        st.success("Image generated successfully!")
