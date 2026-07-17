import streamlit as st
from src.diffusion.generator import generate_image, load_model


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


if "show_image" not in st.session_state:
    st.session_state.show_image = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

with st.sidebar:
    left, right = st.columns([1, 6])
    with left:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.image("public/logo.png", width=100)
    with right:
        st.title("TensorRoom")

    if not st.session_state.model_loaded:
        model_status = st.empty()
        model_status.info("Loading model...")
        with st.spinner("Loading model..."):
            diffusion_model = get_model()
        model_status.success("Model loaded")
        st.session_state.model_loaded = True
    else:
        diffusion_model = get_model()

    if st.button("Generate Image"):
        st.session_state.show_image = False
        generation_status = st.empty()
        generation_status.info("Generating image...")
        with st.spinner("Generating image..."):
            generate_image(diffusion_model)
        generation_status.success("Image generated")
        st.session_state.show_image = True

if st.session_state.show_image:
    st.image("data/images/output.png", width=512)