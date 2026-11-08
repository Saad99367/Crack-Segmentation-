from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"

st.set_page_config(page_title="Crack Segmentation AI", page_icon="🔍", layout="wide")
st.title("🔍 Crack Segmentation AI")
st.caption("YOLO11s instance segmentation for surface crack inspection")

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Place best.pt in models/.")
    model = YOLO(str(MODEL_PATH))
    if model.task != "segment":
        raise ValueError("best.pt must be a YOLO segmentation model")
    return model

with st.sidebar:
    st.header("Inference settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.selectbox("Image size", [640, 960, 1024], index=0)
    st.caption("Larger images may reveal fine cracks but use more memory and time.")

uploaded = st.file_uploader("Upload a surface image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        original = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not read the image: {exc}")
        st.stop()

    if st.button("Run segmentation", type="primary"):
        try:
            with st.spinner("Detecting cracks..."):
                model = load_model()
                result = model.predict(
                    source=np.asarray(original),
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )[0]
                # Ultralytics plot() returns BGR; PIL expects RGB.
                annotated = Image.fromarray(result.plot()[..., ::-1].copy())
                count = len(result.boxes) if result.boxes is not None else 0

            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.image(original, use_container_width=True)
            with right:
                st.subheader("Segmentation prediction")
                st.image(annotated, use_container_width=True)

            st.metric("Detected crack instances", count)
            if count == 0:
                st.info("No cracks detected at this confidence threshold.")

            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            st.download_button(
                "Download prediction",
                data=buffer.getvalue(),
                file_name=f"{Path(uploaded.name).stem}_prediction.png",
                mime="image/png",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
