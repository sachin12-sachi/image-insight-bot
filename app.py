import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.title("🖼️ Hugging Face Image Insight Bot")

# Load model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Upload section
uploaded_file = st.file_uploader("Upload an image (e.g., X-ray, chart, photo)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🧠 Generate Insight"):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"🔍 Insight: {caption}")
