
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Generative Media Detector",
    page_icon="🔍",
    layout="centered"
)

# Model settings
MODEL_PATH = Path(__file__).parent / "resnet50_genimage_finetuned.pth"
DEVICE = torch.device("cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load trained model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# App interface
st.title("🔍 Generative Media Detector")
st.write(
    "Upload an image to check whether it appears to be "
    "real or AI-generated."
)

st.info(
    "This is an experimental image classifier. "
    "Its predictions may be incorrect."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                input_tensor = transform(image).unsqueeze(0)

                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                # Model class mapping: FAKE = 0, REAL = 1
                fake_score = probabilities[0].item()
                real_score = probabilities[1].item()

            st.subheader("Analysis Result")

            if fake_score >= real_score:
                st.error("🤖 Prediction: AI-GENERATED")
            else:
                st.success("📷 Prediction: REAL IMAGE")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "AI-generated score",
                    f"{fake_score * 100:.2f}%"
                )
                st.progress(fake_score)

            with col2:
                st.metric(
                    "Real image score",
                    f"{real_score * 100:.2f}%"
                )
                st.progress(real_score)

            st.caption(
                "Scores represent the model's relative outputs, "
                "not verified probabilities or proof of authenticity."
            )

    except Exception as error:
        st.error(f"Something went wrong: {error}")
