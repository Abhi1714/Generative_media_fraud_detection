
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Generative Media Detector",
    page_icon="🔍",
    layout="centered"
)

# ---------------- MODEL SETTINGS ----------------
MODEL_REPO = "Abhiram240506/Generative_media_fraud_detection"
MODEL_FILENAME = "resnet50_genimage_finetuned.pth"
DEVICE = torch.device("cpu")

# ---------------- IMAGE PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# ---------------- APPLICATION INTERFACE ----------------
st.title("🔍 Generative Media Detector")

st.write(
    "Upload an image to check whether it appears to be "
    "a real photograph or an AI-generated image."
)

st.info(
    "Experimental AI classifier. Predictions may be incorrect "
    "and should not be treated as proof of authenticity."
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
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Loading model and analyzing image..."):
                model = load_model()
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                # Model classes: FAKE = 0, REAL = 1
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
                "Scores are model outputs, not verified probabilities. "
                "The classifier can make mistakes, especially on "
                "unfamiliar image generators or image types."
            )

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.info(
            "Please check the model repository and deployment logs "
            "if this problem continues."
        )
