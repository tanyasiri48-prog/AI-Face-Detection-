import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(
    page_title="AI Face Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================
# Title
# =====================================

st.markdown("""
<h1 style='
text-align:center;
color:#173B57;
font-size:55px;
'>
🤖 AI Face Detection
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='
text-align:center;
font-size:22px;
color:gray;
'>
ระบบตรวจจับภาพจริงและภาพที่สร้างด้วย AI
</p>
""", unsafe_allow_html=True)


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.write(f"อุปกรณ์ที่ใช้ประมวลผล: {device}")

import os

st.write(os.getcwd())

# =====================================
# Load Model
# =====================================

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

st.success("โหลดโมเดลสำเร็จ")


# =====================================
# Image Transform
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ลำดับคลาส
class_names = ["FAKE", "REAL"]


# =====================================
# Upload Image
# =====================================

uploaded_file = st.file_uploader(
    "เลือกรูปภาพ",
    type=["jpg", "jpeg", "png"]
)


# =====================================
# Predict
# =====================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="รูปภาพที่เลือก",
        use_container_width=True
    )

    if st.button("ตรวจสอบภาพ"):

        # เตรียมภาพ
        input_tensor = transform(image)

        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # ทำนายผล
        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            probabilities[predicted_index].item()
            * 100
        )

        fake_probability = (
            probabilities[0].item()
            * 100
        )

        real_probability = (
            probabilities[1].item()
            * 100
        )

        # =====================================
        # Result
        # =====================================

        st.subheader("ผลการตรวจสอบ")

        if predicted_class == "REAL":

            st.success(
                "ผลลัพธ์ : REAL (ภาพจริง)"
            )

        else:

            st.error(
                "ผลลัพธ์ : FAKE (ภาพที่สร้างด้วย AI)"
            )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.write(
            f"FAKE : {fake_probability:.2f}%"
        )

        st.progress(
            fake_probability / 100
        )

        st.write(
            f"REAL : {real_probability:.2f}%"
        )

        st.progress(
            real_probability / 100
        )