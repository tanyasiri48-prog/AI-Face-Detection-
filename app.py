import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ==================================================
# 1. ตั้งค่าหน้าเว็บ
# ==================================================

st.set_page_config(
    page_title="AI Face Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CSS ตกแต่งหน้าเว็บ
# ==================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(63, 131, 248, 0.14),
                transparent 35%
            ),
            radial-gradient(
                circle at bottom right,
                rgba(98, 213, 174, 0.10),
                transparent 35%
            ),
            #f6f8fc;
        color: #000000;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 56px;
        font-weight: 800;
        color: #000000;
        line-height: 1.15;
        margin-bottom: 8px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #000000;
        margin-bottom: 30px;
    }

    .device-box {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dbe5f0;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(31, 50, 81, 0.06);
        color: #000000;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e7ef;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(31, 50, 81, 0.07);
        margin-bottom: 16px;
        color: #000000;
    }

    .real-result {
        background: linear-gradient(
            135deg,
            #e7f8ed,
            #d7f4e1
        );
        border: 2px solid #35a65f;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        color: #000000;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .fake-result {
        background: linear-gradient(
            135deg,
            #fdecec,
            #fbdada
        );
        border: 2px solid #e05555;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        color: #000000;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .result-description {
        font-size: 17px;
        font-weight: 500;
        margin-top: 5px;
        color: #000000;
    }

    .file-info {
        background-color: #f8fafc;
        border: 1px solid #dde6ef;
        border-radius: 12px;
        padding: 13px 16px;
        margin-top: 12px;
        color: #000000;
    }

    .footer {
        text-align: center;
        color: #000000;
        font-size: 14px;
        margin-top: 45px;
        padding-top: 22px;
        border-top: 1px solid #dce4ec;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e7ef;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(31, 50, 81, 0.05);
        color: #000000;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: #000000;
    }

    h1, h2, h3, h4, p, label, span {
        color: #000000;
    }
* กล่องอัปโหลด: คงพื้นหลังเดิม */
div[data-testid="stFileUploaderDropzone"] {
    color: #ffffff !important;
}

/* ข้อความรายละเอียดไฟล์ */
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p {
    color: #ffffff !important;
}

/* ปุ่ม Upload ให้เป็นพื้นเข้ม ตัวอักษรขาว */
div[data-testid="stFileUploaderDropzone"] button {
    background-color: #171a22 !important;
    border: 1px solid #4b5563 !important;
    color: #ffffff !important;
}

/* ข้อความและไอคอนในปุ่ม */
div[data-testid="stFileUploaderDropzone"] button *,
div[data-testid="stFileUploaderDropzone"] button span,
div[data-testid="stFileUploaderDropzone"] button p {
    color: #ffffff !important;
}

div[data-testid="stFileUploaderDropzone"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 3. หัวเว็บ
# ==================================================

st.markdown(
    """
    <div class="main-title">
        🤖 AI Face Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        ระบบตรวจจับภาพใบหน้าจริงและภาพใบหน้าที่สร้างด้วยปัญญาประดิษฐ์
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 4. กำหนดอุปกรณ์ประมวลผล
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.markdown(
    f"""
    <div class="device-box">
        <b>💻 อุปกรณ์ที่ใช้ประมวลผล:</b> {device}
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 5. โหลดโมเดล ResNet50
# ==================================================

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    state_dict = torch.load(
        "best_model.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


try:
    model = load_model()
    st.success("โหลดโมเดล ResNet50 สำเร็จ")

except Exception as error:
    st.error("ไม่สามารถโหลดโมเดลได้")
    st.code(str(error))
    st.stop()


# ==================================================
# 6. เตรียมข้อมูลภาพ
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["FAKE", "REAL"]


# ==================================================
# 7. แบ่งหน้าจอซ้าย–ขวา
# ==================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ==================================================
# 8. ฝั่งซ้าย: อัปโหลดและแสดงรูปภาพ
# ==================================================

with left_column:

    st.subheader("📤 อัปโหลดรูปภาพ")

    uploaded_file = st.file_uploader(
        "เลือกรูปภาพที่ต้องการตรวจสอบ",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="รูปภาพที่เลือก",
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class="file-info">
                <b>ชื่อไฟล์:</b> {uploaded_file.name}<br>
                <b>ขนาดภาพต้นฉบับ:</b>
                {image.width} × {image.height} พิกเซล
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# 9. ฝั่งขวา: ตรวจสอบและแสดงผล
# ==================================================

with right_column:

    st.subheader("🔍 ผลการตรวจสอบ")

    if uploaded_file is None:

        st.info(
            "กรุณาอัปโหลดรูปภาพก่อนทำการตรวจสอบ"
        )

    else:

        analyze_button = st.button(
            "🔍 ตรวจสอบภาพ",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner(
                "กำลังวิเคราะห์รูปภาพ..."
            ):

                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to(device)

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


            # ==========================================
            # 10. กล่องผลลัพธ์
            # ==========================================

            if predicted_class == "REAL":

                st.markdown(
                    """
                    <div class="real-result">
                        ✅ REAL
                        <div class="result-description">
                            ภาพใบหน้าจริง
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="fake-result">
                        ⚠️ FAKE
                        <div class="result-description">
                            ภาพใบหน้าที่สร้างด้วย AI
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ==========================================
            # 11. Confidence
            # ==========================================

            st.metric(
                label="ค่าความเชื่อมั่นของโมเดล",
                value=f"{confidence:.2f}%"
            )


            # ==========================================
            # 12. Probability
            # ==========================================

            st.write("### ความน่าจะเป็นของแต่ละคลาส")

            st.write(
                f"FAKE: {fake_probability:.2f}%"
            )

            st.progress(
                fake_probability / 100
            )

            st.write(
                f"REAL: {real_probability:.2f}%"
            )

            st.progress(
                real_probability / 100
            )


            # ==========================================
            # 13. ตารางสรุปผล
            # ==========================================

            st.write("### ตารางสรุปผล")

            result_data = {
                "รายการ": [
                    "ผลการทำนาย",
                    "ค่าความเชื่อมั่น",
                    "ความน่าจะเป็น FAKE",
                    "ความน่าจะเป็น REAL",
                    "ขนาดภาพที่ส่งเข้าโมเดล",
                    "อุปกรณ์ประมวลผล"
                ],
                "ผลลัพธ์": [
                    predicted_class,
                    f"{confidence:.2f}%",
                    f"{fake_probability:.2f}%",
                    f"{real_probability:.2f}%",
                    "224 × 224 พิกเซล",
                    str(device)
                ]
            }

            st.table(result_data)


            # ==========================================
            # 14. ระดับความมั่นใจ
            # ==========================================

            if confidence < 60:

                st.warning(
                    "โมเดลมีความมั่นใจต่ำ "
                    "ควรตรวจสอบด้วยวิธีอื่นเพิ่มเติม"
                )

            elif confidence < 80:

                st.info(
                    "โมเดลมีความมั่นใจในระดับปานกลาง"
                )

            else:

                st.success(
                    "โมเดลมีความมั่นใจในผลการตรวจสอบค่อนข้างสูง"
                )


# ==================================================
# 15. ขั้นตอนการทำงาน
# ==================================================

st.write("---")

st.subheader("📌 ขั้นตอนการทำงานของระบบ")

st.markdown(
    """
    1. ผู้ใช้อัปโหลดรูปภาพใบหน้าที่ต้องการตรวจสอบ  
    2. ระบบแปลงรูปภาพเป็นโหมดสี RGB  
    3. ระบบปรับขนาดภาพเป็น 224 × 224 พิกเซล  
    4. ระบบแปลงรูปภาพให้อยู่ในรูปแบบ Tensor  
    5. รูปภาพถูกส่งเข้าสู่โมเดล ResNet50  
    6. โมเดลคำนวณความน่าจะเป็นของคลาส REAL และ FAKE  
    7. ระบบเลือกคลาสที่มีค่าความน่าจะเป็นสูงที่สุด  
    8. หน้าเว็บแสดงผลการตรวจสอบพร้อมค่าความเชื่อมั่น  
    """
)


# ==================================================
# 16. หมายเหตุ
# ==================================================

st.warning(
    "ผลลัพธ์ของระบบเป็นการวิเคราะห์จากแบบจำลองปัญญาประดิษฐ์ "
    "จึงอาจเกิดความคลาดเคลื่อนได้ โดยเฉพาะภาพที่มีคุณภาพต่ำ "
    "มีแสงไม่เหมาะสม หรือมีลักษณะแตกต่างจากข้อมูลที่ใช้ฝึกโมเดล"
)


# ==================================================
# 17. Footer
# ==================================================

st.markdown(
    """
    <div class="footer">
        AI-Generated Face Detection System<br>
        Developed using PyTorch, ResNet50 and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)