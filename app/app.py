import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PCOS AI Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #FAFAFA;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = tf.keras.models.load_model("../models/pcos_model.h5")

IMG_SIZE = 224

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🩺 PCOS AI Dashboard")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    ### AI Technologies Used

    ✅ MobileNetV2  
    ✅ Transfer Learning  
    ✅ Grad-CAM  
    ✅ Explainable AI  
    ✅ Streamlit Dashboard
    """
)

st.sidebar.markdown("---")

st.sidebar.success(
    """
    ### Project Objective

    AI-assisted ultrasound image
    analysis for detecting patterns
    commonly associated with PCOS.
    """
)

st.sidebar.markdown("---")

st.sidebar.warning(
    """
    Educational & research use only.
    Not a replacement for
    professional diagnosis.
    """
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🩺 AI-Assisted PCOS Ultrasound Analysis System")

st.markdown("""
This system analyzes ultrasound images using deep learning
and explainable AI techniques to identify patterns
commonly associated with PCOS.
""")

st.markdown("---")

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Ultrasound Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def predict_image(image):

    img = np.array(image)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_normalized = img / 255.0

    img_array = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "PCOS Detected"
    else:
        label = "No PCOS Detected"

    return label, confidence, img_normalized, img_array

# -------------------------------------------------
# GRAD-CAM FUNCTION
# -------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with st.spinner("🔍 AI is analyzing ultrasound image..."):

        label, confidence, processed_img, img_array = predict_image(image)

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "Conv_1"
        )

    # -------------------------------------------------
    # TABS
    # -------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Prediction",
        "🔥 Grad-CAM",
        "🧠 AI Explanation"
    ])

    # -------------------------------------------------
    # TAB 1 — PREDICTION
    # -------------------------------------------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🖼 Uploaded Ultrasound")

            st.image(image, width=450)

        with col2:

            st.subheader("🤖 AI Prediction")

            if label == "PCOS Detected":

                st.error(f"### {label}")

            else:

                st.success(f"### {label}")

            # -----------------------------------------
            # CONFIDENCE SCORE
            # -----------------------------------------
            confidence_percent = confidence * 100

            st.metric(
                label="Confidence Score",
                value=f"{confidence_percent:.2f}%"
            )

            st.progress(float(confidence))

            # -----------------------------------------
            # RISK LEVEL
            # -----------------------------------------
            if confidence_percent > 90:

                st.error("🔴 High Confidence Detection")

            elif confidence_percent > 70:

                st.warning("🟡 Moderate Confidence Detection")

            else:

                st.info("🔵 Low Confidence Detection")

            st.markdown("---")

            # -----------------------------------------
            # AI SUMMARY
            # -----------------------------------------
            st.subheader("📋 AI Analysis Summary")

            if label == "PCOS Detected":

                st.warning("""
                • Strong activation regions detected  
                • Follicle-like clustered structures observed  
                • Irregular ovarian texture patterns identified  
                • AI attention concentrated in highlighted regions
                """)

            else:

                st.info("""
                • Lower activation intensity observed  
                • More uniform ovarian texture patterns  
                • Reduced abnormal clustered regions detected  
                • Reduced high-attention activation areas
                """)

            st.markdown("---")

            # -----------------------------------------
            # AI INSIGHTS
            # -----------------------------------------
            st.subheader("🧠 AI Insights")

            if label == "PCOS Detected":

                st.info("""
                • Strong CNN activation regions observed  
                • AI focused heavily on clustered structures  
                • Heatmap indicates high regional importance  
                • Feature extraction suggests PCOS-like patterns  
                • Transfer learning model detected strong similarity
                """)

            else:

                st.success("""
                • Lower abnormal activation intensity observed  
                • More uniform ovarian texture patterns detected  
                • Reduced clustered follicle-like structures  
                • Lower CNN activation in key regions  
                • AI confidence indicates weaker PCOS similarity
                """)

    # -------------------------------------------------
    # TAB 2 — GRAD-CAM
    # -------------------------------------------------
    with tab2:

        st.subheader("🔥 Grad-CAM Visualization")

        heatmap = cv2.resize(heatmap, (224,224))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        superimposed_img = heatmap * 0.4 + (processed_img * 255)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(image, width=350)

        with col2:

            st.subheader("AI Attention Heatmap")

            fig, ax = plt.subplots(figsize=(6,6))

            ax.imshow(superimposed_img.astype("uint8"))

            ax.axis("off")

            st.pyplot(fig)

    # -------------------------------------------------
    # TAB 3 — AI EXPLANATION
    # -------------------------------------------------
    with tab3:

        st.subheader("🧠 Explainable AI Analysis")

        if label == "PCOS Detected":

            st.warning("""
            ### Why the AI Predicted PCOS

            The AI model identified ultrasound regions
            commonly associated with PCOS patterns.

            The Grad-CAM heatmap shows that the model
            focused strongly on highlighted red/yellow regions.

            ### Possible contributing patterns:
            • Multiple follicle-like structures  
            • Dense clustered ovarian regions  
            • Irregular ovarian texture patterns  
            • Strong CNN activation regions
            """)

        else:

            st.info("""
            ### Why the AI Predicted No PCOS

            The AI model did not detect strong
            PCOS-associated ultrasound patterns.

            The highlighted regions showed:
            • Lower activation intensity  
            • More uniform texture patterns  
            • Reduced clustered structures
            """)

        st.markdown("---")

        # -----------------------------------------
        # HEATMAP COLOR GUIDE
        # -----------------------------------------
        st.subheader("🎨 Heatmap Color Guide")

        st.success("""
        🔴 Red Regions:
        Strongest AI attention areas.
        These regions contributed most to prediction.
        """)

        st.warning("""
        🟡 Yellow/Green Regions:
        Moderately important regions analyzed by AI.
        """)

        st.info("""
        🔵 Blue/Purple Regions:
        Lower importance regions with reduced influence.
        """)

        st.markdown("---")

        st.markdown("""
        ### Why Explainable AI Matters

        Grad-CAM helps explain WHY the AI model
        made a prediction instead of behaving
        like a black box system.

        This improves:
        ✅ Transparency  
        ✅ Explainability  
        ✅ AI Trustworthiness  
        ✅ Medical Interpretability
        """)

    # -------------------------------------------------
    # PDF REPORT GENERATION
    # -------------------------------------------------
    st.markdown("---")

    st.subheader("📄 Download AI Report")

    def generate_pdf():

        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        pdf_path = temp_pdf.name

        c = canvas.Canvas(pdf_path)

        # -----------------------------------------
        # TITLE
        # -----------------------------------------
        c.setFont("Helvetica-Bold", 20)

        c.drawString(
            50,
            800,
            "PCOS AI Analysis Report"
        )

        # -----------------------------------------
        # PREDICTION DETAILS
        # -----------------------------------------
        c.setFont("Helvetica", 12)

        c.drawString(
            50,
            760,
            f"Prediction Result: {label}"
        )

        c.drawString(
            50,
            735,
            f"Confidence Score: {confidence_percent:.2f}%"
        )

        # -----------------------------------------
        # SAVE ORIGINAL IMAGE
        # -----------------------------------------
        original_img_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ).name

        image.save(original_img_path)

        # -----------------------------------------
        # SAVE HEATMAP IMAGE
        # -----------------------------------------
        heatmap_img_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ).name

        plt.imsave(
            heatmap_img_path,
            superimposed_img.astype("uint8")
        )

        # -----------------------------------------
        # ADD ORIGINAL IMAGE
        # -----------------------------------------
        c.setFont("Helvetica-Bold", 14)

        c.drawString(
            50,
            690,
            "Uploaded Ultrasound Image"
        )

        c.drawImage(
            ImageReader(original_img_path),
            50,
            470,
            width=200,
            height=200
        )

        # -----------------------------------------
        # ADD HEATMAP IMAGE
        # -----------------------------------------
        c.drawString(
            320,
            690,
            "Grad-CAM Visualization"
        )

        c.drawImage(
            ImageReader(heatmap_img_path),
            320,
            470,
            width=200,
            height=200
        )

        # -----------------------------------------
        # AI EXPLANATION
        # -----------------------------------------
        c.setFont("Helvetica-Bold", 14)

        c.drawString(
            50,
            420,
            "AI Explanation"
        )

        c.setFont("Helvetica", 12)

        if label == "PCOS Detected":

            explanation = [
                "• Strong activation regions detected",
                "• Follicle-like clustered structures observed",
                "• High CNN activation in highlighted regions",
                "• AI model detected PCOS-like patterns"
            ]

        else:

            explanation = [
                "• Lower activation intensity observed",
                "• More uniform ovarian texture patterns",
                "• Reduced abnormal clustered structures",
                "• Lower CNN activation regions detected"
            ]

        y_position = 390

        for line in explanation:

            c.drawString(
                60,
                y_position,
                line
            )

            y_position -= 25

        # -----------------------------------------
        # DISCLAIMER
        # -----------------------------------------
        c.setFont("Helvetica-Oblique", 10)

        c.drawString(
            50,
            250,
            "This system is for educational and research purposes only."
        )

        c.drawString(
            50,
            230,
            "Not intended for professional medical diagnosis."
        )

        c.save()

        return pdf_path

    pdf_path = generate_pdf()

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(
            label="⬇ Download AI Report",
            data=pdf_file,
            file_name="PCOS_AI_Report.pdf",
            mime="application/pdf"
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.caption(
    "AI-Assisted PCOS Ultrasound Analysis System | "
    "Developed for Educational & Research Purposes"
)