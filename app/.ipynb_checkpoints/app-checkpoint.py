import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("../models/pcos_model.h5")

IMG_SIZE = 224

# App title
st.title("PCOS Ultrasound Detection System")

st.write("Upload an ultrasound image to predict PCOS.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an ultrasound image",
    type=["jpg", "png", "jpeg"]
)

# Prediction function
def predict_image(image):

    img = np.array(image)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_normalized = img / 255.0

    img_array = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_array)

    confidence = prediction[0][0]

    if confidence > 0.5:
        label = "Infected (PCOS)"
    else:
        label = "Not Infected"

    return label, confidence, img_normalized, img_array

# Grad-CAM function
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

# Main app
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence, processed_img, img_array = predict_image(image)

    st.subheader("Prediction")

    st.write(f"Result: {label}")

    st.write(f"Confidence Score: {confidence:.4f}")

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        "Conv_1"
    )

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Overlay heatmap
    superimposed_img = heatmap * 0.4 + (processed_img * 255)

    st.subheader("Grad-CAM Heatmap")

    fig, ax = plt.subplots()

    ax.imshow(superimposed_img.astype("uint8"))

    ax.axis("off")

    st.pyplot(fig)