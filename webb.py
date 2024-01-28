import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

class_names = ["Normal", "Pneumonia"]

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "C:\\Users\\Allan\\Documents\\Phase5project\\Pneumonia_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.markdown(
    """
    <style>
        body {
            background-color: #1E90FF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("""
        # Pneumonia Identification Web App
        Upload your chest scan and click the "Predict" button to identify pneumonia.
        """
)

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model, class_names):
    # Resize the image to the required size
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the expected input shape of the model
    img_resize = (cv2.resize(img, dsize=(180, 180), interpolation=cv2.INTER_CUBIC))/255.

    # Add channel dimension and reshape for prediction
    img_reshape = img_resize[np.newaxis, ...]

    # Make the prediction
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Add a "Predict" button
    if st.button("Predict"):
        with st.spinner("Predicting..."):

            predictions = import_and_predict(image, model, class_names)
            score = tf.nn.softmax(predictions[0])
            st.write(predictions)
            st.write(score)
            st.write(
             f"This chest scan is classified as {class_names[np.argmax(score)]} with {100 * np.max(score):.2f}% confidence."
    )
