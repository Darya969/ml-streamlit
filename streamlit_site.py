import io
import base64
import streamlit as st
import pathlib
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from PIL import Image
from st_pages import Page, show_pages

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def add_logo():
    image = Image.open('images/pngwing.com.png')
    
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url(data:image/png;base64,{image_to_base64(image)});
                background-repeat: no-repeat;
                background-size: 80%;
                padding-top: 100%;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

show_pages(
    [
        Page("streamlit_site.py", "Home", "🏠"),
        Page("pages/info.py", "Information", ":books:"),
    ]
)

class_names = {
    0: "маргаритка",
    1: "одуванчик", 
    2: "розы", 
    3: "подсолнухи", 
    4: "тюльпаны"}

st.title('🌺 Flower Recognition Assistant 🌸')

# Define the model architecture
model = Sequential([
    Rescaling(1./255, input_shape=(180, 180, 3)),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5)  # assuming 5 classes
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Rebuild the model to match the original one before loading weights
model.build((None, 180, 180, 3))

model.load_weights("my_flowers_model.weights.h5")

def load_img():
    uploaded_file = st.file_uploader(label="Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB').resize((180, 180))
        st.image(img)
        return img
    else:
        return None
    
img = load_img()
result = st.button("Recognize")

if result and img is not None:
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names.get(predicted_class_index, "Unknown")

    st.write("На изображении скорее всего {} ({:.2f}% вероятность)".format(predicted_class_name, 100 * np.max(score)))