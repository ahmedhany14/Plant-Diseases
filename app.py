import streamlit as st

import pandas as pd
import numpy as np
import cv2
import os
import json

import tensorflow as tf
from io import StringIO
from PIL import Image


st.title("Plant Disease classification")


def load_model():
    model = tf.keras.models.load_model("./models/model.keras")
    return model


model = load_model()
classes = json.load(open("./dataset/classes.json"))


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    img = Image.open(uploaded_file)

    # get the image as a numpy array to be able to use it with opencv
    img = np.array(img)
    st.image(img, caption="Uploaded Image.", use_column_width=True)


def predict(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    prediction = np.argmax(prediction)
    return classes[str(prediction)]


if uploaded_file is not None:
    prediction = predict(img)
    st.write("The predicted class is: ", prediction)