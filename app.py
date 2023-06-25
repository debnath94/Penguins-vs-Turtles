# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 21:05:04 2023

@author: debna
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('P.h2')

def main():
    st.title("Penguins vs Turtles")
    st.sidebar.title("Options")

    # Add input options to the sidebar
    upload_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button = st.sidebar.button('Predict')

        if predict_button:
            # Preprocess the image
            image = np.array(image.resize((75, 75)))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Make predictions
            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            classes = ['Penguin', 'Turtle']
            result = classes[class_index]

            st.success(f"Prediction: {result}")

if __name__ == '__main__':
    main()
