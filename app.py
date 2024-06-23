import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import io 
import requests

# Load the VGG-16 model
try:
    model = VGG16(weights='imagenet')
except Exception as e:
    st.error(f"Error: {e} while loading the model")
    st.stop()

# Fetch the labels
labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
response = requests.get(labels_url)
target_info = response.json()
target_labels = [target_info[str(i)][1] for i in range(len(target_info))]

def load_and_preprocess(image):
    try:
        img = image.resize((224, 224))  # Resize the image to the target size
        img_arr = img_to_array(img)  # Convert the image to an array
        img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension
        img_arr = preprocess_input(img_arr)  # Preprocess the image for the model
        return img_arr
    except Exception as e:
        st.error(f"Error in preprocessing the image: {e}")
        return None

# Streamlit application process
st.title("Let's Classify Some Images")

uploaded_file = st.file_uploader('Choose an image you want to classify under a category', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write('Classifying...')

        processed = load_and_preprocess(img)
        if processed is not None:
            predictions = model.predict(processed)
            decoded_pred = decode_predictions(predictions, top=1)
            
            for id, name, score in decoded_pred[0]:
                st.write(f'Your image belongs to ›››››› {name}:{score*100:.2f}')
        else:
            st.error("Failed to preprocess the image.")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
