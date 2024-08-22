import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv

def configure():
    load_dotenv()

configure()    

# Load your model (change 'your_model.h5' to your actual model file)
model = tf.keras.models.load_model('AIR_Aug.keras')
# Correctly access the secret
api_key = st.secrets["my_secret"]

genai.configure(api_key=api_key)
# Define the image size expected by your model
IMG_SIZE = (224, 224)  # Update this according to your model's input size

# Define class names based on your model
class_names = np.array([
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
    'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
    'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
    'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse',
    'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot',
    'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat',
    'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
    'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale',
    'wolf', 'wombat', 'woodpecker', 'zebra'
])

# Function to capitalize the first letter of each string
def capitalize_first_letter(strings):
    # Vectorize the string operation
    vectorized_capitalize = np.vectorize(lambda s: s.capitalize())
    return vectorized_capitalize(strings)

class_names = capitalize_first_letter(class_names)

def preprocess_image(img):
    # Convert image to RGB
    img = img.convert('RGB')
    # Resize image to the expected input shape
    img = img.resize((IMG_SIZE))
    # Convert image to numpy array
    img_array = np.array(img)
    # Scale the image
    img_array = img_array
    # Expand dimensions to fit model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(img):
    # Preprocess the image
    img_array = preprocess_image(img)
    # Make prediction
    preds = model.predict(img_array)
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Streamlit application
st.title('Animal Classification Web App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = make_prediction(image)
    google = genai.GenerativeModel('gemini-1.5-flash')
    response = google.generate_content(f'In about 40 words tell me something fun about {prediction}s')
    
    # Display prediction
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Fun Fact: 📖** {response.text}")

    st.write("")
    st.write("By RickmwasOfficial 2024")
