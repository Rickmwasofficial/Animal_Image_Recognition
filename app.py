import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Configuration and Setup
def configure():
    load_dotenv()
    # Configure API key from Streamlit secrets
    api_key = st.secrets["my_secret"]
    genai.configure(api_key=api_key)

# Model Configuration
IMG_SIZE = (224, 224)  # Update according to model's input size

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

def capitalize_first_letter(strings):
    vectorized_capitalize = np.vectorize(lambda s: s.capitalize())
    return vectorized_capitalize(strings)

def load_model_and_classes():
    # Load the model
    model = tf.keras.models.load_model('AIR_Aug.keras')
    # Capitalize class names
    global class_names
    class_names = capitalize_first_letter(class_names)
    return model

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(model, img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

def get_fun_fact(animal):
    google = genai.GenerativeModel('gemini-1.5-flash')
    response = google.generate_content(f'In about 40 words tell me something fun about {animal}s')
    return response.text

def main():
    st.title('Animal Classification Web App')
    
    # Initialize configuration
    configure()
    
    # Load model
    model = load_model_and_classes()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("")
        st.write("Classifying...")
        
        # Make prediction and get fun fact
        prediction = make_prediction(model, image)
        fun_fact = get_fun_fact(prediction)
        
        # Display results
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Fun Fact: 📖** {fun_fact}")
        st.write("")
        st.write("By RickmwasOfficial 2024")

if __name__ == "__main__":
    main()
