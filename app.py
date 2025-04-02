import streamlit as st
import numpy as np
import cv2

import requests
import base64
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import tensorflow as tf

# TMDb API Key
TMDB_API_KEY = "fb2d107a97a8b9fccccdf093d0829d91"

# Load Model
model = load_model("VGG2.h5")

# Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Emotion to Movie Genre Mapping
emotion_to_genre = {
    "Angry": (28, "Action"),
    "Disgust": (27, "Horror"),
    "Fear": (9648, "Thriller"),
    "Happy": (35, "Comedy"),
    "Neutral": (18, "Drama"),
    "Sad": (10749, "Romance"),
    "Surprise": (878, "Sci-Fi")
}

# Get Movies from TMDb
def get_movies(emotion, num_movies=15):
    genre_id, genre_name = emotion_to_genre.get(emotion, (35, "Comedy"))
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=popularity.desc"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        movies = [
            {
                "title": movie["title"], 
                "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}", 
                "id": movie["id"], 
                "overview": movie.get("overview", "No description available")
            }
            for movie in data.get("results", [])[:num_movies] if movie.get("poster_path")
        ]
        return movies, genre_name
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movies: {e}")
        return [], "Comedy"

# Background Image
background_image_url = "https://www.shutterstock.com/image-vector/old-cinema-background-vector-illustration-600w-770497021.jpg"

# Function to Encode Image
def get_base64_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

# Page Config
st.set_page_config(page_title="Emotion-Based Movie Recommender", layout="wide")

# CSS for Background, Image Centering, and Animations
st.markdown(
    f"""
    <style>
        body {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: cover;
            color: white;
        }}
        .stApp {{
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
        }}
        .uploaded-img {{
            display: block;
            margin: auto;
            width: 50%;
            max-height: 50vh;
            border-radius: 20px;
            transition: transform 0.3s;
        }}
        .uploaded-img:hover {{
            transform: scale(1.05);
        }}
        .movie-card {{
            text-align: center;
            transition: transform 0.3s ease-in-out;
        }}
        .movie-card:hover {{
            transform: scale(1.1);
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
    """, 
    unsafe_allow_html=True
)

# Title
st.title("🎭 Emotion-Based Movie Recommender")
st.write("Choose how to provide an image: Upload a file or capture using your camera.")

# Image Upload or Camera Input
image_source = st.radio("Choose image source:", ["Upload Image", "Use Camera"])
img = None

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif image_source == "Use Camera":
    camera_input = st.camera_input("Take a picture...")
    if camera_input:
        img = np.frombuffer(camera_input.getvalue(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

# Emotion Detection and Movie Recommendation
if img is not None:
    resized_img = cv2.resize(img, (48, 48))  # Keep color image
    img_array = np.expand_dims(resized_img, axis=0).astype("float32") / 255.0

    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_label] if predicted_label < len(emotion_labels) else "Unknown"

    # Convert image to base64 and display properly centered
    base64_img = get_base64_image(img)
    st.markdown(f'<img class="uploaded-img" src="data:image/jpeg;base64,{base64_img}">', unsafe_allow_html=True)
    
    st.markdown(f"## 🎭 **Predicted Emotion: {predicted_emotion}**")

    recommended_movies, genre_name = get_movies(predicted_emotion)
    
    st.markdown(f"### 🎬 Since you're feeling **{predicted_emotion}**, here are some **{genre_name}** movies for you!")
    
    if recommended_movies:
        cols = st.columns(5)
        for i, movie in enumerate(recommended_movies[:15]):
            with cols[i % 5]:
                movie_url = f"https://www.themoviedb.org/movie/{movie['id']}"
                st.markdown(f"""
                    <div class="movie-card">
                        <a href="{movie_url}" target="_blank">
                            <img src="{movie['poster']}" alt="{movie['title']}" width="90%" style="border-radius:10px;">
                        </a>
                        <p><strong>{movie['title']}</strong></p>
                        <p>{movie['overview'][:150]}...</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No movies found. Try again!")

else:
    st.write("Please upload or capture an image to get predictions.")
