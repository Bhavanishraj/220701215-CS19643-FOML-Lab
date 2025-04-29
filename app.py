import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile

# Load models
gender_model = joblib.load("model.pkl")
emotion_model = joblib.load("emotion_model.pkl")  # NEW model for emotion/sentiment

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

st.set_page_config(page_title="Gender & Sentiment Recognition", page_icon="🎙️")
st.title("🎙️ Gender and Sentiment Recognition App")
st.write("Upload a `.wav` file, and the model will predict the speaker's gender and emotional tone!")

# Upload audio file
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)

        # Gender Prediction
        gender_pred = gender_model.predict(features[:, :20])[0]  # Use first 20 features
        gender = "Female 👩" if gender_pred == 0 else "Male 👨"

        # Sentiment Prediction
        sentiment_pred = emotion_model.predict(features)[0]
        sentiment_label = {
            0: "Neutral 😐",
            1: "Happy 😄",
            2: "Sad 😢",
            3: "Angry 😡"
        }[sentiment_pred]

        st.success(f"**Gender:** {gender}")
        st.info(f"**Sentiment:** {sentiment_label}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
