import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    print(f"Loaded audio file with shape: {y.shape}, sample rate: {sr}")
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

X, y = [], []

for file in os.listdir("emotion_dataset/"):
    if file.endswith(".wav"):
        parts = file.split("-")
        if len(parts) > 2:  # Ensure the filename has enough segments
            emotion_code = parts[2]
            if emotion_code in emotions:
                print(f"Processing file: {file}")
                feature = extract_features(f"emotion_dataset/{file}")
                X.append(feature)
                label = list(emotions.keys()).index(emotion_code)
                y.append(label)
            else:
                print(f"Skipping file with unknown emotion code: {file}")
        else:
            print(f"Skipping file with invalid format: {file}")

print(f"Total valid samples: {len(X)}")
print(f"Training data labels: {y}")

# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# Print the shape of the feature matrix
print(f"Feature shape: {X.shape}")

if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid data found in 'emotion_dataset/'. Ensure the directory contains properly formatted .wav files.")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
emotion_model = RandomForestClassifier()
emotion_model.fit(X_train, y_train)
joblib.dump(emotion_model, "emotion_model.pkl")
print("Emotion model saved as emotion_model.pkl ✅")

y_pred = emotion_model.predict(X_test)

# Determine unique classes in y_test
labels = np.unique(y_test)  # Get unique labels in y_test
target_names = [list(emotions.values())[i] for i in labels]

print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

test_features = extract_features("emotion_dataset/male-01-01-01.wav").reshape(1, -1)
predicted_emotion = emotion_model.predict(test_features)[0]
print(f"Predicted emotion: {list(emotions.values())[predicted_emotion]}")
