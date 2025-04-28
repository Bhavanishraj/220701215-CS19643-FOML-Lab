def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    print(f"Loaded audio file with shape: {y.shape}, sample rate: {sr}")
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])