import sys
import numpy as np
import librosa
import joblib

def extract_features(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))).astype(np.float32)

if len(sys.argv) < 2:
    print("Usage: python predict_one.py <path_to_audio.wav>")
    sys.exit(1)

audio_path = sys.argv[1]
model = joblib.load("artifacts/echoguard_model.pkl")

features = extract_features(audio_path).reshape(1, -1)
expected_features = getattr(model, "n_features_in_", features.shape[1])
if features.shape[1] != expected_features:
    raise ValueError(
        f"Feature length mismatch: model expects {expected_features}, got {features.shape[1]}"
    )

proba = model.predict_proba(features)[0]
pred = int(np.argmax(proba))

label = "FAKE (AI)" if pred == 1 else "REAL (Human)"
print("File:", audio_path)
print("Prediction:", label)
print(f"Confidence -> Real: {proba[0]:.2f}, Fake: {proba[1]:.2f}")
