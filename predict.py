import numpy as np
import joblib
import librosa

MODEL_PATH = "artifacts/echoguard_model.pkl"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


model = load_model()


def extract_features(file_path, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        if y is None or len(y) == 0:
            raise ValueError("Uploaded audio file is empty or unreadable.")

        duration = librosa.get_duration(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [centroid, bandwidth, rolloff, rms, zcr]
        ]).astype(np.float32)

        return {
            "features": features.reshape(1, -1),
            "sample_rate": sr,
            "duration": duration
        }

    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")


def predict_audio(file_path):
    extracted = extract_features(file_path)

    features = extracted["features"]
    sample_rate = extracted["sample_rate"]
    duration = extracted["duration"]

    expected_features = getattr(model, "n_features_in_", features.shape[1])
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Feature length mismatch: model expects {expected_features}, got {features.shape[1]}"
        )

    try:
        prediction = int(model.predict(features)[0])

        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        class_index = list(classes).index(prediction)
        confidence = float(probabilities[class_index])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "sample_rate": int(sample_rate),
            "duration": round(float(duration), 2)
        }

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")