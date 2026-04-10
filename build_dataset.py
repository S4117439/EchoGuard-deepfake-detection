import os
import numpy as np
import librosa

DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
OUT_DIR = "artifacts"

os.makedirs(OUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")


def extract_features(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=16000)

    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty or unreadable.")

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

    return features


X = []
labels = []
files_used = []
files_skipped = []


def process_folder_recursive(folder_path, label, label_name):
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            rel_dir = os.path.relpath(root, folder_path)
            rel_path = os.path.join(label_name, rel_dir, file) if rel_dir != "." else os.path.join(label_name, file)

            if not file.lower().endswith(ALLOWED_EXTENSIONS):
                files_skipped.append(f"{rel_path} - skipped (unsupported extension)")
                continue

            path = os.path.join(root, file)

            try:
                features = extract_features(path)
                X.append(features)
                labels.append(label)
                files_used.append(rel_path)
            except Exception as e:
                files_skipped.append(f"{rel_path} - skipped ({str(e)})")


process_folder_recursive(REAL_DIR, 0, "real")
process_folder_recursive(FAKE_DIR, 1, "fake")

X = np.array(X)
y = np.array(labels)

print("Total samples:", len(X))
print("Real samples:", int(np.sum(y == 0)))
print("Fake samples:", int(np.sum(y == 1)))
print("Feature vector length:", X.shape[1] if len(X) > 0 else 0)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

with open(os.path.join(OUT_DIR, "files_used.txt"), "w") as f:
    for line in files_used:
        f.write(line + "\n")

with open(os.path.join(OUT_DIR, "files_skipped.txt"), "w") as f:
    for line in files_skipped:
        f.write(line + "\n")

print("Saved to artifacts/X.npy and artifacts/y.npy")
print("Saved file lists to artifacts/files_used.txt and artifacts/files_skipped.txt")