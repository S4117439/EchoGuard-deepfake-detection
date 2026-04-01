import librosa
import numpy as np

audio_path = "data/test.wav"

# Load audio
y, sr = librosa.load(audio_path, sr=16000)

# Extract MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

print("Audio loaded")
print("Sample rate:", sr)
print("MFCC shape:", mfcc.shape)

# Convert to fixed-size feature vector
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_std = np.std(mfcc, axis=1)
features = np.concatenate((mfcc_mean, mfcc_std))

print("Feature vector length:", len(features))