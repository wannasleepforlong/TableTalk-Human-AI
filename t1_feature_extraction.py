import numpy as np
import os
import librosa
import soundfile as sf
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = "CREMA-D"
SAVE_PATH = "processed"

os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------------------
# Feature Extraction
# -------------------------------
import numpy as np
import librosa
import soundfile as sf

def extract_features(file_path, n_mfcc=40):
    """
    Extracts a rich set of audio features for emotion recognition.
    Features included:
      - MFCC + Delta MFCC (mean & std)
      - Chroma (mean & std)
      - Spectral centroid, bandwidth, rolloff, contrast (mean & std)
      - Zero Crossing Rate (mean & std)
      - RMS energy (mean & std)
      - Pitch (fundamental frequency) (mean & std)
      - Duration (seconds)
      - Tonnetz (mean & std)
    """
    try:
        audio, sr = sf.read(file_path)

        # Stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Trim silence
        audio, _ = librosa.effects.trim(audio)

        feats = []

        # Duration
        duration = len(audio) / sr
        feats.append(duration)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc, axis=1))
        feats.extend(np.mean(mfcc_delta, axis=1))
        feats.extend(np.std(mfcc_delta, axis=1))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        feats.extend(np.mean(chroma, axis=1))
        feats.extend(np.std(chroma, axis=1))

        # Spectral features
        centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        contrast  = librosa.feature.spectral_contrast(y=audio, sr=sr)
        feats.extend([np.mean(centroid), np.std(centroid)])
        feats.extend([np.mean(bandwidth), np.std(bandwidth)])
        feats.extend([np.mean(rolloff), np.std(rolloff)])
        feats.extend([np.mean(contrast), np.std(contrast)])

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        feats.extend([np.mean(zcr), np.std(zcr)])

        # RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        feats.extend([np.mean(rms), np.std(rms)])

        # Pitch (fundamental frequency)
        try:
            f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'),
                             fmax=librosa.note_to_hz('C7'), sr=sr)
            voiced = f0[f0 > 0]
            pitch_mean = float(np.mean(voiced)) if len(voiced) else 0.0
            pitch_std  = float(np.std(voiced))  if len(voiced) else 0.0
        except:
            pitch_mean, pitch_std = 0.0, 0.0
        feats.extend([pitch_mean, pitch_std])

        # Tonnetz (optional)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            feats.extend(np.mean(tonnetz, axis=1))
            feats.extend(np.std(tonnetz, axis=1))
        except:
            feats.extend([0]*12)  # 6 mean + 6 std

        return np.array(feats, dtype=np.float32)

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None        
        

def get_label(file_name):
    return file_name.split("_")[2]

# -------------------------------
# Build Dataset
# -------------------------------
features = []
labels = []

files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]

for file in tqdm(files):
    file_path = os.path.join(DATASET_PATH, file)

    feat = extract_features(file_path)
    if feat is None:
        continue

    features.append(feat)
    labels.append(get_label(file))

X = np.array(features)
y = np.array(labels)

print("Final dataset shape:", X.shape, y.shape)

import pandas as pd
num_mfcc = 40
mfcc_cols       = [f"mfcc_mean_{i+1}" for i in range(num_mfcc)] + [f"mfcc_std_{i+1}" for i in range(num_mfcc)]
mfcc_delta_cols = [f"mfcc_delta_mean_{i+1}" for i in range(num_mfcc)] + [f"mfcc_delta_std_{i+1}" for i in range(num_mfcc)]
chroma_cols     = [f"chroma_mean_{i+1}" for i in range(12)] + [f"chroma_std_{i+1}" for i in range(12)]
spectral_cols   = ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
                   "rolloff_mean", "rolloff_std", "contrast_mean", "contrast_std"]
zcr_rms_cols    = ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
pitch_cols      = ["pitch_mean", "pitch_std"]
tonnetz_cols    = [f"tonnetz_mean_{i+1}" for i in range(6)] + [f"tonnetz_std_{i+1}" for i in range(6)]
duration_col    = ["duration"]

all_columns = duration_col + mfcc_cols + mfcc_delta_cols + chroma_cols + spectral_cols + zcr_rms_cols + pitch_cols + tonnetz_cols + ["label"]
df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]), columns=all_columns)
csv_path = os.path.join(SAVE_PATH, "features.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Features saved as CSV: {csv_path}")


np.save(os.path.join(SAVE_PATH, "X.npy"), X)
np.save(os.path.join(SAVE_PATH, "y.npy"), y)

print("✅ Dataset saved in 'processed/'")