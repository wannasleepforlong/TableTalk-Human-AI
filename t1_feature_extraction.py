import numpy as np
import os
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = "test"
SAVE_PATH = "outputs"
PROCESSED_PATH = "test_processed"
SEGMENT_DURATION = 15.0  
SEGMENT_HOP      = 15.0  # Changed to match SEGMENT_DURATION for non-overlapping chunks

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# -------------------------------
# Segmentation Helper
# -------------------------------
def segment_audio(audio, sr, max_duration=SEGMENT_DURATION, hop=SEGMENT_HOP):
    """
    Splits `audio` into overlapping segments of `max_duration` seconds,
    stepping by `hop` seconds.  If the clip is already ≤ max_duration it
    is returned as-is (single-element list).

    Returns
    -------
    List[np.ndarray]
        Each element is one audio segment.
    """
    total_duration = len(audio) / sr
    if total_duration <= max_duration:
        return [audio]

    segments = []
    max_samples = int(max_duration * sr)
    hop_samples = int(hop * sr)
    start = 0

    while start < len(audio):
        end = start + max_samples
        segment = audio[start:end]
        # Discard very short tail segments (< 1 s)
        if len(segment) / sr >= 1.0:
            segments.append(segment)
        start += hop_samples

    return segments if segments else [audio]

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features_from_array(audio, sr, n_mfcc=40):
    """
    Extracts a rich set of audio features for emotion recognition
    from a raw audio array.

    Features included:
      - Duration (seconds)
      - MFCC + Delta MFCC (mean & std)
      - Chroma (mean & std)
      - Spectral centroid, bandwidth, rolloff, contrast (mean & std)
      - Zero Crossing Rate (mean & std)
      - RMS energy (mean & std)
      - Pitch / F0 (mean & std)
      - Tonnetz (mean & std)
    """
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
    feats.extend([np.mean(centroid),  np.std(centroid)])
    feats.extend([np.mean(bandwidth), np.std(bandwidth)])
    feats.extend([np.mean(rolloff),   np.std(rolloff)])
    feats.extend([np.mean(contrast),  np.std(contrast)])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    feats.extend([np.mean(zcr), np.std(zcr)])

    # RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    feats.extend([np.mean(rms), np.std(rms)])

    # Pitch
    try:
        f0 = librosa.yin(audio,
                         fmin=librosa.note_to_hz('C2'),
                         fmax=librosa.note_to_hz('C7'),
                         sr=sr)
        voiced = f0[f0 > 0]
        pitch_mean = float(np.mean(voiced)) if len(voiced) else 0.0
        pitch_std  = float(np.std(voiced))  if len(voiced) else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0
    feats.extend([pitch_mean, pitch_std])

    # Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr
        )
        feats.extend(np.mean(tonnetz, axis=1))
        feats.extend(np.std(tonnetz, axis=1))
    except Exception:
        feats.extend([0.0] * 12)   # 6 mean + 6 std

    return np.array(feats, dtype=np.float32)


def extract_features(file_path, n_mfcc=40, save_audio=False, processed_path=None):
    """
    Loads an audio file, trims silence, applies segmentation for clips
    longer than SEGMENT_DURATION seconds, and returns a list of feature
    vectors — one per segment.
    
    If save_audio is True, it also saves the segments to processed_path.

    Returns
    -------
    List[np.ndarray] | None
        None if the file could not be processed.
    """
    try:
        # Using librosa.load for better format support (mp3, wav, etc.)
        audio, sr = librosa.load(file_path, sr=None)
        # Stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Trim leading/trailing silence
        audio, _ = librosa.effects.trim(audio)
        # Segment if necessary
        segments = segment_audio(audio, sr,
                                  max_duration=SEGMENT_DURATION,
                                  hop=SEGMENT_HOP)
        # Save audio if requested
        if save_audio and processed_path:
            file_name = os.path.basename(file_path)
            save_segments(file_name, segments, sr, processed_path)
        feature_list = []
        for seg in segments:
            feat = extract_features_from_array(seg, sr, n_mfcc=n_mfcc)
            feature_list.append(feat)
        return feature_list

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def save_segments(file_name, segments, sr, processed_path):
    """
    Saves audio segments to the processed audio path.
    If only one segment, saves with original name.
    If multiple, saves with _chunk_n suffix.
    """
    base_name = os.path.splitext(file_name)[0]
    
    if len(segments) == 1:
        out_path = os.path.join(processed_path, f"{base_name}.wav")
        sf.write(out_path, segments[0], sr)
    else:
        for i, seg in enumerate(segments):
            out_path = os.path.join(processed_path, f"{base_name}_chunk_{i+1}.wav")
            sf.write(out_path, seg, sr)


def get_label(file_name):
    try:
        return file_name.split("_")[2]
    except (IndexError, AttributeError):
        # Fallback for files without '_' or unexpected format
        return "unknown"


features = []
labels   = []
segment_counts = {}   # track how many segments each file produced

files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".wav", ".mp3"))]

for file in tqdm(files, desc="Extracting features"):
    file_path = os.path.join(DATASET_PATH, file)
    label     = get_label(file)

    feat_list = extract_features(file_path, save_audio=True, processed_path=PROCESSED_PATH)
    if feat_list is None:
        continue

    segment_counts[file] = len(feat_list)
    for feat in feat_list:
        features.append(feat)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

print(f"\nRaw dataset shape : {X.shape}  |  labels: {y.shape}")
multi_seg = {f: n for f, n in segment_counts.items() if n > 1}
print(f"Files segmented   : {len(multi_seg)} / {len(files)}")


scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

print(f"Scaled  — mean ≈ {X_scaled.mean():.4f}  std ≈ {X_scaled.std():.4f}")

# Save the scaler so it can be reused at inference time
scaler_path = os.path.join(SAVE_PATH, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved  : {scaler_path}")

num_mfcc        = 40
mfcc_cols       = ([f"mfcc_mean_{i+1}"       for i in range(num_mfcc)] +
                   [f"mfcc_std_{i+1}"        for i in range(num_mfcc)])
mfcc_delta_cols = ([f"mfcc_delta_mean_{i+1}" for i in range(num_mfcc)] +
                   [f"mfcc_delta_std_{i+1}"  for i in range(num_mfcc)])
chroma_cols     = ([f"chroma_mean_{i+1}"     for i in range(12)] +
                   [f"chroma_std_{i+1}"      for i in range(12)])
spectral_cols   = ["centroid_mean",  "centroid_std",
                   "bandwidth_mean", "bandwidth_std",
                   "rolloff_mean",   "rolloff_std",
                   "contrast_mean",  "contrast_std"]
zcr_rms_cols    = ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
pitch_cols      = ["pitch_mean", "pitch_std"]
tonnetz_cols    = ([f"tonnetz_mean_{i+1}" for i in range(6)] +
                   [f"tonnetz_std_{i+1}"  for i in range(6)])
duration_col    = ["duration"]

all_columns = (duration_col + mfcc_cols + mfcc_delta_cols +
               chroma_cols + spectral_cols + zcr_rms_cols +
               pitch_cols + tonnetz_cols + ["label"])

# Save raw (unscaled) CSV with labels
df_raw = pd.DataFrame(
    np.hstack([X, y.reshape(-1, 1)]),
    columns=all_columns
)
csv_raw = os.path.join(SAVE_PATH, "features.csv")
df_raw.to_csv(csv_raw, index=False)
print(f"✅ Raw features CSV    : {csv_raw}")

# Save scaled CSV without label column (label-free, ready for ML)
df_scaled = pd.DataFrame(X_scaled, columns=all_columns[:-1])
df_scaled["label"] = y
csv_scaled = os.path.join(SAVE_PATH, "features_scaled.csv")
df_scaled.to_csv(csv_scaled, index=False)
print(f"✅ Scaled features CSV : {csv_scaled}")

np.save(os.path.join(SAVE_PATH, "X.npy"),        X)         # raw
np.save(os.path.join(SAVE_PATH, "X_scaled.npy"), X_scaled)  # standardized
np.save(os.path.join(SAVE_PATH, "y.npy"),        y)

print("\n✅ All artifacts saved in 'processed/'")