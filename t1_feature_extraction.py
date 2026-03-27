import numpy as np
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import pandas as pd

DATASET_PATH = "test"
PROCESSED_PATH = "test_processed"
SEGMENT_DURATION = 15.0  
SEGMENT_HOP      = 12.0 

os.makedirs(PROCESSED_PATH, exist_ok=True)

def segment_audio(audio, sr, max_duration=SEGMENT_DURATION, hop=SEGMENT_HOP):
    """Splits audio into overlapping segments."""
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
        if len(segment) / sr >= 1.0:  # discard very short segments
            segments.append(segment)
        start += hop_samples

    return segments if segments else [audio]

def extract_features_from_array(audio, sr, n_mfcc=40):
    """Extracts audio features from a raw audio array."""
    feats = []

    # Duration
    duration = len(audio) / sr
    feats.append(duration)

    # MFCC + delta
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
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        feats.extend(np.mean(tonnetz, axis=1))
        feats.extend(np.std(tonnetz, axis=1))
    except Exception:
        feats.extend([0.0] * 12)  # fallback for tonnetz

    return np.array(feats, dtype=np.float32)

def extract_features(file_path, n_mfcc=40, save_audio=False, processed_path=None):
    """Load audio, segment if needed, extract features."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio, _ = librosa.effects.trim(audio)
        segments = segment_audio(audio, sr)

        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        if save_audio and processed_path:
            save_segments(file_name, segments, sr, processed_path)

        results = []
        for i, seg in enumerate(segments):
            feat = extract_features_from_array(seg, sr, n_mfcc=n_mfcc)
            chunk_name = f"{base_name}_chunk_{i+1}.wav" if len(segments) > 1 else f"{base_name}.wav"
            results.append((chunk_name, feat))
        return results

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_segments(file_name, segments, sr, processed_path):
    base_name = os.path.splitext(file_name)[0]
    if len(segments) == 1:
        sf.write(os.path.join(processed_path, f"{base_name}.wav"), segments[0], sr)
    else:
        for i, seg in enumerate(segments):
            sf.write(os.path.join(processed_path, f"{base_name}_chunk_{i+1}.wav"), seg, sr)

if __name__ == "__main__":
    filenames = []
    features  = []

    files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".flac", ".mp3"))]

    for file in tqdm(files, desc="Extracting features"):
        file_path = os.path.join(DATASET_PATH, file)
        processed_results = extract_features(file_path, save_audio=True, processed_path=PROCESSED_PATH)
        if processed_results is None:
            continue
        for chunk_name, feat in processed_results:
            filenames.append(chunk_name)
            features.append(feat)

    X = np.array(features)

    # Define CSV column names
    num_mfcc = 40
    mfcc_cols       = [f"mfcc_mean_{i+1}"       for i in range(num_mfcc)] + [f"mfcc_std_{i+1}"        for i in range(num_mfcc)]
    mfcc_delta_cols = [f"mfcc_delta_mean_{i+1}" for i in range(num_mfcc)] + [f"mfcc_delta_std_{i+1}"  for i in range(num_mfcc)]
    chroma_cols     = [f"chroma_mean_{i+1}"     for i in range(12)] + [f"chroma_std_{i+1}"      for i in range(12)]
    spectral_cols   = ["centroid_mean",  "centroid_std",
                       "bandwidth_mean", "bandwidth_std",
                       "rolloff_mean",   "rolloff_std",
                       "contrast_mean",  "contrast_std"]
    zcr_rms_cols    = ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
    pitch_cols      = ["pitch_mean", "pitch_std"]
    tonnetz_cols    = [f"tonnetz_mean_{i+1}" for i in range(6)] + [f"tonnetz_std_{i+1}"  for i in range(6)]
    duration_col    = ["duration"]

    all_columns = ["filename"] + duration_col + mfcc_cols + mfcc_delta_cols + chroma_cols + spectral_cols + zcr_rms_cols + pitch_cols + tonnetz_cols

    df = pd.DataFrame(X, columns=all_columns[1:])
    df.insert(0, "filename", filenames)
    csv_path = os.path.join("outputs", "features.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Features CSV saved: {csv_path}")