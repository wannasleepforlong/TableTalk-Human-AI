import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📌 Emotion mapping
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# 📌 Feature Extraction (MFCC)
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

# 📌 Load Dataset
data_path = "CREMA-D"   # change if needed

features = []
labels = []

for file in tqdm(os.listdir(data_path)):
    if file.endswith(".wav"):
        file_path = os.path.join(data_path, file)
        
        # Extract emotion code
        parts = file.split("_")
        emotion_code = parts[2]
        
        if emotion_code in emotion_map:
            feature = extract_features(file_path)
            
            if feature is not None:
                features.append(feature)
                labels.append(emotion_map[emotion_code])

# Convert to numpy
X = np.array(features)
y = np.array(labels)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 📌 Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 📌 Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 📌 Predictions
y_pred = model.predict(X_test)

# 📌 Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", accuracy)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))