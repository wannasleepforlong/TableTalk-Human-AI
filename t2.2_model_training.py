import os
import pickle
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils     import to_categorical
import librosa
import soundfile as sf
from t1_feature_extraction import extract_features_from_array
import random
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "test_processed"
OUTPUT_DIR    = "outputs"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "emotion_model.keras")  
ENCODER_PATH  = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
SCALER_PATH   = os.path.join(OUTPUT_DIR, "scaler.pkl")
PLOT_PATH     = os.path.join(OUTPUT_DIR, "training_history.png")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
features_df = pd.read_csv(os.path.join("outputs", "features_scaled.csv"))
labels_df   = pd.read_csv(os.path.join("outputs", "bart_output.csv"))

features_df['filename'] = features_df['filename'].astype(str)
labels_df['filename']   = labels_df['filename'].astype(str)

merged_df = pd.merge(features_df, labels_df[['filename', 'tone']], 
                     on='filename', how='inner')
feature_files = set(features_df['filename'])
label_files   = set(labels_df['filename'])

not_in_labels   = feature_files - label_files
not_in_features = label_files - feature_files

print(f"❌ Not found in labels      : {len(not_in_labels)}")
print(f"❌ Not found in features    : {len(not_in_features)}")

X_df = merged_df.drop(columns=['filename', 'tone'])
X = X_df.values.astype(np.float32)

y_raw = merged_df['tone'].values

if len(y_raw) == 0:
    print("❌ ERROR: No matching samples found!")
    exit(1)

print(f"\nFinal X shape: {X.shape},  y shape: {y_raw.shape}")
print(f"Classes : {np.unique(y_raw)}")

if len(y_raw) == 0:
    print("❌ ERROR: No matching samples found after join!")
    print(f"Features head:\n{features_df['filename'].head()}")
    print(f"Labels head:\n{labels_df['filename'].head()}")
    exit(1)

print(f"Final X shape: {X.shape},  y shape: {y_raw.shape}")
print(f"Classes : {np.unique(y_raw)}")

# ─── ENCODE LABELS ────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
n_classes = len(le.classes_)
y_cat     = to_categorical(y_encoded, num_classes=n_classes)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"Classes ({n_classes}): {list(le.classes_)}")

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, y_enc_train, y_enc_test = train_test_split(
    X, y_cat, y_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_encoded,
)

# ─── SCALE FEATURES (fit on train only — no leakage) ─────────────────────────
# NOTE: If you ran extract_features.py with scaling already applied,
#       skip the block below and load the saved scaler instead.
#       Here we refit on the training split to be safe.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print("Scaler fit on training data and saved.")

# ─── CLASS WEIGHTS ────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_enc_train),
    y=y_enc_train,
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ─── MODEL ────────────────────────────────────────────────────────────────────
input_dim = X_train.shape[1]   # 193 with the improved feature set

model = Sequential([
    Input(shape=(input_dim,)),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    # Output
    Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,              # EarlyStopping will stop well before this
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1,
)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*40}")
print(f"Test Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print(f"{'='*40}\n")

# Predictions
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# 🔍 DEBUG (important)
print("Unique y_test:", np.unique(y_enc_test))
print("Unique y_pred:", np.unique(y_pred))
print("All classes  :", list(le.classes_))

# ✅ FIX: Use LabelEncoder classes directly
print("📊 Classification Report:\n")
report = classification_report(
    y_enc_test,
    y_pred,
    labels=np.arange(len(le.classes_)),   # ensures correct alignment
    target_names=le.classes_,
    digits=4,
    zero_division=0
)
print(report)

# ─── CONFUSION MATRIX ────────────────────────────────────────────────────────
cm = confusion_matrix(
    y_enc_test,
    y_pred,
    labels=np.arange(len(le.classes_))
)

print("📊 Confusion Matrix:\n")
print(cm)

cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\n📊 Confusion Matrix (Readable):\n")
print(cm_df)

# ─── PER-CLASS ACCURACY ──────────────────────────────────────────────────────
print("\n📊 Per-class Accuracy:\n")
for i, cls in enumerate(le.classes_):
    mask = (y_enc_test == i)
    if np.sum(mask) == 0:
        print(f"{cls:<20s}: No samples")
        continue
    cls_acc = np.mean(y_pred[mask] == i)
    print(f"{cls:<20s}: {cls_acc*100:.2f}%")