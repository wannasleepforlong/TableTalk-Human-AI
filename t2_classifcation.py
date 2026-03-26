import os
import pickle
import numpy as np

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

PROCESSED_DIR = "processed"
MODEL_PATH    = "emotion_model.keras"  
ENCODER_PATH  = "label_encoder.pkl"
SCALER_PATH   = os.path.join(PROCESSED_DIR, "scaler.pkl")
PLOT_PATH     = "training_history.png"

TARGET_SR = 22050
N_MFCC    = 40

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y_raw = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

print(f"Loaded X: {X.shape},  y: {y_raw.shape}")
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
    test_size=0.2,
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

    # Block 1
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    # Block 2
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    # Block 3
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    # Block 4
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
    epochs=150,              # EarlyStopping will stop well before this
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    # callbacks=callbacks,
    verbose=1,
)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"Test Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print(f"{'='*40}\n")

# Per-class accuracy
y_pred      = np.argmax(model.predict(X_test, verbose=0), axis=1)
for i, cls in enumerate(le.classes_):
    mask     = y_enc_test == i
    cls_acc  = np.mean(y_pred[mask] == i)
    print(f"  {cls:10s}: {cls_acc*100:.1f}%  ({mask.sum()} samples)")


from t1_feature_extraction import extract_features

def predict_emotion(file_path: str) -> str:
    """Predict emotion from a WAV file path."""
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    peak  = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    feat    = extract_features(y, sr).reshape(1, -1)
    feat    = scaler.transform(feat)
    pred_id = np.argmax(model.predict(feat, verbose=0), axis=1)
    return le.inverse_transform(pred_id)[0]


print(predict_emotion("CREMA-D/1091_WSI_NEU_XX.wav"))