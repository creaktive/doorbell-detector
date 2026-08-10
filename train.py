"""Train a 1D CNN to classify doorbell audio into downstairs / upstairs / environment."""

import json
import os
from collections import Counter

import numpy as np
import librosa
import tensorflow as tf

DATA_DIR = "data"
SAMPLE_RATE = 8000
MFCC_FRAMES = 40
MAX_T = 128
BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = "model.h5"


def load_dataset(data_dir):
    """Walk data_dir, load wav files, extract MFCCs. Returns X, y."""
    label_map = {}
    idx = 0
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        label_map[label] = idx
        idx += 1

    xs, ys = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(label_dir, fname)
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=MFCC_FRAMES)
            if mfcc.shape[1] < MAX_T:
                pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
                mfcc = np.hstack([mfcc, pad])
            else:
                mfcc = mfcc[:, :MAX_T]
            xs.append(mfcc)
            ys.append(label_map[label])

    X = np.stack(xs).transpose(0, 2, 1).astype("float32")  # (N, MAX_T, MFCC_FRAMES)
    y = np.array(ys, dtype="int32")
    return X, y, label_map


def build_model(n_classes):
    """Minimal 1D CNN on MFCCs: Conv1D → MaxPool × 2 → Flatten → Dense."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_T, MFCC_FRAMES)),
        tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Loading dataset...")
    X, y, label_map = load_dataset(DATA_DIR)
    n_classes = len(label_map)
    counts = Counter(y.tolist())
    print(f"Samples: {X.shape[0]}, Classes: {n_classes}, Label map: {label_map}")
    print(f"Class distribution: {dict(counts)}")

    # Compute class weights to handle imbalance (ponytail: inverse frequency weighting)
    total = len(y)
    class_weight = {}
    for c, count in counts.items():
        class_weight[c] = total / (n_classes * count)
    print(f"Class weights: {class_weight}")

    # Shuffle deterministically so validation split isn't biased by file order
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    model = build_model(n_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping("val_accuracy", patience=10, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.15,
                        shuffle=False, class_weight=class_weight, callbacks=callbacks)

    # Report final val accuracy
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    print(f"Best epoch: {best_epoch}, val_accuracy: {history.history['val_accuracy'][best_epoch-1]:.4f}")

    model.save(MODEL_PATH)
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)
    print(f"Saved model to {MODEL_PATH} and label_map.json")


if __name__ == "__main__":
    main()
