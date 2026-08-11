#!/usr/bin/env python

"""Train a 1D CNN to classify doorbell audio into downstairs / upstairs / environment."""

import os
from collections import Counter

import librosa
import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, EPOCHS, LABELS, MAX_T, MFCC_FRAMES, MODEL_TFLITE_PATH, SAMPLE_RATE
from inferencer import mfcc_features

def load_dataset(data_dir):
    """Walk data_dir, load wav files, extract MFCCs. Returns X, y."""
    label_map = {name: i for i, name in enumerate(LABELS)}

    xs, ys = [], []
    for label in sorted(os.listdir(data_dir)):
        if label not in label_map:
            continue
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(label_dir, fname)
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            X = mfcc_features(y)
            xs.append(X)
            ys.append(label_map[label])

    # Silence produces an all-extreme-negative MFCC pattern (log(0)=-inf clipped).
    # The model was never trained on this pattern and confidently misclassifies it as "upstairs".
    # Add genuinely silent samples so the model learns silence = environment.
    n_silence = len(xs) // 4  # supplement with 25% of dataset size in silence samples
    window_samples = int(SAMPLE_RATE * 2)  # 2 seconds @ 8kHz = 16000
    for _ in range(n_silence):
        silence = np.zeros(window_samples, dtype="float32")
        X = mfcc_features(silence)
        xs.append(X)
        ys.append(label_map["environment"])

    X = np.concatenate(xs).astype("float32")  # (N, MAX_T, MFCC_FRAMES)
    y = np.array(ys, dtype="int32")
    return X, y

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
    X, y = load_dataset(DATA_DIR)
    n_classes = len(LABELS)
    counts = Counter(y.tolist())
    print(f"Samples: {X.shape[0]}, Classes: {n_classes}, Labels: {LABELS}")
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

    # Convert to INT8 TFLite for LiteRT inference
    print("Converting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {MODEL_TFLITE_PATH} ({len(tflite_model)} bytes)")

    # Print tensor info for CompiledModel API
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"Input name:  '{input_details['name']}', shape={list(input_details['shape'])}, dtype={input_details['dtype']}")
    print(f"Output name: '{output_details['name']}', shape={list(output_details['shape'])}, dtype={output_details['dtype']}")

if __name__ == "__main__":
    main()
