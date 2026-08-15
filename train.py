#!/usr/bin/env python

"""Train a 1D CNN to classify doorbell audio into downstairs / upstairs / environment."""

import os
import re
from collections import Counter

import librosa
import numpy as np
import tensorflow as tf

from ai_edge_litert.compiled_model import CompiledModel

from config import (
    BATCH_SIZE, DATA_DIR, EPOCHS, FRAME_LENGTH, FRAME_STEP, LABELS,
    MAX_T, MODEL_TFLITE_PATH, N_MELS, SAMPLE_RATE, WINDOW_SAMPLES,
)

rng = np.random.default_rng(42)


def load_audio(path):
    """Yield a single 1s window from wav file via random offset."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < WINDOW_SAMPLES:
        raise ValueError(f"Audio too short ({len(audio)} samples) in {path}")
    offset = rng.integers(0, len(audio) - WINDOW_SAMPLES + 1)
    yield audio[offset:offset + WINDOW_SAMPLES].astype("float32")


def load_dataset(data_dir):
    """Walk data_dir, load wav files as raw PCM. Returns X, y, file_ids."""
    label_map = {name: i for i, name in enumerate(LABELS)}

    xs, ys, fids = [], [], []
    for label in sorted(os.listdir(data_dir)):
        if label not in label_map:
            continue
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(data_dir, label, fname)
            match = re.search(r"^([a-z-]+[0-9]+)", fname)
            base_fname = match.group(1) if match else fname
            for sample in load_audio(path):
                xs.append(sample)
                ys.append(label_map[label])
                fids.append(base_fname)

    X = np.array(xs, dtype="float32")
    y = np.array(ys, dtype="int32")
    return X, y, np.array(fids, dtype="U256")

class AudioFrontend(tf.keras.layers.Layer):
    """Computes Mel-Spectrograms directly inside the TF graph."""

    def __init__(self, sample_rate, n_mels, frame_length, frame_step, max_t, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.max_t = max_t
        self.n_mels = n_mels
        # Precompute the exact number of samples needed for exactly max_t frames
        self.target_samples = (self.max_t - 1) * self.frame_step + self.frame_length
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels,
            num_spectrogram_bins=frame_length // 2 + 1,
            sample_rate=sample_rate,
            lower_edge_hertz=400.0, # doorbell tones are all above this
            upper_edge_hertz=sample_rate / 2.0
        )

    def call(self, raw_audio):
        # Pad each sample in the batch to target_samples
        pad_amount = self.target_samples - tf.shape(raw_audio)[1]
        audio = tf.concat([raw_audio, tf.zeros((tf.shape(raw_audio)[0], tf.maximum(0, pad_amount)), dtype=tf.float32)], axis=1)

        stft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step)
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        return tf.math.log(mel_spectrogram + 1e-6)


def build_model():
    """End-to-end model: raw audio → Mel-spectrogram → Residual 1D CNN."""
    inputs = tf.keras.layers.Input(shape=(WINDOW_SAMPLES,), dtype=tf.float32, name="audio")

    x = AudioFrontend(sample_rate=SAMPLE_RATE, n_mels=N_MELS, frame_length=FRAME_LENGTH,
                      frame_step=FRAME_STEP, max_t=MAX_T)(inputs)
    # x shape: (batch, MAX_T, N_MELS)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.SeparableConv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    res = x
    x = tf.keras.layers.SeparableConv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, res])
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("Loading dataset...")
    X, y, file_ids = load_dataset(DATA_DIR)
    print(f"Samples: {X.shape[0]}, Classes: {len(LABELS)}, Labels: {LABELS}")
    counts = Counter(y.tolist())
    print(f"Class distribution: {dict(counts)}")

    # Compute class weights to handle imbalance (inverse frequency weighting)
    total = len(y)
    class_weight = {c: total / (len(LABELS) * cnt) for c, cnt in counts.items()}
    print(f"Class weights: {class_weight}")

    # Group-aware train/val split - prevents data leakage from augmented variants
    unique_files = sorted(set(file_ids))
    rng.shuffle(unique_files)
    n_val = max(1, int(len(unique_files) * 0.15))
    val_files = set(unique_files[-n_val:])

    train_mask = np.array([fid not in val_files for fid in file_ids])
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]} (from {len(unique_files)} unique source files)")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping("val_accuracy", patience=10, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(X_val, y_val), class_weight=class_weight, callbacks=callbacks)

    # Report final val accuracy
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    print(f"Best epoch: {best_epoch}, val_accuracy: {history.history['val_accuracy'][best_epoch-1]:.4f}")

    # Float16 quantization for Pi Zero: halves model size, faster ARM inference.
    # Float16 avoids the int8 I/O complexity while still cutting memory bandwidth in half.
    print("Converting to FP16 TFLite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {MODEL_TFLITE_PATH} ({len(tflite_model)} bytes)")

    # Print tensor info via LiteRT CompiledModel API (same as runtime inference).
    compiled = CompiledModel.from_file(MODEL_TFLITE_PATH)
    sigs = compiled.get_signature_list()
    sig_key = list(sigs.keys())[0]
    details = compiled.get_input_tensor_details(sig_key)
    inp = details[sigs[sig_key]["inputs"][0]]
    out_details = compiled.get_output_tensor_details(sig_key)
    out = out_details[sigs[sig_key]["outputs"][0]]
    print(f"Input name:  '{inp['name']}', shape={list(inp['shape'])}, dtype={inp['dtype']}")
    print(f"Output name: '{out['name']}', shape={list(out['shape'])}, dtype={out['dtype']}")

if __name__ == "__main__":
    main()
