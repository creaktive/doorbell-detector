"""Predict category for a single wav file using the trained model."""

import os
import sys
import json
import numpy as np
import librosa
import tensorflow as tf

MFCC_FRAMES = 40
MAX_T = 256
MODEL_PATH = "model.h5"
LABEL_MAP_PATH = "label_map.json"


def predict(path, model, label_map):
    """Load wav, extract MFCCs, return (predicted_label, confidence)."""
    y, sr = librosa.load(path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FRAMES)

    # Pad or truncate to MAX_T
    if mfcc.shape[1] < MAX_T:
        pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_T]

    X = mfcc.astype("float32").reshape(1, MFCC_FRAMES, MAX_T)
    probs = model.predict(X, verbose=0)[0]
    idx = np.argmax(probs)
    label = list(label_map.keys())[list(label_map.values()).index(idx)]
    return label, float(probs[idx])


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python predict.py <wav_file> [wav_file ...]")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        label, conf = predict(path, model, label_map)
        print(f"{os.path.basename(path)} → {label} ({conf:.2%})")


if __name__ == "__main__":
    main()
