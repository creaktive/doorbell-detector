#!/usr/bin/env python

"""Predict category for a single wav file using the trained model."""

import os
import sys

import librosa
import numpy as np
import tensorflow as tf

from config import SAMPLE_RATE
from model_io import load_model, mfcc_features

def predict(path, model, idx_to_label):
    """Load wav, extract MFCCs, return (predicted_label, confidence)."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    X = mfcc_features(y)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx_to_label[idx], float(probs[idx])

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python predict.py <wav_file> [wav_file ...]")
        sys.exit(1)

    model, idx_to_label = load_model()

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        label, conf = predict(path, model, idx_to_label)
        print(f"{label}\t{conf:.2%}\t{os.path.basename(path)}")


if __name__ == "__main__":
    main()
