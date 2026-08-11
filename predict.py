#!/usr/bin/env python

"""Predict category for a single wav file using the trained model."""

import os
import sys

import librosa
import numpy as np

from config import SAMPLE_RATE
from inferencer import Inferencer


def predict(path, inferencer):
    """Load wav, extract MFCCs, return (predicted_label, confidence)."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return inferencer.predict(y)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python predict.py <wav_file> [wav_file ...]")
        sys.exit(1)

    inferencer = Inferencer()

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        label, conf = predict(path, inferencer)
        print(f"{label}\t{conf:.2%}\t{os.path.basename(path)}")


if __name__ == "__main__":
    main()
