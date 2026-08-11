"""Shared model I/O utilities for doorbell scripts."""

import json
import logging

import numpy as np
import tensorflow as tf

from config import MAX_T, MFCC_FRAMES, MODEL_PATH, LABEL_MAP_PATH


def mfcc_features(audio: np.ndarray) -> np.ndarray:
    """Extract MFCCs from raw audio and return shaped tensor (1, MAX_T, MFCC_FRAMES)."""
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=8000, n_mfcc=MFCC_FRAMES)
    if mfcc.shape[1] < MAX_T:
        pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_T]
    return mfcc.astype("float32").T.reshape(1, MAX_T, MFCC_FRAMES)


def load_model():
    """Load trained model and label map. Returns (model, idx_to_label)."""
    tf.get_logger().setLevel(logging.ERROR)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_label = {int(v): k for k, v in label_map.items()}
    return model, idx_to_label
