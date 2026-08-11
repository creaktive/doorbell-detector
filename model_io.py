"""Shared model I/O utilities for doorbell scripts."""

import librosa
import numpy as np
import tensorflow as tf

from config import LABELS, MAX_T, MFCC_FRAMES, MODEL_PATH, SAMPLE_RATE

def mfcc_features(audio: np.ndarray) -> np.ndarray:
    """Extract MFCCs from raw audio and return shaped tensor (1, MAX_T, MFCC_FRAMES)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FRAMES)
    if mfcc.shape[1] < MAX_T:
        pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_T]
    return mfcc.astype("float32").T.reshape(1, MAX_T, MFCC_FRAMES)

def load_model():
    """Load trained model and label map. Returns (model, idx_to_label)."""
    model = tf.keras.models.load_model(MODEL_PATH)
    idx_to_label = {i: name for i, name in enumerate(LABELS)}
    return model, idx_to_label


def get_label_map():
    """Return label_name -> index mapping."""
    return {name: i for i, name in enumerate(LABELS)}
