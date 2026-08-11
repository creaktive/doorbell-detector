"""Live classification from microphone at 8kHz using a trained 1D CNN."""

import json
import logging
import signal
import sys
import threading
import time

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

SAMPLE_RATE = 8000
MFCC_FRAMES = 40
MAX_T = 128
BUFFER_SEC = 16
CLASSIFY_HZ = 1
MODEL_PATH = "model.keras"
LABEL_MAP_PATH = "label_map.json"


class AudioRing:
    """Lock-free-ish ring buffer for audio samples."""

    def __init__(self, sr=SAMPLE_RATE, duration=BUFFER_SEC):
        self.sr = sr
        self.size = int(sr * duration)
        self.buf = np.zeros(self.size, dtype="float32")
        self.idx = 0
        self.lock = threading.Lock()

    def push(self, samples: np.ndarray):
        with self.lock:
            n = len(samples)
            start = self.idx % self.size
            if start + n <= self.size:
                self.buf[start:start + n] = samples
            else:
                self.buf[start:] = samples[:self.size - start]
                self.buf[:n - (self.size - start)] = samples[self.size - start:]
            self.idx += n

    def get_latest(self, duration_sec: float) -> np.ndarray:
        with self.lock:
            n = int(self.sr * duration_sec)
            end = self.idx % self.size
            if n >= self.size:
                return self.buf.copy()
            start = (end - n + self.size) % self.size
            if start < end:
                return self.buf[start:end]
            else:
                return np.concatenate([self.buf[start:], self.buf[:end]])


def mfcc_features(audio: np.ndarray):
    """Extract MFCCs and pad/truncate to MAX_T, matching train.py exactly."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FRAMES)
    if mfcc.shape[1] < MAX_T:
        pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_T]
    return mfcc.astype("float32").T.reshape(1, MAX_T, MFCC_FRAMES)


def classify(model, idx_to_label, audio: np.ndarray):
    """Run inference on audio samples, return (label, confidence)."""
    X = mfcc_features(audio)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx_to_label[idx], float(probs[idx])


def main():
    # Load model and labels
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_label = {int(v): k for k, v in label_map.items()}

    # Setup ring buffer and audio stream
    ring = AudioRing()
    print(f"Starting microphone at {SAMPLE_RATE}Hz mono...")

    stop_event = threading.Event()

    def callback(indata, frames, _info, _status):
        if _status:
            print(f"[audio status] {_status}")
        ring.push(indata[:, 0])

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    stream.start()

    # Graceful exit on Ctrl-C
    def handle_signal(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Classifying every {1/CLASSIFY_HZ:.0f}s (latest {BUFFER_SEC}s of audio). Press Ctrl-C to exit.\n")

    try:
        while not stop_event.is_set():
            # Grab the last ~8 seconds of audio for classification
            audio = ring.get_latest(8.0)
            if len(audio) < SAMPLE_RATE * 2:
                time.sleep(1 / CLASSIFY_HZ)
                continue

            label, conf = classify(model, idx_to_label, audio)
            print(f"{label:<20s} ({conf:.2%})")
            time.sleep(1 / CLASSIFY_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
