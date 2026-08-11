#!/usr/bin/env python

"""Live classification from microphone at 8kHz using a trained 1D CNN."""

import platform
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from config import BUFFER_SEC, CLASSIFY_HZ, SAMPLE_RATE
from model_io import load_model, mfcc_features, run_inference

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


def classify(model, idx_to_label, audio: np.ndarray):
    """Run inference on audio samples, return (label, confidence)."""
    X = mfcc_features(audio)
    probs = run_inference(model, X)
    idx = int(np.argmax(probs))
    return idx_to_label[idx], float(probs[idx])


def main():
    # Load model and labels
    print("Loading model...")
    model, idx_to_label = load_model()

    # Setup ring buffer and audio stream
    ring = AudioRing()
    print(f"Starting microphone at {SAMPLE_RATE}Hz mono...")

    stop_event = threading.Event()

    def callback(indata, frames, _info, _status):
        if _status:
            print(f"[audio status] {_status}")
        ring.push(indata[:, 0])

    if platform.system() == "Linux":
        device = "plug:dsnoop"
    else:
        device = None

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, device=device)
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
