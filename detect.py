#!/usr/bin/env python
"""Stream prediction on 16-bit PCM raw audio at 16kHz mono via stdin."""

import datetime
import os
import threading
import urllib.parse
import urllib.request
import wave

import numpy as np

from config import SAMPLE_RATE, WINDOW_SAMPLES
from inferencer import Inferencer


BUF_SIZE = SAMPLE_RATE * 2          # read 1s of 16-bit PCM chunks from stdin
STRIDE = SAMPLE_RATE // 10          # trigger detection rate 10 Hz
COOLDOWN_SAMPLES = SAMPLE_RATE * 10 # 10s cooldown after detection
CONF_THRESHOLD = 0.9                # force "environment" below this
DETECTION_STREAK = 8                # detect 5x in a row to trigger notification


def notify(label):
    """Fire Pushsafer notification in a background thread."""
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{ts}\t{label}", flush=True)

    key = os.environ.get("PUSHSAFER_KEY")
    if not key:
        return
    data = urllib.parse.urlencode({
        "k": key,
        "pr": 2,
        "m": label,
    }).encode()
    req = urllib.request.Request(
        "https://www.pushsafer.com/api",
        data=data,
        method="POST",
    )
    threading.Thread(target=_send, args=(req,), daemon=True).start()


def _send(req):
    try:
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


def main():
    inferencer = Inferencer()

    buf = np.empty(0, dtype=np.int16)
    streak_label = None
    streak_count = 0
    cooldown_samples_left = 0

    while True:
        chunk = os.read(0, BUF_SIZE)
        if not chunk:
            break
        new_samples = np.frombuffer(chunk, dtype=np.int16)
        buf = np.concatenate([buf, new_samples])

        # Process sliding windows as long as we have enough data in the buffer
        while len(buf) >= WINDOW_SAMPLES:

            # Skip inference if we are in a cooldown period
            if cooldown_samples_left > 0:
                cooldown_samples_left -= STRIDE
                buf = buf[STRIDE:] # Slide buffer forward and discard old samples
                continue

            # Run inference on the current window
            window = buf[:WINDOW_SAMPLES]
            label, conf = inferencer.predict(
                window.astype(np.float32) / 32768.0
            )

            # Confidence floor - below threshold => force "environment"
            if conf < CONF_THRESHOLD:
                label = "environment"

            # Streak tracking
            if label == streak_label and label != "environment":
                streak_count += 1
            else:
                streak_label = label
                streak_count = 0 if label == "environment" else 1

            # Streak confirmed → print + notify, enter cooldown
            if streak_count >= DETECTION_STREAK:
                # Write detected window to a file if requested via DUMP_DETECTED env var
                if os.environ.get("DUMP_DETECTED"):
                    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    fname = f"detected/doorbell-{label}-{ts}.wav"
                    os.makedirs("detected", exist_ok=True)
                    with wave.open(fname, "wb") as f:
                        f.setnchannels(1)
                        f.setsampwidth(2)
                        f.setframerate(SAMPLE_RATE)
                        f.writeframes(window.tobytes())

                notify(f"{label.upper()} DOORBELL")
                cooldown_samples_left = COOLDOWN_SAMPLES
                streak_count = 0
                streak_label = None

            # Slide buffer forward by STRIDE to prepare for the next iteration
            buf = buf[STRIDE:]


if __name__ == "__main__":
    main()
