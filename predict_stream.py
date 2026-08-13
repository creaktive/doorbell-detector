#!/usr/bin/env python
"""Stream prediction on 16-bit PCM raw audio at 8kHz mono via stdin."""

import datetime
import os
import threading
import urllib.parse
import urllib.request

import numpy as np

from inferencer import Inferencer


WINDOW = 16000       # 2 seconds @ 8kHz
STRIDE = 800         # trigger detection rate 10 Hz @ 8kHz
BUF_SIZE = 32768     # read 32KB chunks from stdin (4096 int16, ~0.5s)
COOLDOWN_SAMPLES = 80000   # 10s cooldown after detection
CONF_THRESHOLD = 0.9       # force "environment" below this
DETECTION_STREAK = 10      # detect 10x in a row to trigger notification


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
    abs_offset = 0
    streak_label = None
    streak_count = 0
    cooldown_until = -COOLDOWN_SAMPLES   # not in cooldown at start

    while True:
        chunk = os.read(0, BUF_SIZE)
        if not chunk:
            break
        new_samples = np.frombuffer(chunk, dtype=np.int16)
        buf = np.concatenate([buf, new_samples])

        while abs_offset + WINDOW <= len(buf):
            # Skip inference during cooldown
            if abs_offset < cooldown_until:
                abs_offset += STRIDE
                continue

            start = abs_offset
            end = abs_offset + WINDOW
            label, conf = inferencer.predict(
                buf[start:end].astype(np.float32) / 32768.0
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
                notify(f"{label.upper()} DOORBELL")
                cooldown_until = abs_offset + COOLDOWN_SAMPLES
                streak_count = 0
                streak_label = None

            abs_offset += STRIDE

        # Discard processed samples, keep the rest
        trim = min(abs_offset, len(buf))
        if trim > 0:
            buf = buf[trim:]
            abs_offset -= trim


if __name__ == "__main__":
    main()
