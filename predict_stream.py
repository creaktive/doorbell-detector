#!/usr/bin/env python

"""Stream prediction on 16-bit PCM raw audio at 8kHz mono via stdin."""

import os

import numpy as np

from inferencer import Inferencer


WINDOW = 16000   # 2 seconds @ 8kHz
STRIDE = 800     # trigger detection rate 10 Hz @ 8kHz
BUF_SIZE = 32768 # read 32KB chunks from stdin (4096 int16, ~0.5s)


def main():
    inferencer = Inferencer()

    buf = np.empty(0, dtype=np.int16)
    abs_offset = 0  # the next sample appended to buf will have this index

    while True:
        chunk = os.read(0, BUF_SIZE)
        if not chunk:
            break
        new_samples = np.frombuffer(chunk, dtype=np.int16)
        buf = np.concatenate([buf, new_samples])

        # Classify every STRIDE samples as soon as a full window is available
        while abs_offset + WINDOW <= len(buf):
            start = abs_offset
            end = abs_offset + WINDOW
            label, conf = inferencer.predict(
                buf[start:end].astype(np.float32) / 32768.0
            )
            print(f"{label}\t{conf:.2%}", flush=True)
            abs_offset += STRIDE

        # Discard processed samples, keep the rest
        trim = min(abs_offset, len(buf))
        if trim > 0:
            buf = buf[trim:]
            abs_offset -= trim


if __name__ == "__main__":
    main()
