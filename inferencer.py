"""Shared model I/O utilities for doorbell scripts."""

import numpy as np

from ai_edge_litert.compiled_model import CompiledModel

from config import LABELS, MODEL_TFLITE_PATH, WINDOW_SAMPLES


class Inferencer:
    """Encapsulates a loaded LiteRT model and its signature metadata."""

    def __init__(self):
        self.model = CompiledModel.from_file(MODEL_TFLITE_PATH)
        sigs = self.model.get_signature_list()
        self.sig_key = list(sigs.keys())[0]
        self.input_name = sigs[self.sig_key]["inputs"][0]
        self.output_name = sigs[self.sig_key]["outputs"][0]

        # Output shape is always (1, num_classes) - derived from LABELS since
        # get_output_tensor_details() doesn't exist on Linux/RPi builds of ai-edge-litert
        self.output_shape = [1, len(LABELS)]

        self.idx_to_label = {i: name for i, name in enumerate(LABELS)}

    def run(self, X):
        """Run inference on input tensor. Returns probs array."""
        input_buf = self.model.create_input_buffer_by_name(self.sig_key, self.input_name)
        output_buf = self.model.create_output_buffer_by_name(self.sig_key, self.output_name)

        input_buf.write(X)
        self.model.run_by_name(
            self.sig_key, {self.input_name: input_buf}, {self.output_name: output_buf}
        )

        num_elements = 1
        for s in self.output_shape:
            num_elements *= int(s)
        result = output_buf.read(num_elements, np.float32)
        return result.reshape(self.output_shape)[0]

    def predict(self, audio):
        """Run inference on raw PCM audio (float32, WINDOW_SAMPLES samples @ 16kHz). Returns (label, confidence)."""
        if len(audio) < WINDOW_SAMPLES:
            audio = np.concatenate([audio, np.zeros(WINDOW_SAMPLES - len(audio), dtype="float32")])
        X = audio[:WINDOW_SAMPLES].astype("float32").reshape(1, -1)
        probs = self.run(X)
        idx = int(np.argmax(probs))
        return self.idx_to_label[idx], float(probs[idx])
