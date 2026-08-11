"""Shared model I/O utilities for doorbell scripts."""

import librosa
import numpy as np

from ai_edge_litert.compiled_model import CompiledModel

from config import LABELS, MAX_T, MFCC_FRAMES, MODEL_TFLITE_PATH, SAMPLE_RATE


def mfcc_features(audio: np.ndarray) -> np.ndarray:
    """Extract MFCCs from raw audio and return shaped tensor (1, MAX_T, MFCC_FRAMES)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FRAMES)
    if mfcc.shape[1] < MAX_T:
        pad = np.zeros((MFCC_FRAMES, MAX_T - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :MAX_T]
    return mfcc.astype("float32").T.reshape(1, MAX_T, MFCC_FRAMES)


class Inferencer:
    """Encapsulates a loaded LiteRT model and its signature metadata."""

    def __init__(self):
        self.model = CompiledModel.from_file(MODEL_TFLITE_PATH)
        sigs = self.model.get_signature_list()
        self.sig_key = list(sigs.keys())[0]
        self.input_name = sigs[self.sig_key]["inputs"][0]
        self.output_name = sigs[self.sig_key]["outputs"][0]

        # Output shape is always (1, num_classes) — derived from LABELS since
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
        """Extract features and run inference. Returns (label, confidence)."""
        X = mfcc_features(audio)
        probs = self.run(X)
        idx = int(np.argmax(probs))
        return self.idx_to_label[idx], float(probs[idx])

