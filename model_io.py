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


# Cached signature metadata set by load_model()
_litert_sig_key = None
_litert_input_name = None
_litert_output_name = None


def load_model():
    """Load trained LiteRT model and label map. Returns (compiled_model, idx_to_label)."""
    global _litert_sig_key, _litert_input_name, _litert_output_name

    model = CompiledModel.from_file(MODEL_TFLITE_PATH)
    sigs = model.get_signature_list()
    _litert_sig_key = list(sigs.keys())[0]
    _litert_input_name = sigs[_litert_sig_key]["inputs"][0]
    _litert_output_name = sigs[_litert_sig_key]["outputs"][0]

    idx_to_label = {i: name for i, name in enumerate(LABELS)}
    return model, idx_to_label


def run_inference(model, X):
    """Run inference on a compiled model. Returns probs array."""
    input_buf = model.create_input_buffer_by_name(_litert_sig_key, _litert_input_name)
    output_buf = model.create_output_buffer_by_name(_litert_sig_key, _litert_output_name)

    input_buf.write(X)
    model.run_by_name(_litert_sig_key, {_litert_input_name: input_buf}, {_litert_output_name: output_buf})

    out_details = output_buf.get_tensor_details()
    num_elements = 1
    for s in out_details["shape"]:
        num_elements *= int(s)
    result = output_buf.read(num_elements, "float32")
    return result.reshape(out_details["shape"])[0]


def get_label_map():
    """Return label_name -> index mapping."""
    return {name: i for i, name in enumerate(LABELS)}
