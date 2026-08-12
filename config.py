"""Shared constants across all doorbell scripts."""

BATCH_SIZE = 32
DATA_DIR = "data"
EPOCHS = 100
FRAME_LENGTH = 256
FRAME_STEP = 128
LABELS = ["downstairs", "upstairs", "environment"]
MAX_T = 128
MODEL_TFLITE_PATH = "doorbell.tflite"
N_MELS = 40
SAMPLE_RATE = 8000
