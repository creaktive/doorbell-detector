"""Shared constants across all doorbell scripts."""

BATCH_SIZE = 32
DATA_DIR = "data"
EPOCHS = 100
FRAME_LENGTH = 512
FRAME_STEP = 256
LABELS = ["downstairs", "upstairs", "environment"]
MAX_T = 128
MODEL_TFLITE_PATH = "doorbell.tflite"
N_MELS = 40
SAMPLE_RATE = 16000
WINDOW_DURATION = 1
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_DURATION
