"""Shared constants across all doorbell scripts."""

BATCH_SIZE = 32
BUFFER_SEC = 16
CLASSIFY_HZ = 1
DATA_DIR = "data"
EPOCHS = 100
LABELS = ["downstairs", "upstairs", "environment"]
MAX_T = 128
MFCC_FRAMES = 40
MODEL_TFLITE_PATH = "doorbell_int8.tflite"
SAMPLE_RATE = 8000
