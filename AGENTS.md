# Doorbell Detector — Project Context

## What This Is

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). Runs on a Pi Zero via LiteRT with an FP16 TFLite model (~47 KB).

The entire feature extraction pipeline — STFT + Mel-filterbank — lives inside the TF graph. The exported `.tflite` accepts raw PCM audio and returns classification. No librosa needed at inference time.

## Data Flow

```
Training:  wav → librosa.load() → raw PCM (16000 samples) → AudioFrontend layer → Mel-spectrogram → CNN → class
Inference: wav → librosa.load() [predict.py only] → raw PCM → TFLite model → class
Stream:    stdin (int16 @ 8kHz) → float32 normalize → TFLite model → class
```

## Architecture

```
Input(16000,) raw PCM → AudioFrontend (STFT + Mel-filterbank) → (128, 40) mel spectrogram
                     → BatchNorm → SeparableConv1D(64, k=5) → BatchNorm → MaxPool(2)
                     → SeparableConv1D(64, k=5) → BatchNorm → Residual Add → MaxPool(2)
                     → GlobalAveragePooling1D → Dropout(0.2) → Softmax(3)
```

**Total parameters:** ~8,200 (~47 KB FP16 TFLite)

### Key design decisions
- **Mel-spectrogram in-graph**: Self-contained model, no feature extraction deps at inference. STFT params: frame_length=256, frame_step=128, n_mels=40, freq range 400–4000 Hz (doorbell tones are all above 400 Hz).
- **SeparableConv + residual**: Lower parameter count than standard Conv1D while maintaining accuracy.
- **GlobalAveragePooling instead of Flatten**: Fewer parameters, less overfitting.
- **FP16 quantization**: Halves model size and memory bandwidth for Pi Zero without int8 I/O complexity.
- **Class weights (inverse frequency)**: Dataset is heavily imbalanced toward "environment" (~90%). Weights compensate during training.
- **Silence augmentation**: 25% of dataset is synthetic silence labeled as "environment". Without this, the model misclassifies silence as "upstairs" because it was never trained on flat-zero spectrograms.

## Files

| File | Role | Dependencies |
|------|------|-------------|
| `config.py` | Shared constants (sample rate, model paths, labels, STFT params) | — |
| `train.py` | Training script: loads data, builds end-to-end model with in-graph Mel extraction, trains, converts to FP16 TFLite | librosa, numpy, tensorflow, ai-edge-litert |
| `inferencer.py` | LiteRT inference wrapper. `Inferencer.predict(audio)` takes raw float32 PCM (16000 samples) and returns `(label, confidence)` | numpy, ai-edge-litert |
| `predict.py` | Batch prediction on `.wav` files. Uses librosa only for wav I/O, delegates to Inferencer | librosa, ai-edge-litert |
| `predict_stream.py` | Real-time stream prediction via stdin (16-bit PCM @ 8kHz mono). 2s windows, 10 Hz trigger rate, sliding stride of 800 samples (~100ms overlap) | numpy, ai-edge-litert |
| `augment.sh` | Audio augmentation with sox: speed/tempo ±10%, pitch ±200 cents, volume ±30%, reverb, echo, flanger, overdrive, compand, lowpass/highpass/bandpass filtering, EQ dip, proximity effect, padding. Applies 20 transforms per file | bash, sox, find |
| `get-env-data.sh` | Downloads ESC-50 environmental sounds into `data/environment/` | curl, bsdtar |

## Conventions & Gotchas

### Audio format
- All audio is **8kHz mono**. Training loads with `librosa.load(path, sr=8000, mono=True)`.
- Fixed 2-second windows: exactly 16000 samples. Shorter files are zero-padded; longer files are truncated.
- Stream input (`predict_stream.py`) is **16-bit signed int** normalized to float32 by dividing by 32768.

### Model I/O
- Input tensor name: `"audio"`, shape `(1, 16000)`, dtype `float32`
- Output tensor name: softmax probabilities over `[downstairs, upstairs, environment]`
- The Inferencer handles padding to 16000 samples internally if needed.

### Training specifics
- Deterministic shuffle with `np.random.default_rng(42)` so validation split isn't biased by file order.
- Early stopping monitors `val_accuracy` (patience=10, max mode) with weight restoration.
- ReduceLROnPlateau on `val_loss` (factor=0.5, patience=5).
- Validation split: 15%. Training data is NOT shuffled again after splitting (`shuffle=False` in `model.fit`).

### Platform quirks
- `get_output_tensor_details()` doesn't exist on Linux/RPi builds of `ai-edge-litert`. Output shape is derived from `LABELS` instead.
- The CompiledModel API uses buffer-based I/O: `create_input_buffer_by_name()`, `write()`, `run_by_name()`, `read()`.

### Data directory structure
```
data/
├── downstairs/    # doorbell-downstairs-*.wav (+ augmented copies)
├── environment/   # ESC-50 background noise files
└── upstairs/      # doorbell-upstairs-*.wav (+ augmented copies)
```

Augmented files follow naming: `doorbell-downstairs-1-aug-speed-0-9.wav`. These are gitignored.

### Dependencies by env
- **Training**: librosa, numpy, tensorflow (plus ai-edge-litert for tensor inspection post-conversion)
- **Inference**: numpy, ai-edge-litert only — no librosa, no feature extraction libs

## Running

```bash
# Training (full pipeline: data → model → .tflite)
python train.py

# Batch prediction on wav files
python predict.py file1.wav [file2.wav ...]
# Output: label\tconfidence\tfilename per line

# Real-time stream from stdin (raw 16-bit PCM @ 8kHz mono)
cat audio.raw | python predict_stream.py
# Output: label\tconfidence per detection window (~10 Hz)
```

## Augmentation

Run `./augment.sh [data_dir]` to apply 20 transforms to every `.wav` in the data directory. Each transform produces a copy with `-aug-<name>.wav` suffix. The script uses `find -print0` for safe filename handling and reports success/failure counts.

## Gitignore

```
.venv/
__pycache__/
data/**/*-aug-*.wav    # augmented files (regenerated)
data/environment/[1-5]-*.wav   # raw ESC-50 files (downloaded by script)
*.raw                  # test artifacts
```

## Model File

`doorbell.tflite` — FP16 quantized, ~47 KB. This is the deployable artifact for edge inference. Regenerated each time `train.py` runs successfully.
