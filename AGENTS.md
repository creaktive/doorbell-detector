# Doorbell Detector — Project Context

## What This Is

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). Runs on a Pi Zero via LiteRT with an FP16 TFLite model (~57 KB).

The entire feature extraction pipeline — STFT + Mel-filterbank — lives inside the TF graph. The exported `.tflite` accepts raw PCM audio and returns classification. No librosa needed at inference time.

## Data Flow

```
Training:  wav → librosa.load(mono=False) → each channel yields one sample → random 1s slice → AudioFrontend layer → Mel-spectrogram → CNN → class
Inference: wav → raw PCM → TFLite model → class
Stream:    stdin (int16 @ 16kHz) → float32 normalize → TFLite model → confidence floor (<90% → "environment") → streak detection (8 frames) → cooldown (10s) → Pushsafer notification
```

## Architecture

```
Input(16000,) raw PCM → AudioFrontend (STFT + Mel-filterbank) → (~62, 40) mel spectrogram
                       → BatchNorm → SeparableConv1D(64, k=5) → BatchNorm → MaxPool(2)
                       → SeparableConv1D(64, k=5) → BatchNorm → Residual Add → MaxPool(2)
                       → GlobalAveragePooling1D → Dropout(0.2) → Softmax(3)
```

**Total parameters:** ~8,200 (~57 KB FP16 TFLite)

### Key design decisions
- **Mel-spectrogram in-graph**: Self-contained model, no feature extraction deps at inference. STFT params: frame_length=512, frame_step=256, n_mels=40, freq range 400–16000 Hz (doorbell tones are all above 400 Hz).
- **SeparableConv + residual**: Lower parameter count than standard Conv1D while maintaining accuracy.
- **GlobalAveragePooling instead of Flatten**: Fewer parameters, less overfitting.
- **FP16 quantization**: Halves model size and memory bandwidth for Pi Zero without int8 I/O complexity.
- **Class weights (inverse frequency)**: Dataset is heavily imbalanced toward "environment" (~90%). Weights compensate during training.

## Files

| File | Role | Dependencies |
|------|------|-------------|
| `config.py` | Shared constants (sample rate, model paths, labels, STFT params) | — |
| `train.py` | Training script: loads data, builds end-to-end model with in-graph Mel extraction, trains, converts to FP16 TFLite | librosa, numpy, tensorflow, ai-edge-litert |
| `inferencer.py` | LiteRT inference wrapper. `Inferencer.predict(audio)` takes raw float32 PCM (16000 samples @ 16kHz) and returns `(label, confidence)` | numpy, ai-edge-litert |
| `detect.py` | Real-time stream prediction via stdin (16-bit PCM @ 16kHz mono). 1s windows, 10 Hz trigger rate, sliding stride of 1600 samples (~100ms), confidence floor <90% → "environment", streak confirmation (8 frames = ~0.8s), cooldown mode (10s after detection), Pushsafer notifications, optional ALSA live capture, DUMP_DETECTED WAV export | numpy, ai-edge-litert, pyalsaaudio (optional), stdlib (threading, urllib, wave) |
| `augment.sh` | Audio augmentation with sox: speed/tempo ±10%, pitch ±200 cents, volume ±30%, overdrive, compand, lowpass/highpass/bandpass filtering, EQ dip, proximity effect, reverb, echo, flanger, chorus. Applies 20 transforms per file | bash, sox, find |
| `get-env-data.sh` | Downloads ESC-50 environmental sounds into `data/environment/` | curl, bsdtar |
| `test.sh` | Quick test: runs detect.py on each `.wav` in `data/test/` (16-bit PCM @ 16kHz mono) | bash, sox |

## Conventions & Gotchas

### Audio format
- All audio is **16kHz**. Training loads with `librosa.load(path, sr=16000, mono=False)` — multi-channel files yield one sample per channel.
- Fixed 1-second windows: random offset slice from each channel (no padding). Files shorter than 1s raise an error.
- Stream input (`detect.py`) is **16-bit signed int** normalized to float32 by dividing by 32768.

### Model I/O
- Input tensor name: `"audio"`, shape `(1, 16000)`, dtype `float32`
- Output tensor name: softmax probabilities over `[downstairs, upstairs, environment]`
- The Inferencer handles padding to 16000 samples internally if needed.

### Training specifics
- Group-aware validation split: unique source files are shuffled (`np.random.default_rng(42)`) and 15% go to val. Augmented variants from the same file never leak across train/val boundaries. Each channel of each file yields one or more random 1s windows during training (multiple for files >3s).
- Early stopping monitors `val_accuracy` (patience=10, max mode) with weight restoration.
- ReduceLROnPlateau on `val_loss` (factor=0.5, patience=5).

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
- **Stream** (`detect.py`): stdlib `threading`, `urllib.request`, `urllib.parse` for Pushsafer notifications (daemon thread, 3s timeout, silent failure)

## Running

```bash
# Training (full pipeline: data → model → .tflite)
./train.py

# Real-time stream from stdin (raw 16-bit PCM @ 16kHz mono)
cat audio.raw | ./detect.py
# Output: YYYY-MM-DDTHH:MM:SS\tLABEL DOORBELL on confirmed detection (~once per 10s cooldown)
```

## Augmentation

Run `./augment.sh [data_dir]` to apply 20 transforms to every `.wav` in the data directory. Each transform produces a copy with `-aug-<name>.wav` suffix. The script uses `find -print0` for safe filename handling and reports success/failure counts.

## Gitignore

```
.DS_Store
.venv/
__pycache__/
data/**/*-aug-*.wav    # augmented files (regenerated)
data/environment/[1-5]-*.wav   # raw ESC-50 files (downloaded by script)
data.bak/              # backup directory
```

## Model File

`doorbell.tflite` — FP16 quantized, ~57 KB. This is the deployable artifact for edge inference. Regenerated each time `train.py` runs successfully.

## Specific instructions for agents

When the model is changed, be sure to update README.md and AGENTS.md accordingly. Be extra careful with model file size and the number of parmeters.
