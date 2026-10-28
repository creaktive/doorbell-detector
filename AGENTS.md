# Doorbell Detector — Project Context

## What This Is

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). Runs on a Pi Zero via LiteRT with an optimized TFLite model (~45 KB, float32 input/output).

The entire feature extraction pipeline — STFT + Mel-filterbank — lives inside the TF graph. The exported `.tflite` accepts raw PCM audio and returns classification. No librosa needed at inference time.

## Data Flow

```
Training:  wav → librosa.load(mono=False) → each channel yields one sample → random 1s slice → AudioFrontend layer → Mel-spectrogram → CNN → class
Inference: wav → raw PCM → TFLite model → class
Stream:    stdin (int16 @ 16kHz) → float32 normalize → TFLite model → confidence floor (<90% → "environment") → streak detection (8 frames) → cooldown (10s) → Pushsafer notification
```

## Architecture

```
Input(16000,) raw PCM → AudioFrontend (STFT + Mel-filterbank) → (61, 40) log-mel spectrogram (MAX_T = 1 + (WINDOW_SAMPLES - FRAME_LENGTH) // FRAME_STEP = 61 frames, no zero padding)
                       → LayerNorm → SeparableConv1D(64, k=5) → BatchNorm → ReLU → MaxPool(2)
                       → SeparableConv1D(64, k=5) → BatchNorm → Residual Add → ReLU → MaxPool(2)
                       → GlobalAveragePooling1D → Dropout(0.2) → Softmax(3)
```

**Total parameters:** ~8,100 (~45 KB optimized TFLite)

### Key design decisions
- **Mel-spectrogram in-graph**: Self-contained model, no feature extraction deps at inference. STFT params: frame_length=512, frame_step=256, n_mels=40, freq range 400–8000 Hz (Nyquist) (doorbell tones are all above 400 Hz).
- **SeparableConv + residual**: Lower parameter count than standard Conv1D while maintaining accuracy.
- **GlobalAveragePooling instead of Flatten**: Fewer parameters, less overfitting.
- **Runtime quantization**: The exported model keeps float32 input/output while quantizing internal tensors to reduce the deployable file size (`Optimize.DEFAULT` dynamic range: large constants become int8). The FP16 comments and print in `train.py` are stale; the float16 line is commented out.
- **Class weights (inverse frequency)**: Computed from per-class window counts at train time to compensate for class imbalance.

## Files

| File | Role | Dependencies |
|------|------|-------------|
| `config.py` | Shared constants (sample rate, model paths, labels, STFT params) | — |
| `train.py` | Training script: loads data, builds end-to-end model with in-graph Mel extraction, trains, converts to optimized TFLite | librosa, numpy, tensorflow |
| `inferencer.py` | LiteRT inference wrapper. `Inferencer.predict(audio)` takes raw float32 PCM (16000 samples @ 16kHz) and returns `(label, confidence)` | numpy, ai-edge-litert |
| `detect.py` | Real-time stream prediction via stdin (16-bit PCM @ 16kHz mono). 1s windows, 10 Hz trigger rate, sliding stride of 1600 samples (~100ms), confidence floor <90% → "environment", streak confirmation (8 frames = ~0.8s), cooldown mode (10s after detection), Pushsafer notifications, optional ALSA live capture, DUMP_DETECTED WAV export. Auto-detects systemd `JOURNAL_STREAM` env var for journald-compatible operation | numpy, ai-edge-litert, pyalsaaudio (optional), stdlib (threading, urllib, wave) |
| `test-false-positives.sh` | Scans `$HOME` for `.ac3`, `.avi`, `.mkv`, and `.mp3` files, converts them with ffmpeg, and feeds them to `detect.py` | bash, ffmpeg, find |
| `augment.sh` | Audio augmentation with sox: speed ±10%, tempo ±5%, pitch ±200 cents, volume ±30%, overdrive, compand, lowpass/highpass/bandpass filtering, EQ dip, proximity effect, reverb, echo, flanger, chorus. Applies 19 transforms per file in `data/downstairs/` and `data/upstairs/`, also downloads ESC-50 to `data/environment/` and writes 31 pure sine tones there (850-1150 Hz, 5 s, `synthetic-sine-*.wav`) as negatives. Accepts optional `.wav` paths as arguments to process only those files (skips ESC-50 and the sines) | bash, sox, find, curl, bsdtar |
| `test.sh` | Quick test: runs detect.py on each `.wav` in `data/test/` with brown noise mix for robustness checks (16-bit PCM @ 16kHz mono) | bash, sox |
| `doorbell-detector.service` | systemd user service for running detect.py as a persistent background daemon. Captures from ALSA `default` unless `ALSAAUDIO_DEVICE` is set; supports PUSHSAFER_KEY and DUMP_DETECTED env vars | — |

## Conventions & Gotchas

### Audio format
- All audio is **16kHz**. Training loads with `librosa.load(path, sr=16000, mono=False)` — multi-channel files yield one sample per channel.
- Fixed 1-second windows: random offset slice from each channel; short files are zero-padded to 16000 samples.
- Stream input (`detect.py`) is **16-bit signed int** normalized to float32 by dividing by 32768.

### Model I/O
- Input tensor name: `"audio"`, shape `(1, 16000)`, dtype `float32`
- Output tensor name: `"output_0"` (signature `serving_default`), softmax probabilities over `[downstairs, upstairs, environment]`
- `Inferencer.predict` does not pad: it raises `ValueError` unless given exactly 16000 samples.

### Training specifics
- Group-aware, class-stratified validation split: each class's unique source files are shuffled separately (`np.random.default_rng(42)`) and 15% of each class goes to val. Augmented variants from the same file never leak across train/val boundaries, and val keeps them. Each channel of each file yields one or more random 1s windows during training (multiple for files >3s). Tried on 2026-10-28 with the same seed: one window per file, or val on originals only with per-class weights, raised false positives on media (25 and 21 detections vs 14 over the 85 files `test-false-positives.sh` scans).
- File names define the groups: `train.py` takes the regex `^(?:[0-9]+-)?([a-z-]+[0-9]+|[0-9]+)` of each name, so `upstairs-042-pi.wav`, `upstairs-042-imac.wav` (same number, other device) and their `-aug-` copies land on the same side. ESC-50 files group by Freesound clip ID: `1-100210-A-36.wav` and `1-100210-B-36.wav` share the group `100210`, and so do the 4 clip IDs that ESC-50 puts in two folds (`2-131943-A-38.wav`, `3-131943-A-37.wav`). Name new files that way; false positives in `data/environment/` follow it too (`false-positive-upstairs-271.wav`). A name the regex misses (uppercase, no digits) becomes its own group and leaks its augmented copies into val.
- Early stopping monitors `val_loss` (patience=10) with weight restoration, so training runs about 50 epochs (about 10 minutes). Not `val_accuracy`: it reaches 1.0 within 4-12 epochs, and restoring that first 1.0 epoch kept undertrained models that detected `upstairs-test-pi.wav` 0-3 times in 10 `test.sh` runs (10/10 with `val_loss`).
- ReduceLROnPlateau on `val_loss` (factor=0.5, patience=5).
- Seeded: `train.py` calls `tf.keras.utils.set_random_seed(42)` and `tf.config.experimental.enable_op_determinism()`, so the same data, machine and TF version give a byte-identical `doorbell.tflite` (same md5 over four runs). Another machine or TF version can differ, and re-running `augment.sh` changes the data (sox dithers its 16-bit output with random noise). The same `rng` draws the random windows before the val shuffle, so adding or removing any file changes the split.

### Platform quirks
- `get_output_tensor_details()` doesn't exist on Linux/RPi builds of `ai-edge-litert`. Output shape is derived from `LABELS` instead.
- The CompiledModel API uses buffer-based I/O: `create_input_buffer_by_name()`, `write()`, `run_by_name()`, `read()`.
- `ai-edge-litert` has no wheel for Intel macOS (x86_64, Python 3.12), and the training env doesn't install it. Run `detect.py` and `test.sh` on the Pi.
- `detect.py` ignores piped stdin when `JOURNAL_STREAM` is set and captures from ALSA instead (meant for systemd). Unset it if it leaks into your shell.

### Data directory structure
```
data/
├── downstairs/    # doorbell-downstairs-*.wav (+ augmented copies)
├── environment/   # ESC-50 background noise files + synthetic-sine-*.wav negatives
└── upstairs/      # doorbell-upstairs-*.wav (+ augmented copies)
```

Augmented files follow naming: `doorbell-downstairs-1-aug-speed-0-9.wav`. These are gitignored.

### Dependencies by env
- **Training**: librosa, numpy, tensorflow
- **Inference**: numpy, ai-edge-litert only — no librosa, no feature extraction libs
- **Stream** (`detect.py`): stdlib `threading`, `urllib.request`, `urllib.parse` for Pushsafer notifications (daemon thread, 3s timeout, silent failure)
- **False-positive scan** (`test-false-positives.sh`): ffmpeg and standard Unix `find`

## Running

Run from the repo root with the venv active: the scripts use `#!/usr/bin/env python`, and `doorbell.tflite`, `data/` and `detected/` are relative paths.

```bash
source .venv/bin/activate

# Training (full pipeline: data → model → .tflite); overwrites the tracked doorbell.tflite and prints its size in bytes
./train.py

# Real-time stream from stdin (raw 16-bit PCM @ 16kHz mono)
cat audio.raw | ./detect.py
# Output: YYYY-MM-DDTHH:MM:SS\tLABEL DOORBELL on confirmed detection (~once per 10s cooldown)

# Every data/test/*.wav through detect.py, mixed with brown noise (NOISE_VOL default 0.9, 0 = clean)
./test.sh [NOISE_VOL]

# One clip
sox data/test/downstairs-test-pi.wav -c 1 -r 16000 -e signed-integer -b 16 -t raw - | ./detect.py

# Scan local media for false positives
./test-false-positives.sh

# Bytes per tensor in the exported model
python -c 'import tensorflow as tf; tf.lite.experimental.Analyzer.analyze(model_path="doorbell.tflite")'
```

- No unit tests, linter or CI. `test.sh` asserts nothing: it prints each file name, then `DOWNSTAIRS DOORBELL` / `UPSTAIRS DOORBELL` if detected. The expected label is the file-name prefix; a name with nothing after it is a miss.
- `test.sh` mixes in new random brown noise on every run, and training is unseeded, so one run proves little. Count detections of the hardest clip, `upstairs-test-pi.wav`, over 10 runs (expect 10): `for i in $(seq 10); do ./test.sh; echo; done | grep -c -E 'upstairs-test-pi\.wav[[:space:]]+[^[:space:]]+[[:space:]]+UPSTAIRS'`
- `test.sh` sends `detect.py` stderr to `/dev/null`, so a crash shows as one file name and exit status 1. Run the one-clip command to see the error.
- A clip triggers only if it has at least 1.7 s of audio: one 1 s window plus 7 strides of 100 ms for the 8-frame streak.

## Augmentation

Run `./augment.sh` to apply 19 transforms to every `.wav` in `data/downstairs/` and `data/upstairs/`, then download ESC-50 to `data/environment/` and write the sine negatives there: 31 pure tones from 850 to 1150 Hz in 10 Hz steps, 5 s each, made repeatable with `sox -R`. The upstairs bell is 976 Hz plus odd harmonics, and the model fired UPSTAIRS on a bare 1 kHz sine. Each transform produces a copy with `-aug-<name>.wav` suffix. The script uses `find -print0` for safe filename handling and reports success/failure counts.

Pass specific `.wav` paths as arguments to augment only those files (skips ESC-50 and the sines):

```bash
./augment.sh data/downstairs/downstairs-061-pi.wav data/upstairs/upstairs-052-pi.wav
```

Without arguments the script also re-augments existing `-aug-` files, because its `find` does not exclude them. Pass only the new files, or first run `find data -name '*-aug-*.wav' -delete` (they are regenerated).

## Gitignore

```
.DS_Store
.venv/
__pycache__/
data/**/*-aug-*.wav    # augmented files (regenerated)
data/environment/[1-5]-*.wav   # raw ESC-50 files (downloaded by script)
data/environment/synthetic-*.wav   # sine negatives (generated by script)
data.bak/              # backup directory
detected/              # DUMP_DETECTED output from detect.py
```

## Model File

`doorbell.tflite` — optimized TFLite, ~45 KB with float32 input/output and quantized internal tensors. This is the deployable artifact for edge inference. Regenerated each time `train.py` runs successfully.

Most of the file is not CNN weights. About half is graph metadata (the in-graph frontend is 45 of the 80 ops). The largest tensor is the int8 mel filterbank, `(FRAME_LENGTH/2+1) × N_MELS` bytes (10,280 now), followed by the float32 Hann window, `FRAME_LENGTH × 4` bytes. The parameter count excludes these frontend constants, so changing `FRAME_LENGTH` or `N_MELS` moves the file size more than the parameter count suggests. Measure with the Analyzer command under Running.

## Specific instructions for agents

When the model is changed, be sure to update README.md and AGENTS.md accordingly. Be extra careful with model file size and the number of parmeters.
