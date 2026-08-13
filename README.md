# Doorbell Detector

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). The deployed model runs on a Raspberry Pi Zero as a ~47 KB FP16 TFLite file, accepting raw PCM audio and returning a classification - no feature extraction libraries needed at inference time.

## Quick Start

```bash
# Classify wav files
python predict.py doorbell-downstairs-1.wav background-noise.wav

# Stream live audio from stdin (raw 16-bit PCM @ 8kHz mono)
cat audio.raw | python predict_stream.py
```

## Requirements

- Python ≥3.9
- [sox](https://sourceforge.net/p/sox/wiki/Home/) - audio augmentation and format conversion
- `bsdtar` (Linux: `sudo apt install libarchive-tools`, macOS: built-in) - environment sounds extraction

## Setup

### Training Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install librosa numpy tensorflow ai-edge-litert
```

### Inference Environment (Pi Zero, etc.)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy ai-edge-litert
```

## Data Preparation

Place `.wav` files into `data/{downstairs,upstairs,environment}/`:

```
data/
├── downstairs/       # doorbell-downstairs-*.wav
├── environment/      # background noise (ESC-50 or your own)
└── upstairs/         # doorbell-upstairs-*.wav
```

### Augment your recordings

Apply 20 transforms per file (speed, pitch, volume, reverb, filters, echo, etc.):

```bash
./augment.sh [data_dir]
```

Augmented files get a `-aug-<name>.wav` suffix and are gitignored.

### Download environmental sounds

```bash
./get-env-data.sh
```

Downloads ESC-50 background noise into `data/environment/`.

## Training

```bash
python train.py
```

Trains an end-to-end model (raw audio → Mel-spectrogram → CNN → class) and exports `doorbell.tflite` (~47 KB FP16).

## Usage

### Batch Prediction

Classify one or more `.wav` files:

```bash
python predict.py file1.wav [file2.wav ...]
```

Output is tab-separated: `label\tconfidence\tfilename`

Example:
```
downstairs	99.85%	doorbell-downstairs-30.wav
upstairs	97.20%	doorbell-upstairs-12.wav
environment	100.00%	background-noise.wav
```

### Real-time Stream Prediction

```bash
cat audio.raw | python predict_stream.py
```

Output on confirmed detection only: `YYYY-MM-DDTHH:MM:SS\tLABEL DOORBELL`

Detection logic:
- Predictions below 90% confidence are treated as `"environment"`
- Requires 10 consecutive frames with the same label (~1 second) before triggering
- 10-second cooldown after each detection
- Optional Pushsafer notifications via `PUSHSAFER_KEY` env var
