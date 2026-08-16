# Doorbell Detector

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). The deployed model runs on a Raspberry Pi Zero as a ~57 KB FP16 TFLite file, accepting raw PCM audio and returning a classification - no feature extraction libraries needed at inference time.

## Quick Start

```bash
# Stream live audio from stdin (raw 16-bit PCM @ 8kHz mono)
arecord -c1 -D plug:dsnoop -f S16_LE -r 16000 -t raw | ./detect.py
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
./train.py
```

Trains an end-to-end model (raw audio → Mel-spectrogram → CNN → class) and exports `doorbell.tflite` (~57 KB FP16).

## Usage

### Real-time Stream Prediction

```bash
sox data/test/downstairs-test-pi.wav -c 1 -r 16000 -e signed-integer -b 16 -t raw - | ./detect.py
```

Output on confirmed detection only: `YYYY-MM-DDTHH:MM:SS\tLABEL DOORBELL`

Detection logic:
- Predictions below 90% confidence are treated as `"environment"`
- Requires 5 consecutive frames with the same label (~0.5 second) before triggering
- 10-second cooldown after each detection
- Optional Pushsafer notifications via `PUSHSAFER_KEY` env var
