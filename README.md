# Doorbell Detector

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). The deployed model runs on a Raspberry Pi Zero as a ~45 KB optimized TFLite file with float32 input and output, accepting raw PCM audio and returning a classification - no feature extraction libraries needed at inference time.

## Quick Start

```bash
# Stream live audio from stdin (raw 16-bit PCM @ 16kHz mono)
arecord -c1 -D plug:dsnoop -f S16_LE -r 16000 -t raw | ./detect.py
```

## Requirements

- Python ≥3.9
- [sox](https://sourceforge.net/p/sox/wiki/Home/) - audio augmentation and format conversion
- [ffmpeg](https://ffmpeg.org/) - media conversion for false-positive scanning
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
pip install numpy ai-edge-litert pyalsaaudio
```

## Data Preparation

Place `.wav` files into `data/{downstairs,upstairs,environment}/`:

```
data/
├── downstairs/       # doorbell-downstairs-*.wav
├── environment/      # background noise (ESC-50 or your own)
└── upstairs/         # doorbell-upstairs-*.wav
```

### Augment your recordings & download environment data

Apply 20 transforms per file (speed, pitch, volume, reverb, filters, echo, etc.) to `data/downstairs/` and `data/upstairs/`, then download ESC-50 into `data/environment/`:

```bash
./augment.sh
```

Pass specific `.wav` paths as arguments to augment only those files (skips ESC-50 download):

```bash
./augment.sh data/downstairs/my-recording.wav data/upstairs/test.wav
```

Augmented files get a `-aug-<name>.wav` suffix and are gitignored.

## Training

```bash
./train.py
```

Trains an end-to-end model (raw audio → Mel-spectrogram → CNN → class) and exports `doorbell.tflite` (~45 KB optimized TFLite).

## Usage

### Testing

```bash
./test.sh
```

Runs detect.py on each `.wav` in `data/test/`, mixing brown noise for robustness checks.

### False-positive scan

```bash
./test-false-positives.sh
```

Scans the home directory for `.ac3`, `.avi`, `.mkv`, and `.mp3` files, converts each to 16 kHz mono PCM with `ffmpeg`, and runs the detector. Uncomment `export DUMP_DETECTED=1` in the script to save confirmed detections under `detected/` for inspection.

### Offline Stream Prediction

```bash
sox data/test/downstairs-test-pi.wav -c 1 -r 16000 -e signed-integer -b 16 -t raw - | ./detect.py
```

Output on confirmed detection only: `YYYY-MM-DDTHH:MM:SS\tLABEL DOORBELL`

Detection logic:
- Predictions below 90% confidence are treated as `"environment"`
- Requires 8 consecutive frames with the same label (~0.8 second) before triggering
- 10-second cooldown after each detection
- Optional Pushsafer notifications via `PUSHSAFER_KEY` env var
- Optional WAV dump of detected audio via `DUMP_DETECTED=1` env var (saved to `detected/`)

### Live Capture from Microphone

```bash
# Requires pyalsaaudio installed and ALSA device available
./detect.py
```

Automatically detects when stdin is not a pipe and falls back to ALSA capture.

### Systemd Service (Persistent Daemon)

Run as a background service that auto-starts on boot:

```bash
mkdir -p ~/.config/systemd/user
ln -rs doorbell-detector.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now doorbell-detector.service
loginctl enable-linger $USER
```

Configure environment variables in the service file:

| Variable | Description |
|----------|-------------|
| `ALSAAUDIO_DEVICE` | ALSA capture device (default: auto-detect) |
| `DUMP_DETECTED=1` | Save detected audio to WAV files |
| `PUSHSAFER_KEY` | Pushsafer API key for notifications |

The service automatically falls back to ALSA when stdout is connected to journald.
