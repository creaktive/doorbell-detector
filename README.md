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
pip install librosa numpy tensorflow
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

Apply 14 transforms per file (volume, overdrive, reverb, filters, echo, a quiet brown/pink/white noise mix on a third of the files, etc.; no speed/tempo/pitch shifts) to `data/downstairs/` and `data/upstairs/`, then download ESC-50 and write pure sine tones around 1 kHz (negatives for the 976 Hz upstairs bell) into `data/environment/`:

```bash
./augment.sh
```

Pass specific `.wav` paths as arguments to augment only those files (skips ESC-50 and the sine tones):

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
- Requires 6 consecutive frames with the same label (~0.6 second) before triggering
- 10-second cooldown after each detection
- Optional Pushsafer notifications via `PUSHSAFER_KEY` env var
- Optional WAV dump of detected audio via `DUMP_DETECTED=1` env var (saved to `detected/`)

### Live Capture from Microphone

```bash
# Requires pyalsaaudio installed and ALSA device available
./detect.py
```

Automatically detects when stdin is not a pipe and falls back to ALSA capture.

### Optional: `.asoundrc` for the ReSpeaker 2-Mics Pi HAT

The [ReSpeaker 2-Mics Pi HAT](https://www.seeedstudio.com/ReSpeaker-2-Mics-Pi-HAT.html) captures in stereo. Asking a generic `plug` device for 1 channel averages both mics, a mix the model never saw in training (training uses each channel alone). The bundled `.asoundrc` defines:

- `mic_dsnoop`: a shared (`dsnoop`) capture of the HAT at 16 kHz, 2 channels, S16_LE, so the codec records at the model's rate and ALSA does no resampling
- `mic_left`: a `plug` on top of it that passes only the left mic (`ttable.0.0 1`) as mono

Install and use it:

```bash
arecord -l                     # card name must match hw:seeed2micvoicec; edit .asoundrc if not
ln -rs .asoundrc ~/.asoundrc   # or merge into an existing ~/.asoundrc
arecord -D mic_left -r 16000 -c 1 -f S16_LE -d 1 -v /dev/null 2>&1 | grep -iE '<-|Snoop'
# expect "0 <- 0" (not "0*0.5 + 1*0.5") and "Direct Snoop PCM"
ALSAAUDIO_DEVICE=mic_left ./detect.py
```

`dsnoop` is shared, so `arecord -D mic_left` works while the service is running. For the systemd service, set `Environment="ALSAAUDIO_DEVICE=mic_left"`.

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
| `ALSAAUDIO_DEVICE` | ALSA capture device (default: ALSA `default`; `mic_left` with the bundled `.asoundrc`) |
| `DUMP_DETECTED=1` | Save detected audio to WAV files |
| `PUSHSAFER_KEY` | Pushsafer API key for notifications |

The service automatically falls back to ALSA when stdout is connected to journald.
