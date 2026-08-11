# Doorbell Detector

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). Uses MFCC features extracted at 8kHz sample rate.

## Requirements

- Python ≥3.9
- [sox](https://sourceforge.net/p/sox/wiki/Home/) - audio augmentation and format conversion
- `bsdtar` (Linux: `sudo apt install libarchive-tools`, macOS: built-in) - environment sounds extraction
- PortAudio library (Linux: `sudo apt install portaudio19-dev`, macOS: `brew install portaudio`)

## Setup

### Training (TensorFlow)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install librosa numpy tensorflow
```

### Prediction (LiteRT)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install librosa numpy sounddevice ai-edge-litert
```

## Data Preparation

### 1. Augment existing data

Run augmentation on your own doorbell recordings to increase dataset diversity:

```bash
./augment.sh
```

This applies 8 transforms per `.wav` file (speed ±10%, pitch ±200 cents, volume ±30%, reverb, lowpass/highpass filtering) and saves them alongside the originals with `-aug-<transform>.wav` suffixes.

### 2. Enrich with environmental sounds

Download background noise samples from [ESC-50](https://github.com/karoldvl/ESC-50):

```bash
./get-env-data.sh
```

This downloads ESC-50 audio files into `data/environment/`.

### Directory structure

After data preparation, your dataset should look like:

```
data/
├── downstairs/
│   ├── doorbell-downstairs-1.wav
│   └── ...
├── environment/
│   ├── esc50_0.wav
│   └── ...
└── upstairs/
    ├── doorbell-upstairs-1.wav
    └── ...
```

## Training

Train the model on your prepared dataset:

```bash
python train.py
```

The training process:
1. Loads all `.wav` files from `data/{downstairs,upstairs,environment}/`
2. Extracts MFCC features (40 frames, padded/truncated to 128 time steps)
3. Computes class weights to handle dataset imbalance
4. Trains a 1D CNN with early stopping and learning rate reduction
5. Converts the trained model to INT8 quantized TFLite for LiteRT inference

Output includes sample counts, class distribution, class weights, training progress, final validation accuracy, and tensor details for the exported model. The converted model is saved as `doorbell_int8.tflite` (~320 KB from ~1.2 MB uncompressed).

## Batch Prediction

Classify one or more `.wav` files:

```bash
python predict.py file1.wav [file2.wav ...]
```

Output is tab-separated for easy parsing: `label\tconfidence\tfilename`

Example:
```
downstairs	99.85%	doorbell-downstairs-30.wav
upstairs	97.20%	doorbell-upstairs-12.wav
environment	100.00%	background-noise.wav
```

## Live Classification

Run continuous classification from your microphone:

```bash
python predict-live.py
```

This streams audio at 8kHz, classifies the latest 8 seconds every second, and prints results:

```
Starting microphone at 8000Hz mono...
Classifying every 1s (latest 16s of audio). Press Ctrl-C to exit.

downstairs          (99.85%)
environment         (100.00%)
upstairs            (97.20%)
```

Press **Ctrl-C** to stop. On Linux, the microphone defaults to ALSA `plug:dsnoop` for shared access; on macOS/Windows it uses the system default device.

## Model Architecture

```
Input(128, 40) → Conv1D(64, k=5, relu, same) → MaxPool(2)
               → Conv1D(128, k=5, relu, same) → MaxPool(2)
               → Flatten → Dense(64, relu) → Dropout(0.3) → Softmax(3)
```

**Total parameters:** 316,355 (320 KB INT8 TFLite)

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Shared constants (sample rate, model paths, labels) |
| `model_io.py` | MFCC extraction and LiteRT inference utilities |
| `train.py` | Training script - loads data, builds model, trains, converts to INT8 TFLite |
| `predict.py` | Batch prediction on `.wav` files using LiteRT |
| `predict-live.py` | Live microphone classification using LiteRT |
| `augment.sh` | Audio augmentation with sox (speed, pitch, volume, reverb, filters) |
| `get-env-data.sh` | Download ESC-50 environmental sounds |
