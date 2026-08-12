# Doorbell Detector

A 1D CNN that classifies doorbell audio into three categories: **downstairs**, **upstairs**, or **environment** (background noise). Uses MFCC features extracted at 8kHz sample rate.

## Requirements

- Python ≥3.9
- [sox](https://sourceforge.net/p/sox/wiki/Home/) - audio augmentation and format conversion
- `bsdtar` (Linux: `sudo apt install libarchive-tools`, macOS: built-in) - environment sounds extraction

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
pip install librosa numpy ai-edge-litert
```

## Data Preparation

### 1. Augment existing data

Run augmentation on your own doorbell recordings to increase dataset diversity:

```bash
./augment.sh
```

This applies 20+ transforms per `.wav` file (speed/tempo ±10%, pitch ±200 cents, volume ±30%, reverb, echo, flanger, overdrive, compand, lowpass/highpass/bandpass filtering, EQ dip, proximity effect, padding) and saves them alongside the originals with `-aug-<transform>.wav` suffixes.

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
2. Extracts MFCC features (13 frames, padded/truncated to 128 time steps)
3. Computes class weights to handle dataset imbalance
4. Trains a 1D CNN with early stopping and learning rate reduction
5. Converts the trained model to FP16 quantized TFLite for LiteRT inference

Output includes sample counts, class distribution, class weights, training progress, final validation accuracy, and tensor details for the exported model. The converted model is saved as `doorbell.tflite` (~173 KB).

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

## Model Architecture

```
Input(128, 13) → BatchNorm
               → SeparableConv1D(64, k=5, relu, same) → BatchNorm → MaxPool(2)
               → SeparableConv1D(64, k=5, relu, same) → BatchNorm → Residual Add → MaxPool(2)
               → GlobalAveragePooling1D → Dropout(0.2) → Softmax(3)
```

**Total parameters:** ~6,200 (25 KB FP16 TFLite)

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Shared constants (sample rate, model paths, labels) |
| `inferencer.py` | MFCC extraction and LiteRT inference utilities |
| `train.py` | Training script - loads data, builds model, trains, converts to FP16 TFLite |
| `predict.py` | Batch prediction on `.wav` files using LiteRT |
| `predict_stream.py` | Real-time stream prediction via stdin (2s windows, 2 Hz trigger rate) |
| `augment.sh` | Audio augmentation with sox (speed, pitch, volume, reverb, filters, echo, flanger, and more) |
| `get-env-data.sh` | Download ESC-50 environmental sounds |
