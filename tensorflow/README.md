# ChengetAI TensorFlow Models

Agricultural intelligence models for Zimbabwe's farming sector. Part of the ChengetAI platform.

## Models

| Model | Task | Input | Output |
|-------|------|-------|--------|
| Crop Disease Detection | Image classification | Crop leaf images | Disease class + confidence |
| Yield Prediction | Regression | Tabular sensor/field data | Predicted yield (tonnes/ha) |

## Project Structure

```
tensorflow/
├── config.yaml           # Model and training configuration
├── train.py              # Training entry point
├── predict.py            # Inference entry point
├── requirements.txt      # Python dependencies
├── data/
│   ├── preprocessing.py  # Data loading and augmentation
│   └── raw/              # Place training data here
├── models/
│   ├── crop_disease.py   # Disease classification model
│   └── yield_prediction.py # Yield regression model
├── utils/
│   └── config.py         # Configuration loader
├── saved_models/         # Trained model checkpoints
├── logs/                 # TensorBoard logs
└── notebooks/            # Jupyter notebooks for exploration
```

## Setup

```bash
cd tensorflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

### Crop Disease Classification

1. Organize images into class folders under `data/raw/`:
   ```
   data/raw/
       healthy/
       leaf_blight/
       rust/
       ...
   ```

2. Run training:
   ```bash
   python train.py --task crop_disease
   ```

### Yield Prediction

```bash
python train.py --task yield_prediction
```

## Inference

```bash
# Single image
python predict.py --model saved_models/best_model.keras --image path/to/image.jpg

# Batch prediction
python predict.py --model saved_models/best_model.keras --image-dir path/to/images/

# With class names
python predict.py --model saved_models/best_model.keras --image photo.jpg --classes "healthy,blight,rust"
```

## Configuration

Edit `config.yaml` to adjust:
- Image size, batch size, data splits
- Model backbone (`resnet50`, `mobilenetv2`, `efficientnetb0`)
- Training epochs, learning rate, early stopping
- File paths for data and checkpoints

## TensorBoard

```bash
tensorboard --logdir logs/
```
