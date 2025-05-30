# CSI-Predictor

A modular PyTorch project for predicting 6-zone CSI (Chest X-ray Severity Index) scores on chest X-ray images.

## Project Structure

```
CSI-PREDICTOR/
├── docs/                     # Documentation files
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Main source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── data.py             # Data loading and preprocessing
│   ├── models/             # Neural network models
│   │   ├── __init__.py
│   │   ├── backbones.py    # Feature extraction backbones
│   │   └── head.py         # Classification heads
│   ├── train.py            # Training logic
│   ├── evaluate.py         # Evaluation logic
│   └── utils.py            # Utility functions
├── .env.template           # Environment variables template
├── config.ini              # Configuration file
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore rules
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CSI-Predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your specific paths and settings
   ```

## Configuration

The project uses a centralized configuration system that loads settings from multiple sources with the following priority:
1. **Environment Variables** (highest priority)
2. **config.ini file** 
3. **Default values** (lowest priority)

### Configuration Files

**config.ini**:
```ini
[TRAINING]
BATCH_SIZE = 32
N_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
OPTIMIZER = adam

[MODEL]
MODEL_ARCH = resnet50
MODELS_FOLDER = ./models
```

**Environment Variables** (copy `env_template.txt` to `.env`):
```bash
# Device configuration
DEVICE=cuda
LOAD_DATA_TO_MEMORY=False

# Data configuration
DATA_SOURCE=local
DATA_PATH=./data

# Model storage
MODELS_FOLDER=./models

# CSV configuration
CSV_PATH=./data/metadata.csv
LABELS_CSV=./data/labels.csv
```

### Model Path Configuration

The project now uses `MODELS_FOLDER` to specify where models are stored. To get the full path of a specific model:

```python
from src.config import cfg

# Get path for a specific model
model_path = cfg.get_model_path("best_model")  # Returns: ./models/best_model.pth
model_path = cfg.get_model_path("resnet50_trained", "pt")  # Returns: ./models/resnet50_trained.pt

# During training, models are automatically saved to the models folder
# Example: best model saved as ./models/best_model.pth
```

### Configuration Usage

```python
from src.config import cfg

# Access configuration values
print(f"Batch size: {cfg.batch_size}")
print(f"Models folder: {cfg.models_folder}")
print(f"Best model path: {cfg.get_model_path('best_model')}")
```

## Usage

### Training
Train a new model:
```bash
python main.py --mode train
```

### Evaluation
Evaluate an existing model:
```bash
python main.py --mode eval
```

### Both Training and Evaluation
Run complete pipeline:
```bash
python main.py --mode both
```

### Custom Configuration
Use custom config files:
```bash
python main.py --config custom_config.ini --env custom.env
```

## Model Architecture

The project supports multiple backbone architectures for feature extraction:
- ResNet (resnet18, resnet34, resnet50, resnet101)
- EfficientNet
- DenseNet
- Vision Transformers

The model predicts CSI scores for 6 zones of chest X-rays using a multi-output regression head.

## Data Format

Expected data structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── metadata.csv
└── labels.csv
```

Labels CSV should contain:
- Image filename
- CSI scores for 6 zones
- Additional metadata

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]