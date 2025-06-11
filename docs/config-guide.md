# Configuration Guide

CSI-Predictor uses a flexible, hierarchical configuration system that supports multiple sources and formats. This guide covers all configuration options and best practices.

## Configuration Hierarchy

The configuration system loads settings from multiple sources in this priority order:

1. **Environment Variables** (highest priority)
2. **INI Configuration Files** (medium priority)  
3. **Default Values** (lowest priority)

This allows you to:
- Set production defaults in INI files
- Override specific settings with environment variables
- Use different configurations for different environments

## Configuration Sources

### 1. Environment Variables (.env file)

Create a `.env` file in your project root:

```bash
# Device and Performance
DEVICE=cuda
LOAD_DATA_TO_MEMORY=True

# Data Paths
DATA_SOURCE=local
DATA_DIR=/home/pyuser/data/Paradise_Images
CSV_DIR=/home/pyuser/data/Paradise_CSV
MODELS_DIR=./models
INI_DIR=./config/
GRAPH_DIR=./graphs

# Labels Configuration
LABELS_CSV=Labeled_Data_RAW.csv
LABELS_CSV_SEPARATOR=;

# Training Parameters (can override INI settings)
BATCH_SIZE=32
N_EPOCHS=100
LEARNING_RATE=0.001
MODEL_ARCH=resnet50
```

### 2. INI Configuration Files

Create a `config.ini` file:

```ini
[TRAINING]
# Core training parameters
BATCH_SIZE = 32
N_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001
OPTIMIZER = adam

[MODEL]
# Model architecture settings  
MODEL_ARCH = resnet50
USE_OFFICIAL_PROCESSOR = false

[DATA]
# Data loading settings (usually better in .env)
# LOAD_DATA_TO_MEMORY = true

[OPTIMIZATION]
# Hyperparameter optimization settings
OPTUNA_N_TRIALS = 50
WANDB_PROJECT = csi-predictor
```

### 3. Command Line Arguments

Many scripts accept INI file paths:

```bash
# Use custom configuration file
python -m src.train --ini experiments/resnet_config.ini
python -m src.evaluate --ini production_config.ini

# Legacy entry point
python main.py --mode train --config custom_config.ini
```

## Complete Configuration Reference

### Device and Performance

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `DEVICE` | string | `"cuda"` | Device for training: `"cuda"`, `"cpu"`, `"mps"` |
| `LOAD_DATA_TO_MEMORY` | bool | `True` | Cache images in memory for faster training |

**Examples:**
```bash
# Use CPU only
DEVICE=cpu

# Use Apple Silicon GPU (Mac M1/M2)
DEVICE=mps

# Disable memory caching for large datasets
LOAD_DATA_TO_MEMORY=False
```

### Data Paths

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `DATA_SOURCE` | string | `"local"` | Data source type |
| `DATA_DIR` | string | Required | Directory containing chest X-ray images |
| `CSV_DIR` | string | Required | Directory containing label CSV files |
| `MODELS_DIR` | string | `"./models"` | Directory for saving/loading models |
| `INI_DIR` | string | `"./"` | Directory for configuration files |
| `GRAPH_DIR` | string | `"./graphs"` | Directory for saving plots and graphs |

**Examples:**
```bash
# Local development setup
DATA_DIR=/home/user/datasets/csi/images
CSV_DIR=/home/user/datasets/csi/labels
MODELS_DIR=./models

# Production setup with shared storage
DATA_DIR=/mnt/shared/csi_data/images
CSV_DIR=/mnt/shared/csi_data/labels  
MODELS_DIR=/mnt/shared/csi_models
```

### Labels Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `LABELS_CSV` | string | `"Labeled_Data_RAW.csv"` | Name of the labels CSV file |
| `LABELS_CSV_SEPARATOR` | string | `";"` | CSV delimiter character |

**Examples:**
```bash
# Comma-separated CSV
LABELS_CSV=csi_labels.csv
LABELS_CSV_SEPARATOR=,

# Tab-separated file
LABELS_CSV=labels.tsv
LABELS_CSV_SEPARATOR=\t
```

### Training Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `BATCH_SIZE` | int | `32` | Training batch size |
| `N_EPOCHS` | int | `100` | Maximum number of training epochs |
| `PATIENCE` | int | `10` | Early stopping patience |
| `LEARNING_RATE` | float | `0.001` | Initial learning rate |
| `OPTIMIZER` | string | `"adam"` | Optimizer: `"adam"`, `"adamw"`, `"sgd"` |

**Examples:**
```ini
[TRAINING]
# Small GPU setup
BATCH_SIZE = 16
N_EPOCHS = 50
PATIENCE = 5

# Large GPU setup  
BATCH_SIZE = 128
N_EPOCHS = 200
PATIENCE = 20

# Fast prototyping
BATCH_SIZE = 8
N_EPOCHS = 10
PATIENCE = 3
```

### Model Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `MODEL_ARCH` | string | `"resnet50"` | Model architecture |
| `USE_OFFICIAL_PROCESSOR` | bool | `False` | Use official RadDINO processor |

**Supported Architectures:**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `efficientnet_b0` to `efficientnet_b4`
- `densenet121`, `densenet169`, `densenet201`
- `chexnet` (DenseNet121 pretrained on CheXpert)
- `custom1` (simple 5-layer CNN)
- `raddino` (Microsoft's chest X-ray specialist)
- `vit_base_patch16_224`, `vit_large_patch16_224`

**Examples:**
```bash
# Fast baseline
MODEL_ARCH=resnet50

# Best accuracy
MODEL_ARCH=raddino

# Quick testing
MODEL_ARCH=custom1
```

## Usage in Code

```python
from src.config import cfg
from src.utils import print_config

# Access configuration
print(f"Batch size: {cfg.batch_size}")
print(f"Model: {cfg.model_arch}")
print(f"Device: {cfg.device}")

# Pretty print all settings
print_config(cfg)

# Check if using memory caching
if cfg.load_data_to_memory:
    print("Data will be cached in memory")
```

## Environment-Specific Configurations

### Development
```bash
# .env.development
DEVICE=cuda
BATCH_SIZE=16
N_EPOCHS=10
LOAD_DATA_TO_MEMORY=True
```

### Production
```bash
# .env.production
DEVICE=cuda
BATCH_SIZE=64
N_EPOCHS=200
LOAD_DATA_TO_MEMORY=False
```

### Testing
```bash
# .env.test
DEVICE=cpu
BATCH_SIZE=2
N_EPOCHS=1
LOAD_DATA_TO_MEMORY=False
``` 