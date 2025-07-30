<div align="center">

# Project Structure

This document describes the modular architecture and organization of the CSI-Predictor project.

</div>

The CSI-Predictor project follows a modular design pattern with clear separation of concerns. Each module has a single responsibility and can be developed, tested, and maintained independently.

## 📁 Directory Structure

```
CSI-Predictor/
├── src/                          # Main source code (modular)
│   ├── config/                   # Configuration management
│   │   ├── config.py            # Main config class
│   │   ├── config_loader.py     # Config loading logic
│   │   ├── validation.py        # Config validation
│   │   └── __init__.py          # Main config module
│   ├── data/                     # Data pipeline
│   │   ├── dataset.py           # CSIDataset class
│   │   ├── dataloader.py        # Data loader creation
│   │   ├── transforms.py        # Image transformations
│   │   ├── preprocessing.py     # Data preprocessing
│   │   ├── splitting.py         # Data splitting utilities
│   │   └── __init__.py          # Data package
│   ├── models/                   # Model architectures
│   │   ├── factory.py           # Model factory
│   │   ├── backbones/           # Feature extraction backbones
│   │   │   ├── custom.py        # Custom CNN backbone
│   │   │   ├── resnet.py        # ResNet backbones
│   │   │   ├── densenet.py      # DenseNet/CheXNet backbones
│   │   │   └── raddino.py       # RadDINO backbone
│   │   ├── heads/               # Classification heads
│   │   │   ├── csi_head.py      # CSI classification head
│   │   │   └── regression_head.py # Regression head
│   │   ├── complete/            # Complete models
│   │   │   └── csi_models.py    # Complete CSI models
│   │   └── __init__.py          # Models package
│   ├── training/                 # Training pipeline
│   │   ├── trainer.py           # Main training logic
│   │   ├── optimizer.py         # Optimizer management
│   │   ├── loss.py              # Loss functions
│   │   ├── metrics.py           # Training metrics
│   │   ├── callbacks.py         # Training callbacks
│   │   └── __init__.py          # Training package
│   ├── evaluation/               # Evaluation pipeline
│   │   ├── evaluator.py         # Main evaluation logic
│   │   ├── metrics/             # Evaluation metrics
│   │   │   ├── classification.py # Classification metrics
│   │   │   ├── confusion_matrix.py # Confusion matrix utilities
│   │   │   └── f1_score.py      # F1 score calculations
│   │   ├── visualization/       # Evaluation visualization
│   │   │   ├── plots.py         # Plotting utilities
│   │   │   └── confusion_matrix.py # Confusion matrix plots
│   │   ├── wandb_logging.py     # W&B logging
│   │   └── __init__.py          # Evaluation package
│   ├── optimization/             # Hyperparameter optimization
│   │   ├── hyperopt.py          # Optuna hyperparameter optimization
│   │   ├── wandb_sweep.py       # W&B sweep integration
│   │   └── __init__.py          # Optimization package
│   ├── utils/                    # Utility functions
│   │   ├── logging.py           # Logging setup
│   │   ├── checkpoint.py        # Model checkpointing
│   │   ├── visualization.py     # General visualization
│   │   ├── file_utils.py        # File operations
│   │   ├── seed.py              # Random seed management
│   │   └── __init__.py          # Utils package
│   ├── cli/                      # Command-line interface
│   │   ├── main.py              # Main CLI entry point
│   │   ├── train.py             # Training CLI
│   │   ├── evaluate.py          # Evaluation CLI
│   │   ├── optimize.py          # Optimization CLI
│   │   └── __init__.py          # CLI package
│   └── __init__.py               # Main package
├── scripts/                      # Utility scripts
│   ├── debug/                   # Debugging scripts
│   │   ├── debug_images.py      # Image debugging
│   │   └── diagnose_raddino.py  # RadDINO diagnostics
│   ├── data/                    # Data processing scripts
│   │   └── download_archimed.py # ArchiMed downloader
│   ├── tests/                   # Testing scripts
│   │   ├── test_metrics.py      # Metrics tests
│   │   └── test_raddino.py      # RadDINO tests
│   └── __init__.py              # Scripts package
├── config/                       # Configuration files
│   ├── config.ini               # Main configuration
│   ├── config_example.ini       # Example configuration
│   └── sweep_config.yaml        # W&B sweep configuration
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── logs/                         # Log files
├── models/                       # Trained models
├── main.py                       # Legacy entry point
└── requirements.txt              # Dependencies
```

## 🧩 Module Descriptions

### 📋 Configuration Module (`src/config/`)

**Purpose**: Centralized configuration management with validation and loading.

**Key Components**:
- `config.py`: Main `Config` dataclass with all configuration options
- `config_loader.py`: Loading logic for environment variables and INI files
- `validation.py`: Configuration validation and error checking
- `__init__.py`: Singleton access and main configuration functions

**Usage**:
```python
from src.config import cfg, get_config

# Use singleton configuration
print(cfg.model_arch)

# Or get fresh configuration
config = get_config()
```

### 📊 Data Module (`src/data/`)

**Purpose**: Complete data pipeline for loading, preprocessing, and transforming medical images.

**Key Components**:
- `dataset.py`: `CSIDataset` class for PyTorch data loading
- `dataloader.py`: DataLoader creation and data splitting
- `transforms.py`: Image transformations and preprocessing
- `preprocessing.py`: Data preprocessing utilities
- `splitting.py`: Stratified data splitting functions

**Usage**:
```python
from src.data.dataloader import create_data_loaders
from src.data.dataset import CSIDataset

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config)
```

### 🤖 Models Module (`src/models/`)

**Purpose**: Model architectures and factory functions for building complete models.

**Key Components**:
- `factory.py`: Model factory functions for building complete models
- `backbones/`: Feature extraction backbones
  - `custom.py`: Custom CNN backbone
  - `resnet.py`: ResNet50 backbone
  - `densenet.py`: DenseNet121/CheXNet backbone
  - `raddino.py`: RadDINO backbone
- `heads/`: Classification heads
  - `csi_head.py`: CSI classification head
  - `regression_head.py`: Regression head
- `complete/`: Complete model implementations
  - `csi_models.py`: Complete CSI models

**Usage**:
```python
from src.models.factory import build_model
from src.models.backbones import get_backbone

# Build complete model
model = build_model(config)

# Get specific backbone
backbone = get_backbone("resnet50")
```

### 🏋️ Training Module (`src/training/`)

**Purpose**: Complete training pipeline with loss functions, metrics, and callbacks.

**Key Components**:
- `trainer.py`: Main training logic and pipeline
- `loss.py`: Loss functions (WeightedCSILoss)
- `metrics.py`: Training metrics computation
- `optimizer.py`: Optimizer and scheduler management
- `callbacks.py`: Training callbacks (EarlyStopping, MetricsTracker)

**Usage**:
```python
from src.training.trainer import train_model
from src.training.loss import WeightedCSILoss

# Train model
train_model(config)

# Use custom loss
criterion = WeightedCSILoss(unknown_weight=0.3)
```

### 📈 Evaluation Module (`src/evaluation/`)

**Purpose**: Model evaluation, metrics computation, and visualization.

**Key Components**:
- `evaluator.py`: Main evaluation logic and pipeline
- `metrics/`: Evaluation metrics
  - `classification.py`: Classification metrics
  - `confusion_matrix.py`: Confusion matrix utilities
  - `f1_score.py`: F1 score calculations
- `visualization/`: Evaluation visualization
  - `plots.py`: ROC curves, PR curves, training curves
  - `confusion_matrix.py`: Confusion matrix plots
- `wandb_logging.py`: Weights & Biases logging

**Usage**:
```python
from src.evaluation.evaluator import evaluate_model
from src.evaluation.metrics import compute_confusion_matrix

# Evaluate model
evaluate_model(config)

# Compute metrics
conf_matrix = compute_confusion_matrix(predictions, targets)
```

### 🔧 Optimization Module (`src/optimization/`)

**Purpose**: Hyperparameter optimization using Optuna and W&B sweeps.

**Key Components**:
- `hyperopt.py`: Optuna hyperparameter optimization
- `wandb_sweep.py`: Weights & Biases sweep integration

**Usage**:
```python
from src.optimization.hyperopt import optimize_hyperparameters
from src.optimization.wandb_sweep import initialize_sweep

# Run hyperparameter optimization
study = optimize_hyperparameters(n_trials=100)

# Initialize W&B sweep
sweep_id = initialize_sweep(project="my-project")
```

### 🛠️ Utils Module (`src/utils/`)

**Purpose**: Utility functions for logging, checkpointing, and visualization.

**Key Components**:
- `logging.py`: Logging setup and configuration
- `checkpoint.py`: Model checkpointing utilities
- `visualization.py`: General visualization functions
- `file_utils.py`: File operation utilities
- `seed.py`: Random seed management

**Usage**:
```python
from src.utils.logging import logger
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import seed_everything

# Setup logging
logger.info("Starting training")

# Save checkpoint
save_checkpoint(model, optimizer, epoch, path)

# Set random seeds
seed_everything(42)
```

### 💻 CLI Module (`src/cli/`)

**Purpose**: Command-line interface for all project operations.

**Key Components**:
- `main.py`: Main CLI entry point and routing
- `train.py`: Training CLI commands
- `evaluate.py`: Evaluation CLI commands
- `optimize.py`: Optimization CLI commands

**Usage**:
```bash
# Use main CLI
python -m src.cli.main --mode train

# Use individual CLI modules
python -m src.cli.train
python -m src.cli.evaluate
python -m src.cli.optimize
```

## 🔄 Migration from Monolithic Structure

### Backward Compatibility

All existing code continues to work without modification:

```python
# These imports still work
from src.train import train_model
from src.evaluate import evaluate_model
from src.data import create_data_loaders
from src.models import build_model
from src.utils import logger
from src.config import cfg
```

### New Modular Imports (Recommended)

```python
# Recommended new imports
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.data.dataloader import create_data_loaders
from src.models.factory import build_model
from src.utils.logging import logger
from src.config import cfg
```

## 🎯 Design Principles

### 1. Single Responsibility Principle
Each module has one clear purpose and responsibility.

### 2. Dependency Inversion
High-level modules don't depend on low-level modules. Both depend on abstractions.

### 3. Interface Segregation
Clients aren't forced to depend on interfaces they don't use.

### 4. Open/Closed Principle
Modules are open for extension but closed for modification.

### 5. DRY (Don't Repeat Yourself)
Code duplication is eliminated through proper abstraction.

### 6. KISS (Keep It Simple, Stupid)
Simple solutions are preferred over complex ones.

## 🔗 Module Dependencies

```
src/
├── config/          # No dependencies (base module)
├── utils/           # Depends on config
├── data/            # Depends on config, utils
├── models/          # Depends on config, utils
├── training/        # Depends on config, utils, data, models
├── evaluation/      # Depends on config, utils, data, models, training
├── optimization/    # Depends on config, utils, data, models, training
└── cli/             # Depends on all other modules
```

## 🧪 Testing Strategy

Each module can be tested independently:

```python
# Test individual modules
python -m pytest tests/test_data.py
python -m pytest tests/test_models.py
python -m pytest tests/test_training.py
```

## 📚 Documentation

- [Quick Start Guide](quick-start.md) - Getting started
- [Training Guide](training.md) - Training procedures
- [Evaluation Guide](evaluation.md) - Evaluation procedures
- [API Reference](api-reference.md) - Function documentation
- [Configuration Guide](config-guide.md) - Configuration options

## 🚀 Benefits of Modular Architecture

1. **Maintainability**: Smaller, focused files are easier to understand and modify
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Components can be easily reused across the project
4. **Scalability**: New features can be added without affecting existing code
5. **Documentation**: Clear structure makes it easier to understand and document
6. **Team Development**: Multiple developers can work on different modules simultaneously
7. **Code Quality**: Modular structure encourages better code organization and practices 