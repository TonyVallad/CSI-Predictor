<div align="center">

# CSI-Predictor

A modular PyTorch project for predicting 6-zone CSI (Chest Severity Index) scores on chest X-ray images using deep learning.

</div>

## 🏗️ Project Structure

The CSI-Predictor project has been reorganized into a clean, modular architecture:

```
CSI-Predictor/
├── src/                          # Main source code
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

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CSI-Predictor
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the project:**
   ```bash
   cp config/config_example.ini config/config.ini
   # Edit config/config.ini with your settings
   ```

### Basic Usage

#### Training
```bash
# Train a model
python main.py --mode train

# Train with optimized hyperparameters
python main.py --mode train-optimized --hyperparams models/hyperopt/best_params.json
```

#### Evaluation
```bash
# Evaluate a trained model
python main.py --mode eval
```

#### Hyperparameter Optimization
```bash
# Run Optuna hyperparameter optimization
python main.py --mode hyperopt --n-trials 100

# Run W&B sweep
python main.py --mode sweep
```

#### Using the CLI
```bash
# Use the new CLI structure
python -m src.cli.main --mode train
python -m src.cli.main --mode evaluate
python -m src.cli.main --mode optimize
```

## 📚 Documentation

- **[Project Structure Analysis](project_structure_analysis.md)** - Detailed analysis of the project structure
- **[Migration Documentation](MIGRATION_PHASE1_README.md)** - Phase 1 migration details
- **[Migration Progress](MIGRATION_PHASE2_PROGRESS.md)** - Phase 2 & 3 migration progress
- **[Phase 3 Completion](MIGRATION_PHASE3_COMPLETION.md)** - Phase 3 completion report

## 🔧 Configuration

The project uses a hierarchical configuration system:

1. **Environment variables** (`.env` file)
2. **Configuration file** (`config/config.ini`)
3. **Code defaults** (`src/config/config.py`)

Key configuration options:
- `data_dir`: Path to data directory
- `model_arch`: Model architecture (ResNet50, CheXNet, RadDINO, Custom_1)
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `n_epochs`: Number of training epochs

## 🧪 Testing

### Debug Scripts
```bash
# Debug image loading and visualization
python scripts/debug/debug_images.py

# Diagnose RadDINO availability
python scripts/debug/diagnose_raddino.py
```

### Test Scripts
```bash
# Test metrics functionality
python scripts/tests/test_metrics.py

# Test RadDINO functionality
python scripts/tests/test_raddino.py
```

## 🔄 Migration Notes

This project has been migrated from a monolithic structure to a modular architecture:

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RadDINO team for the medical vision transformer
- PyTorch community for the deep learning framework
- Weights & Biases for experiment tracking
- Optuna for hyperparameter optimization

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.