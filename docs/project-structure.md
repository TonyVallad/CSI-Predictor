# Project Structure

This document provides a comprehensive overview of the CSI-Predictor project organization, file structure, and module responsibilities.

## 📁 Directory Overview

```
CSI-Predictor/
├── 📁 docs/                     # Documentation files
├── 📁 logs/                     # Training and application logs
├── 📁 models/                   # Saved model checkpoints and artifacts
├── 📁 notebooks/                # Jupyter notebooks for analysis and prototyping
├── 📁 src/                      # Main source code directory
│   ├── 📁 models/               # Model architectures and implementations
│   └── 📄 *.py                  # Core application modules
├── 📁 test_logs/                # Test-specific logs
├── 📄 main.py                   # Legacy entry point script
├── 📄 config.ini                # Configuration file
├── 📄 requirements.txt          # Python dependencies
├── 📄 sweep_config.yaml         # W&B sweep configuration
└── 📄 *.py                      # Utility and diagnostic scripts
```

## 🗂️ Detailed Structure

### Root Directory Files

| File | Purpose | Description |
|------|---------|-------------|
| `main.py` | Legacy Entry Point | Unified CLI for training, evaluation, and optimization |
| `config.ini` | Configuration | Default configuration settings |
| `.env` / `.env.template` | Environment | Environment variables and templates |
| `requirements.txt` | Dependencies | Python package requirements |
| `sweep_config.yaml` | W&B Configuration | Weights & Biases sweep parameters |
| `diagnose_raddino.py` | Diagnostic Tool | RadDINO model debugging utility |
| `download_archimed_images.py` | Data Tool | ArchiMed dataset downloading script |
| `test_*.py` | Testing Scripts | Model testing and validation utilities |

### 📁 `src/` - Core Application

The main source directory containing all application logic:

#### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Configuration Management | Settings loading, validation, environment handling |
| `data.py` | Data Pipeline | Dataset loading, preprocessing, augmentation |
| `train.py` | Training Logic | Model training, checkpointing, metrics tracking |
| `evaluate.py` | Evaluation System | Model testing, metrics computation, reporting |
| `metrics.py` | Metrics Computation | F1 scores, confusion matrices, medical metrics |
| `utils.py` | Utilities | Helper functions, model naming, logging setup |

#### Advanced Features

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `hyperopt.py` | Hyperparameter Optimization | Optuna integration, automated tuning |
| `wandb_sweep.py` | W&B Sweeps | Weights & Biases sweep management |
| `train_optimized.py` | Optimized Training | Training with best hyperparameters |
| `data_split.py` | Data Splitting | Train/validation/test split utilities |

#### Entry Points

| File | Purpose | Description |
|------|---------|-------------|
| `__init__.py` | Package Initialization | Module exports and package metadata |
| `__main__.py` | Modern Entry Point | Python module execution (`python -m src`) |

### 📁 `src/models/` - Model Architecture

Model implementations and neural network components:

| File | Purpose | Components |
|------|---------|------------|
| `__init__.py` | Model Registry | Available model list and factory functions |
| `backbones.py` | Backbone Networks | ResNet, EfficientNet, DenseNet, ViT implementations |
| `head.py` | Classification Heads | Multi-output heads for 6-zone CSI prediction |
| `rad_dino.py` | RadDINO Integration | Microsoft's specialized chest X-ray model |

### 📁 `docs/` - Documentation

Comprehensive project documentation:

#### User Documentation
- `installation.md` - Setup and installation guide
- `quick-start.md` - Getting started tutorial
- `config-guide.md` - Configuration reference
- `training.md` - Training guide and best practices
- `evaluation.md` - Model evaluation and metrics
- `data-format.md` - Expected data structure

#### Technical Documentation
- `model-architectures.md` - Available models and specifications
- `hyperparameter_optimization.md` - HPO with Optuna and W&B
- `legacy-entry-point.md` - Backward compatibility features
- `model_naming_format.md` - Model naming conventions

#### Developer Documentation
- `contributing.md` - Development guidelines
- `api-reference.md` - API documentation
- `project-structure.md` - This document

#### Implementation Details
- `Image Preprocessing.md` - Data preprocessing pipeline
- `ResNet50 Implementation in this project.md` - ResNet50 adaptation
- `ArchiMed Python Connector.md` - ArchiMed integration

### 📁 `models/` - Model Artifacts

Directory for storing trained models and checkpoints:

```
models/
├── 📁 checkpoints/              # Training checkpoints
├── 📁 hyperopt/                 # Hyperparameter optimization results
├── 📁 wandb/                    # W&B artifacts
└── 📄 *.pth                     # Final trained models
```

**Model Naming Convention**: `[YYYYMMDD_HHMMSS]_[ModelName]_[TaskTag]_[ExtraInfo].pth`

Example: `20250611_093054_ResNet50_Tr.pth`

### 📁 `notebooks/` - Analysis & Prototyping

Jupyter notebooks for data exploration and model development:

| Notebook | Purpose |
|----------|---------|
| `CSV_Exploration.ipynb` | Label data analysis |
| `CSV_Preprocessing*.ipynb` | Data preprocessing development |
| `Image_Exploration*.ipynb` | Image dataset analysis |
| `Download ArchiMed Images*.ipynb` | Dataset downloading workflows |
| `data_pipeline_demo.ipynb` | Data pipeline demonstration |

### 📁 `logs/` - Application Logs

Structured logging output:

```
logs/
├── 📄 csi_predictor_YYYY-MM-DD.log    # Daily application logs
└── 📁 archived/                        # Compressed old logs
```

### 📁 `test_logs/` - Test Logs

Separate logging for testing and validation:

```
test_logs/
└── 📄 csi_predictor_YYYY-MM-DD.log    # Test-specific logs
```

## 🔧 Module Dependencies

### Core Data Flow

```
config.py → data.py → train.py → evaluate.py
     ↓         ↓         ↓          ↓
   utils.py  models/  metrics.py  utils.py
```

### Advanced Workflows

```
hyperopt.py → train_optimized.py
wandb_sweep.py → train.py
main.py → [any module]
```

## 🚀 Entry Points

### Modern Entry Points (Recommended)

```bash
# Direct module execution
python -m src.train
python -m src.evaluate
python -m src.hyperopt

# Package execution
python -m src  # Uses __main__.py
```

### Legacy Entry Point

```bash
# Unified CLI
python main.py --mode train
python main.py --mode eval
python main.py --mode hyperopt
```

## 📦 Development Workflow

### Adding New Models

1. Implement in `src/models/backbones.py`
2. Add to model registry in `src/models/__init__.py`
3. Update documentation in `docs/model-architectures.md`

### Adding New Features

1. Create module in `src/`
2. Add CLI support in `__main__.py` or `main.py`
3. Update configuration in `config.py`
4. Add documentation in `docs/`

### Testing Changes

1. Use utility scripts in root directory
2. Check logs in `logs/` and `test_logs/`
3. Validate with notebooks in `notebooks/`

## 🔒 File Conventions

### Python Files
- **Snake case**: `train_optimized.py`
- **Clear naming**: Purpose evident from filename
- **Module docstrings**: Clear description at top

### Configuration
- **INI format**: `config.ini`
- **Environment variables**: `.env`
- **YAML for complex configs**: `sweep_config.yaml`

### Documentation
- **Kebab case**: `project-structure.md`
- **Descriptive names**: Purpose clear from filename
- **Markdown format**: Consistent structure

This structure supports both simple usage (via legacy entry point) and advanced development (via modular architecture), making the codebase accessible to users with different needs and expertise levels. 