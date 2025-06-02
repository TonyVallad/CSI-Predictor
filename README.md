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
│   ├── data_split.py       # Pure PyTorch data splitting
│   ├── metrics.py          # Pure PyTorch metrics
│   ├── models/             # Neural network models
│   │   ├── __init__.py
│   │   ├── backbones.py    # Feature extraction backbones
│   │   └── head.py         # Classification heads
│   ├── train.py            # Training logic
│   ├── evaluate.py         # Evaluation logic
│   └── utils.py            # Utility functions
├── .env                    # Environment variables (create from template)
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
   Create a `.env` file with your paths and settings:
   ```bash
   # Device configuration
   DEVICE=cuda
   
   # Data loading configuration
   LOAD_DATA_TO_MEMORY=True
   
   # Data source and paths
   DATA_SOURCE=local
   DATA_DIR=/home/pyuser/data/Paradise_Images
   MODELS_DIR=./models
   CSV_DIR=/home/pyuser/data/Paradise_CSV
   INI_DIR=./config/
   
   # Labels configuration
   LABELS_CSV=Labeled_Data_RAW.csv
   LABELS_CSV_SEPARATOR=;
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
N_EPOCHS = 3
PATIENCE = 2
LEARNING_RATE = 0.001
OPTIMIZER = adam

[MODEL]
MODEL_ARCH = resnet50
```

**Environment Variables** (in `.env` file):
```bash
# Device configuration
DEVICE=cuda
LOAD_DATA_TO_MEMORY=True

# Data source and paths
DATA_SOURCE=local
DATA_DIR=/home/pyuser/data/Paradise_Images
MODELS_DIR=./models
CSV_DIR=/home/pyuser/data/Paradise_CSV
INI_DIR=./config/

# Labels configuration
LABELS_CSV=Labeled_Data_RAW.csv
LABELS_CSV_SEPARATOR=;
```

### Model Path Configuration

The project uses `MODELS_DIR` to specify where models are stored. To get the full path of a specific model:

```python
from src.config import cfg

# Get path for a specific model
model_path = cfg.get_model_path("best_model")  # Returns: ./models/best_model.pth
model_path = cfg.get_model_path("resnet50_trained", "pt")  # Returns: ./models/resnet50_trained.pt

# During training, models are automatically saved to the models directory
# Example: best model saved as ./models/ResNet50 - 2024-01-01 12:30:45.pth
```

### Configuration Usage

```python
from src.config import cfg

# Access configuration values
print(f"Batch size: {cfg.batch_size}")
print(f"Models directory: {cfg.models_dir}")
print(f"CSV path: {cfg.csv_path}")  # Computed from CSV_DIR + LABELS_CSV
print(f"Best model path: {cfg.get_model_path('best_model')}")
```

## Usage

### Training
Train a new model with comprehensive logging:
```bash
# Basic training
python -m src.train

# With custom configuration
python -m src.train --ini custom_config.ini

# The training will automatically:
# - Set up Loguru logging with rotating files in ./logs/
# - Initialize Weights & Biases tracking 
# - Save best model with timestamped name
# - Log detailed metrics and model artifacts
```

### Evaluation
Comprehensive model evaluation with confusion matrices and WandB logging:
```bash
# Evaluate latest model
python -m src.evaluate

# With custom configuration
python -m src.evaluate --ini custom_config.ini

# The evaluation will:
# - Load the most recent model automatically
# - Compute per-zone confusion matrices and classification reports
# - Log detailed results to WandB with heatmaps and tables
# - Generate comprehensive evaluation reports
# - Save predictions and metrics to files
```

### Hyperparameter Optimization

Automated hyperparameter optimization using [Optuna](https://optuna.readthedocs.io/) with Bayesian optimization:

```bash
# Quick optimization (50 trials)
python main.py --mode hyperopt --n-trials 50

# Advanced optimization with custom settings
python main.py --mode hyperopt \
    --study-name "csi_advanced_optimization" \
    --n-trials 100 \
    --max-epochs 30 \
    --sampler tpe \
    --pruner median \
    --wandb-project "csi-hyperopt"

# Train final model with best hyperparameters
python main.py --mode train-optimized \
    --hyperparams models/hyperopt/csi_optimization_best_params.json
```

**Optimized Hyperparameters:**
- **Model Architecture**: ResNet18/34/50/101, DenseNet121/169
- **Optimizer**: Adam, AdamW, SGD with weight decay and momentum
- **Learning Rate**: 1e-5 to 1e-1 (log scale)
- **Batch Size**: 16, 32, 64, 128
- **Loss Function**: WeightedCSILoss unknown_weight (0.1 to 1.0)
- **Early Stopping**: Patience (5 to 20 epochs)

**Optimization Features:**
- **Bayesian Optimization**: TPE sampler for intelligent parameter search
- **Early Pruning**: Automatically stops poor trials to save computation
- **WandB Integration**: Real-time monitoring and parameter importance analysis
- **Resumable Studies**: Continue optimization across sessions
- **Visualization**: Interactive plots for optimization history and parameter relationships

**Results:**
After optimization, you get:
- `models/hyperopt/study_name_best_params.json`: Best hyperparameters
- Interactive HTML plots: optimization history, parameter importance, parallel coordinates
- WandB dashboard with detailed trial analysis

See `docs/hyperparameter_optimization.md` for detailed documentation.

### Advanced Usage Examples

**Quick Model Testing:**
```bash
# Train for just a few epochs for testing
# Edit config.ini: N_EPOCHS = 3, PATIENCE = 2
python -m src.train
```

**Resuming Training:**
The system automatically saves the best model during training. While automatic resuming isn't implemented yet, you can manually restart training with the same configuration.

**Custom Logging Setup:**
```python
from src.utils import setup_logging

# Setup custom logging directory and level  
setup_logging(log_dir="./custom_logs", log_level="DEBUG")
```

**Configuration Utilities:**
```python
from src.config import cfg
from src.utils import print_config, make_run_name

# Pretty print current configuration
print_config(cfg)

# Generate run name for experiments
run_name = make_run_name(cfg)  # e.g., "resnet50_20241201_143025"
```

### Legacy Entry Point
Use the main.py entry point for backward compatibility:
```bash
python main.py --mode train
python main.py --mode eval
python main.py --mode both
```

## Monitoring and Logging

### Weights & Biases Integration

The project includes comprehensive Weights & Biases integration for experiment tracking:

**Training Tracking:**
- Real-time loss and F1 metrics (overall and per-zone)
- Learning rate schedules
- Model artifacts and checkpoints
- Configuration logging
- Run naming with timestamps

**Evaluation Tracking:**
- Confusion matrices as interactive heatmaps for each CSI zone
- Classification reports as structured tables
- Per-zone accuracy and F1 scores
- Comparative analysis between validation and test sets

**WandB Project Structure:**
- Training runs: `csi-predictor` project
- Evaluation runs: `csi-predictor-eval` project
- Run names: `train_ResNet50_20241201_143025` or `eval_ResNet50_20241201_143026`

**Access your dashboards:**
```bash
# After running training/evaluation, WandB will provide URLs like:
# https://wandb.ai/your-username/csi-predictor
# https://wandb.ai/your-username/csi-predictor-eval
```

### Loguru Logging System

Enhanced logging with automatic file rotation and structured output:

**Log Files:**
- Location: `./logs/csi_predictor_YYYY-MM-DD.log`
- Rotation: Daily rotation with 30-day retention
- Compression: Automatic ZIP compression of old logs
- Format: Timestamped with module, function, and line information

**Log Levels:**
- Console: Colored output with INFO level by default
- Files: All levels captured (DEBUG, INFO, WARNING, ERROR)

**Example log output:**
```
2024-12-01 14:30:25 | INFO     | src.train:train_model:245 - Starting epoch 1/100
2024-12-01 14:30:25 | INFO     | src.utils:log_config:156 - Configuration loaded: ...
2024-12-01 14:30:26 | WARNING  | src.evaluate:log_to_wandb:123 - Could not log to WandB: ...
```

### Model Artifacts and Checkpointing

**Automatic Model Saving:**
- Best models saved with timestamps: `ResNet50 - 2024-01-01 12:30:45.pth`
- Location: Configurable via `MODELS_DIR` (default: `./models/`)
- Includes: Model weights, optimizer state, configuration, metrics

**WandB Model Artifacts:**
- Automatic upload of best model checkpoints
- Versioned model storage with run associations
- Easy model downloading and deployment

### Evaluation Reports

**Comprehensive Reports Generated:**
- `validation_comprehensive_report.txt`: Detailed validation analysis
- `test_comprehensive_report.txt`: Final test set evaluation
- `validation_predictions.csv`: Per-sample predictions and ground truth
- `test_predictions.csv`: Test set predictions for further analysis

**Report Contents:**
- Overall accuracy and F1 scores
- Per-zone metrics (6 CSI zones)
- Confusion matrices for each zone
- Detailed classification reports (precision, recall, F1 per class)
- Class distribution analysis
- Summary statistics

## Model Architecture

The project supports multiple backbone architectures for feature extraction:
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet_b0 through efficientnet_b4
- **DenseNet**: densenet121, densenet169, densenet201 (including CheXNet)
- **Vision Transformers**: vit_base_patch16_224, vit_large_patch16_224
- **Custom architectures**: Custom_1 (simple CNN baseline), Rad_DINO

The model predicts CSI scores for 6 zones of chest X-rays using multi-output classification (5 classes per zone: 0-3 for severity, 4 for ungradable).

## Data Format

Expected data structure:
```
/home/pyuser/data/
├── Paradise_Images/          # Images directory (DATA_DIR)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Paradise_CSV/             # CSV directory (CSV_DIR)
    └── Labeled_Data_RAW.csv  # Labels file (LABELS_CSV)
```

The labels CSV should contain:
- **FileID**: Image filename
- **right_sup, left_sup**: Right and left superior zone CSI scores
- **right_mid, left_mid**: Right and left middle zone CSI scores  
- **right_inf, left_inf**: Right and left inferior zone CSI scores
- CSV separator should match `LABELS_CSV_SEPARATOR` (default: `;`)

Values can be:
- 0-3: CSI severity scores
- NaN/empty: Automatically converted to class 4 (ungradable)

## Key Features

- **Pure PyTorch implementation**: No scikit-learn dependency
- **Comprehensive metrics**: F1 scores, accuracy with zone-specific reporting
- **Data augmentation**: Configurable transforms for training
- **Early stopping**: Prevent overfitting with patience-based stopping
- **Experiment tracking**: Weights & Biases integration
- **Model checkpointing**: Automatic saving with timestamps
- **Stratified splitting**: Smart data splitting based on unknown value patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]