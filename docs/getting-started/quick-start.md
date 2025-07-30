<div align="center">

# Quick Start Guide

Welcome to CSI-Predictor! This guide will help you get up and running quickly.

</div>

This guide will help you get started with CSI-Predictor quickly. The project has been reorganized into a clean, modular architecture for better maintainability and scalability.

## 🚀 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CSI-Predictor
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Project

```bash
# Copy example configuration
cp config/config_example.ini config/config.ini

# Edit configuration file with your settings
# See config/config.ini for all available options
```

## ⚙️ Configuration

The project uses a hierarchical configuration system:

1. **Environment variables** (`.env` file)
2. **Configuration file** (`config/config.ini`)
3. **Code defaults** (`src/config/config.py`)

### Key Configuration Options

Edit `config/config.ini` to set your preferences:

```ini
[Data]
data_dir = /path/to/your/data
csv_dir = /path/to/your/csv
labels_csv = labels.csv
image_format = nifti
image_extension = .nii.gz

[Model]
model_arch = ResNet50
pretrained = true
num_classes = 5

[Training]
batch_size = 32
learning_rate = 0.001
n_epochs = 100
device = cuda

[Paths]
models_dir = models/
logs_dir = logs/
```

## 🏃‍♂️ Quick Start Examples

### Basic Training

```bash
# Train a model with default settings
python main.py --mode train

# Train with specific configuration
python main.py --mode train --config config/config.ini
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --mode eval

# Evaluate with specific model
python main.py --mode eval --model-path models/best_model.pth
```

### Hyperparameter Optimization

```bash
# Run Optuna hyperparameter optimization
python main.py --mode hyperopt --n-trials 50

# Run with W&B logging
python main.py --mode hyperopt --n-trials 50 --wandb-project my-project
```

### W&B Sweeps

```bash
# Initialize a sweep
python main.py --mode sweep --sweep-name my_sweep

# Run a sweep agent
python main.py --mode sweep-agent --sweep-id <sweep_id>
```

## 🔧 Using the New CLI Structure

The project now supports a modern CLI structure:

```bash
# Use the new CLI modules directly
python -m src.cli.main --mode train
python -m src.cli.main --mode evaluate
python -m src.cli.main --mode optimize

# Or use individual CLI modules
python -m src.cli.train
python -m src.cli.evaluate
python -m src.cli.optimize
```

## 📊 Data Preparation

### 1. Organize Your Data

```
data/
├── images/
│   ├── patient_001.nii.gz
│   ├── patient_002.nii.gz
│   └── ...
└── labels.csv
```

### 2. Prepare Labels CSV

Your `labels.csv` should contain:

```csv
filename,R_Sup,L_Sup,R_Mid,L_Mid,R_Inf,L_Inf
patient_001.nii.gz,0,1,2,0,1,3
patient_002.nii.gz,1,0,0,2,1,0
...
```

### 3. Update Configuration

Update `config/config.ini` with your data paths:

```ini
[Data]
data_dir = /path/to/your/data
csv_dir = /path/to/your/data
labels_csv = labels.csv
```

## 🧪 Testing and Debugging

### Debug Image Loading

```bash
# Debug image loading and visualization
python scripts/debug/debug_images.py
```

### Test RadDINO Availability

```bash
# Check RadDINO dependencies
python scripts/debug/diagnose_raddino.py
```

### Test Metrics

```bash
# Test metrics functionality
python scripts/tests/test_metrics.py
```

## 📈 Monitoring Training

### Using Weights & Biases

1. **Install W&B:**
   ```bash
   pip install wandb
   ```

2. **Login to W&B:**
   ```bash
   wandb login
   ```

3. **Run training with W&B:**
   ```bash
   python main.py --mode train --wandb-project my-project
   ```

### Using TensorBoard

Training logs are automatically saved to `logs/` directory. View them with:

```bash
tensorboard --logdir logs/
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size in `config/config.ini`
   - Use gradient accumulation
   - Enable mixed precision training

2. **Import Errors:**
   - Ensure virtual environment is activated
   - Check that all dependencies are installed
   - Verify Python version compatibility

3. **Data Loading Issues:**
   - Check file paths in configuration
   - Verify CSV format and column names
   - Ensure image files exist and are readable

### Getting Help

- Check the [API Reference](api-reference.md) for detailed function documentation
- Review the [Project Structure](project-structure.md) for architecture overview
- See [Training Guide](training.md) for advanced training options
- Consult [Evaluation Guide](evaluation.md) for evaluation procedures

## 🎯 Next Steps

1. **Read the Documentation:**
   - [Project Structure](project-structure.md) - Understanding the architecture
   - [Training Guide](training.md) - Advanced training options
   - [Evaluation Guide](evaluation.md) - Model evaluation procedures
   - [API Reference](api-reference.md) - Function documentation

2. **Explore Examples:**
   - Check `notebooks/` for Jupyter notebook examples
   - Review `scripts/` for utility scripts
   - Examine `config/` for configuration examples

3. **Customize for Your Use Case:**
   - Modify configuration for your data
   - Adjust model architecture as needed
   - Customize training parameters

## 🔄 Migration Notes

If you're migrating from the old monolithic structure:

### Backward Compatibility

All existing code continues to work:

```python
# These imports still work
from src.train import train_model
from src.evaluate import evaluate_model
from src.data import create_data_loaders
from src.models import build_model
from src.config import cfg
```

### New Modular Imports (Recommended)

```python
# Recommended new imports
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.data.dataloader import create_data_loaders
from src.models.factory import build_model
from src.config import cfg
```

### Configuration Changes

- Configuration files moved to `config/` directory
- Default config path is now `config/config.ini`
- All CLI tools updated to use new paths

The migration maintains full backward compatibility while providing a much cleaner, more maintainable structure. 