# Quick Start Guide

Get CSI-Predictor up and running in minutes! This guide assumes you have already completed the [installation](installation.md).

## 🚀 30-Second Start

```bash
# 1. Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Train a model (using built-in test data if available)
python -m src.train

# 3. Evaluate the model
python -m src.evaluate
```

## 📁 Data Setup

### Option 1: Use Your Own Data

1. **Prepare your data structure**:
   ```
   /your/data/path/
   ├── Paradise_Images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── Paradise_CSV/
       └── Labeled_Data_RAW.csv
   ```

2. **Update your `.env` file**:
   ```bash
   DATA_DIR=/your/data/path/Paradise_Images
   CSV_DIR=/your/data/path/Paradise_CSV
   LABELS_CSV=Labeled_Data_RAW.csv
   ```

### Option 2: Generate Test Data

```bash
# Generate synthetic test data for experimentation
python -c "
from src.utils import create_debug_dataset
create_debug_dataset(num_samples=100, output_dir='./debug_data')
"

# Update .env to use test data
echo "DATA_DIR=./debug_data/images" >> .env
echo "CSV_DIR=./debug_data" >> .env
echo "LABELS_CSV=debug_labels.csv" >> .env
echo "LABELS_CSV_SEPARATOR=," >> .env
```

## 🏃‍♂️ First Training Run

### Basic Training

```bash
# Train with default settings (ResNet50)
python -m src.train

# Or use the legacy entry point
python main.py --mode train
```

### Quick Test Training

For rapid testing, create a `quick_config.ini`:
```ini
[TRAINING]
BATCH_SIZE = 16
N_EPOCHS = 3
PATIENCE = 2
LEARNING_RATE = 0.001
OPTIMIZER = adam

[MODEL]
MODEL_ARCH = resnet50
```

Then run:
```bash
python -m src.train --ini quick_config.ini
```

## 📊 First Evaluation

```bash
# Evaluate the latest trained model
python -m src.evaluate

# Or specify a custom config
python -m src.evaluate --ini quick_config.ini
```

## 🎯 Try Different Models

### ResNet50 (Fast, Good Baseline)
```bash
echo "MODEL_ARCH = resnet50" > model_config.ini
python -m src.train --ini model_config.ini
```

### CheXNet (Medical Domain Optimized)
```bash
echo "MODEL_ARCH = chexnet" > model_config.ini
python -m src.train --ini model_config.ini
```

### RadDINO (State-of-the-Art for Chest X-rays)
```bash
# First install transformers if not already done
pip install transformers>=4.30.0

echo "MODEL_ARCH = raddino" > model_config.ini
python -m src.train --ini model_config.ini
```

## 🔧 Quick Hyperparameter Optimization

### W&B Sweeps (Recommended)
```bash
# Initialize a sweep
python main.py --mode sweep --sweep-name "quick_test" --sweep-project "csi-quick"

# Run the sweep (will print sweep ID)
python main.py --mode sweep-agent --sweep-id YOUR_SWEEP_ID --count 5
```

### Optuna (Alternative)
```bash
# Quick 10-trial optimization
python main.py --mode hyperopt --n-trials 10 --max-epochs 5

# Train with best parameters
python main.py --mode train-optimized --hyperparams models/hyperopt/csi_optimization_best_params.json
```

## 📈 Monitor Your Training

### Weights & Biases Dashboard

1. **Sign up** at [wandb.ai](https://wandb.ai) (free)
2. **Login** in terminal: `wandb login`
3. **Run training** - dashboard URL will be printed
4. **View real-time metrics**, confusion matrices, and model artifacts

### Local Logs

```bash
# View training logs
tail -f logs/csi_predictor_$(date +%Y-%m-%d).log

# View all log files
ls -la logs/
```

## 🎛️ Configuration Options

### Quick Configuration Changes

```bash
# Change batch size
echo "BATCH_SIZE = 64" >> config.ini

# Use GPU
echo "DEVICE=cuda" >> .env

# Change model
echo "MODEL_ARCH = raddino" >> config.ini

# Reduce epochs for testing
echo "N_EPOCHS = 5" >> config.ini
```

### View Current Configuration

```bash
python -c "from src.config import cfg; from src.utils import print_config; print_config(cfg)"
```

## 🔍 Verify Everything Works

### Check Installation
```bash
python -c "
import torch
from src.config import cfg
from src.models import build_model

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Current device: {cfg.device}')
print(f'Model architecture: {cfg.model_arch}')

# Test model creation
model = build_model(cfg)
print(f'Model created successfully: {type(model).__name__}')
"
```

### Test Data Loading
```bash
python -c "
from src.config import cfg
from src.data import create_data_loaders

try:
    train_loader, val_loader, test_loader = create_data_loaders(cfg)
    print(f'Data loaders created successfully')
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
except Exception as e:
    print(f'Data loading error: {e}')
    print('Make sure your data paths in .env are correct')
"
```

## 🎯 Common First-Time Issues

### Issue: "No module named 'src'"
**Solution**: Make sure you're in the project root directory
```bash
cd CSI-Predictor
python -m src.train
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
echo "BATCH_SIZE = 8" >> config.ini
```

### Issue: "Data directory not found"
**Solution**: Check your `.env` file paths
```bash
cat .env  # Verify DATA_DIR and CSV_DIR paths
ls -la $DATA_DIR  # Check if directory exists
```

### Issue: "RadDINO not available"
**Solution**: Install transformers
```bash
pip install transformers>=4.30.0
```

## 🎊 Next Steps

Once you have basic training working:

1. **[Configure](configuration.md)** the project for your specific needs
2. **[Understand the data format](data-format.md)** requirements
3. **[Explore model architectures](model-architectures.md)** available
4. **[Set up monitoring](monitoring-logging.md)** for production use
5. **[Run hyperparameter optimization](hyperparameter-optimization.md)** for best results

## 💡 Pro Tips

### Faster Iteration
```bash
# Use smaller datasets during development
echo "LOAD_DATA_TO_MEMORY=False" >> .env

# Reduce epochs for quick testing
echo "N_EPOCHS = 3" >> config.ini
echo "PATIENCE = 2" >> config.ini
```

### Save Disk Space
```bash
# Disable detailed logging
echo "LOG_LEVEL=WARNING" >> .env

# Clean up old models
find models/ -name "*.pth" -mtime +7 -delete  # Remove models older than 7 days
```

### Multiple Experiments
```bash
# Create experiment-specific configs
cp config.ini experiments/resnet_experiment.ini
cp config.ini experiments/raddino_experiment.ini

# Run with specific configs
python -m src.train --ini experiments/resnet_experiment.ini
```

## 🆘 Need Help?

- **Documentation**: Check other docs in the `docs/` folder
- **Issues**: Something broken? [Open an issue](../../issues)
- **Questions**: General questions? [Start a discussion](../../discussions)
- **Examples**: Look at `examples/` folder (if available)

Happy training! 🚀 