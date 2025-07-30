<div align="center">

# Training Guide

This guide covers the training pipeline for CSI-Predictor models. The training system has been reorganized into a modular architecture for better maintainability and extensibility.

</div>

## 🏗️ Training Architecture

The training system is organized into several focused modules:

```
src/training/
├── trainer.py           # Main training logic and pipeline
├── optimizer.py         # Optimizer and scheduler management
├── loss.py              # Loss functions
├── metrics.py           # Training metrics computation
├── callbacks.py         # Training callbacks
└── __init__.py          # Training package
```

## 🚀 Quick Start

### Basic Training

```bash
# Train with default settings
python main.py --mode train

# Train with specific configuration
python main.py --mode train --config config/config.ini

# Train with optimized hyperparameters
python main.py --mode train-optimized --hyperparams models/hyperopt/best_params.json
```

### Using the New CLI

```bash
# Use the new CLI structure
python -m src.cli.main --mode train

# Use individual training CLI
python -m src.cli.train
```

## 📋 Training Configuration

### Configuration Options

Edit `config/config.ini` to customize training:

```ini
[Training]
batch_size = 32
learning_rate = 0.001
n_epochs = 100
device = cuda
num_workers = 4
pin_memory = true

[Model]
model_arch = ResNet50
pretrained = true
num_classes = 5

[Optimization]
optimizer = adam
weight_decay = 0.0001
scheduler = reduce_lr_on_plateau
patience = 10
factor = 0.5

[Loss]
unknown_weight = 0.3
```

### Key Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `batch_size` | Training batch size | 32 | 8-128 |
| `learning_rate` | Initial learning rate | 0.001 | 0.0001-0.01 |
| `n_epochs` | Number of training epochs | 100 | 10-500 |
| `device` | Training device | cuda | cuda, cpu |
| `unknown_weight` | Weight for unknown class | 0.3 | 0.1-1.0 |

## 🧠 Model Architectures

### Available Backbones

1. **ResNet50** - Standard ResNet50 with ImageNet pretraining
2. **CheXNet** - DenseNet121 adapted for chest X-rays
3. **RadDINO** - Medical vision transformer
4. **Custom_1** - Simple 5-layer CNN

### Model Selection

```python
from src.models.factory import build_model
from src.config import cfg

# Build model with configuration
model = build_model(cfg)

# Or specify architecture directly
from src.models.backbones import get_backbone
backbone = get_backbone("resnet50")
```

## 🎯 Loss Functions

### WeightedCSILoss

The primary loss function is `WeightedCSILoss`, which reduces the importance of the "unknown" class:

```python
from src.training.loss import WeightedCSILoss

# Create loss function
criterion = WeightedCSILoss(unknown_weight=0.3)

# Use in training
loss = criterion(predictions, targets)
```

### Loss Configuration

```ini
[Loss]
unknown_weight = 0.3  # Weight for unknown class (0.1-1.0)
```

## 📊 Training Metrics

### Available Metrics

The training pipeline computes several metrics:

1. **F1 Scores**: Macro, weighted, and per-class F1 scores
2. **Precision/Recall**: Per-class precision and recall
3. **CSI Average Metrics**: MAE, RMSE, correlation for CSI scores
4. **AHF Classification**: Acute Heart Failure classification metrics

### Metrics Computation

```python
from src.training.metrics import (
    compute_f1_metrics,
    compute_precision_recall,
    compute_csi_average_metrics,
    compute_ahf_classification_metrics
)

# Compute F1 metrics
f1_metrics = compute_f1_metrics(predictions, targets)

# Compute CSI average metrics
csi_metrics = compute_csi_average_metrics(predictions, targets, file_ids, csv_data)
```

## ⚙️ Optimizers and Schedulers

### Available Optimizers

1. **Adam** - Adaptive learning rate optimizer
2. **SGD** - Stochastic gradient descent
3. **AdamW** - Adam with weight decay

### Learning Rate Schedulers

1. **ReduceLROnPlateau** - Reduce LR when validation loss plateaus
2. **StepLR** - Step-based learning rate decay
3. **CosineAnnealingLR** - Cosine annealing learning rate

### Configuration

```ini
[Optimization]
optimizer = adam
learning_rate = 0.001
weight_decay = 0.0001
scheduler = reduce_lr_on_plateau
patience = 10
factor = 0.5
```

## 🔄 Training Callbacks

### Early Stopping

Automatically stops training when validation loss stops improving:

```python
from src.training.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)
```

### Metrics Tracking

Tracks and logs training metrics:

```python
from src.training.callbacks import MetricsTracker

metrics_tracker = MetricsTracker()
```

### Configuration

```ini
[Callbacks]
early_stopping_patience = 10
early_stopping_min_delta = 0.001
restore_best_weights = true
```

## 📈 Training Monitoring

### Weights & Biases Integration

```bash
# Train with W&B logging
python main.py --mode train --wandb-project my-project

# Or set in configuration
wandb_project = my-project
wandb_entity = my-username
```

### TensorBoard Logging

Training logs are automatically saved to `logs/` directory:

```bash
# View training logs
tensorboard --logdir logs/
```

### Custom Logging

```python
from src.utils.logging import logger

logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, F1 = {f1:.4f}")
```

## 🔧 Advanced Training Options

### Custom Training Loop

```python
from src.training.trainer import train_epoch, validate_epoch
from src.training.optimizer import create_optimizer, create_scheduler

# Create optimizer and scheduler
optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)

# Custom training loop
for epoch in range(config.n_epochs):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
    
    # Update scheduler
    scheduler.step(val_metrics['val_loss'])
```

### Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🎯 Hyperparameter Optimization

### Optuna Integration

```bash
# Run hyperparameter optimization
python main.py --mode hyperopt --n-trials 100

# With W&B logging
python main.py --mode hyperopt --n-trials 100 --wandb-project my-project
```

### W&B Sweeps

```bash
# Initialize sweep
python main.py --mode sweep --sweep-name my_sweep

# Run sweep agent
python main.py --mode sweep-agent --sweep-id <sweep_id>
```

## 📊 Training Results

### Model Checkpoints

Models are automatically saved during training:

- `models/best_model.pth` - Best validation model
- `models/last_model.pth` - Last epoch model
- `models/checkpoint_epoch_X.pth` - Epoch checkpoints

### Training History

Training history is saved to:

- `logs/training_history.json` - Training metrics
- `logs/validation_history.json` - Validation metrics

### Loading Trained Models

```python
from src.utils.checkpoint import load_checkpoint
from src.models.factory import build_model

# Load model
model = build_model(config)
checkpoint = load_checkpoint(model, 'models/best_model.pth')
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Not Converging**
   - Check learning rate
   - Verify data preprocessing
   - Adjust loss weights

3. **Overfitting**
   - Increase regularization
   - Use data augmentation
   - Reduce model complexity

### Debug Tools

```bash
# Debug image loading
python scripts/debug/debug_images.py

# Check model architecture
python -c "from src.models.factory import build_model; print(build_model(config))"
```

## 📚 API Reference

### Main Training Functions

```python
from src.training.trainer import train_model, train_epoch, validate_epoch

# Complete training pipeline
train_model(config)

# Individual training/validation steps
train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
```

### Loss Functions

```python
from src.training.loss import WeightedCSILoss

criterion = WeightedCSILoss(unknown_weight=0.3)
loss = criterion(predictions, targets)
```

### Optimizers and Schedulers

```python
from src.training.optimizer import create_optimizer, create_scheduler

optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)
```

### Callbacks

```python
from src.training.callbacks import EarlyStopping, MetricsTracker, AverageMeter

early_stopping = EarlyStopping(patience=10)
metrics_tracker = MetricsTracker()
```

## 🔄 Migration Notes

### Backward Compatibility

All existing training code continues to work:

```python
# These imports still work
from src.train import train_model, WeightedCSILoss
```

### New Modular Imports (Recommended)

```python
# Recommended new imports
from src.training.trainer import train_model
from src.training.loss import WeightedCSILoss
from src.training.metrics import compute_f1_metrics
from src.training.callbacks import EarlyStopping
```

The training system has been completely modularized while maintaining full backward compatibility. 