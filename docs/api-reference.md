# API Reference

Comprehensive reference for CSI-Predictor's Python API.

## Configuration (`src.config`)

### Main Configuration Class

```python
from src.config import cfg, get_config

# Access global configuration
print(cfg.model_arch)
print(cfg.batch_size)

# Load custom configuration
custom_cfg = get_config(ini_path="custom.ini")
```

**Key Properties:**
- `model_arch`: Model architecture name
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `device`: Training device (cuda/cpu/mps)
- `data_dir`: Path to image directory
- `csv_dir`: Path to CSV directory
- `models_dir`: Path to model storage directory

## Data Loading (`src.data`)

### Core Functions

```python
from src.data import create_data_loaders, CSIDataset

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(cfg)

# Custom dataset
dataset = CSIDataset(
    image_dir="/path/to/images",
    labels_csv="/path/to/labels.csv",
    transform=transforms,
    indices=[0, 1, 2, ...]  # Optional subset
)
```

### Dataset Classes

**CSIDataset**
- Main dataset class for chest X-ray images
- Handles image loading and label processing
- Supports data augmentation transforms

## Model Building (`src.models`)

### Model Factory

```python
from src.models import build_model

# Build model from configuration
model = build_model(cfg)

# Get model information
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
```

### Available Architectures

- ResNet family: `resnet18`, `resnet50`, etc.
- Medical models: `chexnet`, `raddino`
- Efficient models: `efficientnet_b0` to `efficientnet_b4`
- Vision transformers: `vit_base_patch16_224`
- Custom: `custom1`

## Training (`src.train`)

### Main Training Function

```python
from src.train import train_model

# Train with configuration
best_model, history = train_model(cfg)

# Custom training loop
model = build_model(cfg)
train_loader, val_loader, test_loader = create_data_loaders(cfg)
best_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=cfg
)
```

## Evaluation (`src.evaluate`)

### Evaluation Functions

```python
from src.evaluate import evaluate_model, compute_metrics

# Evaluate trained model
results = evaluate_model(model, cfg)

# Custom evaluation
predictions, targets = model_inference(model, test_loader)
metrics = compute_metrics(predictions, targets)
```

### Metrics

```python
from src.metrics import calculate_f1_score, calculate_accuracy

# Per-zone metrics
f1_scores = calculate_f1_score(y_true, y_pred, num_zones=6)
accuracies = calculate_accuracy(y_true, y_pred, num_zones=6)
```

## Utilities (`src.utils`)

### Logging and Setup

```python
from src.utils import setup_logging, logger

# Setup logging
setup_logging(log_dir="./logs", log_level="INFO")

# Use logger
logger.info("Training started")
logger.error("Error occurred")
```

### Model Utilities

```python
from src.utils import (
    save_checkpoint, 
    load_checkpoint,
    make_model_name,
    count_parameters
)

# Save model
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    best_f1=0.85,
    filepath="model.pth"
)

# Load model
model, optimizer, epoch, best_f1 = load_checkpoint("model.pth", cfg)

# Generate model name
name = make_model_name(cfg, task_tag="Tr", extra_info="batch64")
# Returns: "20250611_093054_ResNet50_Tr_batch64"

# Count parameters
num_params = count_parameters(model)
```

### Configuration Utilities

```python
from src.utils import print_config, log_config

# Pretty print configuration
print_config(cfg)

# Log configuration to file
log_config(cfg, logger)
```

### Data Utilities

```python
from src.utils import (
    create_debug_dataset,
    visualize_data_distribution,
    show_batch
)

# Create synthetic test data
create_debug_dataset(
    num_samples=100,
    output_dir="./debug_data"
)

# Visualize data distribution
visualize_data_distribution(labels_df, save_path="distribution.png")

# Show sample batch
show_batch(data_loader, num_samples=8)
```

## Data Splitting (`src.data_split`)

### Stratified Splitting

```python
from src.data_split import create_stratified_splits

# Create stratified splits
train_indices, val_indices, test_indices = create_stratified_splits(
    labels_df=df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)
```

## Example Usage Patterns

### Complete Training Pipeline

```python
from src.config import cfg
from src.data import create_data_loaders
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import setup_logging

# Setup
setup_logging()

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(cfg)

# Build model
model = build_model(cfg)

# Train model
best_model, history = train_model(model, train_loader, val_loader, cfg)

# Evaluate model
results = evaluate_model(best_model, cfg)
print(f"Test F1: {results['overall_f1']:.4f}")
```

### Custom Model Evaluation

```python
from src.config import cfg
from src.models import build_model
from src.data import create_data_loaders
from src.utils import load_checkpoint

# Load specific model
model = build_model(cfg)
model, _, _, _ = load_checkpoint("path/to/model.pth", cfg)

# Create test data loader
_, _, test_loader = create_data_loaders(cfg)

# Evaluate
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        outputs = model(images)
        predictions.append(outputs.cpu())
        targets.append(labels.cpu())

predictions = torch.cat(predictions)
targets = torch.cat(targets)

# Compute metrics
from src.metrics import calculate_f1_score
f1_scores = calculate_f1_score(targets, predictions, num_zones=6)
print(f"Per-zone F1 scores: {f1_scores}")
```

### Configuration Management

```python
from src.config import get_config

# Load different configurations
dev_cfg = get_config(ini_path="configs/development.ini")
prod_cfg = get_config(ini_path="configs/production.ini")

# Override specific values
import os
os.environ['BATCH_SIZE'] = '64'  # Override batch size
os.environ['DEVICE'] = 'cuda'    # Override device

cfg = get_config()  # Loads with overrides
```

For detailed parameter specifications and advanced usage, refer to the source code docstrings and type hints. 