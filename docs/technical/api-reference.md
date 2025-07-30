<div align="center">

# API Reference

Complete API reference for CSI-Predictor modules and functions.

</div>

This document provides a comprehensive API reference for the CSI-Predictor project. The API has been reorganized into a modular structure for better maintainability and clarity.

## 📋 Table of Contents

- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Optimization](#optimization)
- [Utils](#utils)
- [CLI](#cli)

## ⚙️ Configuration

### Main Configuration

```python
from src.config import cfg, get_config, copy_config_on_training_start

# Singleton configuration instance
print(cfg.model_arch)
print(cfg.batch_size)

# Get fresh configuration
config = get_config()

# Copy configuration for reproducibility
copy_config_on_training_start()
```

### Configuration Classes

#### `Config` Dataclass

```python
from src.config.config import Config

# Configuration dataclass with all settings
config = Config(
    model_arch="ResNet50",
    batch_size=32,
    learning_rate=0.001,
    n_epochs=100,
    device="cuda"
)
```

#### `ConfigLoader` Class

```python
from src.config.config_loader import ConfigLoader

# Load configuration from files
loader = ConfigLoader()
config = loader.load_config("config/config.ini")
```

### Configuration Validation

```python
from src.config.validation import validate_config, validate_paths

# Validate configuration
validate_config(config)

# Validate file paths
validate_paths(config)
```

## 📊 Data Pipeline

### Data Loading

```python
from src.data.dataloader import create_data_loaders, load_and_split_data

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config)

# Load and split data
train_data, val_data, test_data = load_and_split_data(config)
```

### Dataset

```python
from src.data.dataset import CSIDataset

# Create dataset
dataset = CSIDataset(
    csv_data=csv_data,
    data_dir=config.data_dir,
    transform=transforms,
    load_to_memory=config.load_data_to_memory
)

# Get item
image, labels = dataset[0]
```

### Data Preprocessing

```python
from src.data.preprocessing import (
    get_normalization_parameters,
    load_csv_data,
    filter_existing_files,
    convert_nans_to_unknown
)

# Get normalization parameters
mean, std = get_normalization_parameters(config.normalization_strategy)

# Load CSV data
csv_data = load_csv_data(config.csv_dir, config.labels_csv)

# Filter existing files
filtered_data = filter_existing_files(csv_data, config.data_dir)

# Convert NaN values
cleaned_data = convert_nans_to_unknown(filtered_data)
```

### Data Splitting

```python
from src.data.splitting import create_stratification_key, split_data_stratified

# Create stratification key
strat_key = create_stratification_key(csv_data)

# Split data
train_idx, val_idx, test_idx = split_data_stratified(
    csv_data, strat_key, 
    train_size=0.7, val_size=0.15, test_size=0.15
)
```

### Transforms

```python
from src.data.transforms import get_default_transforms, get_raddino_processor

# Get default transforms
transforms = get_default_transforms(config)

# Get RadDINO processor
processor = get_raddino_processor()
```

## 🤖 Models

### Model Factory

```python
from src.models.factory import (
    build_model,
    build_zone_focus_model,
    build_zone_masking_model,
    get_model_info
)

# Build complete model
model = build_model(config)

# Build zone focus model
model = build_zone_focus_model(config, focus_zones=[0, 1, 2])

# Build zone masking model
model = build_zone_masking_model(config, mask_zones=[4, 5])

# Get model information
info = get_model_info(model)
```

### Backbones

```python
from src.models.backbones import get_backbone, get_backbone_feature_dim

# Get backbone
backbone = get_backbone("resnet50", pretrained=True)

# Get feature dimension
feature_dim = get_backbone_feature_dim("resnet50")
```

#### Custom CNN Backbone

```python
from src.models.backbones.custom import CustomCNNBackbone

backbone = CustomCNNBackbone(
    input_channels=3,
    feature_dim=512
)
```

#### ResNet Backbone

```python
from src.models.backbones.resnet import ResNet50Backbone

backbone = ResNet50Backbone(pretrained=True)
```

#### DenseNet Backbone

```python
from src.models.backbones.densenet import CheXNetBackbone

backbone = CheXNetBackbone(pretrained=True)
```

#### RadDINO Backbone

```python
from src.models.backbones.raddino import RadDINOBackbone

backbone = RadDINOBackbone(pretrained=True)
```

### Heads

#### CSI Classification Head

```python
from src.models.heads.csi_head import CSIHead

head = CSIHead(
    input_dim=512,
    num_classes=5,
    dropout_rate=0.5
)
```

#### Regression Head

```python
from src.models.heads.regression_head import CSIRegressionHead

head = CSIRegressionHead(
    input_dim=512,
    output_dim=6
)
```

### Complete Models

```python
from src.models.complete.csi_models import CSIModel, CSIModelWithZoneMasking

# Standard CSI model
model = CSIModel(backbone, head)

# CSI model with zone masking
model = CSIModelWithZoneMasking(backbone, head, mask_zones=[4, 5])
```

## 🏋️ Training

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

# Create weighted loss function
criterion = WeightedCSILoss(unknown_weight=0.3)

# Compute loss
loss = criterion(predictions, targets)
```

### Training Metrics

```python
from src.training.metrics import (
    compute_f1_metrics,
    compute_precision_recall,
    compute_csi_average_metrics,
    compute_ahf_classification_metrics
)

# Compute F1 metrics
f1_metrics = compute_f1_metrics(predictions, targets)

# Compute precision and recall
pr_metrics = compute_precision_recall(predictions, targets)

# Compute CSI average metrics
csi_metrics = compute_csi_average_metrics(predictions, targets, file_ids, csv_data)

# Compute AHF metrics
ahf_metrics = compute_ahf_classification_metrics(predictions, targets, file_ids, csv_data)
```

### Optimizers and Schedulers

```python
from src.training.optimizer import create_optimizer, create_scheduler, get_learning_rate

# Create optimizer
optimizer = create_optimizer(model, config)

# Create scheduler
scheduler = create_scheduler(optimizer, config)

# Get current learning rate
lr = get_learning_rate(optimizer)
```

### Training Callbacks

```python
from src.training.callbacks import EarlyStopping, MetricsTracker, AverageMeter

# Early stopping
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

# Metrics tracking
metrics_tracker = MetricsTracker()

# Average meter
loss_meter = AverageMeter()
```

## 📈 Evaluation

### Main Evaluation Functions

```python
from src.evaluation.evaluator import (
    evaluate_model,
    load_trained_model,
    evaluate_model_on_loader,
    compute_zone_metrics,
    compute_overall_metrics,
    save_predictions
)

# Complete evaluation pipeline
evaluation_results = evaluate_model(config)

# Load trained model
model = load_trained_model(config)

# Evaluate on data loader
predictions, targets, file_ids = evaluate_model_on_loader(model, test_loader, config)

# Compute zone metrics
zone_metrics = compute_zone_metrics(predictions, targets, file_ids, csv_data)

# Compute overall metrics
overall_metrics = compute_overall_metrics(predictions, targets)

# Save predictions
save_predictions(predictions, targets, file_ids, output_path)
```

### Evaluation Metrics

```python
from src.evaluation.metrics.classification import compute_accuracy
from src.evaluation.metrics.f1_score import (
    compute_pytorch_f1_metrics,
    compute_per_class_f1_scores,
    compute_enhanced_f1_metrics
)
from src.evaluation.metrics.confusion_matrix import compute_confusion_matrix

# Compute accuracy
accuracy = compute_accuracy(predictions, targets)

# Compute F1 metrics
f1_metrics = compute_pytorch_f1_metrics(predictions, targets)

# Compute per-class F1 scores
per_class_f1 = compute_per_class_f1_scores(predictions, targets)

# Compute enhanced F1 metrics
enhanced_f1 = compute_enhanced_f1_metrics(predictions, targets)

# Compute confusion matrix
conf_matrix = compute_confusion_matrix(predictions, targets)
```

### Evaluation-Specific Metrics

```python
from src.evaluation.metrics.evaluation_metrics import (
    compute_confusion_matrices_per_zone,
    create_classification_report_per_zone
)

# Compute confusion matrices per zone
conf_matrices = compute_confusion_matrices_per_zone(predictions, targets, file_ids, csv_data)

# Create classification reports per zone
class_reports = create_classification_report_per_zone(predictions, targets, file_ids, csv_data)
```

### Visualization

```python
from src.evaluation.visualization.plots import (
    create_roc_curves,
    create_precision_recall_curves,
    plot_training_curves
)
from src.evaluation.visualization.confusion_matrix import (
    create_confusion_matrix_grid,
    create_overall_confusion_matrix,
    save_confusion_matrix_graphs
)

# Create ROC curves
roc_curves = create_roc_curves(predictions, targets, class_names)

# Create precision-recall curves
pr_curves = create_precision_recall_curves(predictions, targets, class_names)

# Plot training curves
plot_training_curves(train_history, val_history, metrics=['loss', 'f1_score'])

# Create confusion matrix grid
conf_matrix_grid = create_confusion_matrix_grid(predictions, targets, class_names)

# Create overall confusion matrix
overall_conf_matrix = create_overall_confusion_matrix(predictions, targets)

# Save confusion matrix graphs
save_confusion_matrix_graphs(conf_matrices, output_dir="evaluation_results/")
```

### W&B Logging

```python
from src.evaluation.wandb_logging import log_to_wandb

# Log evaluation results
log_to_wandb(
    zone_metrics=zone_metrics,
    overall_metrics=overall_metrics,
    confusion_matrices=confusion_matrices,
    predictions=predictions,
    targets=targets
)
```

## 🔧 Optimization

### Hyperparameter Optimization

```python
from src.optimization.hyperopt import (
    optimize_hyperparameters,
    create_study,
    objective,
    save_best_hyperparameters,
    clear_data_cache,
    get_search_space_info,
    OptunaPruningCallback
)

# Run hyperparameter optimization
study = optimize_hyperparameters(
    study_name="csi_optimization",
    n_trials=100,
    max_epochs=30,
    config_path="config/config.ini"
)

# Create Optuna study
study = create_study(study_name="csi_optimization", direction="maximize")

# Objective function
best_value = objective(trial)

# Save best hyperparameters
save_best_hyperparameters(study, output_path="models/hyperopt/best_params.json")

# Clear data cache
clear_data_cache()

# Get search space info
search_info = get_search_space_info()
```

### W&B Sweeps

```python
from src.optimization.wandb_sweep import (
    initialize_sweep,
    run_sweep_agent,
    create_and_run_sweep,
    get_sweep_config,
    train_sweep_run,
    clear_data_cache
)

# Initialize sweep
sweep_id = initialize_sweep(
    project="csi-sweeps",
    sweep_name="my_sweep",
    config_path="config/config.ini"
)

# Run sweep agent
run_sweep_agent(
    sweep_id=sweep_id,
    project="csi-sweeps",
    config_path="config/config.ini"
)

# Create and run sweep
sweep_id = create_and_run_sweep(
    project="csi-sweeps",
    sweep_name="my_sweep",
    config_path="config/config.ini"
)

# Get sweep configuration
sweep_config = get_sweep_config()

# Train sweep run
train_sweep_run(config, model, train_loader, val_loader)
```

## 🛠️ Utils

### Logging

```python
from src.utils.logging import setup_logging, logger

# Setup logging
setup_logging(log_level="INFO", log_file="logs/app.log")

# Use logger
logger.info("Training started")
logger.warning("Low GPU memory")
logger.error("Training failed")
```

### Checkpointing

```python
from src.utils.checkpoint import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    path="models/checkpoint.pth"
)

# Load checkpoint
checkpoint = load_checkpoint(model, "models/checkpoint.pth")
```

### Visualization

```python
from src.utils.visualization import (
    show_batch,
    visualize_data_distribution,
    analyze_missing_data,
    plot_training_curves,
    plot_training_curves_grid,
    create_summary_dashboard
)

# Show batch of images
show_batch(images, labels, title="Training Batch")

# Visualize data distribution
visualize_data_distribution(csv_data, output_path="data_distribution.png")

# Analyze missing data
missing_analysis = analyze_missing_data(csv_data)

# Plot training curves
plot_training_curves(train_history, val_history)

# Plot training curves grid
plot_training_curves_grid(train_history, val_history)

# Create summary dashboard
create_summary_dashboard(evaluation_results, output_path="dashboard.html")
```

### File Utils

```python
from src.utils.file_utils import (
    create_dirs,
    save_training_history,
    load_training_history
)

# Create directories
create_dirs(["models/", "logs/", "evaluation_results/"])

# Save training history
save_training_history(train_history, "logs/training_history.json")

# Load training history
history = load_training_history("logs/training_history.json")
```

### Seed Management

```python
from src.utils.seed import set_seed, seed_everything

# Set random seed
set_seed(42)

# Seed everything (PyTorch, NumPy, Python random)
seed_everything(42)
```

## 💻 CLI

### Main CLI

```python
from src.cli.main import main

# Run main CLI
main()
```

### Training CLI

```python
from src.cli.train import train_cli, create_train_parser

# Create training parser
parser = create_train_parser()
args = parser.parse_args()

# Run training CLI
train_cli(args)
```

### Evaluation CLI

```python
from src.cli.evaluate import evaluate_cli, create_evaluate_parser

# Create evaluation parser
parser = create_evaluate_parser()
args = parser.parse_args()

# Run evaluation CLI
evaluate_cli(args)
```

### Optimization CLI

```python
from src.cli.optimize import optimize_cli, create_optimize_parser

# Create optimization parser
parser = create_optimize_parser()
args = parser.parse_args()

# Run optimization CLI
optimize_cli(args)
```

## 🔄 Backward Compatibility

### Legacy Imports (Still Work)

```python
# These imports still work for backward compatibility
from src.train import train_model, WeightedCSILoss
from src.evaluate import evaluate_model, load_trained_model
from src.data import create_data_loaders, CSIDataset
from src.models import build_model, get_backbone
from src.utils import logger, save_checkpoint, load_checkpoint
from src.config import cfg, get_config
from src.metrics import compute_confusion_matrix, compute_f1_metrics
from src.hyperopt import optimize_hyperparameters
from src.wandb_sweep import initialize_sweep, run_sweep_agent
```

### New Modular Imports (Recommended)

```python
# Recommended new modular imports
from src.training.trainer import train_model
from src.training.loss import WeightedCSILoss
from src.evaluation.evaluator import evaluate_model, load_trained_model
from src.data.dataloader import create_data_loaders
from src.data.dataset import CSIDataset
from src.models.factory import build_model
from src.models.backbones import get_backbone
from src.utils.logging import logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.config import cfg, get_config
from src.evaluation.metrics.confusion_matrix import compute_confusion_matrix
from src.training.metrics import compute_f1_metrics
from src.optimization.hyperopt import optimize_hyperparameters
from src.optimization.wandb_sweep import initialize_sweep, run_sweep_agent
```

## 📚 Examples

### Complete Training Example

```python
from src.config import cfg
from src.data.dataloader import create_data_loaders
from src.models.factory import build_model
from src.training.trainer import train_model

# Load configuration
config = cfg

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config)

# Build model
model = build_model(config)

# Train model
train_model(config)
```

### Complete Evaluation Example

```python
from src.config import cfg
from src.evaluation.evaluator import evaluate_model

# Load configuration
config = cfg

# Evaluate model
evaluation_results = evaluate_model(config)
```

### Custom Training Loop Example

```python
from src.training.trainer import train_epoch, validate_epoch
from src.training.loss import WeightedCSILoss
from src.training.optimizer import create_optimizer, create_scheduler

# Create loss function
criterion = WeightedCSILoss(unknown_weight=0.3)

# Create optimizer and scheduler
optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)

# Custom training loop
for epoch in range(config.n_epochs):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
    
    scheduler.step(val_metrics['val_loss'])
```

This API reference provides comprehensive documentation for all the functions and classes available in the CSI-Predictor project. The modular structure makes it easy to import only what you need while maintaining full backward compatibility. 