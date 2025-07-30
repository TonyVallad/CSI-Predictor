<div align="center">

# Evaluation Guide

This guide covers the evaluation pipeline for CSI-Predictor models. The evaluation system has been reorganized into a modular architecture for better maintainability and extensibility.

</div>

## 🏗️ Evaluation Architecture

The evaluation system is organized into several focused modules:

```
src/evaluation/
├── evaluator.py         # Main evaluation logic and pipeline
├── metrics/             # Evaluation metrics
│   ├── classification.py # Classification metrics
│   ├── confusion_matrix.py # Confusion matrix utilities
│   └── f1_score.py      # F1 score calculations
├── visualization/       # Evaluation visualization
│   ├── plots.py         # Plotting utilities
│   └── confusion_matrix.py # Confusion matrix plots
├── wandb_logging.py     # W&B logging
└── __init__.py          # Evaluation package
```

## 🚀 Quick Start

### Basic Evaluation

```bash
# Evaluate with default settings
python main.py --mode eval

# Evaluate with specific model
python main.py --mode eval --model-path models/best_model.pth

# Evaluate with specific configuration
python main.py --mode eval --config config/config.ini
```

### Using the New CLI

```bash
# Use the new CLI structure
python -m src.cli.main --mode evaluate

# Use individual evaluation CLI
python -m src.cli.evaluate
```

## 📊 Evaluation Metrics

### Classification Metrics

The evaluation pipeline computes comprehensive classification metrics:

#### F1 Scores
- **Macro F1**: Average F1 score across all classes
- **Weighted F1**: F1 score weighted by class frequency
- **Per-class F1**: Individual F1 scores for each class

#### Precision and Recall
- **Per-class Precision**: Precision for each congestion class
- **Per-class Recall**: Recall for each congestion class
- **Overall Accuracy**: Overall classification accuracy

#### CSI Average Metrics
- **MAE**: Mean Absolute Error for CSI scores
- **RMSE**: Root Mean Square Error for CSI scores
- **Correlation**: Pearson correlation coefficient

#### AHF Classification Metrics
- **AHF Accuracy**: Acute Heart Failure classification accuracy
- **AHF F1**: F1 score for AHF classification
- **AHF Confusion Matrix**: Confusion matrix for AHF classification

### Metrics Computation

```python
from src.evaluation.metrics.classification import compute_accuracy
from src.evaluation.metrics.f1_score import compute_pytorch_f1_metrics
from src.evaluation.metrics.confusion_matrix import compute_confusion_matrix

# Compute accuracy
accuracy = compute_accuracy(predictions, targets)

# Compute F1 metrics
f1_metrics = compute_pytorch_f1_metrics(predictions, targets)

# Compute confusion matrix
conf_matrix = compute_confusion_matrix(predictions, targets)
```

## 🎯 Per-Zone Evaluation

### Zone-Specific Metrics

The evaluation provides detailed metrics for each of the 6 chest zones:

1. **Right Superior (R_Sup)**
2. **Left Superior (L_Sup)**
3. **Right Middle (R_Mid)**
4. **Left Middle (L_Mid)**
5. **Right Inferior (R_Inf)**
6. **Left Inferior (L_Inf)**

### Zone Metrics Computation

```python
from src.evaluation.evaluator import compute_zone_metrics

# Compute metrics per zone
zone_metrics = compute_zone_metrics(predictions, targets, file_ids, csv_data)

# Access specific zone metrics
r_sup_f1 = zone_metrics['R_Sup']['f1_score']
l_inf_accuracy = zone_metrics['L_Inf']['accuracy']
```

## 📈 Visualization

### Confusion Matrices

#### Per-Zone Confusion Matrices
```python
from src.evaluation.visualization.confusion_matrix import create_confusion_matrix_grid

# Create confusion matrix grid for all zones
conf_matrix_grid = create_confusion_matrix_grid(
    predictions, targets, 
    class_names=['Normal', 'Mild', 'Moderate', 'Severe', 'Unknown']
)
```

#### Overall Confusion Matrix
```python
from src.evaluation.visualization.confusion_matrix import create_overall_confusion_matrix

# Create overall confusion matrix
overall_conf_matrix = create_overall_confusion_matrix(predictions, targets)
```

### ROC Curves

```python
from src.evaluation.visualization.plots import create_roc_curves

# Create ROC curves for all classes
roc_curves = create_roc_curves(predictions, targets, class_names)
```

### Precision-Recall Curves

```python
from src.evaluation.visualization.plots import create_precision_recall_curves

# Create precision-recall curves
pr_curves = create_precision_recall_curves(predictions, targets, class_names)
```

### Training Curves

```python
from src.evaluation.visualization.plots import plot_training_curves

# Plot training history
plot_training_curves(
    train_history, val_history,
    metrics=['loss', 'f1_score', 'accuracy']
)
```

## 🔍 Evaluation Pipeline

### Complete Evaluation Process

```python
from src.evaluation.evaluator import evaluate_model

# Run complete evaluation
evaluation_results = evaluate_model(config)
```

### Step-by-Step Evaluation

```python
from src.evaluation.evaluator import (
    load_trained_model,
    evaluate_model_on_loader,
    compute_zone_metrics,
    compute_overall_metrics
)

# 1. Load trained model
model = load_trained_model(config)

# 2. Evaluate on test set
test_loader = create_test_loader(config)
predictions, targets, file_ids = evaluate_model_on_loader(model, test_loader, config)

# 3. Compute zone metrics
zone_metrics = compute_zone_metrics(predictions, targets, file_ids, csv_data)

# 4. Compute overall metrics
overall_metrics = compute_overall_metrics(predictions, targets)
```

## 📊 Evaluation Reports

### Classification Reports

```python
from src.evaluation.metrics.evaluation_metrics import create_classification_report_per_zone

# Create classification reports for each zone
classification_reports = create_classification_report_per_zone(
    predictions, targets, file_ids, csv_data
)
```

### Confusion Matrix Reports

```python
from src.evaluation.metrics.evaluation_metrics import compute_confusion_matrices_per_zone

# Compute confusion matrices for each zone
confusion_matrices = compute_confusion_matrices_per_zone(
    predictions, targets, file_ids, csv_data
)
```

## 🎨 Visualization Outputs

### Saved Visualizations

The evaluation pipeline automatically saves visualizations to:

- `evaluation_results/confusion_matrices/` - Confusion matrix plots
- `evaluation_results/roc_curves/` - ROC curve plots
- `evaluation_results/pr_curves/` - Precision-recall curves
- `evaluation_results/training_curves/` - Training history plots

### Custom Visualization

```python
from src.evaluation.visualization.confusion_matrix import save_confusion_matrix_graphs

# Save confusion matrix visualizations
save_confusion_matrix_graphs(
    confusion_matrices, 
    output_dir="custom_evaluation/",
    class_names=['Normal', 'Mild', 'Moderate', 'Severe', 'Unknown']
)
```

## 📈 Weights & Biases Integration

### W&B Logging

```python
from src.evaluation.wandb_logging import log_to_wandb

# Log evaluation results to W&B
log_to_wandb(
    zone_metrics=zone_metrics,
    overall_metrics=overall_metrics,
    confusion_matrices=confusion_matrices,
    predictions=predictions,
    targets=targets
)
```

### W&B Configuration

```ini
[WandB]
project = csi-evaluation
entity = your-username
log_predictions = true
log_confusion_matrices = true
log_roc_curves = true
```

## 📋 Evaluation Configuration

### Configuration Options

Edit `config/config.ini` to customize evaluation:

```ini
[Evaluation]
model_path = models/best_model.pth
output_dir = evaluation_results/
save_predictions = true
save_visualizations = true
compute_zone_metrics = true
compute_overall_metrics = true

[Visualization]
save_confusion_matrices = true
save_roc_curves = true
save_pr_curves = true
save_training_curves = true
figure_size = 10,8
dpi = 300
```

### Key Evaluation Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model_path` | Path to trained model | `models/best_model.pth` | Any .pth file |
| `output_dir` | Output directory | `evaluation_results/` | Any directory |
| `save_predictions` | Save predictions to file | `true` | true, false |
| `save_visualizations` | Save visualization plots | `true` | true, false |
| `compute_zone_metrics` | Compute per-zone metrics | `true` | true, false |

## 🔧 Advanced Evaluation

### Custom Evaluation Metrics

```python
from src.evaluation.metrics.classification import compute_custom_metrics

# Define custom metrics
def custom_metric(predictions, targets):
    # Your custom metric implementation
    return metric_value

# Use in evaluation
custom_metrics = compute_custom_metrics(predictions, targets)
```

### Batch Evaluation

```python
from src.evaluation.evaluator import evaluate_model_on_loader

# Evaluate on multiple data loaders
for loader_name, loader in data_loaders.items():
    predictions, targets, file_ids = evaluate_model_on_loader(
        model, loader, config, loader_name=loader_name
    )
```

### Cross-Validation Evaluation

```python
from sklearn.model_selection import KFold
from src.evaluation.evaluator import evaluate_model_on_loader

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # Create fold-specific data loaders
    train_loader = create_loader_from_indices(train_idx)
    val_loader = create_loader_from_indices(val_idx)
    
    # Train model on this fold
    model = train_model_on_fold(train_loader, val_loader)
    
    # Evaluate on validation set
    predictions, targets, file_ids = evaluate_model_on_loader(model, val_loader, config)
    
    cv_results.append(compute_metrics(predictions, targets))
```

## 📊 Results Analysis

### Metrics Summary

```python
from src.evaluation.evaluator import create_evaluation_summary

# Create comprehensive evaluation summary
summary = create_evaluation_summary(
    zone_metrics=zone_metrics,
    overall_metrics=overall_metrics,
    confusion_matrices=confusion_matrices
)

print(summary)
```

### Performance Analysis

```python
# Analyze performance by zone
for zone, metrics in zone_metrics.items():
    print(f"{zone}: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f}")

# Analyze performance by class
for class_name, metrics in overall_metrics.items():
    print(f"{class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model file path
   - Verify model architecture compatibility
   - Ensure model was saved correctly

2. **Memory Issues**
   - Reduce batch size for evaluation
   - Use CPU for evaluation if GPU memory is limited
   - Process data in smaller batches

3. **Metric Computation Errors**
   - Check data format compatibility
   - Verify class labels are correct
   - Ensure predictions and targets have same shape

### Debug Tools

```bash
# Debug evaluation process
python -c "from src.evaluation.evaluator import load_trained_model; print('Model loaded successfully')"

# Check evaluation data
python scripts/debug/debug_images.py
```

## 📚 API Reference

### Main Evaluation Functions

```python
from src.evaluation.evaluator import (
    evaluate_model,
    load_trained_model,
    evaluate_model_on_loader,
    compute_zone_metrics,
    compute_overall_metrics
)

# Complete evaluation pipeline
evaluation_results = evaluate_model(config)

# Individual evaluation steps
model = load_trained_model(config)
predictions, targets, file_ids = evaluate_model_on_loader(model, test_loader, config)
zone_metrics = compute_zone_metrics(predictions, targets, file_ids, csv_data)
overall_metrics = compute_overall_metrics(predictions, targets)
```

### Metrics Functions

```python
from src.evaluation.metrics.classification import compute_accuracy
from src.evaluation.metrics.f1_score import compute_pytorch_f1_metrics
from src.evaluation.metrics.confusion_matrix import compute_confusion_matrix

accuracy = compute_accuracy(predictions, targets)
f1_metrics = compute_pytorch_f1_metrics(predictions, targets)
conf_matrix = compute_confusion_matrix(predictions, targets)
```

### Visualization Functions

```python
from src.evaluation.visualization.plots import (
    create_roc_curves,
    create_precision_recall_curves,
    plot_training_curves
)
from src.evaluation.visualization.confusion_matrix import (
    create_confusion_matrix_grid,
    create_overall_confusion_matrix
)

roc_curves = create_roc_curves(predictions, targets, class_names)
pr_curves = create_precision_recall_curves(predictions, targets, class_names)
conf_matrix_grid = create_confusion_matrix_grid(predictions, targets, class_names)
```

## 🔄 Migration Notes

### Backward Compatibility

All existing evaluation code continues to work:

```python
# These imports still work
from src.evaluate import evaluate_model, load_trained_model
```

### New Modular Imports (Recommended)

```python
# Recommended new imports
from src.evaluation.evaluator import evaluate_model, load_trained_model
from src.evaluation.metrics import compute_confusion_matrix
from src.evaluation.visualization import create_roc_curves
```

The evaluation system has been completely modularized while maintaining full backward compatibility. 