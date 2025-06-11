# Model Evaluation

Comprehensive guide to evaluating CSI prediction models with detailed metrics and analysis.

## Quick Evaluation

```bash
# Evaluate latest trained model
python -m src.evaluate

# Evaluate with custom configuration
python -m src.evaluate --ini config.ini

# Legacy entry point
python main.py --mode eval
```

## Evaluation Features

### 📊 Comprehensive Metrics
- **Per-Zone Analysis**: Individual metrics for all 6 chest X-ray zones
- **Overall Performance**: Combined accuracy and F1-scores
- **Confusion Matrices**: Interactive heatmaps for each zone
- **Classification Reports**: Precision, recall, F1-score per congestion class
- **Statistical Analysis**: Class distribution and performance statistics

### 📈 Visual Analytics
- **Confusion Matrix Heatmaps**: Per-zone confusion visualization
- **ROC Curves**: Receiver Operating Characteristic curves for each zone and class
- **Precision-Recall Curves**: PR curves for imbalanced class analysis
- **Training Curves**: Loss, accuracy, and F1-score progression during training
- **Performance Comparisons**: Validation vs test set analysis
- **Class Distribution**: Label frequency analysis
- **Interactive Dashboards**: Real-time WandB visualizations

### 📄 Detailed Reports
- **Comprehensive Text Reports**: Detailed analysis saved to files
- **Prediction Exports**: CSV files with per-sample predictions
- **Performance Summaries**: Executive summaries of model performance
- **Error Analysis**: Detailed misclassification analysis

## Understanding CSI Zones

The model predicts congestion scores for 6 anatomical zones:

| Zone | Anatomical Region | Classes |
|------|------------------|---------|
| **right_sup** | Right Superior | 0-3 (congestion), 4 (ungradable) |
| **left_sup** | Left Superior | 0-3 (congestion), 4 (ungradable) |
| **right_mid** | Right Middle | 0-3 (congestion), 4 (ungradable) |
| **left_mid** | Left Middle | 0-3 (congestion), 4 (ungradable) |
| **right_inf** | Right Inferior | 0-3 (congestion), 4 (ungradable) |
| **left_inf** | Left Inferior | 0-3 (congestion), 4 (ungradable) |

**Congestion Scale:**
- **0**: Normal (no congestion)
- **1**: Mild congestion
- **2**: Moderate congestion
- **3**: Severe congestion
- **4**: Ungradable (poor image quality, obscured)

## Advanced Curve Analysis

### 🎯 ROC Curves (Receiver Operating Characteristic)

**Purpose**: Evaluate binary classification performance for each congestion class using One-vs-Rest approach.

**Generated Curves**:
- **Per-Zone ROC**: Individual ROC curves for each anatomical zone
- **Per-Class ROC**: ROC curves for each congestion severity level (0-3)
- **Macro-Averaged ROC**: Overall performance across all zones
- **AUC Scores**: Area Under Curve metrics for quantitative comparison

**Interpretation**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (diagonal line)
- **Higher AUC**: Better discrimination between classes

**Files Generated**:
```
graphs/[model_name]/
├── validation_right_sup_roc_curves.png
├── validation_left_sup_roc_curves.png
├── validation_overall_roc_curves.png
├── test_right_sup_roc_curves.png
└── test_overall_roc_curves.png
```

### 📊 Precision-Recall Curves

**Purpose**: Analyze performance on imbalanced medical data where certain congestion levels are rare.

**Generated Curves**:
- **Per-Zone PR**: Individual PR curves for each anatomical zone  
- **Per-Class PR**: PR curves for each congestion severity level
- **Macro-Averaged PR**: Overall performance across all zones
- **Average Precision (AP)**: Summary metric for each curve

**Medical Relevance**:
- **High Precision**: Few false positive diagnoses (important for patient safety)
- **High Recall**: Few missed congestion cases (important for comprehensive care)
- **Better than ROC for imbalanced classes**: More appropriate for medical CSI data

**Files Generated**:
```
graphs/[model_name]/
├── validation_right_sup_pr_curves.png
├── validation_left_sup_pr_curves.png  
├── validation_overall_pr_curves.png
├── test_right_sup_pr_curves.png
└── test_overall_pr_curves.png
```

### 📈 Training Curves

**Purpose**: Monitor model learning progression and detect overfitting/underfitting.

**Generated Plots**:
- **Loss Curves**: Training vs validation loss over epochs
- **Accuracy Curves**: Training vs validation accuracy progression
- **F1-Score Curves**: Training vs validation F1-score evolution

**Usage for Model Development**:
- **Overfitting Detection**: Validation metrics plateau while training improves
- **Underfitting Detection**: Both training and validation metrics plateau early
- **Learning Rate Optimization**: Smooth convergence vs oscillation patterns
- **Early Stopping Validation**: Optimal stopping point identification

**Files Generated**:
```
graphs/training_curves/
└── [run_name]_training_curves.png
```

## Evaluation Process

### 1. Model Loading
```
Loading model: 20250611_093054_ResNet50_Tr.pth
Model architecture: ResNet50
Device: cuda
Parameters: 23.5M
```

### 2. Data Processing
```
Loading test dataset...
Test samples: 1,234
Batch size: 32
Processing complete: 100%
```

### 3. Inference and Metrics
```
Computing predictions...
Calculating per-zone metrics...
Generating confusion matrices...
Creating ROC curves...
Creating Precision-Recall curves...
Creating comprehensive reports...
```

### 4. Results Export
```
Reports saved to: docs/evaluation/
- validation_comprehensive_report.txt
- test_comprehensive_report.txt
- validation_predictions.csv
- test_predictions.csv

Graphs saved to: graphs/[model_name]/
- Confusion matrices (PNG)
- ROC curves (PNG)  
- Precision-Recall curves (PNG)
- Training curves (PNG)
```

## Evaluation Outputs

### Comprehensive Reports

**Location**: `docs/evaluation/test_comprehensive_report.txt`

**Contents**:
```
=== CSI PREDICTION EVALUATION REPORT ===
Generated: 2025-06-11 09:30:54
Model: 20250611_093054_ResNet50_Tr
Dataset: Test Set (1,234 samples)

=== OVERALL PERFORMANCE ===
Overall Accuracy: 0.8456
Overall F1 Score: 0.8234

=== PER-ZONE PERFORMANCE ===
Zone: right_sup
- Accuracy: 0.8567
- F1 Score: 0.8345
- Precision: 0.8456
- Recall: 0.8234

[Detailed per-zone metrics for all 6 zones]

=== CONFUSION MATRICES ===
[Per-zone confusion matrices]

=== CLASS DISTRIBUTION ===
[Label frequency analysis]
```

### Prediction CSV Files

**Location**: `docs/evaluation/test_predictions.csv`

**Format**:
```csv
FileID,right_sup_true,right_sup_pred,left_sup_true,left_sup_pred,...
image001.jpg,1,1,0,0,2,2,1,1,0,1,3,3
image002.jpg,0,0,1,2,1,1,0,0,2,2,4,4
```

**Uses**:
- Error analysis and debugging
- Post-processing and ensemble methods
- Custom metric calculations
- Detailed performance investigation

### Confusion Matrix Visualizations

**Location**: `graphs/confusion_matrices/`

**Features**:
- **Interactive Heatmaps**: Hover details and zooming
- **Per-Zone Matrices**: Individual analysis for each anatomical zone
- **Normalized Views**: Both raw counts and percentage views
- **Export Options**: High-resolution PNG and PDF formats

## WandB Integration

### Real-time Dashboard

Evaluation automatically logs to Weights & Biases:

```bash
# Dashboard URL will be displayed:
wandb: 🚀 View evaluation at https://wandb.ai/your-username/csi-predictor-eval/runs/...
```

**Dashboard Features**:
- **Interactive Confusion Matrices**: Heatmaps for each zone
- **Performance Tables**: Structured metrics display
- **Comparison Tools**: Compare multiple model evaluations
- **Artifact Storage**: Automatic model and report storage

### Logged Metrics

- **Per-zone accuracy and F1-scores**
- **Overall performance metrics**
- **Confusion matrices as interactive tables**
- **Classification reports as structured data**
- **Sample predictions for error analysis**

## Advanced Evaluation

### Custom Evaluation Script

```python
from src.config import cfg
from src.evaluate import evaluate_model
from src.utils import load_checkpoint

# Load specific model
model_path = "models/20250611_093054_ResNet50_Tr.pth"
model = load_checkpoint(model_path, cfg)

# Run evaluation
results = evaluate_model(model, cfg)

# Access detailed results
print(f"Overall F1: {results['overall_f1']:.4f}")
for zone in ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']:
    print(f"{zone} F1: {results[f'{zone}_f1']:.4f}")
```

### Batch Evaluation

```bash
# Evaluate multiple models
for model in models/*.pth; do
    echo "Evaluating $model"
    python -m src.evaluate --model "$model"
done
```

### Custom Metrics

```python
# Custom evaluation with additional metrics
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

def extended_evaluation(y_true, y_pred):
    """Extended evaluation with additional metrics."""
    metrics = {}
    
    for zone_idx, zone in enumerate(['right_sup', 'left_sup', 'right_mid', 
                                   'left_mid', 'right_inf', 'left_inf']):
        zone_true = y_true[:, zone_idx]
        zone_pred = y_pred[:, zone_idx]
        
        # Standard metrics (already computed)
        metrics[f'{zone}_accuracy'] = accuracy_score(zone_true, zone_pred)
        metrics[f'{zone}_f1'] = f1_score(zone_true, zone_pred, average='weighted')
        
        # Additional metrics
        metrics[f'{zone}_kappa'] = cohen_kappa_score(zone_true, zone_pred)
        metrics[f'{zone}_mcc'] = matthews_corrcoef(zone_true, zone_pred)
    
    return metrics
```

## Performance Interpretation

### Understanding F1-Scores

| F1-Score Range | Interpretation | Recommendation |
|----------------|----------------|----------------|
| **0.90-1.00** | Excellent | Production ready |
| **0.80-0.89** | Good | Consider optimization |
| **0.70-0.79** | Fair | Needs improvement |
| **0.60-0.69** | Poor | Major changes needed |
| **< 0.60** | Very Poor | Restart with different approach |

### Zone-Specific Analysis

Different zones may have different performance characteristics:

- **Superior zones**: Often clearer, better performance expected
- **Middle zones**: May have overlapping anatomy, moderate difficulty
- **Inferior zones**: Can be obscured by diaphragm, potentially challenging

### Class Imbalance Considerations

- **Normal cases (class 0)**: Usually most frequent
- **Severe cases (class 3)**: Often least frequent but most important
- **Ungradable (class 4)**: Variable frequency, important for quality control

## Troubleshooting Evaluation

### Common Issues

#### Model Loading Error
```bash
# Check model file exists
ls -la models/

# Verify model compatibility
python -c "
from src.config import cfg
from src.utils import load_checkpoint
model = load_checkpoint('path/to/model.pth', cfg)
print('Model loaded successfully')
"
```

#### Memory Issues
```bash
# Reduce batch size for evaluation
EVAL_BATCH_SIZE = 16  # if implemented

# Use CPU if GPU memory insufficient
DEVICE = cpu
```

#### Poor Performance Analysis
```bash
# Check data quality
python -c "
from src.data import create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(cfg)
print(f'Test samples: {len(test_loader.dataset)}')
"

# Verify label distribution
python -c "
import pandas as pd
df = pd.read_csv('path/to/labels.csv', sep=';')
print(df.describe())
"
```

## Evaluation Best Practices

### 1. Multiple Evaluation Runs
```bash
# Evaluate same model multiple times to check consistency
for i in {1..5}; do
    python -m src.evaluate --run-id "eval_$i"
done
```

### 2. Cross-Validation Analysis
```bash
# If multiple fold models available
for fold in {1..5}; do
    python -m src.evaluate --model "models/fold_${fold}_model.pth"
done
```

### 3. Error Analysis
```python
# Analyze specific error patterns
import pandas as pd

# Load predictions
df = pd.read_csv('docs/evaluation/test_predictions.csv')

# Find systematic errors
for zone in ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']:
    true_col = f'{zone}_true'
    pred_col = f'{zone}_pred'
    
    # Most common errors
    errors = df[df[true_col] != df[pred_col]]
    error_patterns = errors.groupby([true_col, pred_col]).size().sort_values(ascending=False)
    print(f"\n{zone} most common errors:")
    print(error_patterns.head())
```

### 4. Performance Monitoring
```bash
# Track performance over time
python scripts/track_performance.py  # if available

# Compare with baseline models
python scripts/compare_models.py  # if available
```

## Model Comparison

### Comparing Multiple Models

```python
# Compare different architectures
models = {
    'ResNet50': 'models/20250611_093054_ResNet50_Tr.pth',
    'CheXNet': 'models/20250611_094512_CheXNet_Tr.pth', 
    'RadDINO': 'models/20250611_095234_RadDINO_Tr.pth'
}

results = {}
for name, path in models.items():
    # Evaluate each model
    result = evaluate_model(load_checkpoint(path, cfg), cfg)
    results[name] = result

# Compare results
comparison_df = pd.DataFrame(results).T
print(comparison_df[['overall_f1', 'right_sup_f1', 'left_sup_f1']])
```

### Statistical Significance Testing

```python
# Test if performance differences are significant
from scipy.stats import ttest_rel

# Compare two models on same test set
model1_f1_scores = [...]  # Per-sample F1 scores
model2_f1_scores = [...]  # Per-sample F1 scores

statistic, p_value = ttest_rel(model1_f1_scores, model2_f1_scores)
print(f"Statistical significance: p = {p_value:.4f}")
```

## Next Steps

After evaluation:

1. **[Hyperparameter Optimization](hyperparameter-optimization.md)** if performance needs improvement
2. **[Model Architecture Analysis](model-architectures.md)** to understand model behavior
3. **[Production Deployment](monitoring-logging.md)** if performance is satisfactory
4. **Error Analysis** to identify improvement opportunities 