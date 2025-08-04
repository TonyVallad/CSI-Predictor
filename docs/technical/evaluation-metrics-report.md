# CSI-Predictor Evaluation Metrics Report

## Table of Contents

1. [Overview](#overview)
2. [CSI Classification System](#csi-classification-system)
3. [Data Structure and Encoding](#data-structure-and-encoding)
4. [Evaluation Metrics Overview](#evaluation-metrics-overview)
5. [F1 Score Implementation](#f1-score-implementation)
6. [Zone-Specific Evaluation](#zone-specific-evaluation)
7. [CSI Average Metrics](#csi-average-metrics)
8. [AHF Risk Classification](#ahf-risk-classification)
9. [Training and Validation Process](#training-and-validation-process)
10. [Metric Interpretation Guidelines](#metric-interpretation-guidelines)
11. [Current Performance Analysis](#current-performance-analysis)
12. [Recommendations for Improvement](#recommendations-for-improvement)

## Overview

The CSI-Predictor project evaluates deep learning models for predicting Congestion Score Index (CSI) across 6 lung zones (3 per lung: superior, middle, inferior). This report provides a comprehensive analysis of the evaluation metrics, their implementation, and interpretation guidelines.

## CSI Classification System

### CSI Score Categories

The model predicts CSI scores across 5 categories:

| Class | CSI Score | Description | Clinical Significance |
|-------|-----------|-------------|---------------------|
| 0 | 0 | Normal | No congestion detected |
| 1 | 1 | Mild | Slight congestion |
| 2 | 2 | Moderate | Moderate congestion |
| 3 | 3 | Severe | Severe congestion |
| 4 | N/A | Unknown/Ungradable | Image quality insufficient |

### Lung Zone Mapping

The model predicts CSI for 6 distinct lung zones:

| Zone | Lung | Region | Clinical Importance |
|------|------|--------|-------------------|
| Zone 1 | Left | Superior | Upper left lung |
| Zone 2 | Left | Middle | Middle left lung |
| Zone 3 | Left | Inferior | Lower left lung |
| Zone 4 | Right | Superior | Upper right lung |
| Zone 5 | Right | Middle | Middle right lung |
| Zone 6 | Right | Inferior | Lower right lung |

## Data Structure and Encoding

### Input Data Format

```python
# Model predictions: [batch_size, n_zones, n_classes]
predictions = torch.tensor([[[0.1, 0.2, 0.6, 0.1, 0.0],  # Zone 1 probabilities
                            [0.8, 0.1, 0.05, 0.05, 0.0], # Zone 2 probabilities
                            # ... for all 6 zones
                           ]])

# Ground truth targets: [batch_size, n_zones]
targets = torch.tensor([[2, 0, 1, 3, 1, 2]])  # Actual CSI scores
```

### Class Encoding Strategy

- **Classes 0-3**: Represent actual CSI scores (0=Normal, 1=Mild, 2=Moderate, 3=Severe)
- **Class 4**: Special "unknown" class for ungradable images
- **Default handling**: Class 4 is typically ignored in evaluation (`ignore_index=4`)

## Evaluation Metrics Overview

The evaluation system computes multiple complementary metrics:

### Primary Metrics
1. **F1 Score (Weighted)**: Primary optimization metric for sweeps
2. **F1 Score (Macro)**: Equal weight for all classes
3. **Accuracy**: Overall classification accuracy

### Secondary Metrics
1. **Precision/Recall**: Per-class performance analysis
2. **CSI Average Metrics**: Regression-style evaluation
3. **AHF Risk Classification**: Clinical outcome prediction

### Zone-Specific Metrics
1. **Per-zone F1 scores**: Individual zone performance
2. **Zone-specific accuracy**: Regional performance analysis

## F1 Score Implementation

### Mathematical Foundation

The F1 score is the harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- **Precision** = True Positives / (True Positives + False Positives)
- **Recall** = True Positives / (True Positives + False Negatives)

### Implementation Details

```python
def compute_f1_from_confusion_matrix(cm: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    # True positives, false positives, false negatives
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    
    # Precision and recall
    precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
```

### Averaging Methods

#### Macro F1
- **Formula**: `F1_macro = (F1_class_0 + F1_class_1 + F1_class_2 + F1_class_3) / 4`
- **Characteristics**: Equal weight for all classes
- **Use case**: When all classes are equally important

#### Weighted F1
- **Formula**: `F1_weighted = Σ(w_i × F1_i)` where `w_i` is class frequency
- **Characteristics**: Accounts for class imbalance
- **Use case**: Primary metric for sweep optimization

#### Micro F1
- **Formula**: Global precision/recall across all classes
- **Characteristics**: Treats all samples equally regardless of class
- **Use case**: Overall performance assessment

## Zone-Specific Evaluation

### Per-Zone F1 Computation

The system computes F1 scores for each lung zone independently:

```python
zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]

for i, zone_name in enumerate(zone_names):
    zone_mask = mask[:, i]  # Valid samples for this zone
    zone_pred = pred_classes[:, i][zone_mask]
    zone_true = targets[:, i][zone_mask]
    
    # Compute confusion matrix for this zone
    cm = compute_confusion_matrix(zone_pred, zone_true, num_classes=5)
    
    # Compute F1 scores
    zone_f1_macro = compute_f1_from_confusion_matrix(cm, average='macro')
    zone_f1_weighted = compute_f1_from_confusion_matrix(cm, average='weighted')
```

### Zone Performance Analysis

Each zone is evaluated independently, allowing identification of:
- **Zone-specific strengths/weaknesses**
- **Regional performance patterns**
- **Clinical relevance by lung region**

## CSI Average Metrics

### Average CSI Calculation

The system computes average CSI scores for each sample:

```python
def calculate_csi_average(predictions, targets):
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    for i in range(pred_classes.shape[0]):
        sample_preds = pred_classes[i]  # [n_zones]
        sample_targets = targets[i]     # [n_zones]
        
        # Calculate average excluding unknown class (4)
        pred_valid_mask = sample_preds != 4
        target_valid_mask = sample_targets != 4
        
        if pred_valid_mask.sum() > 0:
            pred_avg = sample_preds[pred_valid_mask].float().mean().item()
        else:
            pred_avg = 0.0  # Default if all zones are unknown
```

### Regression Metrics

The system computes regression-style metrics on average CSI:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual CSI
- **Root Mean Square Error (RMSE)**: Square root of mean squared error
- **Correlation**: Pearson correlation between predicted and actual averages

## AHF Risk Classification

### AHF Classification System

Based on average CSI scores, patients are classified into AHF risk categories:

| AHF Class | Average CSI Range | Risk Level | Clinical Action |
|-----------|-------------------|------------|-----------------|
| 0 | ≤ 1.3 | Low Risk | Routine monitoring |
| 1 | 1.3 < CSI ≤ 2.2 | Medium Risk | Enhanced monitoring |
| 2 | > 2.2 | High Risk | Immediate intervention |

### AHF Metrics Computation

```python
def calculate_ahf_class(avg_csi: float) -> int:
    if avg_csi <= 1.3:
        return 0
    elif avg_csi <= 2.2:
        return 1
    else:
        return 2
```

### AHF Performance Metrics

- **AHF Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall**: Performance for each risk level
- **Macro F1**: Equal weight for all risk categories
- **Confusion Matrix**: Detailed error analysis

## Training and Validation Process

### Epoch-Level Evaluation

During each training/validation epoch:

1. **Forward Pass**: Model generates predictions for all zones
2. **Loss Computation**: Cross-entropy loss with optional class weighting
3. **Metric Computation**: All metrics computed on epoch data
4. **Logging**: Metrics logged to wandb for sweep optimization

### Metric Computation Flow

```python
def validate_epoch(model, val_loader, criterion, device, csv_data, config):
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    
    for batch_data in val_loader:
        images, targets = batch_data[:2]
        outputs = model(images)  # [batch_size, n_zones, n_classes]
        
        # Store for metric computation
        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())
    
    # Concatenate all batches
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    
    # Compute all metrics
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    pr_metrics = compute_precision_recall(all_pred_tensor, all_target_tensor)
    csi_avg_metrics = compute_csi_average_metrics(all_pred_tensor, all_target_tensor)
    ahf_metrics = compute_ahf_classification_metrics(all_pred_tensor, all_target_tensor)
    
    return {**f1_metrics, **pr_metrics, **csi_avg_metrics, **ahf_metrics}
```

### Sweep Optimization

For wandb sweeps, the primary optimization metric is `val_f1_weighted`:

```python
# Log final metric for sweep optimization
if is_wandb_run:
    final_val_f1 = float(best_val_f1)
    wandb.log({'val_f1_weighted': final_val_f1}, step=cfg.n_epochs)
```

## Metric Interpretation Guidelines

### F1 Score Interpretation

| F1 Range | Performance Level | Clinical Interpretation |
|----------|------------------|------------------------|
| 0.0 - 0.3 | Poor | Unreliable for clinical use |
| 0.3 - 0.5 | Fair | Limited clinical utility |
| 0.5 - 0.7 | Good | Suitable for clinical use |
| 0.7 - 0.9 | Very Good | High clinical reliability |
| 0.9 - 1.0 | Excellent | Near-perfect performance |

### Current Performance Analysis

Based on sweep results showing F1 scores of 0.30-0.56:

- **Best runs**: ~0.56 (fair to good performance)
- **Average runs**: ~0.40-0.45 (fair performance)
- **Poor runs**: ~0.30 (poor performance)

### Zone-Specific Interpretation

- **High zone variance**: Indicates inconsistent performance across lung regions
- **Zone-specific patterns**: May reveal anatomical or pathological differences
- **Clinical relevance**: Different zones may have different clinical importance

### AHF Risk Classification Interpretation

- **High-risk sensitivity**: Critical for patient safety
- **Low-risk specificity**: Important for resource allocation
- **Overall accuracy**: General performance indicator

## Current Performance Analysis

### Strengths

1. **Comprehensive evaluation**: Multiple complementary metrics
2. **Zone-specific analysis**: Detailed regional performance
3. **Clinical relevance**: AHF risk classification
4. **Robust implementation**: Handles edge cases (unknown class)

### Limitations

1. **F1 score limitations**: 
   - Treats ordinal CSI scores as categorical
   - Doesn't account for clinical impact of errors
   - May not reflect medical decision-making needs

2. **Class imbalance**: 
   - Severe cases (class 3) may be underrepresented
   - Weighted F1 may mask poor performance on critical cases

3. **Zone weighting**: 
   - All zones treated equally
   - No clinical weighting for different lung regions

## Recommendations for Improvement

### 1. Enhanced Metrics

#### Medical-Specific Metrics
```python
def compute_medical_metrics(predictions, targets):
    # Severity-weighted error
    severity_weights = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
    weighted_mae = compute_weighted_mae(predictions, targets, severity_weights)
    
    # False negative rate for severe cases
    fn_rate_severe = compute_false_negative_rate(predictions, targets, class=3)
    
    # Clinical accuracy (AHF risk classification)
    clinical_accuracy = compute_ahf_risk_accuracy(predictions, targets)
    
    return {
        'weighted_mae': weighted_mae,
        'fn_rate_severe': fn_rate_severe,
        'clinical_accuracy': clinical_accuracy
    }
```

#### Zone-Weighted Metrics
```python
def compute_zone_weighted_metrics(predictions, targets, zone_weights):
    # Weight zones by clinical importance
    weighted_f1 = sum(w * f1 for w, f1 in zip(zone_weights, zone_f1_scores))
    return weighted_f1
```

### 2. Improved Evaluation Strategy

#### Primary Metrics (Recommended)
1. **Weighted MAE**: Better for ordinal CSI scores
2. **Severity-weighted F1**: Higher penalty for severe case errors
3. **AHF Risk Accuracy**: Direct clinical outcome measure

#### Secondary Metrics
1. **Per-class F1 scores**: Detailed class performance
2. **Zone-specific analysis**: Regional performance
3. **Correlation metrics**: Linear relationship assessment

### 3. Clinical Validation

#### Medical Expert Review
- **Case-by-case analysis**: Review of prediction errors
- **Clinical impact assessment**: Real-world consequences
- **Threshold optimization**: Adjust classification boundaries

#### Performance Benchmarks
- **Radiologist comparison**: Human expert performance
- **Clinical outcome correlation**: Patient outcomes
- **Resource utilization**: Impact on healthcare resources

### 4. Implementation Improvements

#### Enhanced Logging
```python
# Log comprehensive metrics for analysis
wandb.log({
    'val_f1_weighted': f1_weighted,
    'val_mae': mae,
    'val_severity_weighted_mae': severity_weighted_mae,
    'val_ahf_accuracy': ahf_accuracy,
    'val_fn_rate_severe': fn_rate_severe,
    'val_zone_f1_variance': zone_f1_variance
})
```

#### Real-time Monitoring
- **Performance alerts**: Notify when metrics drop below thresholds
- **Drift detection**: Monitor for performance degradation
- **A/B testing**: Compare different model versions

## Conclusion

The current evaluation system provides a solid foundation for CSI prediction assessment. However, the F1 score-based approach has limitations for medical applications. Implementing medical-specific metrics, zone weighting, and clinical validation will provide more meaningful evaluation for real-world deployment.

The recommended approach combines:
1. **Weighted MAE** as primary metric for sweep optimization
2. **Severity-weighted evaluation** for clinical relevance
3. **AHF risk classification** for outcome prediction
4. **Zone-specific analysis** for regional performance
5. **Clinical validation** for real-world assessment

This comprehensive evaluation framework will ensure the CSI-Predictor model meets the rigorous standards required for medical AI applications.

---

*Report generated on: August 4, 2025*  
*CSI-Predictor Version: 1.0.0*  
*Author: CSI-Predictor Team* 