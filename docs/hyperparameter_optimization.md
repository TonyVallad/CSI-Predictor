# Hyperparameter Optimization with Optuna

This document describes the hyperparameter optimization capabilities of the CSI-Predictor project using [Optuna](https://optuna.readthedocs.io/), a state-of-the-art hyperparameter optimization framework.

## Overview

The CSI-Predictor includes automated hyperparameter optimization using Bayesian optimization through Optuna. This system intelligently searches the hyperparameter space to find the best combination of parameters for your specific dataset and task.

### Key Features

- **Bayesian Optimization**: Uses TPE (Tree-structured Parzen Estimator) sampler for efficient search
- **Automatic Pruning**: Stops unpromising trials early to save computational resources
- **Multi-objective Support**: Optimize validation F1 score while considering computational efficiency
- **WandB Integration**: Seamless logging and visualization with Weights & Biases
- **Resumable Studies**: Continue optimization across multiple sessions
- **Production Ready**: Automatic training with best hyperparameters found

## Quick Start

### 1. Install Dependencies

```bash
pip install optuna plotly  # For visualization
```

### 2. Run Hyperparameter Optimization

```bash
# Basic optimization with 50 trials
python main.py --mode hyperopt --n-trials 50

# Advanced optimization with custom settings
python main.py --mode hyperopt \
    --study-name "csi_advanced_optimization" \
    --n-trials 100 \
    --max-epochs 30 \
    --sampler tpe \
    --pruner median \
    --wandb-project "csi-hyperopt"
```

### 3. Train with Best Hyperparameters

```bash
# Train final model with optimized hyperparameters
python main.py --mode train-optimized \
    --hyperparams models/hyperopt/csi_optimization_best_params.json
```

## Optimized Hyperparameters

The optimization searches over the following hyperparameter space:

### Model Architecture
- **Choices**: `ResNet50`, `CheXNet`, `Custom_1`
- **Impact**: Different architectures balance capacity vs. computational efficiency
- **Note**: Available architectures are limited to those implemented in the project

### Optimizer Settings
- **Optimizer**: `adam`, `adamw`, `sgd`
- **Learning Rate**: `1e-5` to `1e-1` (log scale)
- **Weight Decay**: `1e-6` to `1e-2` (for AdamW and SGD)
- **Momentum**: `0.5` to `0.99` (for SGD)

### Training Parameters
- **Batch Size**: `16`, `32`, `64`, `128`
- **Early Stopping Patience**: `5` to `20` epochs

### Loss Function
- **Unknown Weight**: `0.1` to `1.0`
  - Controls the importance of unknown class (class 4) in loss calculation
  - Lower values emphasize clear CSI classifications (0-3)

## Performance Optimizations

### Smart Data Caching
The optimization system includes intelligent caching to avoid re-loading and pre-processing images for every trial:

- **One-time Caching**: Images are cached only once at the start of optimization
- **Memory Efficient**: Cached datasets are reused across all trials
- **Batch Size Flexibility**: Different batch sizes reuse the same cached data
- **Significant Speedup**: Reduces trial time by 60-80% after initial caching

### Automatic Resource Management
- **GPU Memory**: Automatically manages CUDA memory between trials
- **Early Pruning**: Stops unpromising trials early based on validation performance
- **Error Recovery**: Failed trials don't stop the optimization process

## Advanced Usage

### Custom Optimization Study

```python
from src.hyperopt import optimize_hyperparameters

# Run optimization with custom parameters
study = optimize_hyperparameters(
    study_name="custom_csi_study",
    n_trials=200,
    max_epochs=50,
    sampler='tpe',  # or 'random', 'cmaes'
    pruner='median',  # or 'successive_halving', 'none'
    wandb_project="my-csi-project"
)

print(f"Best validation F1: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

### Distributed Optimization

For large-scale optimization across multiple machines:

```bash
# On machine 1
python -m src.hyperopt --storage "sqlite:///shared_study.db" --n-trials 50

# On machine 2
python -m src.hyperopt --storage "sqlite:///shared_study.db" --n-trials 50
```

### Resume Previous Study

```bash
# Continue previous optimization (automatically resumes if study exists)
python main.py --mode hyperopt --study-name "previous_study_name" --n-trials 50
```

## Optimization Strategies

### Samplers

1. **TPE (Tree-structured Parzen Estimator)** - Default
   - Most effective for continuous and categorical parameters
   - Balances exploration and exploitation intelligently
   - Best for most use cases

2. **Random Sampler**
   - Good baseline, especially for initial exploration
   - Useful when parameter space is not well understood

3. **CMA-ES (Covariance Matrix Adaptation)**
   - Excellent for continuous parameters
   - May be slower but more thorough

### Pruners

1. **Median Pruner** - Default
   - Prunes trials performing worse than median at each epoch
   - Good balance of efficiency and thoroughness

2. **Successive Halving**
   - More aggressive pruning
   - Faster but may miss slower-converging good configurations

3. **No Pruning**
   - Run all trials to completion
   - Most thorough but computationally expensive

## Monitoring and Analysis

### Real-time Monitoring

When using WandB integration:

```bash
python main.py --mode hyperopt --wandb-project "csi-optimization"
```

Visit your WandB dashboard to see:
- Real-time optimization progress
- Parameter importance analysis
- Trial comparison and filtering
- Best hyperparameter combinations

### Visualization

After optimization, interactive plots are automatically generated:

```
models/hyperopt/
├── csi_optimization_optimization_history.html  # Progress over time
├── csi_optimization_param_importances.html     # Parameter importance
├── csi_optimization_parallel_coordinate.html   # Parameter relationships
└── csi_optimization_best_params.json          # Best hyperparameters
```

### Manual Analysis

```python
import optuna

# Load and analyze study
study = optuna.load_study(study_name="csi_optimization")

# Get best trial
best_trial = study.best_trial
print(f"Best value: {best_trial.value}")
print(f"Best params: {best_trial.params}")

# Analyze parameter importance
importance = optuna.importance.get_param_importances(study)
print("Parameter importance:", importance)
```

## Best Practices

### 1. Start Small
- Begin with 20-50 trials to understand the parameter landscape
- Use shorter `max_epochs` (20-30) during exploration

### 2. Iterative Refinement
```bash
# Phase 1: Broad exploration
python main.py --mode hyperopt --n-trials 50 --max-epochs 20 --sampler random

# Phase 2: Focused optimization
python main.py --mode hyperopt --n-trials 100 --max-epochs 30 --sampler tpe

# Phase 3: Fine-tuning
python main.py --mode hyperopt --n-trials 50 --max-epochs 50 --pruner successive_halving
```

### 3. Monitor Resource Usage
- Use pruning to avoid wasting computation on poor trials
- Consider batch size impact on GPU memory
- Monitor training time per trial

### 4. Validate Results
```bash
# Train multiple models with best hyperparameters to ensure consistency
for i in {1..3}; do
    python main.py --mode train-optimized \
        --hyperparams models/hyperopt/csi_optimization_best_params.json
done
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce max batch size in optimization: modify `batch_size` choices in `src/hyperopt.py`
   - Use gradient accumulation or mixed precision training

2. **Slow Optimization**
   - Enable pruning: use `median` or `successive_halving` pruner
   - Reduce `max_epochs` for initial exploration
   - Use smaller model architectures first

3. **Poor Results**
   - Increase number of trials (100-200+)
   - Verify data quality and preprocessing
   - Check validation set representativeness

### Performance Tips

1. **Faster Trials**
   ```python
   # Reduce image size during optimization
   # Modify data transforms in create_optuna_config()
   ```

2. **Better Convergence**
   ```python
   # Increase n_startup_trials for TPE sampler
   sampler = TPESampler(seed=42, n_startup_trials=20)
   ```

3. **Memory Optimization**
   ```python
   # Use smaller models during initial search
   model_arch_choices = ['resnet18', 'resnet34']  # Instead of larger models
   ```

## Integration with Existing Workflow

The hyperparameter optimization seamlessly integrates with your existing training pipeline:

1. **Data Pipeline**: Uses your existing data loading and preprocessing
2. **Model Architecture**: Leverages your model definitions and configurations
3. **Training Loop**: Reuses training and validation logic
4. **Evaluation**: Compatible with your metrics and evaluation scripts
5. **Logging**: Integrates with WandB and your existing logging setup

## Example Workflow

Complete optimization and training workflow:

```bash
# 1. Run hyperparameter optimization
python main.py --mode hyperopt \
    --study-name "production_optimization" \
    --n-trials 100 \
    --max-epochs 30 \
    --wandb-project "csi-production"

# 2. Train final model with best hyperparameters
python main.py --mode train-optimized \
    --hyperparams models/hyperopt/production_optimization_best_params.json

# 3. Evaluate the optimized model
python main.py --mode eval

# 4. Compare with baseline
python -m src.evaluate --model-path models/optimized_model.pth
```

This systematic approach ensures you get the best possible performance from your CSI prediction model while maintaining reproducibility and efficiency. 