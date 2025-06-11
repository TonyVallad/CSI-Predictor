# Model Naming Format

## Overview

CSI-Predictor uses a structured model naming format for consistent identification and organization of trained models, evaluation runs, and other artifacts.

## Format Structure

```
[YYYYMMDD_HHMMSS]_[ModelName]_[TaskTag]_[ExtraInfo]
```

### Components

| Component | Description | Example | Required |
|-----------|-------------|---------|----------|
| **YYYYMMDD_HHMMSS** | Compact, sortable timestamp | `20250611_093054` | ✅ Yes |
| **ModelName** | Model architecture name | `ResNet50`, `RadDINO`, `ViT-B` | ✅ Yes |
| **TaskTag** | Task/operation identifier | `Tr`, `Eval`, `Va`, `Te`, `Infer` | ✅ Yes |
| **ExtraInfo** | Optional additional information | `batch64`, `518x518`, `aug_v2` | ❌ Optional |

### Task Tags

| Tag | Description | Usage Context |
|-----|-------------|---------------|
| `Tr` | Training | Model saved during training |
| `Eval` | Evaluation | Evaluation runs and reports |
| `Va` | Validation | Validation-specific operations |
| `Te` | Test | Test set evaluation |
| `Infer` | Inference | Production inference runs |

## Examples

### Basic Examples

```
20250611_093054_ResNet50_Tr.pth           # Training model
20250611_093054_ResNet50_Eval.pth         # Evaluation run
20250611_093054_RadDINO_Infer.pth         # Inference model
```

### With Extra Information

```
20250611_093054_ResNet50_Tr_batch64.pth       # Training with batch size info
20250611_093054_RadDINO_Eval_518x518.pth      # Evaluation with resolution info
20250611_093054_ViT-B_Infer_aug_heavy.pth     # Inference with augmentation tag
20250611_152330_CheXNet_Tr_optuna_trial42.pth # Hyperparameter optimization trial
```

## Implementation

The naming format is implemented through utility functions in `src/utils.py`:

### `make_model_name(cfg, task_tag, extra_info="")`

Creates a complete model name with the structured format.

```python
from src.utils import make_model_name
from src.config import cfg

# Basic training model name
model_name = make_model_name(cfg, "Tr")
# Result: "20250611_093054_ResNet50_Tr"

# Evaluation with extra info
eval_name = make_model_name(cfg, "Eval", "batch64")
# Result: "20250611_093054_ResNet50_Eval_batch64"
```

### `make_run_name(cfg)`

Creates a run name for WandB and other tracking systems (defaults to training task).

```python
from src.utils import make_run_name
from src.config import cfg

run_name = make_run_name(cfg)
# Result: "20250611_093054_ResNet50_Tr"
```

## Migration from Old Format

### Old Format (Deprecated)
```
ResNet50 - 2024-01-01 12:30:45.pth
```

### New Format
```
20250111_123045_ResNet50_Tr.pth
```

### Benefits of New Format

1. **Sortable**: Timestamps sort chronologically
2. **Parseable**: Easy to extract components programmatically
3. **Consistent**: Uniform structure across all model types
4. **Compact**: Shorter names, easier to work with
5. **Informative**: Clear task identification
6. **Extensible**: Optional extra info for specific needs

## File Organization

With the new naming format, models are naturally organized:

```
models/
├── 20250610_141230_ResNet50_Tr.pth
├── 20250610_141230_ResNet50_Eval.pth
├── 20250610_152000_RadDINO_Tr_batch32.pth
├── 20250610_152000_RadDINO_Eval_batch32.pth
├── 20250611_093054_ViT-B_Tr.pth
└── 20250611_093054_ViT-B_Eval.pth
```

Models from the same session (timestamp) are grouped together, making it easy to match training and evaluation artifacts.

## Usage in Code

### Training
```python
# In training loop when saving best model
model_name = make_model_name(config, "Tr")
save_path = config.get_model_path(model_name)
```

### Evaluation
```python
# When running evaluation
run_name = make_model_name(config, "Eval")
wandb.init(project="csi-predictor-eval", name=run_name)
```

### Hyperparameter Optimization
```python
# For hyperparameter trials
trial_info = f"trial{trial.number}"
model_name = make_model_name(config, "Tr", trial_info)
```

This structured approach ensures consistent, informative, and manageable model naming across the entire CSI-Predictor project. 