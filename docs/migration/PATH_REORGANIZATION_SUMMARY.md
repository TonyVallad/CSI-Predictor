# Path Reorganization Summary

This document summarizes the changes made to reorganize the path configuration system in CSI-Predictor.

## Overview

The path configuration system has been reorganized to:
1. Centralize all paths in the `.env` file
2. Remove path variables from `config.ini`
3. Implement automatic subfolder generation under a master `DATA_DIR`
4. Rename `DATA_DIR` to `NIFTI_DIR` for clarity
5. Add new path variables for better organization

## Changes Made

### 1. New .env Structure

The `.env` file now supports the following path variables:

```bash
# Discord Webhook for model results
DISCORD_WEBHOOK_URL=

# Device configuration
DEVICE=cuda

# Data loading configuration
LOAD_DATA_TO_MEMORY=True

# Data source and paths
DATA_DIR=/home/pyuser/data/Paradise
INI_DIR= (will be DATA_DIR + "/config/" by default)
CSV_DIR= (will be DATA_DIR + "/csv/" by default)
NIFTI_DIR= (will be DATA_DIR + "/nifti/" by default)
PNG_DIR= (will be DATA_DIR + "/png/" by default)
MODELS_DIR=./models (would be DATA_DIR + "/models/" by default)
GRAPH_DIR= (will be DATA_DIR + "/graphs/" by default)
DEBUG_DIR= (will be DATA_DIR + "/debug/" by default)
MASKS_DIR= (will be DATA_DIR + "/masks/" by default)
LOGS_DIR=./logs (would be DATA_DIR + "/logs/" by default)
RUNS_DIR= (will be DATA_DIR + "/runs/" by default)
EVALUATION_DIR= (will be DATA_DIR + "/evaluations/" by default)
WANDB_DIR=./wandb (would be DATA_DIR + "/wandb/" by default)

# Labels configuration
LABELS_CSV=Labeled_Data_RAW.csv (would be "labels.csv" by default)
LABELS_CSV_SEPARATOR=; (default value)
```

### 2. Path Resolution Logic

- If a path variable is empty or not set, it automatically becomes a subfolder of `DATA_DIR`
- If a path variable is set, it uses the specified path
- Some paths (MODELS_DIR, LOGS_DIR, WANDB_DIR) default to relative paths if not set

### 3. Configuration Class Updates

#### New Path Variables Added:
- `nifti_dir`: Directory containing NIFTI images (renamed from `data_dir`)
- `png_dir`: Directory for PNG images
- `masks_dir`: Directory for segmentation masks (renamed from `masks_path`)
- `runs_dir`: Directory for training runs
- `evaluation_dir`: Directory for evaluation outputs
- `wandb_dir`: Directory for Weights & Biases files

#### Legacy Properties:
- `data_path`: Returns `nifti_dir` for backward compatibility
- `masks_path`: Returns `masks_dir` for backward compatibility

### 4. Files Updated

#### Core Configuration:
- `src/config/config.py`: Updated Config class with new path variables
- `src/config/config_loader.py`: Added path resolution logic
- `src/config/validation.py`: Updated validation for new paths
- `src/config/__init__.py`: Updated initialization and helper functions

#### Training & Optimization:
- `src/training/train_optimized.py`: Updated to use new path variables
- `src/training/trainer.py`: Updated directory creation
- `src/optimization/wandb_sweep.py`: Updated path references
- `src/optimization/hyperopt.py`: Updated path references

#### Models:
- `src/models/complete/csi_models.py`: Updated to use `masks_dir`

#### Configuration Files:
- `config/config.ini`: Removed `MASKS_PATH` from ZONES section

#### Legacy Support:
- `src/config.py`: Updated legacy config for backward compatibility

### 5. Backward Compatibility

The system maintains backward compatibility through:
- Legacy properties that map old names to new ones
- Legacy configuration loader that supports both old and new systems
- Automatic migration of old path references

## Migration Guide

### For Users:

1. **Update your `.env` file** with the new structure:
   ```bash
   # Set your master data directory
   DATA_DIR=/path/to/your/data
   
   # Leave other paths empty to use subfolders, or set specific paths
   NIFTI_DIR=
   CSV_DIR=
   MASKS_DIR=
   # etc.
   ```

2. **Remove path variables from `config.ini`** - they should only contain non-path settings

3. **Test your configuration** - the system will automatically create subfolders as needed

### For Developers:

1. **Use new path variables** in new code:
   ```python
   from src.config import get_config
   config = get_config()
   
   # Use new path variables
   nifti_path = config.nifti_dir
   masks_path = config.masks_dir
   ```

2. **Legacy code will continue to work** but should be updated:
   ```python
   # Old way (still works)
   data_path = config.data_dir
   
   # New way (recommended)
   nifti_path = config.nifti_dir
   ```

## Benefits

1. **Centralized Configuration**: All paths in one place (`.env`)
2. **Automatic Organization**: Subfolders created automatically under `DATA_DIR`
3. **Flexibility**: Can override any path with specific values
4. **Clarity**: Better naming (`nifti_dir` vs `data_dir`)
5. **Backward Compatibility**: Existing code continues to work
6. **Maintainability**: Easier to manage and understand path structure

## Testing

The new configuration system has been implemented but not tested on the target machine. Please test:

1. Configuration loading with various `.env` setups
2. Automatic subfolder creation
3. Path resolution with empty vs. set variables
4. Backward compatibility with existing code
5. Training and evaluation workflows

## Future Improvements

1. Add validation for path creation permissions
2. Add automatic cleanup of empty directories
3. Add path migration utilities
4. Add configuration validation for path relationships
5. Add documentation for path organization best practices 