<div align="center">

# Project Structure Analysis

Analysis of the current CSI-Predictor project structure and proposed reorganization.

</div>

## Overview
The CSI-Predictor is a modular PyTorch project for predicting 6-zone CSI (Chest Severity Index) scores on chest X-ray images. The project uses deep learning models to analyze medical images and provide severity assessments across different lung zones.

## Current Project Structure

### Root Level Python Files

#### `main.py` (172 lines)
**Purpose**: Main entry point for the CSI-Predictor application
**Functions**:
- `main()`: Orchestrates training, evaluation, and hyperparameter optimization workflows

#### `debug_images.py` (101 lines)
**Purpose**: Debug script to visualize NIFTI images and check orientation
**Functions**:
- `main()`: Creates data loaders and visualizes sample images for debugging

#### `diagnose_raddino.py` (125 lines)
**Purpose**: Diagnostic script for RadDINO availability issues
**Functions**:
- `main()`: Checks RadDINO dependencies and availability

#### `download_archimed_images.py` (629 lines)
**Purpose**: Downloads medical images from ArchiMed system
**Classes**:
- `ArchiMedDownloader`: Main downloader class
**Methods**:
- `__init__()`: Initialize downloader with configuration
- `import_csv_to_dataframe()`: Import CSV file into DataFrame
- `collect_metadata()`: Collect metadata for files
- `convert_dicom_to_png()`: Convert DICOM files to PNG format
- `download_archimed_files()`: Download files from ArchiMed
- `run()`: Execute the complete download workflow

#### `test_raddino_processor.py` (100 lines)
**Purpose**: Test script for RadDINO processor functionality
**Functions**:
- `test_processor_availability()`: Test processor availability
- `test_transforms()`: Test image transformations
- `test_config()`: Test configuration settings
- `main()`: Run all tests

#### `test_metrics.py` (1 line)
**Purpose**: Test file for metrics functionality (minimal content)

### Source Directory (`src/`)

#### `src/__init__.py` (5 lines)
**Purpose**: Package initialization
**Content**: Version information and package description

#### `src/__main__.py` (36 lines)
**Purpose**: CLI entry point for src module
**Functions**:
- Command-line interface for train and evaluate commands

#### `src/config.py` (680 lines)
**Purpose**: Centralized configuration management
**Classes**:
- `Config`: Immutable configuration dataclass
- `ConfigLoader`: Configuration loading and validation
**Functions**:
- `get_config()`: Get configuration instance
- `copy_config_on_training_start()`: Copy config with timestamp

#### `src/data.py` (841 lines)
**Purpose**: Data pipeline and dataset management
**Functions**:
- `get_normalization_parameters()`: Get normalization mean/std values
- `get_raddino_processor()`: Get RadDINO image processor
- `load_csv_data()`: Load CSV data with specific columns
- `filter_existing_files()`: Filter DataFrame to only include existing files
- `convert_nans_to_unknown()`: Convert NaN values to unknown class
- `create_stratification_key()`: Create stratification key for splitting
- `split_data_stratified()`: Split data with stratification
- `get_default_transforms()`: Get default image transformations
- `load_and_split_data()`: Load and split data into train/val/test
- `create_data_loaders()`: Create PyTorch data loaders
**Classes**:
- `CSIDataset`: Custom PyTorch dataset for CSI data
**Methods**:
- `__init__()`: Initialize dataset
- `_load_nifti_image()`: Load NIFTI image file
- `_cache_images()`: Cache images in memory
- `__len__()`: Get dataset length
- `_get_image_filename()`: Get image filename
- `__getitem__()`: Get dataset item

#### `src/data_split.py` (212 lines)
**Purpose**: Data splitting utilities
**Functions**:
- `create_stratification_groups()`: Create stratification groups
- `stratified_split_indices()`: Create stratified split indices
- `pytorch_train_test_split()`: PyTorch-compatible train/test split
- `pytorch_train_val_test_split()`: PyTorch-compatible train/val/test split

#### `src/train.py` (1260 lines)
**Purpose**: Main training pipeline
**Functions**:
- `set_random_seeds()`: Set random seeds for reproducibility
- `compute_f1_metrics()`: Compute F1 metrics
- `compute_precision_recall()`: Compute precision and recall
- `compute_csi_average_metrics()`: Compute CSI average metrics
- `compute_ahf_classification_metrics()`: Compute AHF classification metrics
- `save_ahf_confusion_matrices()`: Save AHF confusion matrices
- `create_results_analysis_csv()`: Create results analysis CSV
- `train_epoch()`: Train for one epoch
- `validate_epoch()`: Validate for one epoch
- `train_model()`: Main training function
- `main()`: Training entry point

#### `src/train_optimized.py` (222 lines)
**Purpose**: Training with optimized hyperparameters
**Functions**:
- `load_best_hyperparameters()`: Load best hyperparameters from file
- `create_optimized_config()`: Create optimized configuration
- `train_with_optimized_hyperparameters()`: Train with optimized hyperparameters
- `main()`: Entry point for optimized training

#### `src/evaluate.py` (1014 lines)
**Purpose**: Model evaluation pipeline
**Functions**:
- `load_trained_model()`: Load trained model
- `compute_confusion_matrices_per_zone()`: Compute confusion matrices per zone
- `create_classification_report_per_zone()`: Create classification report per zone
- `save_confusion_matrix_graphs()`: Save confusion matrix graphs
- `log_to_wandb()`: Log results to Weights & Biases
- `evaluate_model_on_loader()`: Evaluate model on data loader
- `compute_zone_metrics()`: Compute metrics per zone
- `compute_overall_metrics()`: Compute overall metrics
- `create_evaluation_report()`: Create evaluation report
- `save_predictions()`: Save model predictions
- `evaluate_model()`: Main evaluation function
- `main()`: Evaluation entry point

#### `src/metrics.py` (481 lines)
**Purpose**: Pure PyTorch metrics implementation
**Functions**:
- `compute_confusion_matrix()`: Compute confusion matrix using PyTorch
- `compute_f1_from_confusion_matrix()`: Compute F1 from confusion matrix
- `compute_pytorch_f1_metrics()`: Compute F1 metrics using PyTorch
- `compute_accuracy()`: Compute accuracy metrics
- `compute_precision_recall_metrics()`: Compute precision/recall metrics
- `compute_per_class_f1_scores()`: Compute per-class F1 scores
- `compute_enhanced_f1_metrics()`: Compute enhanced F1 metrics
- `diagnose_f1_issues()`: Diagnose F1 calculation issues

#### `src/hyperopt.py` (913 lines)
**Purpose**: Hyperparameter optimization using Optuna
**Functions**:
- `get_cached_data_loaders()`: Get cached data loaders
- `create_optuna_config()`: Create Optuna configuration
- `objective()`: Optuna objective function
- `create_study()`: Create Optuna study
- `save_best_hyperparameters()`: Save best hyperparameters
- `clear_data_cache()`: Clear data cache
- `get_search_space_info()`: Get search space information
- `optimize_hyperparameters()`: Main optimization function
- `main()`: Hyperparameter optimization entry point

#### `src/wandb_sweep.py` (555 lines)
**Purpose**: Weights & Biases sweep integration
**Functions**:
- `get_cached_data_loaders()`: Get cached data loaders
- `get_sweep_config()`: Get sweep configuration
- `train_sweep_run()`: Train sweep run
- `initialize_sweep()`: Initialize W&B sweep
- `run_sweep_agent()`: Run sweep agent
- `create_and_run_sweep()`: Create and run sweep
- `main()`: Sweep entry point

#### `src/discord_notifier.py` (380 lines)
**Purpose**: Discord notification system
**Functions**:
- `send_training_notification()`: Send training completion notification
- `send_evaluation_notification()`: Send evaluation completion notification

#### `src/utils.py` (2327 lines)
**Purpose**: Utility functions for the entire project
**Functions**:
- `setup_logging()`: Setup logging configuration
- `set_seed()`: Set random seed
- `count_parameters()`: Count model parameters
- `get_learning_rate()`: Get current learning rate
- `save_checkpoint()`: Save model checkpoint
- `load_checkpoint()`: Load model checkpoint
- `calculate_class_weights()`: Calculate class weights
- `format_time()`: Format time duration
- `print_model_summary()`: Print model summary
- `create_dirs()`: Create directories
- `show_batch()`: Show batch of images
- `visualize_data_distribution()`: Visualize data distribution
- `analyze_missing_data()`: Analyze missing data
- `create_debug_dataset()`: Create debug dataset
- `make_model_name()`: Make model name
- `make_run_name()`: Make run name
- `create_model_name_from_existing()`: Create model name from existing
- `seed_everything()`: Seed everything for reproducibility
- `pretty_print_config()`: Pretty print configuration
- `print_config()`: Print configuration
- `log_config()`: Log configuration
- `create_roc_curves()`: Create ROC curves
- `create_precision_recall_curves()`: Create precision-recall curves
- `plot_training_curves()`: Plot training curves
- `create_confusion_matrix_grid()`: Create confusion matrix grid
- `create_roc_curves_grid()`: Create ROC curves grid
- `create_precision_recall_curves_grid()`: Create precision-recall curves grid
- `plot_training_curves_grid()`: Plot training curves grid
- `create_overall_confusion_matrix()`: Create overall confusion matrix
- `create_summary_dashboard()`: Create summary dashboard
- `save_training_history()`: Save training history
- `load_training_history()`: Load training history

### Models Directory (`src/models/`)

#### `src/models/__init__.py` (462 lines)
**Purpose**: Model factory and initialization
**Functions**:
- `build_model()`: Build complete model
- `build_zone_focus_model()`: Build zone focus model
- `build_zone_masking_model()`: Build zone masking model

#### `src/models/backbones.py` (364 lines)
**Purpose**: Feature extraction backbones
**Classes**:
- `CustomCNNBackbone`: Simple 5-layer CNN backbone
- `CheXNetBackbone`: DenseNet121 adapted for chest X-rays
- `ResNet50Backbone`: ResNet50 backbone
- `RadDINOBackbone`: RadDINO backbone wrapper
**Functions**:
- `get_backbone()`: Get backbone by name
- `get_backbone_feature_dim()`: Get backbone feature dimension
- `diagnose_raddino_availability()`: Diagnose RadDINO availability

#### `src/models/head.py` (101 lines)
**Purpose**: Classification heads
**Classes**:
- `CSIHead`: CSI classification head with 6 parallel zone classifiers
- `CSIRegressionHead`: CSI regression head for backward compatibility

#### `src/models/rad_dino.py` (141 lines)
**Purpose**: RadDINO model implementation
**Classes**:
- `RadDINOCSIModel`: Complete RadDINO model for CSI prediction
- `RadDINOBackboneOnly`: RadDINO backbone for standard CSI head

## Suggested Project Reorganization

### Current Issues Identified

1. **Monolithic Files**: Several files are extremely large (utils.py: 2327 lines, train.py: 1260 lines, data.py: 841 lines)
2. **Mixed Responsibilities**: Files contain multiple unrelated functions
3. **Poor Separation of Concerns**: Training, evaluation, and utilities are mixed
4. **Inconsistent Organization**: Some functionality is scattered across multiple files
5. **Root-level Scripts**: Debug and utility scripts are in the root directory

### Proposed New Structure

```
CSI-Predictor/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py              # Main config class
│   │   ├── config_loader.py       # Config loading logic
│   │   └── validation.py          # Config validation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # CSIDataset class
│   │   ├── dataloader.py          # Data loader creation
│   │   ├── transforms.py          # Image transformations
│   │   ├── preprocessing.py       # Data preprocessing
│   │   └── splitting.py           # Data splitting utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py             # Model factory
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base backbone class
│   │   │   ├── resnet.py          # ResNet backbones
│   │   │   ├── densenet.py        # DenseNet/CheXNet backbones
│   │   │   ├── custom.py          # Custom CNN backbone
│   │   │   └── raddino.py         # RadDINO backbone
│   │   ├── heads/
│   │   │   ├── __init__.py
│   │   │   ├── csi_head.py        # CSI classification head
│   │   │   └── regression_head.py # Regression head
│   │   └── complete/
│   │       ├── __init__.py
│   │       └── raddino_csi.py     # Complete RadDINO model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training logic
│   │   ├── optimizer.py           # Optimizer management
│   │   ├── scheduler.py           # Learning rate scheduling
│   │   └── callbacks.py           # Training callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Main evaluation logic
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── classification.py  # Classification metrics
│   │   │   ├── confusion_matrix.py # Confusion matrix utilities
│   │   │   └── f1_score.py        # F1 score calculations
│   │   └── visualization/
│   │       ├── __init__.py
│   │       ├── plots.py           # Plotting utilities
│   │       ├── confusion_matrix.py # Confusion matrix plots
│   │       └── training_curves.py # Training curve plots
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── hyperopt.py            # Optuna hyperparameter optimization
│   │   └── wandb_sweep.py         # W&B sweep integration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py             # Logging setup
│   │   ├── checkpoint.py          # Model checkpointing
│   │   ├── visualization.py       # General visualization
│   │   ├── file_utils.py          # File operations
│   │   └── seed.py                # Random seed management
│   └── cli/
│       ├── __init__.py
│       ├── main.py                # Main CLI entry point
│       ├── train.py               # Training CLI
│       ├── evaluate.py            # Evaluation CLI
│       └── optimize.py            # Optimization CLI
├── scripts/
│   ├── debug/
│   │   ├── debug_images.py        # Image debugging
│   │   └── diagnose_raddino.py    # RadDINO diagnostics
│   ├── data/
│   │   └── download_archimed.py   # ArchiMed downloader
│   └── tests/
│       ├── test_metrics.py        # Metrics tests
│       └── test_raddino.py        # RadDINO tests
├── config/
│   ├── config.ini                 # Main configuration
│   └── config_example.ini         # Example configuration
├── main.py                        # Legacy entry point (deprecated)
└── requirements.txt
```

### Benefits of Reorganization

1. **Modularity**: Each module has a single responsibility
2. **Maintainability**: Smaller, focused files are easier to maintain
3. **Testability**: Isolated components are easier to test
4. **Reusability**: Components can be easily reused across the project
5. **Scalability**: New features can be added without affecting existing code
6. **Documentation**: Clear structure makes it easier to understand and document

### Migration Strategy

1. **Phase 1**: Create new directory structure
2. **Phase 2**: Extract functions from large files into appropriate modules
3. **Phase 3**: Update imports and dependencies
4. **Phase 4**: Move scripts to appropriate directories
5. **Phase 5**: Update documentation and examples
6. **Phase 6**: Remove deprecated files

### Key Principles for Reorganization

1. **Single Responsibility**: Each file should have one clear purpose
2. **Dependency Inversion**: High-level modules should not depend on low-level modules
3. **Interface Segregation**: Clients should not be forced to depend on interfaces they don't use
4. **Open/Closed Principle**: Open for extension, closed for modification
5. **DRY (Don't Repeat Yourself)**: Eliminate code duplication
6. **KISS (Keep It Simple, Stupid)**: Prefer simple solutions over complex ones

This reorganization will significantly improve the project's maintainability, testability, and scalability while making it easier for new contributors to understand and work with the codebase. 