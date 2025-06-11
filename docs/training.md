# Training Models

This guide covers everything you need to know about training CSI prediction models.

## Quick Training

```bash
# Basic training with default settings
python -m src.train

# Train with custom configuration
python -m src.train --ini my_config.ini

# Legacy entry point
python main.py --mode train
```

## Training Features

### 🎯 Multi-Architecture Support
- **ResNet Family**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet_b0 through efficientnet_b4
- **DenseNet**: densenet121, densenet169, densenet201
- **Medical Models**: CheXNet (DenseNet121 on CheXpert), RadDINO
- **Vision Transformers**: vit_base_patch16_224, vit_large_patch16_224
- **Custom**: Custom_1 (5-layer CNN baseline)

### 📊 Comprehensive Metrics
- **Per-Zone F1 Scores**: Individual metrics for all 6 chest zones
- **Overall Accuracy**: Combined performance across all zones
- **Classification Reports**: Precision, recall, F1 per congestion class
- **Real-time Tracking**: Live metrics via Weights & Biases

### 🔄 Advanced Training Features
- **Early Stopping**: Automatic stopping based on validation performance
- **Data Augmentation**: Configurable transforms for robustness
- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Stability for large models like RadDINO
- **Learning Rate Scheduling**: Adaptive learning rate adjustments

## Model Architecture Configuration

### ResNet50 (Recommended Baseline)
```ini
[MODEL]
MODEL_ARCH = resnet50

[TRAINING]
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = adam
```

**Best for**: Fast training, good baseline performance, low memory usage

### CheXNet (Medical Domain)
```ini
[MODEL]
MODEL_ARCH = chexnet

[TRAINING]
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
OPTIMIZER = adam
```

**Best for**: Medical imaging tasks, domain-adapted features

### RadDINO (State-of-the-Art)
```ini
[MODEL]
MODEL_ARCH = raddino
USE_OFFICIAL_PROCESSOR = true

[TRAINING]
BATCH_SIZE = 16  # Requires more memory
LEARNING_RATE = 0.00001
OPTIMIZER = adamw
N_EPOCHS = 50
```

**Best for**: Highest accuracy, production deployment

### Custom_1 (Fast Prototyping)
```ini
[MODEL]
MODEL_ARCH = custom1

[TRAINING]
BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS = 20
```

**Best for**: Quick experiments, resource-constrained environments

## Training Configuration

### Basic Configuration
```ini
[TRAINING]
BATCH_SIZE = 32          # Adjust based on GPU memory
N_EPOCHS = 100           # Maximum epochs
PATIENCE = 10            # Early stopping patience
LEARNING_RATE = 0.001    # Initial learning rate
OPTIMIZER = adam         # adam, adamw, or sgd

[MODEL]
MODEL_ARCH = resnet50    # Model architecture
```

### GPU Memory Optimization
```ini
[TRAINING]
# For 8GB GPU
BATCH_SIZE = 32

# For 4GB GPU  
BATCH_SIZE = 16

# For 2GB GPU or CPU
BATCH_SIZE = 8
```

### Training Speed vs Quality
```ini
# Fast training (development)
[TRAINING]
BATCH_SIZE = 64
N_EPOCHS = 20
PATIENCE = 5

# Quality training (production)
[TRAINING]
BATCH_SIZE = 32
N_EPOCHS = 200
PATIENCE = 20
```

## Training Process

### Automatic Features

1. **Model Naming**: Models saved with structured timestamps
   - Format: `[YYYYMMDD_HHMMSS]_[ModelName]_[TaskTag]_[ExtraInfo]`
   - Example: `20250611_093054_ResNet50_Tr.pth`

2. **Best Model Saving**: Automatically saves the best performing model

3. **Logging**: Comprehensive logging to files and console
   - Location: `./logs/csi_predictor_YYYY-MM-DD.log`
   - Automatic rotation and compression

4. **Experiment Tracking**: Integration with Weights & Biases
   - Real-time metrics visualization
   - Model artifact storage
   - Hyperparameter logging

### Training Workflow

1. **Data Loading**
   ```
   Loading dataset from: /path/to/Paradise_Images
   Loading labels from: /path/to/Paradise_CSV/Labeled_Data_RAW.csv
   Total samples: XXXX
   Train: XX% | Validation: XX% | Test: XX%
   ```

2. **Model Initialization**
   ```
   Creating model: ResNet50
   Model parameters: X.X million
   Moving to device: cuda
   ```

3. **Training Loop**
   ```
   Epoch 1/100:
   Train Loss: 0.456 | Train F1: 0.789
   Val Loss: 0.423 | Val F1: 0.812
   Best model saved: 20250611_093054_ResNet50_Tr.pth
   ```

4. **Early Stopping**
   ```
   Early stopping triggered at epoch 45
   Best validation F1: 0.856 at epoch 37
   ```

## Monitoring Training

### Weights & Biases Dashboard

Access real-time training metrics:
```bash
# After training starts, you'll see:
wandb: 🚀 View run at https://wandb.ai/your-username/csi-predictor/runs/...
```

**Dashboard Features:**
- **Real-time Metrics**: Loss, F1-score, accuracy per zone
- **Learning Curves**: Training and validation performance over time
- **System Metrics**: GPU usage, memory consumption
- **Model Artifacts**: Automatic model checkpoint storage

### Local Monitoring

```bash
# View live training logs
tail -f logs/csi_predictor_$(date +%Y-%m-%d).log

# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
ls -la models/  # See saved model files
```

## Advanced Training Techniques

### Mixed Precision Training

```python
# Automatically enabled for supported models
# Reduces memory usage and increases speed
```

### Gradient Accumulation

```ini
[TRAINING]
# Effectively increases batch size without memory increase
ACCUMULATION_STEPS = 4  # If implemented
EFFECTIVE_BATCH_SIZE = 32  # BATCH_SIZE * ACCUMULATION_STEPS
```

### Learning Rate Scheduling

The system includes automatic learning rate scheduling:
- **Plateau Reduction**: Reduces LR when validation loss plateaus
- **Warmup**: Gradual learning rate increase at the start
- **Cosine Annealing**: Smooth learning rate decay

### Data Augmentation

Built-in augmentations for chest X-rays:
```python
# Training augmentations
- Random rotation (±10 degrees)
- Random horizontal flip
- Random brightness/contrast adjustment
- Normalization to ImageNet statistics

# Validation/Test (no augmentation)
- Resize to model input size
- Center crop
- Normalization only
```

## Troubleshooting Training

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE = 16  # or even 8

# Enable memory cleanup
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### Slow Training
```bash
# Ensure data is cached in memory
LOAD_DATA_TO_MEMORY = True

# Use multiple workers (if available)
NUM_WORKERS = 4

# Check GPU utilization
nvidia-smi
```

#### NaN Loss
```bash
# Reduce learning rate
LEARNING_RATE = 0.0001

# Add gradient clipping (automatic for RadDINO)
# Check for data preprocessing issues
```

#### Poor Performance
```bash
# Try different model architecture
MODEL_ARCH = raddino

# Increase training time
N_EPOCHS = 200
PATIENCE = 30

# Check data quality and labeling
```

### Performance Optimization

#### GPU Optimization
```bash
# Use appropriate batch size for your GPU
# 4GB GPU: BATCH_SIZE = 16
# 8GB GPU: BATCH_SIZE = 32
# 16GB+ GPU: BATCH_SIZE = 64

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### CPU Optimization
```bash
# Set appropriate number of workers
export OMP_NUM_THREADS=4

# Use CPU-optimized batch size
BATCH_SIZE = 16
```

## Training Best Practices

### 1. Start with Baseline
```bash
# Always start with ResNet50 baseline
MODEL_ARCH=resnet50
BATCH_SIZE=32
N_EPOCHS=50
```

### 2. Validate Setup
```bash
# Test with minimal training first
N_EPOCHS=3
PATIENCE=1
```

### 3. Scale Up Gradually
```bash
# Increase complexity step by step
# ResNet50 → CheXNet → RadDINO
```

### 4. Monitor Overfitting
- Watch validation vs training loss divergence
- Use early stopping (always enabled)
- Check per-zone performance consistency

### 5. Resource Management
```bash
# Clean up old models periodically
find models/ -name "*.pth" -mtime +7 -delete

# Monitor disk space
df -h
```

## Next Steps

After successful training:

1. **[Evaluate your model](evaluation.md)** on test data
2. **[Optimize hyperparameters](hyperparameter-optimization.md)** for better performance
3. **[Set up monitoring](monitoring-logging.md)** for production use
4. **[Understand model outputs](model-architectures.md)** in detail

## Training Examples

### Development Training
```bash
# Quick training for testing
cat > dev_config.ini << EOF
[TRAINING]
BATCH_SIZE = 16
N_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 0.01

[MODEL]
MODEL_ARCH = resnet50
EOF

python -m src.train --ini dev_config.ini
```

### Production Training
```bash
# Full training with RadDINO
cat > prod_config.ini << EOF
[TRAINING]
BATCH_SIZE = 32
N_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.0001

[MODEL]
MODEL_ARCH = raddino
USE_OFFICIAL_PROCESSOR = true
EOF

python -m src.train --ini prod_config.ini
``` 