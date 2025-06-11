# Model Architectures

Comprehensive guide to all supported model architectures in CSI-Predictor.

## Overview

CSI-Predictor supports multiple state-of-the-art architectures optimized for medical image analysis:

| Model Family | Architectures | Best For |
|--------------|---------------|----------|
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152 | General baseline, fast training |
| **EfficientNet** | efficientnet_b0 to efficientnet_b4 | Efficient computation |
| **DenseNet** | densenet121, densenet169, densenet201 | Feature reuse |
| **Medical Specialized** | CheXNet, RadDINO | Medical imaging |
| **Vision Transformers** | vit_base_patch16_224, vit_large_patch16_224 | Advanced features |
| **Custom** | Custom_1 | Fast prototyping |

## Quick Selection Guide

### For Beginners
```ini
[MODEL]
MODEL_ARCH = resnet50  # Fast, reliable baseline
```

### For Medical Domain
```ini
[MODEL]
MODEL_ARCH = chexnet  # Pre-trained on chest X-rays
```

### For Best Performance
```ini
[MODEL]
MODEL_ARCH = raddino  # State-of-the-art for chest X-rays
USE_OFFICIAL_PROCESSOR = true
```

### For Quick Testing
```ini
[MODEL]
MODEL_ARCH = custom1  # Lightweight, fast training
```

## Detailed Architecture Guide

### ResNet Family

**ResNet50** (Recommended Baseline)
- **Parameters**: ~23M
- **Input Size**: 224×224
- **Memory**: ~2GB GPU
- **Training Time**: Fast
- **Best For**: General purpose, baseline experiments

```ini
[MODEL]
MODEL_ARCH = resnet50

[TRAINING]
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

**Other ResNet Variants**:
- `resnet18`: Lighter (11M params), faster training
- `resnet34`: Medium size (21M params)
- `resnet101`: Deeper (44M params), potentially better accuracy
- `resnet152`: Deepest (60M params), slower but powerful

### Medical Domain Models

**CheXNet** (Medical Specialist)
- **Base**: DenseNet121 pre-trained on CheXpert dataset
- **Parameters**: ~7M
- **Input Size**: 224×224
- **Specialty**: Chest X-ray pathology detection
- **Best For**: Medical imaging tasks requiring domain knowledge

```ini
[MODEL]
MODEL_ARCH = chexnet

[TRAINING]
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning
```

**RadDINO** (State-of-the-Art)
- **Base**: Vision Transformer specialized for radiology
- **Parameters**: ~86M
- **Input Size**: 518×518 (higher resolution)
- **Specialty**: Microsoft's chest X-ray specialist model
- **Best For**: Production deployment, highest accuracy

```ini
[MODEL]
MODEL_ARCH = raddino
USE_OFFICIAL_PROCESSOR = true

[TRAINING]
BATCH_SIZE = 16  # Requires more memory
LEARNING_RATE = 0.00001  # Very low LR
OPTIMIZER = adamw
```

### EfficientNet Family

**EfficientNet-B0 to B4**
- **Principle**: Efficient scaling of depth, width, and resolution
- **Parameters**: 5M (B0) to 19M (B4)
- **Input Size**: 224×224 (B0) to 380×380 (B4)
- **Best For**: Resource-constrained environments

```ini
# EfficientNet-B2 (good balance)
[MODEL]
MODEL_ARCH = efficientnet_b2

[TRAINING]
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

### Vision Transformers

**ViT-Base**
- **Parameters**: ~86M
- **Input Size**: 224×224
- **Patch Size**: 16×16
- **Best For**: Learning complex visual relationships

```ini
[MODEL]
MODEL_ARCH = vit_base_patch16_224

[TRAINING]
BATCH_SIZE = 16  # Requires more memory
LEARNING_RATE = 0.0001
```

### Custom Architecture

**Custom_1** (Simple Baseline)
- **Design**: 5-layer CNN
- **Parameters**: ~1M
- **Input Size**: 224×224
- **Best For**: Quick experiments, educational purposes

```ini
[MODEL]
MODEL_ARCH = custom1

[TRAINING]
BATCH_SIZE = 64  # Can use larger batches
LEARNING_RATE = 0.01
N_EPOCHS = 20    # Faster training
```

## Performance Comparison

### Accuracy vs Speed

| Model | Accuracy | Speed | Memory | Parameters |
|-------|----------|-------|--------|------------|
| Custom_1 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1M |
| ResNet50 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 23M |
| CheXNet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 7M |
| EfficientNet-B2 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 9M |
| RadDINO | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 86M |
| ViT-Base | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 86M |

### GPU Memory Requirements

| Model | Minimum GPU | Recommended GPU | Max Batch Size |
|-------|-------------|-----------------|----------------|
| Custom_1 | 2GB | 4GB | 128 |
| ResNet50 | 4GB | 8GB | 64 |
| CheXNet | 4GB | 8GB | 64 |
| EfficientNet-B2 | 4GB | 8GB | 48 |
| RadDINO | 8GB | 16GB | 16 |
| ViT-Base | 8GB | 16GB | 16 |

## Model Selection Strategy

### 1. Development Phase
```bash
# Start with fast baseline
MODEL_ARCH=custom1  # Quick iteration
# Then move to
MODEL_ARCH=resnet50  # Reliable baseline
```

### 2. Optimization Phase
```bash
# Try medical domain models
MODEL_ARCH=chexnet   # Domain knowledge
# Or efficient models
MODEL_ARCH=efficientnet_b2  # Efficiency
```

### 3. Production Phase
```bash
# Use best performing model
MODEL_ARCH=raddino   # State-of-the-art
# With careful optimization
```

## Advanced Configuration

### RadDINO Specific Settings

```ini
[MODEL]
MODEL_ARCH = raddino
USE_OFFICIAL_PROCESSOR = true  # Use Microsoft's preprocessing

[TRAINING]
BATCH_SIZE = 16          # Adjust based on GPU memory
LEARNING_RATE = 0.00001  # Very conservative
OPTIMIZER = adamw        # Preferred for transformers
N_EPOCHS = 50           # May converge faster
PATIENCE = 15           # More patience for large models
```

### Memory Optimization

```ini
# For limited GPU memory
[TRAINING]
BATCH_SIZE = 8           # Reduce batch size
ACCUMULATION_STEPS = 4   # Maintain effective batch size

# For CPU training
[TRAINING]
BATCH_SIZE = 16          # CPU can handle moderate batches
```

### Transfer Learning Settings

```ini
# Fine-tuning pre-trained models
[TRAINING]
LEARNING_RATE = 0.0001   # Lower for pre-trained
FREEZE_BACKBONE = false  # Allow backbone training
WARMUP_EPOCHS = 5        # Gradual unfreezing
```

## Implementation Details

### Model Creation

```python
from src.models import build_model
from src.config import cfg

# Create model based on configuration
model = build_model(cfg)

# Model information
print(f"Model: {cfg.model_arch}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Architecture Details

```python
# View model architecture
from torchsummary import summary

model = build_model(cfg)
summary(model, input_size=(3, 224, 224))  # Adjust size for RadDINO: (3, 518, 518)
```

### Model Comparison

```python
import time
import torch

def benchmark_model(model_name, num_runs=10):
    """Benchmark model speed and memory."""
    cfg.model_arch = model_name
    model = build_model(cfg).cuda()
    
    # Dummy input
    input_size = (518, 518) if model_name == 'raddino' else (224, 224)
    x = torch.randn(1, 3, *input_size).cuda()
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"{model_name:15s}: {avg_time*1000:6.1f}ms, {memory_mb:6.1f}MB")

# Benchmark all models
for model in ['custom1', 'resnet50', 'chexnet', 'raddino']:
    benchmark_model(model)
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE = 8

# Use gradient checkpointing (if available)
GRADIENT_CHECKPOINTING = true

# Use CPU for very large models
DEVICE = cpu
```

#### RadDINO Not Available
```bash
# Install transformers
pip install transformers>=4.30.0

# Check availability
python -c "from src.models.backbones import RADDINO_AVAILABLE; print(f'RadDINO available: {RADDINO_AVAILABLE}')"
```

#### Slow Training
```bash
# Use lighter model
MODEL_ARCH = resnet50  # instead of raddino

# Increase batch size (if memory allows)
BATCH_SIZE = 64

# Enable mixed precision (automatic)
```

#### Poor Performance
```bash
# Try medical domain model
MODEL_ARCH = chexnet

# Use pre-trained weights (default)
PRETRAINED = true

# Adjust learning rate
LEARNING_RATE = 0.0001  # for pre-trained models
```

## Best Practices

### 1. Progressive Complexity
```bash
# Start simple, increase complexity
Custom_1 → ResNet50 → CheXNet → RadDINO
```

### 2. Domain Adaptation
```bash
# For medical imaging, prefer domain models
MODEL_ARCH = chexnet   # or raddino
```

### 3. Resource Management
```bash
# Match model to available resources
# 4GB GPU: resnet50, chexnet
# 8GB+ GPU: raddino, vit_base
# CPU only: custom1, resnet18
```

### 4. Hyperparameter Tuning
```bash
# Different models need different settings
# ResNet: LR=0.001, BATCH=32
# RadDINO: LR=0.00001, BATCH=16
# Custom: LR=0.01, BATCH=64
```

This comprehensive guide helps you choose the right architecture for your specific needs and constraints. 