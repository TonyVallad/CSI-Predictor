
## Image Preprocessing Pipeline

### 1. **Input Image Size Mapping**
Your app uses model-specific input sizes defined in `MODEL_INPUT_SIZES`:

```12:60:src/data.py
# Model architecture to input size mapping
MODEL_INPUT_SIZES = {
    'resnet18': (224, 224),
    'resnet34': (224, 224),
    'resnet50': (224, 224),
    'resnet101': (224, 224),
    'resnet152': (224, 224),
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b2': (260, 260),
    'efficientnet_b3': (300, 300),
    'efficientnet_b4': (380, 380),
    'densenet121': (224, 224),
    'densenet169': (224, 224),
    'densenet201': (224, 224),
    'vit_base_patch16_224': (224, 224),
    'vit_large_patch16_224': (224, 224),
    'chexnet': (224, 224),
    'custom1': (224, 224),
    'raddino': (518, 518),  # RadDINO's expected input size from Microsoft
}
```

### 2. **Different Preprocessing for RadDINO vs Other Models**

#### **RadDINO Preprocessing (518×518)**:
```233:250:src/data.py
if model_arch.lower().replace('_', '').replace('-', '') == 'raddino':
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            # RadDINO uses ImageNet normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            # RadDINO uses ImageNet normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
```

#### **Standard Models Preprocessing (mostly 224×224)**:
```252:270:src/data.py
if phase == "train":
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:  # val or test
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

### 3. **Step-by-Step Preprocessing**

1. **Image Loading**: 
   - Images loaded as PNG files from disk
   - Converted to RGB format: `Image.open(image_path).convert('RGB')`

2. **Resizing**: 
   - Model-specific sizes (RadDINO: 518×518, others: mostly 224×224)

3. **Channel Conversion**: 
   - Grayscale X-rays converted to 3-channel RGB for model compatibility

4. **Data Augmentation** (Training only):
   - Random horizontal flip (50% probability)
   - Random rotation (±10 degrees)
   - Color jitter (brightness/contrast ±10%)

5. **Tensor Conversion**: 
   - PIL Image → PyTorch tensor

6. **Normalization**: 
   - **All models use ImageNet normalization**:
     - Mean: `[0.485, 0.456, 0.406]`
     - Std: `[0.229, 0.224, 0.225]`

### 4. **Notable Design Choices**

- **RadDINO gets larger input**: 518×518 vs 224×224 for others
- **Same normalization for all**: Despite RadDINO being chest X-ray specific, it still uses ImageNet normalization
- **No RadDINO-specific processor**: The code loads RadDINO's `AutoImageProcessor` but doesn't use it - instead uses standard PyTorch transforms
- **Grayscale to RGB conversion**: All models expect 3-channel input even for grayscale X-rays

### 5. **Potential Issue with RadDINO**

There's a discrepancy in your RadDINO implementation: the model loads the official `AutoImageProcessor` but doesn't use it:

```35:35:src/models/rad_dino.py
self.processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
```

However, the actual preprocessing uses standard PyTorch transforms instead of this processor. This might be causing suboptimal performance since RadDINO was likely trained with its specific preprocessing pipeline.

The preprocessing is consistent across training/validation/testing phases, with data augmentation only applied during training.