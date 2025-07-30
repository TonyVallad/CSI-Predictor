<div align="center">

# Image Preprocessing

This guide covers image preprocessing techniques used in CSI-Predictor.

</div>

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

## 3. **Current Preprocessing Pipeline (V2.0 - NIFTI Format)**

1. **Image Loading**: 
   - Images loaded as NIFTI files (.nii.gz) from disk
   - Uses nibabel: `nib.load(image_path).get_fdata().astype(np.float32)`
   - Coordinate corrections applied: `np.transpose()` + `np.fliplr()`

2. **Value Preprocessing**:
   - Hounsfield Units already clipped to 99th percentile (applied during NIFTI creation)
   - Normalized to 0-1 range: `(img_data - min) / (max - min)`
   - Converted to PIL Image format for transform compatibility

3. **Resizing**: 
   - Model-specific sizes (RadDINO: 518×518, others: mostly 224×224)

4. **Channel Conversion**: 
   - Single-channel NIFTI converted to 3-channel RGB for model compatibility

5. **Data Augmentation** (Training only):
   - Random horizontal flip (50% probability)
   - Random rotation (±10 degrees)
   - Color jitter (brightness/contrast ±10%)

6. **Tensor Conversion**: 
   - PIL Image → PyTorch tensor

7. **Normalization** (Configurable):
   - **Medical** (default): `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`
   - **ImageNet**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
   - **Simple**: `mean=[0.0, 0.0, 0.0]`, `std=[1.0, 1.0, 1.0]`
   - **Custom**: User-specified values via configuration

### 4. **Notable Design Choices (V2.0)**

- **NIFTI Format**: Preserves full diagnostic precision with float32 Hounsfield Units
- **Coordinate Corrections**: Automatic handling of NIFTI orientation differences
- **Medical-First Normalization**: Default to medical-appropriate normalization instead of ImageNet
- **RadDINO gets larger input**: 518×518 vs 224×224 for others
- **Configurable Normalization**: Supports multiple strategies for different model requirements
- **Value Range Preservation**: Full HU range maintained until final normalization step

### 5. **Improved RadDINO Handling**

The V2.0 implementation properly supports RadDINO's official preprocessing:

```python
# RadDINO with official processor (recommended)
use_official_processor = True  # Uses AutoImageProcessor from microsoft/rad-dino

# RadDINO with standard transforms (fallback)
use_official_processor = False  # Uses standard PyTorch transforms
```

### 6. **Configuration Options**

Set normalization strategy in your configuration:

```python
# Medical imaging default (recommended)
normalization_strategy = "medical"

# Traditional computer vision
normalization_strategy = "imagenet"

# Simple 0-1 normalization
normalization_strategy = "simple"

# Custom values
normalization_strategy = "custom"
custom_mean = [0.5, 0.5, 0.5]
custom_std = [0.3, 0.3, 0.3]
```