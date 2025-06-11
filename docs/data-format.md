# Data Format Guide

This guide explains the expected data structure and format for CSI-Predictor.

## Data Structure Overview

```
/your/data/path/
├── Paradise_Images/          # Images directory (DATA_DIR)
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── image003.png
│   └── ...
└── Paradise_CSV/             # CSV directory (CSV_DIR)
    └── Labeled_Data_RAW.csv  # Labels file (LABELS_CSV)
```

## Image Requirements

### Supported Formats
- **JPEG** (.jpg, .jpeg) - Recommended
- **PNG** (.png) - Supported
- **TIFF** (.tiff, .tif) - Supported

### Image Specifications
- **Resolution**: Any resolution (automatically resized)
- **Color**: Grayscale or RGB (converted to RGB)
- **Orientation**: Portrait or landscape
- **File Size**: No strict limits (larger files take more memory)

### Image Quality Guidelines
- **Minimum Resolution**: 224x224 pixels (higher recommended)
- **Recommended Resolution**: 512x512 or higher
- **Image Quality**: Clear, well-contrasted chest X-rays
- **Positioning**: Standard PA (posterior-anterior) or AP (anterior-posterior) views

## Labels CSV Format

### Required Structure

The labels CSV file must contain the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `FileID` | string | Image filename (with extension) | `image001.jpg` |
| `right_sup` | int/float | Right superior zone CSI score | `0`, `1`, `2`, `3`, `NaN` |
| `left_sup` | int/float | Left superior zone CSI score | `0`, `1`, `2`, `3`, `NaN` |
| `right_mid` | int/float | Right middle zone CSI score | `0`, `1`, `2`, `3`, `NaN` |
| `left_mid` | int/float | Left middle zone CSI score | `0`, `1`, `2`, `3`, `NaN` |
| `right_inf` | int/float | Right inferior zone CSI score | `0`, `1`, `2`, `3`, `NaN` |
| `left_inf` | int/float | Left inferior zone CSI score | `0`, `1`, `2`, `3`, `NaN` |

### CSV Example

```csv
FileID;right_sup;left_sup;right_mid;left_mid;right_inf;left_inf
image001.jpg;0;0;1;1;0;2
image002.jpg;1;0;NaN;2;1;1
image003.jpg;2;3;1;1;2;2
image004.jpg;0;0;0;0;0;0
image005.jpg;;;;;;;  # All empty - converted to ungradable (4)
```

### CSI Score Values

| Value | Meaning | Description |
|-------|---------|-------------|
| **0** | Normal | No congestion detected |
| **1** | Mild | Mild congestion/opacity |
| **2** | Moderate | Moderate congestion/opacity |
| **3** | Severe | Severe congestion/opacity |
| **NaN/Empty** | Ungradable | Poor quality, obscured, or not assessable |

**Important**: 
- NaN, empty values, or any non-numeric values are automatically converted to class **4** (ungradable)
- Values outside 0-3 range are treated as ungradable

### CSV Configuration

Configure CSV parsing in your `.env` file:

```bash
# CSV file settings
LABELS_CSV=Labeled_Data_RAW.csv
LABELS_CSV_SEPARATOR=;  # or ',' or '\t'
```

**Supported Separators**:
- `;` (semicolon) - Default
- `,` (comma) - Common alternative
- `\t` (tab) - Tab-separated values

## Data Validation

### Automatic Validation

The system automatically validates your data:

```python
# Check data loading
python -c "
from src.config import cfg
from src.data import create_data_loaders

try:
    train_loader, val_loader, test_loader = create_data_loaders(cfg)
    print('✅ Data loading successful')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
except Exception as e:
    print(f'❌ Data loading failed: {e}')
"
```

### Manual Validation

```python
import pandas as pd
import os

# Validate CSV structure
def validate_csv(csv_path, separator=';'):
    df = pd.read_csv(csv_path, sep=separator)
    
    required_columns = ['FileID', 'right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    print(f"✅ CSV structure valid")
    print(f"Total samples: {len(df)}")
    
    # Check for missing images
    image_dir = "/path/to/Paradise_Images"
    missing_images = []
    for file_id in df['FileID']:
        if not os.path.exists(os.path.join(image_dir, file_id)):
            missing_images.append(file_id)
    
    if missing_images:
        print(f"⚠️  Missing images: {len(missing_images)}")
        print(f"First few: {missing_images[:5]}")
    else:
        print("✅ All images found")
    
    return True

# Run validation
validate_csv('/path/to/Paradise_CSV/Labeled_Data_RAW.csv')
```

## Data Statistics and Analysis

### Generate Data Statistics

```python
import pandas as pd
import numpy as np

def analyze_dataset(csv_path, separator=';'):
    df = pd.read_csv(csv_path, sep=separator)
    
    print("=== DATASET ANALYSIS ===")
    print(f"Total samples: {len(df)}")
    print()
    
    # Zone-wise statistics
    zones = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']
    
    for zone in zones:
        print(f"--- {zone.upper()} ---")
        values = df[zone].fillna(4)  # NaN becomes 4 (ungradable)
        
        print("Class distribution:")
        for class_val in [0, 1, 2, 3, 4]:
            count = sum(values == class_val)
            percentage = count / len(values) * 100
            print(f"  Class {class_val}: {count:4d} ({percentage:5.1f}%)")
        print()
    
    # Overall statistics
    all_values = []
    for zone in zones:
        values = df[zone].fillna(4).astype(int)
        all_values.extend(values)
    
    print("=== OVERALL STATISTICS ===")
    unique, counts = np.unique(all_values, return_counts=True)
    for val, count in zip(unique, counts):
        percentage = count / len(all_values) * 100
        print(f"Class {val}: {count:5d} ({percentage:5.1f}%)")

# Run analysis
analyze_dataset('/path/to/Paradise_CSV/Labeled_Data_RAW.csv')
```

## Data Preprocessing

### Automatic Preprocessing

The system automatically handles:

1. **Image Loading**: Loads images in various formats
2. **Resizing**: Resizes to model-specific input size
3. **Normalization**: Applies ImageNet normalization
4. **Data Type Conversion**: Converts to PyTorch tensors
5. **Missing Value Handling**: Converts NaN to class 4

### Data Augmentation

**Training Augmentations** (automatic):
```python
# Applied during training only
transforms.RandomRotation(degrees=10)
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.1, contrast=0.1)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Validation/Test** (no augmentation):
```python
# Applied during validation and testing
transforms.Resize(image_size)
transforms.CenterCrop(image_size)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## Data Splitting

### Automatic Stratified Splitting

The system uses intelligent stratified splitting:

```python
# Default split ratios
TRAIN_RATIO = 0.7  # 70% training
VAL_RATIO = 0.15   # 15% validation  
TEST_RATIO = 0.15  # 15% testing
```

**Stratification Strategy**:
- Maintains class distribution across splits
- Handles multi-label stratification (6 zones)
- Accounts for missing values and ungradable cases
- Ensures minimum samples per class in each split

### Custom Data Splitting

```python
from src.data_split import create_stratified_splits

# Custom split ratios
train_indices, val_indices, test_indices = create_stratified_splits(
    labels_df=df,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
)
```

## Common Data Issues and Solutions

### Issue 1: Missing Images
```
Error: Image not found: image001.jpg
```

**Solution**:
```bash
# Check image directory
ls -la /path/to/Paradise_Images/

# Verify image names match CSV exactly
python -c "
import pandas as pd
df = pd.read_csv('labels.csv', sep=';')
print('First 5 FileIDs:')
print(df['FileID'].head())
"
```

### Issue 2: CSV Encoding Issues
```
Error: UnicodeDecodeError
```

**Solution**:
```python
# Try different encodings
df = pd.read_csv('labels.csv', sep=';', encoding='utf-8')
# or
df = pd.read_csv('labels.csv', sep=';', encoding='latin-1')
# or  
df = pd.read_csv('labels.csv', sep=';', encoding='cp1252')
```

### Issue 3: Inconsistent File Extensions
```
Error: Mismatch between CSV and actual file extensions
```

**Solution**:
```python
# Standardize file extensions in CSV
import pandas as pd
import os

df = pd.read_csv('labels.csv', sep=';')
image_dir = '/path/to/images'

for idx, file_id in enumerate(df['FileID']):
    base_name = os.path.splitext(file_id)[0]
    
    # Check for different extensions
    for ext in ['.jpg', '.jpeg', '.png', '.tiff']:
        if os.path.exists(os.path.join(image_dir, base_name + ext)):
            df.loc[idx, 'FileID'] = base_name + ext
            break

df.to_csv('labels_corrected.csv', sep=';', index=False)
```

### Issue 4: Invalid Label Values
```
Warning: Invalid CSI values found
```

**Solution**:
```python
# Clean label values
import pandas as pd
import numpy as np

df = pd.read_csv('labels.csv', sep=';')
zones = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

for zone in zones:
    # Convert invalid values to NaN
    df[zone] = pd.to_numeric(df[zone], errors='coerce')
    
    # Clip values to valid range (0-3), NaN stays NaN
    df[zone] = df[zone].clip(0, 3)

df.to_csv('labels_cleaned.csv', sep=';', index=False)
```

## Creating Test Data

### Generate Synthetic Dataset

```python
from src.utils import create_debug_dataset

# Create test dataset
create_debug_dataset(
    num_samples=100,
    output_dir='./debug_data',
    image_size=(512, 512),
    csv_separator=','
)

# Update .env for test data
echo "DATA_DIR=./debug_data/images" >> .env
echo "CSV_DIR=./debug_data" >> .env
echo "LABELS_CSV=debug_labels.csv" >> .env
echo "LABELS_CSV_SEPARATOR=," >> .env
```

### Manual Test Data Creation

```python
import pandas as pd
import numpy as np
from PIL import Image
import os

def create_test_data(output_dir='test_data', num_samples=50):
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    
    data = []
    
    for i in range(num_samples):
        # Create synthetic chest X-ray (gray image)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8), mode='L')
        img = img.convert('RGB')  # Convert to RGB
        
        filename = f'test_image_{i:03d}.jpg'
        img.save(f'{output_dir}/images/{filename}')
        
        # Generate random but realistic CSI scores
        scores = {}
        scores['FileID'] = filename
        
        for zone in ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']:
            # 70% normal (0), 20% congested (1-3), 10% ungradable (NaN)
            rand = np.random.random()
            if rand < 0.7:
                scores[zone] = 0  # Normal
            elif rand < 0.9:
                scores[zone] = np.random.randint(1, 4)  # Congested
            else:
                scores[zone] = np.nan  # Ungradable
        
        data.append(scores)
    
    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/test_labels.csv', sep=';', index=False)
    
    print(f"Created {num_samples} test images and labels in {output_dir}/")

# Create test data
create_test_data()
```

## Best Practices

### 1. Data Organization
- Keep images and labels in separate, clearly named directories
- Use consistent file naming conventions
- Maintain backup copies of original data

### 2. Quality Control
- Validate data before training
- Check for missing or corrupted images
- Verify label consistency and ranges

### 3. Version Control
- Track data versions and changes
- Document any preprocessing steps
- Keep original raw data unchanged

### 4. Performance Optimization
- Use appropriate image resolutions (not too large)
- Consider data caching for faster training
- Monitor memory usage with large datasets

This comprehensive data format ensures reliable and consistent model training and evaluation. 