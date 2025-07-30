<div align="center">

# ResNet50 Implementation

This document covers the ResNet50 implementation in CSI-Predictor.

</div>

## How ResNet50 is Adapted for CSI Score Prediction on Chest X-rays

### **Architecture Overview**

The ResNet50 model in this project follows a **backbone + head architecture**:

1. **ResNet50 Backbone** (Feature Extractor)
2. **CSI Classification Head** (Task-Specific Predictor)

### **1. ResNet50 Backbone Implementation**

The ResNet50 backbone is implemented in ```139:174:src/models/backbones.py```

**Key adaptations:**

- **Pretrained on ImageNet**: Uses `ResNet50_Weights.IMAGENET1K_V2` for transfer learning
- **Feature Extractor Only**: The final classification layer is removed, keeping only the feature extraction layers
- **Output**: 2048-dimensional feature vectors (from the global average pooling layer)
- **Input Size**: 224×224 pixels (standard for ResNet50)

```python
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet50 
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer, keep feature extraction
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048  # ResNet50 output features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.flatten(features, 1)  # [batch_size, 2048]
        return features
```

### **2. CSI Classification Head**

The medical-specific part is handled by the CSI classification head in ```17:60:src/models/head.py```

**Key features:**

- **6 Parallel Classifiers**: One for each lung zone (right/left × superior/middle/inferior)
- **5 Classes per Zone**: CSI scores 0-4 (0=normal, 1=mild, 2=moderate, 3=severe, 4=ungradable)
- **Independent Predictions**: Each zone is classified separately

```python
class CSIHead(nn.Module):
    def __init__(self, backbone_out_dim: int, n_classes_per_zone: int = 5):
        super().__init__()
        
        self.n_zones = 6  # 6 CSI zones
        
        # Create 6 parallel classification heads (one per zone)
        self.zone_classifiers = nn.ModuleList([
            nn.Linear(backbone_out_dim, n_classes_per_zone) 
            for _ in range(self.n_zones)
        ])
        
        self.dropout = nn.Dropout(p=0.1)
```

### **3. Complete Model Architecture**

The full CSI model combines backbone + head in ```14:50:src/models/__init__.py```

**Data Flow:**
1. **Input**: Chest X-ray image [batch_size, 3, 224, 224]
2. **ResNet50 Backbone**: Extract features [batch_size, 2048]
3. **CSI Head**: Predict 6 zones × 5 classes [batch_size, 6, 5]

```python
class CSIModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using ResNet50 backbone
        features = self.backbone(x)  # [batch_size, 2048]
        
        # Predict CSI scores using classification head
        predictions = self.head(features)  # [batch_size, 6, 5]
        
        return predictions
```

### **4. Training Strategy**

**Loss Function**: Weighted Cross-Entropy Loss (```35:85:src/train.py```)
- Reduces importance of "ungradable" class (class 4) while still learning to predict it
- Emphasizes learning clear CSI classifications (0-3)

**Data Preprocessing**:
- Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data augmentation: random horizontal flips, rotations, color jittering
- Resizing to 224×224 pixels

### **5. Medical Domain Adaptation**

**Transfer Learning Strategy:**
- Start with ImageNet-pretrained ResNet50 (general visual features)
- Fine-tune the entire network on chest X-ray data
- Task-specific head learns medical domain patterns

**Zone-Specific Learning:**
- Each of the 6 lung zones gets its own classifier
- Allows the model to learn zone-specific pathology patterns
- Can handle cases where different zones have different CSI scores

### **6. Key Advantages of This Adaptation**

1. **Transfer Learning**: Leverages powerful ImageNet features as starting point
2. **Medical Specialization**: Task-specific head adapted for CSI scoring
3. **Multi-Zone Prediction**: Handles the 6-zone medical requirement
4. **Robust Classification**: Handles both gradable (0-3) and ungradable (4) cases
5. **Proven Architecture**: ResNet50 is well-established and stable

### **7. Performance Context**

According to the documentation, ResNet50 serves as a **"general baseline"** in this project:
- **Fast**: Efficient training and inference
- **Proven**: Well-established architecture
- **Low Memory**: Reasonable computational requirements
- **Balanced Performance**: Good trade-off between speed and accuracy

For comparison, the project also supports:
- **RadDINO**: Microsoft's chest X-ray specialist (highest accuracy)
- **CheXNet**: DenseNet121 adapted for chest X-rays
- **Custom CNN**: Lightweight baseline