"""
Complete CSI models for CSI-Predictor.

This module contains complete model implementations extracted from the original src/models/__init__.py file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logging import logger
from ..backbones import get_backbone, get_backbone_feature_dim
from ..heads import CSIHead

class CSIModel(nn.Module):
    """Complete CSI prediction model with backbone + head."""
    
    def __init__(self, backbone_arch: str, n_classes_per_zone: int = 5, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize CSI model.
        
        Args:
            backbone_arch: Backbone architecture name
            n_classes_per_zone: Number of classes per zone (default: 5)
            pretrained: Use pretrained backbone
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        
        # Get backbone
        self.backbone = get_backbone(backbone_arch, pretrained)
        
        # Get backbone feature dimension
        backbone_out_dim = get_backbone_feature_dim(backbone_arch)
        
        # Create CSI head with configurable dropout
        self.head = CSIHead(backbone_out_dim, n_classes_per_zone, dropout_rate)
        
        # Store metadata
        self.backbone_arch = backbone_arch
        self.n_classes_per_zone = n_classes_per_zone
        self.pretrained = pretrained
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Logits [batch_size, n_zones, n_classes_per_zone]
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Predict CSI scores using head
        predictions = self.head(features)
        
        return predictions


class CSIModelWithZoneMasking(nn.Module):
    """
    CSI prediction model with zone focus support.
    
    Supports two zone focus methods:
    1. Masking: Creates 6 masked versions of each image, processes separately
       - Grid-based zones and optional segmentation-guided zones
       - Configurable masking strategies (zero masking or attention masking)
       - Memory: High (6x images), Accuracy: High
       
    2. Spatial Reduction: Single image processing + adaptive pooling to 3x2 grid
       - Processes one image, spatially pools to 6 zone features
       - Memory: Low (1x images), Accuracy: Good, Speed: Faster
    """
    
    # Zone mapping: left lung = right side of image (anatomically correct)
    ZONE_MAPPING = {
        0: "right_sup",   # Patient's right superior (left side of image)
        1: "left_sup",    # Patient's left superior (right side of image)
        2: "right_mid",   # Patient's right middle
        3: "left_mid",    # Patient's left middle  
        4: "right_inf",   # Patient's right inferior
        5: "left_inf"     # Patient's left inferior
    }
    
    def __init__(self, cfg, backbone_arch: str, n_classes_per_zone: int = 5, pretrained: bool = True):
        """
        Initialize CSI model with zone focus.
        
        Args:
            cfg: Configuration object with zone focus settings
            backbone_arch: Backbone architecture name
            n_classes_per_zone: Number of classes per zone (default: 5)
            pretrained: Use pretrained backbone
        """
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.zone_focus_method = cfg.zone_focus_method.lower()
        self.use_segmentation_masking = cfg.use_segmentation_masking
        self.masking_strategy = cfg.masking_strategy.lower()
        self.attention_strength = cfg.attention_strength
        self.masks_path = cfg.masks_dir
        
        # Create base model
        self.backbone = get_backbone(backbone_arch, pretrained)
        backbone_out_dim = get_backbone_feature_dim(backbone_arch)
        
        # Create zone-specific components based on focus method
        if self.zone_focus_method == "spatial_reduction":
            # For spatial reduction, we use zone-specific feature transformations
            # Each zone gets its own feature transformation + classifier
            self.zone_feature_transforms = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(backbone_out_dim, backbone_out_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(backbone_out_dim // 2, backbone_out_dim // 4)
                ) for _ in range(6)
            ])
            self.zone_classifiers = nn.ModuleList([
                nn.Linear(backbone_out_dim // 4, n_classes_per_zone) for _ in range(6)
            ])
        else:
            # For masking approach, use the standard head
            self.head = CSIHead(backbone_out_dim, n_classes_per_zone)
        
        # Store metadata
        self.backbone_arch = backbone_arch
        self.n_classes_per_zone = n_classes_per_zone
        self.pretrained = pretrained
        
        logger.info(f"Created CSI model with zone focus:")
        logger.info(f"  - Zone focus method: {self.zone_focus_method}")
        if self.zone_focus_method == "masking":
            logger.info(f"  - Segmentation masking: {self.use_segmentation_masking}")
            logger.info(f"  - Masking strategy: {self.masking_strategy}")
            logger.info(f"  - Attention strength: {self.attention_strength}")
    
    def create_grid_zone_mask(self, batch_size: int, height: int, width: int, zone_idx: int, device: torch.device) -> torch.Tensor:
        """
        Create grid-based zone mask for specified zone.
        
        Args:
            batch_size: Batch size
            height: Image height
            width: Image width
            zone_idx: Zone index (0-5)
            device: Device to create mask on
            
        Returns:
            Zone mask [batch_size, 1, height, width]
        """
        # Create 3x2 grid zones
        zone_height = height // 3
        zone_width = width // 2
        
        # Zone layout (3x2 grid):
        # 0: right_sup (top-left)    1: left_sup (top-right)
        # 2: right_mid (middle-left) 3: left_mid (middle-right)
        # 4: right_inf (bottom-left) 5: left_inf (bottom-right)
        
        row = zone_idx // 2  # 0, 1, 2
        col = zone_idx % 2   # 0, 1
        
        # Create mask
        mask = torch.zeros(batch_size, 1, height, width, device=device)
        
        # Calculate zone boundaries
        start_row = row * zone_height
        end_row = (row + 1) * zone_height if row < 2 else height
        start_col = col * zone_width
        end_col = (col + 1) * zone_width if col < 1 else width
        
        # Set zone region to 1
        mask[:, :, start_row:end_row, start_col:end_col] = 1.0
        
        return mask
    
    def load_segmentation_mask(self, file_id: str, target_height: int, target_width: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load segmentation mask for a specific file ID.
        
        Args:
            file_id: File ID to load mask for
            target_height: Target height for resizing
            target_width: Target width for resizing
            device: Device to load mask on
            
        Returns:
            Tuple of (left_lung_mask, right_lung_mask) or (None, None) if not found
        """
        if not self.use_segmentation_masking or not self.masks_path:
            return None, None
        
        try:
            # Construct mask file paths
            left_mask_path = Path(self.masks_path) / f"{file_id}_left_lung_mask.png"
            right_mask_path = Path(self.masks_path) / f"{file_id}_right_lung_mask.png"
            
            left_mask = None
            right_mask = None
            
            # Load left lung mask
            if left_mask_path.exists():
                left_mask = cv2.imread(str(left_mask_path), cv2.IMREAD_GRAYSCALE)
                left_mask = cv2.resize(left_mask, (target_width, target_height))
                left_mask = torch.from_numpy(left_mask).float() / 255.0
                left_mask = left_mask.to(device)
            
            # Load right lung mask
            if right_mask_path.exists():
                right_mask = cv2.imread(str(right_mask_path), cv2.IMREAD_GRAYSCALE)
                right_mask = cv2.resize(right_mask, (target_width, target_height))
                right_mask = torch.from_numpy(right_mask).float() / 255.0
                right_mask = right_mask.to(device)
            
            return left_mask, right_mask
            
        except Exception as e:
            logger.warning(f"Could not load segmentation mask for {file_id}: {e}")
            return None, None
    
    def create_zone_mask(self, x: torch.Tensor, zone_idx: int, file_ids: Optional[list] = None) -> torch.Tensor:
        """
        Create zone mask for specified zone.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            zone_idx: Zone index (0-5)
            file_ids: List of file IDs for segmentation masking
            
        Returns:
            Zone mask [batch_size, 1, height, width]
        """
        batch_size, _, height, width = x.shape
        device = x.device
        
        if self.use_segmentation_masking and file_ids is not None:
            # Use segmentation-based masking
            masks = []
            for i, file_id in enumerate(file_ids):
                left_mask, right_mask = self.load_segmentation_mask(file_id, height, width, device)
                
                if left_mask is not None and right_mask is not None:
                    # Create zone-specific mask based on zone_idx
                    if zone_idx in [0, 2, 4]:  # Right lung zones (left side of image)
                        zone_mask = left_mask
                    else:  # Left lung zones (right side of image)
                        zone_mask = right_mask
                    
                    # Apply zone-specific filtering (superior, middle, inferior)
                    if zone_idx in [0, 1]:  # Superior zones
                        zone_mask = zone_mask * torch.ones_like(zone_mask)
                        zone_mask[:, height//3:, :] = 0
                    elif zone_idx in [2, 3]:  # Middle zones
                        zone_mask = zone_mask * torch.ones_like(zone_mask)
                        zone_mask[:, :height//3, :] = 0
                        zone_mask[:, 2*height//3:, :] = 0
                    else:  # Inferior zones
                        zone_mask = zone_mask * torch.ones_like(zone_mask)
                        zone_mask[:, :2*height//3, :] = 0
                    
                    masks.append(zone_mask.unsqueeze(0))
                else:
                    # Fallback to grid-based mask
                    grid_mask = self.create_grid_zone_mask(1, height, width, zone_idx, device)
                    masks.append(grid_mask)
            
            # Stack masks
            zone_mask = torch.cat(masks, dim=0)
        else:
            # Use grid-based masking
            zone_mask = self.create_grid_zone_mask(batch_size, height, width, zone_idx, device)
        
        return zone_mask
    
    def apply_zone_masking(self, x: torch.Tensor, zone_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply zone masking to input images.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            zone_mask: Zone mask [batch_size, 1, height, width]
            
        Returns:
            Masked images [batch_size, channels, height, width]
        """
        if self.masking_strategy == "zero_masking":
            # Zero out non-zone regions
            return x * zone_mask
        elif self.masking_strategy == "attention_masking":
            # Apply attention-based masking
            attention_mask = zone_mask * self.attention_strength + (1 - zone_mask) * (1 - self.attention_strength)
            return x * attention_mask
        else:
            # Default to zero masking
            return x * zone_mask
    
    def forward(self, x: torch.Tensor, file_ids: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass through zone-focused model.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            file_ids: List of file IDs for segmentation masking
            
        Returns:
            Logits [batch_size, n_zones, n_classes_per_zone]
        """
        if self.zone_focus_method == "spatial_reduction":
            # Single image processing with spatial pooling
            features = self.backbone(x)  # [batch_size, feature_dim]
            
            # Process each zone through its respective transformation and classifier
            zone_predictions = []
            for zone_idx in range(6):
                zone_feat = self.zone_feature_transforms[zone_idx](features)
                zone_logits = self.zone_classifiers[zone_idx](zone_feat)
                zone_predictions.append(zone_logits)
            
            # Stack predictions: [batch_size, n_zones, n_classes_per_zone]
            return torch.stack(zone_predictions, dim=1)
        
        else:  # masking approach
            # Process each zone separately with masking
            zone_predictions = []
            
            for zone_idx in range(6):
                # Create zone mask
                zone_mask = self.create_zone_mask(x, zone_idx, file_ids)
                
                # Apply masking
                masked_x = self.apply_zone_masking(x, zone_mask)
                
                # Process through backbone and head
                features = self.backbone(masked_x)
                zone_logits = self.head(features)
                
                zone_predictions.append(zone_logits)
            
            # Stack predictions: [batch_size, n_zones, n_classes_per_zone]
            return torch.stack(zone_predictions, dim=1)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 