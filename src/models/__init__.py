"""
Models package for CSI-Predictor.
Contains neural network architectures and components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple

from .backbones import get_backbone, get_backbone_feature_dim
from .head import CSIHead


class CSIModel(nn.Module):
    """Complete CSI prediction model with backbone + head."""
    
    def __init__(self, backbone_arch: str, n_classes_per_zone: int = 5, pretrained: bool = True):
        """
        Initialize CSI model.
        
        Args:
            backbone_arch: Backbone architecture name
            n_classes_per_zone: Number of classes per zone (default: 5)
            pretrained: Use pretrained backbone
        """
        super().__init__()
        
        # Get backbone
        self.backbone = get_backbone(backbone_arch, pretrained)
        
        # Get backbone feature dimension
        backbone_out_dim = get_backbone_feature_dim(backbone_arch)
        
        # Create CSI head
        self.head = CSIHead(backbone_out_dim, n_classes_per_zone)
        
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
    CSI prediction model with zone masking support.
    
    Supports both grid-based zones and segmentation-guided zones with
    configurable masking strategies (zero masking or attention masking).
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
        Initialize CSI model with zone masking.
        
        Args:
            cfg: Configuration object with zone masking settings
            backbone_arch: Backbone architecture name
            n_classes_per_zone: Number of classes per zone (default: 5)
            pretrained: Use pretrained backbone
        """
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.use_segmentation_masking = cfg.use_segmentation_masking
        self.masking_strategy = cfg.masking_strategy.lower()
        self.attention_strength = cfg.attention_strength
        self.masks_path = cfg.masks_path
        
        # Create base model
        self.backbone = get_backbone(backbone_arch, pretrained)
        backbone_out_dim = get_backbone_feature_dim(backbone_arch)
        self.head = CSIHead(backbone_out_dim, n_classes_per_zone)
        
        # Store metadata
        self.backbone_arch = backbone_arch
        self.n_classes_per_zone = n_classes_per_zone
        self.pretrained = pretrained
        
        logger.info(f"Created CSI model with zone masking:")
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
            device: Device for tensor creation
            
        Returns:
            Zone mask tensor [batch_size, 1, height, width]
        """
        # Create zone mask based on 2x3 grid
        mask = torch.zeros(batch_size, 1, height, width, device=device)
        
        # Calculate grid boundaries
        h_third = height // 3
        w_half = width // 2
        
        # Map zone index to grid position
        if zone_idx == 0:  # right_sup (left side of image, top)
            mask[:, :, 0:h_third, 0:w_half] = 1.0
        elif zone_idx == 1:  # left_sup (right side of image, top)  
            mask[:, :, 0:h_third, w_half:width] = 1.0
        elif zone_idx == 2:  # right_mid (left side of image, middle)
            mask[:, :, h_third:2*h_third, 0:w_half] = 1.0
        elif zone_idx == 3:  # left_mid (right side of image, middle)
            mask[:, :, h_third:2*h_third, w_half:width] = 1.0
        elif zone_idx == 4:  # right_inf (left side of image, bottom)
            mask[:, :, 2*h_third:height, 0:w_half] = 1.0
        elif zone_idx == 5:  # left_inf (right side of image, bottom)
            mask[:, :, 2*h_third:height, w_half:width] = 1.0
        
        return mask
    
    def load_segmentation_mask(self, file_id: str, target_height: int, target_width: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load and process segmentation masks for left and right lungs.
        
        Args:
            file_id: File ID to load masks for
            target_height: Target image height
            target_width: Target image width
            device: Device for tensor creation
            
        Returns:
            Tuple of (left_lung_mask, right_lung_mask) or (None, None) if not found
        """
        try:
            left_mask_path = Path(self.masks_path) / f"{file_id}_left_lung_mask.png"
            right_mask_path = Path(self.masks_path) / f"{file_id}_right_lung_mask.png"
            
            if not (left_mask_path.exists() and right_mask_path.exists()):
                logger.warning(f"Segmentation masks not found for {file_id}, using grid-only")
                return None, None
            
            # Load masks
            left_mask = cv2.imread(str(left_mask_path), cv2.IMREAD_GRAYSCALE)
            right_mask = cv2.imread(str(right_mask_path), cv2.IMREAD_GRAYSCALE)
            
            if left_mask is None or right_mask is None:
                logger.warning(f"Failed to load segmentation masks for {file_id}")
                return None, None
            
            # Resize to target dimensions
            left_mask = cv2.resize(left_mask, (target_width, target_height))
            right_mask = cv2.resize(right_mask, (target_width, target_height))
            
            # Convert to tensors and normalize to [0, 1]
            left_tensor = torch.from_numpy(left_mask.astype(np.float32) / 255.0).to(device)
            right_tensor = torch.from_numpy(right_mask.astype(np.float32) / 255.0).to(device)
            
            # Add batch and channel dimensions
            left_tensor = left_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            right_tensor = right_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            return left_tensor, right_tensor
            
        except Exception as e:
            logger.warning(f"Error loading segmentation masks for {file_id}: {e}")
            return None, None
    
    def create_zone_mask(self, x: torch.Tensor, zone_idx: int, file_ids: Optional[list] = None) -> torch.Tensor:
        """
        Create zone mask combining grid and optional segmentation.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            zone_idx: Zone index (0-5)
            file_ids: List of file IDs for segmentation mask loading
            
        Returns:
            Zone mask [batch_size, 1, height, width]
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Create base grid mask
        grid_mask = self.create_grid_zone_mask(batch_size, height, width, zone_idx, device)
        
        if not self.use_segmentation_masking or file_ids is None:
            return grid_mask
        
        # Apply segmentation masking
        final_masks = []
        
        for i, file_id in enumerate(file_ids):
            left_lung_mask, right_lung_mask = self.load_segmentation_mask(
                file_id, height, width, device
            )
            
            if left_lung_mask is None or right_lung_mask is None:
                # Fallback to grid-only for this sample
                final_masks.append(grid_mask[i:i+1])
                continue
            
            # Combine grid mask with appropriate lung mask
            sample_grid_mask = grid_mask[i:i+1]  # [1, 1, H, W]
            
            if zone_idx in [1, 3, 5]:  # Left lung zones (right side of image)
                lung_mask = left_lung_mask
            else:  # Right lung zones (left side of image)
                lung_mask = right_lung_mask
            
            # Intersect grid zone with lung segmentation
            combined_mask = sample_grid_mask * lung_mask
            final_masks.append(combined_mask)
        
        return torch.cat(final_masks, dim=0)
    
    def apply_zone_masking(self, x: torch.Tensor, zone_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply zone masking to input images.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            zone_mask: Zone mask [batch_size, 1, height, width]
            
        Returns:
            Masked images [batch_size, channels, height, width]
        """
        if self.masking_strategy == "zero":
            # Zero masking: set non-zone pixels to 0
            return x * zone_mask
        
        elif self.masking_strategy == "attention":
            # Attention masking: reduce non-zone pixel influence
            attention_weights = (zone_mask * self.attention_strength + 
                               (1 - zone_mask) * (1 - self.attention_strength))
            return x * attention_weights
        
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
    
    def forward(self, x: torch.Tensor, file_ids: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass with zone masking.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            file_ids: Optional list of file IDs for segmentation mask loading
            
        Returns:
            Predictions [batch_size, n_zones, n_classes_per_zone]
        """
        batch_size = x.shape[0]
        
        # Create zone masks for all zones
        zone_masks = []
        for zone_idx in range(6):
            zone_mask = self.create_zone_mask(x, zone_idx, file_ids)
            zone_masks.append(zone_mask)
        
        # Stack zone masks: [6, batch_size, 1, height, width]
        all_zone_masks = torch.stack(zone_masks, dim=0)
        
        # Apply zone masking to create zone-specific inputs
        zone_inputs = []
        for zone_idx in range(6):
            masked_input = self.apply_zone_masking(x, all_zone_masks[zone_idx])
            zone_inputs.append(masked_input)
        
        # Stack all zone inputs: [6 * batch_size, channels, height, width]
        stacked_inputs = torch.cat(zone_inputs, dim=0)
        
        # Single backbone forward pass for all zones
        all_features = self.backbone(stacked_inputs)
        
        # Reshape features back to [6, batch_size, feature_dim]
        feature_dim = all_features.shape[-1]
        zone_features = all_features.view(6, batch_size, feature_dim)
        
        # Process each zone through its respective head
        zone_predictions = []
        for zone_idx in range(6):
            zone_feat = zone_features[zone_idx]  # [batch_size, feature_dim]
            zone_logits = self.head.zone_classifiers[zone_idx](zone_feat)
            zone_predictions.append(zone_logits)
        
        # Stack predictions: [batch_size, n_zones, n_classes_per_zone]
        return torch.stack(zone_predictions, dim=1)


def build_model(cfg, use_zone_masking: bool = None) -> nn.Module:
    """
    Build CSI model from configuration and move to device.
    
    Args:
        cfg: Configuration object with model settings
        use_zone_masking: Whether to use zone masking model (auto-detected if None)
        
    Returns:
        CSI model on the specified device
    """
    logger.info(f"Building CSI model with architecture: {cfg.model_arch}")
    
    # Auto-detect zone masking if not specified
    if use_zone_masking is None:
        use_zone_masking = hasattr(cfg, 'use_segmentation_masking') or hasattr(cfg, 'masking_strategy')
    
    # Create appropriate model
    if use_zone_masking:
        logger.info("Creating CSI model with zone masking support")
        model = CSIModelWithZoneMasking(
            cfg=cfg,
            backbone_arch=cfg.model_arch,
            n_classes_per_zone=5,  # CSI scores: 0, 1, 2, 3, 4
            pretrained=True
        )
    else:
        logger.info("Creating standard CSI model")
        model = CSIModel(
            backbone_arch=cfg.model_arch,
            n_classes_per_zone=5,  # CSI scores: 0, 1, 2, 3, 4
            pretrained=True
        )
    
    # Move to device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully:")
    logger.info(f"  - Backbone: {cfg.model_arch}")
    logger.info(f"  - Zone masking: {use_zone_masking}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Device: {device}")
    
    return model


def build_zone_masking_model(cfg) -> CSIModelWithZoneMasking:
    """
    Build CSI model with zone masking explicitly.
    
    Args:
        cfg: Configuration object with zone masking settings
        
    Returns:
        CSI model with zone masking on the specified device
    """
    return build_model(cfg, use_zone_masking=True)


# Export main components
__all__ = ['CSIModel', 'CSIModelWithZoneMasking', 'build_model', 'build_zone_masking_model'] 