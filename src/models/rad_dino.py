"""
RadDINO model implementation for CSI prediction.
Based on Microsoft's Rad-DINO: A Radiological Diagnosis Network for Chest X-rays.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from loguru import logger
from typing import Tuple


class RadDINOCSIModel(nn.Module):
    """
    Complete RadDINO model for CSI prediction with 6-zone classification.
    
    This model uses Microsoft's Rad-DINO as the backbone and adds:
    - Classification head for overall diagnosis (3 classes)
    - CSI head for 6-zone scoring (5 classes per zone: 0-4)
    - Mean CSI head for overall severity (1 continuous value)
    """
    
    def __init__(self, n_classes_per_zone: int = 5, pretrained: bool = True):
        """
        Initialize RadDINO CSI model.
        
        Args:
            n_classes_per_zone: Number of classes per CSI zone (default: 5 for scores 0-4)
            pretrained: Whether to use pretrained weights (always True for RadDINO)
        """
        super(RadDINOCSIModel, self).__init__()
        
        # Initialize the RadDINO model from Microsoft
        repo = "microsoft/rad-dino"
        self.model = AutoModel.from_pretrained(repo)
        self.processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
        
        # Model configuration
        self.n_classes_per_zone = n_classes_per_zone
        self.n_zones = 6  # Fixed: 6 CSI zones
        self.feature_dim = 768  # RadDINO feature dimension
        
        # Create heads for different outputs
        self.classification_head = nn.Linear(self.feature_dim, 3)  # Overall classification
        self.head_csi = nn.Linear(self.feature_dim, self.n_zones * self.n_classes_per_zone)  # 6 zones * 5 classes
        self.mean_csi = nn.Linear(self.feature_dim, 1)  # Overall severity score
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        
        logger.info("Using model class: RadDINO_CSI")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"CSI zones: {self.n_zones}, Classes per zone: {self.n_classes_per_zone}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RadDINO CSI model.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple of:
            - classification_output: [batch_size, 3] - Overall classification logits
            - csi_scores: [batch_size, n_zones, n_classes_per_zone] - CSI zone logits
            - mean_csi: [batch_size, 1] - Overall severity score
        """
        # For RadDINO, we need to pass the input as pixel_values
        inputs = {'pixel_values': x}
        
        # Get RadDINO outputs
        outputs = self.model(**inputs)
        cls_embeddings = outputs.pooler_output  # [batch_size, 768]
        
        # Apply dropout
        features = self.dropout(cls_embeddings)
        
        # Classification output (overall diagnosis)
        classification_output = self.classification_head(features)
        
        # CSI scores for 6 zones (with ReLU activation for non-negative scores)
        csi_flat = self.head_csi(features)  # [batch_size, n_zones * n_classes_per_zone]
        csi_scores = csi_flat.view(-1, self.n_zones, self.n_classes_per_zone)  # [batch_size, n_zones, n_classes_per_zone]
        
        # Mean CSI (overall severity with ReLU for non-negative values)
        mean_csi = self.relu(self.mean_csi(features))
        
        return classification_output, csi_scores, mean_csi


class RadDINOBackboneOnly(nn.Module):
    """
    RadDINO backbone for use with the standard CSI head architecture.
    This follows the same pattern as other backbones in the project.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize RadDINO backbone only.
        
        Args:
            pretrained: Whether to use pretrained weights (always True for RadDINO)
        """
        super(RadDINOBackboneOnly, self).__init__()
        
        # Initialize the RadDINO model from Microsoft
        repo = "microsoft/rad-dino"
        self.model = AutoModel.from_pretrained(repo)
        self.processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
        
        # RadDINO outputs 768-dimensional features
        self.feature_dim = 768
        
        logger.info(f"Loaded RadDINO backbone from {repo}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RadDINO backbone.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        # For RadDINO, we need to pass the input as pixel_values
        inputs = {'pixel_values': x}
        
        # Get model outputs
        outputs = self.model(**inputs)
        
        # Extract CLS token embeddings (pooled representation)
        features = outputs.pooler_output  # [batch_size, 768]
        
        return features


# Export main components
__all__ = ['RadDINOCSIModel', 'RadDINOBackboneOnly'] 