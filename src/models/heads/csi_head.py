"""
CSI classification head for CSI-Predictor.

This module contains CSI head functionality extracted from the original src/models/head.py file.
"""

import torch
import torch.nn as nn

class CSIHead(nn.Module):
    """
    CSI classification head with 6 parallel zone classifiers.
    
    Predicts CSI scores for 6 zones using separate classification heads,
    where each zone can have scores from 0-4 (5 classes total).
    """
    
    def __init__(self, backbone_out_dim: int, n_classes_per_zone: int = 5, dropout_rate: float = 0.5):
        """
        Initialize CSI classification head.
        
        Args:
            backbone_out_dim: Output dimension from backbone network
            n_classes_per_zone: Number of classes per zone (default: 5 for scores 0-4)
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        
        self.backbone_out_dim = backbone_out_dim
        self.n_classes_per_zone = n_classes_per_zone
        self.n_zones = 6  # Fixed: 6 CSI zones
        
        # Create 6 parallel classification heads (one per zone)
        self.zone_classifiers = nn.ModuleList([
            nn.Linear(backbone_out_dim, n_classes_per_zone) 
            for _ in range(self.n_zones)
        ])
        
        # Add dropout for regularization with configurable rate
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CSI head.
        
        Args:
            x: Feature tensor from backbone [batch_size, backbone_out_dim]
            
        Returns:
            Logits tensor [batch_size, n_zones, n_classes_per_zone]
        """
        # Apply dropout
        x = self.dropout(x)
        
        # Get predictions from each zone classifier
        zone_logits = []
        for zone_classifier in self.zone_classifiers:
            logits = zone_classifier(x)  # [batch_size, n_classes_per_zone]
            zone_logits.append(logits)
        
        # Stack predictions: [batch_size, n_zones, n_classes_per_zone]
        output = torch.stack(zone_logits, dim=1)
        
        return output

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 