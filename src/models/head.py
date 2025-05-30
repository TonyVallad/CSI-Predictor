"""
Regression heads for CSI prediction in CSI-Predictor.
Handles the final layers that predict 6-zone CSI scores.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CSIRegressionHead(nn.Module):
    """Regression head for predicting 6-zone CSI scores."""
    
    def __init__(
        self,
        input_dim: int,
        num_zones: int = 6,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        """
        Initialize CSI regression head.
        
        Args:
            input_dim: Input feature dimension from backbone
            num_zones: Number of CSI zones to predict (default: 6)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function type
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_zones = num_zones
        self.dropout_rate = dropout_rate
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer for 6 zones
        layers.append(nn.Linear(current_dim, num_zones))
        
        self.head = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            CSI predictions [batch_size, num_zones]
        """
        return self.head(x)


class MultiTaskCSIHead(nn.Module):
    """Multi-task head for CSI prediction with auxiliary tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_zones: int = 6,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        predict_severity: bool = True
    ):
        """
        Initialize multi-task CSI head.
        
        Args:
            input_dim: Input feature dimension from backbone
            num_zones: Number of CSI zones to predict
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            predict_severity: Whether to predict overall severity
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_zones = num_zones
        self.predict_severity = predict_severity
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Shared feature extractor
        shared_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(current_dim, hidden_dim))
            shared_layers.append(nn.ReLU(inplace=True))
            shared_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # CSI zone prediction head
        self.csi_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_zones)
        )
        
        # Overall severity prediction head (optional)
        if predict_severity:
            self.severity_head = nn.Sequential(
                nn.Linear(current_dim, hidden_dims[-1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[-1], 1)
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions
        """
        # Shared features
        shared_feat = self.shared_features(x)
        
        # CSI zone predictions
        csi_pred = self.csi_head(shared_feat)
        
        outputs = {"csi_scores": csi_pred}
        
        # Overall severity prediction
        if self.predict_severity:
            severity_pred = self.severity_head(shared_feat)
            outputs["severity"] = severity_pred
        
        return outputs


class AttentionCSIHead(nn.Module):
    """CSI head with attention mechanism for zone-specific predictions."""
    
    def __init__(
        self,
        input_dim: int,
        num_zones: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.2
    ):
        """
        Initialize attention-based CSI head.
        
        Args:
            input_dim: Input feature dimension from backbone
            num_zones: Number of CSI zones to predict
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_zones = num_zones
        self.hidden_dim = hidden_dim
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Zone embeddings (learnable)
        self.zone_embeddings = nn.Parameter(torch.randn(num_zones, hidden_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            CSI predictions [batch_size, num_zones]
        """
        batch_size = x.size(0)
        
        # Project input features
        features = self.input_projection(x)  # [batch_size, hidden_dim]
        
        # Expand zone embeddings for batch
        zone_emb = self.zone_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_zones, hidden_dim]
        
        # Features as key and value, zone embeddings as query
        features_expanded = features.unsqueeze(1).expand(-1, self.num_zones, -1)  # [batch_size, num_zones, hidden_dim]
        
        # Apply attention
        attended_features, _ = self.attention(
            query=zone_emb,
            key=features_expanded,
            value=features_expanded
        )  # [batch_size, num_zones, hidden_dim]
        
        # Project to final predictions
        predictions = self.output_projection(attended_features).squeeze(-1)  # [batch_size, num_zones]
        
        return predictions 