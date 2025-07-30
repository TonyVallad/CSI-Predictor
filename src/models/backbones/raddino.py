"""
RadDINO backbone for CSI-Predictor.

This module contains RadDINO backbone functionality extracted from the original src/models/backbones.py file.
"""

import torch
import torch.nn as nn
from src.utils.logging import logger

# Conditional import for RadDINO (requires transformers library)
RADDINO_AVAILABLE = False
RadDINOBackboneOnly = None

# First check if transformers is available
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    logger.debug(f"Transformers library is available (version {transformers.__version__})")
    
    # Then try to import RadDINO implementation
    try:
        from ..complete.rad_dino import RadDINOBackboneOnly
        RADDINO_AVAILABLE = True
        logger.debug("RadDINO backbone is available (all dependencies found)")
    except ImportError as rad_dino_error:
        logger.warning(f"RadDINO implementation not available: {rad_dino_error}")
        logger.warning("Transformers library is available, but RadDINO implementation failed to import")
        RADDINO_AVAILABLE = False
        RadDINOBackboneOnly = None
        
except ImportError as transformers_error:
    logger.warning(f"Transformers library not available: {transformers_error}")
    logger.warning("RadDINO backbone will not be available. Install with: pip install transformers>=4.30.0")
    RADDINO_AVAILABLE = False
    RadDINOBackboneOnly = None

class RadDINOBackbone(nn.Module):
    """RadDINO backbone wrapper for CSI-Predictor."""
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize RadDINO backbone.
        
        Args:
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        if not RADDINO_AVAILABLE:
            raise ImportError("RadDINO is not available. Please install transformers>=4.30.0")
        
        # Use the RadDINO backbone implementation
        self.backbone = RadDINOBackboneOnly(pretrained=pretrained)
        self.feature_dim = self.backbone.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        return self.backbone(x)

def diagnose_raddino_availability():
    """
    Diagnose RadDINO availability and dependencies.
    
    Returns:
        Dictionary with availability information
    """
    diagnosis = {
        'transformers_available': False,
        'transformers_version': None,
        'raddino_available': False,
        'raddino_error': None,
        'recommendations': []
    }
    
    # Check transformers
    try:
        import transformers
        diagnosis['transformers_available'] = True
        diagnosis['transformers_version'] = transformers.__version__
    except ImportError as e:
        diagnosis['raddino_error'] = str(e)
        diagnosis['recommendations'].append("Install transformers: pip install transformers>=4.30.0")
        return diagnosis
    
    # Check RadDINO implementation
    try:
        from ..complete.rad_dino import RadDINOBackboneOnly
        diagnosis['raddino_available'] = True
    except ImportError as e:
        diagnosis['raddino_error'] = str(e)
        diagnosis['recommendations'].append("Check RadDINO implementation in src/models/complete/rad_dino.py")
    
    return diagnosis

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 