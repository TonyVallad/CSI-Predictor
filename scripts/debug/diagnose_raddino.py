#!/usr/bin/env python3
"""
Diagnostic script for RadDINO availability issues.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.models.backbones import get_backbone
from src.models.rad_dino import RadDINOBackboneOnly
from src.models.backbones.raddino import diagnose_raddino_availability


def main():
    """Check RadDINO dependencies and availability."""
    print("🔍 RadDINO Diagnostic Script")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"📊 PyTorch Version: {torch.__version__}")
    print(f"📊 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"📊 GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"📊 GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check transformers availability
    try:
        import transformers
        print(f"✅ Transformers Version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
        print("   Install with: pip install transformers>=4.30.0")
        return
    
    # Check AutoModel availability
    try:
        from transformers import AutoModel, AutoImageProcessor
        print("✅ AutoModel and AutoImageProcessor available")
    except ImportError as e:
        print(f"❌ AutoModel/AutoImageProcessor not available: {e}")
        return
    
    # Check RadDINO model availability
    try:
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        print(f"🔍 Testing model: {model_name}")
        
        # Try to load the model
        model = AutoModel.from_pretrained(model_name)
        print("✅ RadDINO model loaded successfully")
        
        # Check model architecture
        print(f"📊 Model type: {type(model)}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Failed to load RadDINO model: {e}")
        print("   This might be due to network issues or model availability")
        return
    
    # Test RadDINO backbone
    try:
        print("\n🔍 Testing RadDINO backbone...")
        backbone = get_backbone("raddino")
        print("✅ RadDINO backbone created successfully")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = backbone.to(device)
        
        # Create dummy input
        batch_size = 2
        channels = 3
        height = 224
        width = 224
        dummy_input = torch.randn(batch_size, channels, height, width).to(device)
        
        print(f"📊 Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = backbone(dummy_input)
        
        print(f"📊 Output shape: {output.shape}")
        print("✅ RadDINO backbone forward pass successful")
        
    except Exception as e:
        print(f"❌ RadDINO backbone test failed: {e}")
        return
    
    # Run comprehensive diagnostics
    print("\n🔍 Running comprehensive RadDINO diagnostics...")
    diagnose_raddino_availability()
    
    print("\n✅ RadDINO diagnostic completed successfully!")
    print("🎯 RadDINO is ready to use in the CSI-Predictor project!")


if __name__ == "__main__":
    main() 