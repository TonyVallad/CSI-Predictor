#!/usr/bin/env python3
"""
Test script for RadDINO processor functionality.
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

from src.data.transforms import get_raddino_processor, get_default_transforms
from src.config import Config


def test_processor_availability():
    """Test processor availability."""
    print("Testing RadDINO processor availability...")
    
    try:
        processor = get_raddino_processor()
        print("✅ RadDINO processor created successfully")
        return processor
    except Exception as e:
        print(f"❌ Failed to create RadDINO processor: {e}")
        return None


def test_transforms():
    """Test image transformations."""
    print("\nTesting image transformations...")
    
    # Test default transforms
    try:
        transforms = get_default_transforms()
        print("✅ Default transforms created successfully")
        
        # Create dummy image
        dummy_image = torch.randn(3, 224, 224)
        transformed = transforms(dummy_image)
        print(f"✅ Transform applied successfully, output shape: {transformed.shape}")
        
    except Exception as e:
        print(f"❌ Failed to test transforms: {e}")


def test_config():
    """Test configuration settings."""
    print("\nTesting configuration...")
    
    try:
        config = Config()
        print("✅ Configuration created successfully")
        print(f"📊 Model architecture: {config.model_arch}")
        print(f"📊 Use official processor: {config.use_official_processor}")
        
    except Exception as e:
        print(f"❌ Failed to test configuration: {e}")


def main():
    """Run all tests."""
    print("🧪 RadDINO Test Suite")
    print("=" * 50)
    
    # Test processor availability
    processor = test_processor_availability()
    
    # Test transforms
    test_transforms()
    
    # Test configuration
    test_config()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main() 