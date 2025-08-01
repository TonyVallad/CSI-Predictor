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
from src.config import Config, ANSI


def test_processor_availability():
    """Test processor availability."""
    print(f"{ANSI['B']}Testing RadDINO processor availability...{ANSI['W']}")
    
    try:
        processor = get_raddino_processor()
        print(f"{ANSI['G']}✅ RadDINO processor created successfully{ANSI['W']}")
        return processor
    except Exception as e:
        print(f"{ANSI['R']}❌ Failed to create RadDINO processor:{ANSI['W']} {e}")
        return None


def test_transforms():
    """Test image transformations."""
    print(f"\n{ANSI['B']}Testing image transformations...{ANSI['W']}")
    
    # Test default transforms
    try:
        transforms = get_default_transforms()
        print(f"{ANSI['G']}✅ Default transforms created successfully{ANSI['W']}")
        
        # Create dummy image
        dummy_image = torch.randn(3, 224, 224)
        transformed = transforms(dummy_image)
        print(f"{ANSI['G']}✅ Transform applied successfully, output shape:{ANSI['W']} {transformed.shape}")
        
    except Exception as e:
        print(f"{ANSI['R']}❌ Failed to test transforms:{ANSI['W']} {e}")


def test_config():
    """Test configuration settings."""
    print(f"\n{ANSI['B']}Testing configuration...{ANSI['W']}")
    
    try:
        config = Config()
        print(f"{ANSI['G']}✅ Configuration created successfully{ANSI['W']}")
        print(f"{ANSI['B']}📊 Model architecture:{ANSI['W']} {config.model_arch}")
        print(f"{ANSI['B']}📊 Use official processor:{ANSI['W']} {config.use_official_processor}")
        
    except Exception as e:
        print(f"{ANSI['R']}❌ Failed to test configuration:{ANSI['W']} {e}")


def main():
    """Run all tests."""
    print(f"{ANSI['B']}🧪 RadDINO Test Suite{ANSI['W']}")
    print(f"{ANSI['B']}{'=' * 50}{ANSI['W']}")
    
    # Test processor availability
    processor = test_processor_availability()
    
    # Test transforms
    test_transforms()
    
    # Test configuration
    test_config()
    
    print(f"\n{ANSI['G']}✅ All tests completed!{ANSI['W']}")


if __name__ == "__main__":
    main() 