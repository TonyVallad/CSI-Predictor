#!/usr/bin/env python3
"""
Test script for RadDINO AutoImageProcessor implementation.
This script tests both the standard transforms and official processor approaches.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import get_raddino_processor, get_default_transforms
from src.config import Config

def test_processor_availability():
    """Test if RadDINO processor can be loaded."""
    print("=== Testing RadDINO Processor Availability ===")
    
    processor = get_raddino_processor(use_official=True)
    if processor is not None:
        print("✅ RadDINO AutoImageProcessor loaded successfully")
        print(f"   Processor type: {type(processor)}")
        return True
    else:
        print("❌ Failed to load RadDINO AutoImageProcessor")
        print("   This could be due to:")
        print("   - transformers library not installed")
        print("   - Network issues downloading the processor")
        print("   - Missing dependencies")
        return False

def test_transforms():
    """Test transform configurations."""
    print("\n=== Testing Transform Configurations ===")
    
    # Test standard RadDINO transforms
    transforms_standard = get_default_transforms("train", "raddino", use_official_processor=False)
    print("✅ Standard RadDINO transforms created")
    print(f"   Number of transforms: {len(transforms_standard.transforms)}")
    
    # Test official processor transforms
    transforms_official = get_default_transforms("train", "raddino", use_official_processor=True)
    print("✅ Official processor transforms created")
    print(f"   Number of transforms: {len(transforms_official.transforms)}")
    
    # Test non-RadDINO model
    transforms_resnet = get_default_transforms("train", "resnet50", use_official_processor=False)
    print("✅ ResNet50 transforms created")
    print(f"   Number of transforms: {len(transforms_resnet.transforms)}")

def test_config():
    """Test config with new parameter."""
    print("\n=== Testing Config with use_official_processor ===")
    
    # Test default config
    config = Config()
    print(f"✅ Default use_official_processor: {config.use_official_processor}")
    
    # Test config with processor enabled
    config_with_processor = Config(use_official_processor=True)
    print(f"✅ Custom use_official_processor: {config_with_processor.use_official_processor}")

def main():
    """Run all tests."""
    print("🧪 Testing RadDINO AutoImageProcessor Implementation")
    print("=" * 60)
    
    try:
        # Test processor availability
        processor_available = test_processor_availability()
        
        # Test transforms (should work regardless of processor availability)
        test_transforms()
        
        # Test config
        test_config()
        
        print("\n" + "=" * 60)
        if processor_available:
            print("🎉 All tests passed! RadDINO processor is ready for use.")
            print("\nNext steps:")
            print("1. Run a sweep to compare preprocessing approaches:")
            print("   python main.py --mode sweep --sweep-name 'raddino_processor_test'")
            print("2. The sweep will automatically test both:")
            print("   - use_official_processor: true (Microsoft's preprocessing)")
            print("   - use_official_processor: false (current PyTorch transforms)")
        else:
            print("⚠️  Tests completed with warnings.")
            print("   Standard transforms will be used as fallback.")
            print("   Install transformers>=4.30.0 to enable official processor.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 