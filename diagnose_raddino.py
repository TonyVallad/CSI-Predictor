#!/usr/bin/env python3
"""
Diagnostic script for RadDINO availability issues.
Run this on your Linux server to identify what's causing the RadDINO import failure.
"""

import sys
import traceback

def main():
    print("=== CSI-Predictor RadDINO Diagnostic ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Check transformers library
    print("1. Checking transformers library...")
    try:
        import transformers
        print(f"   ✅ Transformers version: {transformers.__version__}")
        
        try:
            from transformers import AutoModel, AutoImageProcessor
            print("   ✅ Required transformers components imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import transformers components: {e}")
            traceback.print_exc()
            return
            
    except ImportError as e:
        print(f"   ❌ Transformers library not available: {e}")
        print("   💡 Install with: pip install transformers>=4.30.0")
        return
    
    print()
    
    # Check torch
    print("2. Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"   ❌ PyTorch not available: {e}")
        return
    
    print()
    
    # Check if we can import the project modules
    print("3. Checking project structure...")
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.models import backbones
        print("   ✅ Backbones module imported successfully")
        
        print(f"   RadDINO available flag: {backbones.RADDINO_AVAILABLE}")
        
    except ImportError as e:
        print(f"   ❌ Failed to import project modules: {e}")
        traceback.print_exc()
        return
    
    print()
    
    # Check rad_dino module specifically
    print("4. Checking RadDINO implementation...")
    try:
        from src.models.rad_dino import RadDINOBackboneOnly
        print("   ✅ RadDINO implementation imported successfully")
        
        try:
            print("   🔄 Attempting to create RadDINO backbone...")
            backbone = RadDINOBackboneOnly(pretrained=True)
            print(f"   ✅ RadDINO backbone created successfully!")
            print(f"   📊 Feature dimension: {backbone.feature_dim}")
            
        except Exception as e:
            print(f"   ❌ Failed to create RadDINO backbone: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"   ❌ Failed to import RadDINO implementation: {e}")
        traceback.print_exc()
    
    print()
    
    # Test backbone factory
    print("5. Testing backbone factory...")
    try:
        from src.models.backbones import get_backbone
        
        # Test other architectures first
        for arch in ['ResNet50', 'CheXNet', 'Custom_1']:
            try:
                backbone = get_backbone(arch, pretrained=False)
                print(f"   ✅ {arch}: OK")
            except Exception as e:
                print(f"   ❌ {arch}: {e}")
        
        # Test RadDINO
        try:
            backbone = get_backbone('RadDINO', pretrained=False)
            print(f"   ✅ RadDINO: OK")
        except Exception as e:
            print(f"   ❌ RadDINO: {e}")
            
    except Exception as e:
        print(f"   ❌ Backbone factory test failed: {e}")
        traceback.print_exc()
    
    print()
    print("=== Diagnostic Complete ===")
    print()
    print("💡 If RadDINO is failing:")
    print("   1. Make sure you're in the correct directory")
    print("   2. Check that src/models/rad_dino.py exists")
    print("   3. Verify transformers>=4.30.0 is installed")
    print("   4. Check for any missing dependencies")
    print("   5. Try running: pip install --upgrade transformers torch")


if __name__ == "__main__":
    main() 