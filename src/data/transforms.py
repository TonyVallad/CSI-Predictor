"""
Image transformations for CSI-Predictor.

This module contains image transformation functionality extracted from the original src/data.py file.
"""

import torch
from torchvision import transforms
from typing import Optional
from ..config import Config, cfg, ANSI
from .preprocessing import get_normalization_parameters

# Model architecture to input size mapping
MODEL_INPUT_SIZES = {
    'resnet18': (224, 224),
    'resnet34': (224, 224),
    'resnet50': (224, 224),
    'resnet101': (224, 224),
    'resnet152': (224, 224),
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b2': (260, 260),
    'efficientnet_b3': (300, 300),
    'efficientnet_b4': (380, 380),
    'densenet121': (224, 224),
    'densenet169': (224, 224),
    'densenet201': (224, 224),
    'vit_base_patch16_224': (224, 224),
    'vit_large_patch16_224': (224, 224),
    'chexnet': (224, 224),
    'custom1': (224, 224),
    'raddino': (518, 518),  # RadDINO's expected input size from Microsoft
}

# Try to import transformers for RadDINO processor
try:
    from transformers import AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoImageProcessor = None

def get_raddino_processor(use_official: bool = False):
    """
    Get RadDINO image processor.
    
    Args:
        use_official: Whether to use the official AutoImageProcessor
        
    Returns:
        AutoImageProcessor or None
    """
    if use_official and TRANSFORMERS_AVAILABLE:
        try:
            repo = "microsoft/rad-dino"
            processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
            return processor
        except Exception as e:
            print(f"{ANSI['Y']}Warning: Failed to load RadDINO processor:{ANSI['W']} {e}")
            return None
    return None


def get_default_transforms(phase: str = "train", model_arch: str = "resnet50", use_official_processor: bool = False, config: Optional[Config] = None) -> transforms.Compose:
    """
    Get default image transformations for different phases.
    
    Args:
        phase: Phase name ('train', 'val', 'test')
        model_arch: Model architecture name for input size
        use_official_processor: Whether to use official RadDINO processor (RadDINO only)
        config: Configuration object for normalization parameters
        
    Returns:
        Composed transformations
    """
    # Get input size for model architecture
    input_size = MODEL_INPUT_SIZES.get(model_arch, (224, 224))
    
    # Get normalization parameters
    mean, std = get_normalization_parameters(config)
    
    # Special handling for RadDINO with official processor
    if (model_arch.lower().replace('_', '').replace('-', '') == 'raddino' and 
        use_official_processor):
        # Return minimal transforms - the official processor will handle everything
        return transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor (processor expects PIL images)
        ])
    
    # Special handling for RadDINO which has its own preprocessing
    if model_arch.lower().replace('_', '').replace('-', '') == 'raddino':
        if phase == "train":
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=10),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                # RadDINO uses ImageNet normalization
                transforms.Normalize(mean=mean, std=std)
            ])
        else:  # val or test
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
                transforms.ToTensor(),
                # RadDINO uses ImageNet normalization
                transforms.Normalize(mean=mean, std=std)
            ])
    
    # Standard transforms for other models
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 