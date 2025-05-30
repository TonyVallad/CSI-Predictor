"""
Training logic for CSI-Predictor.
Handles model training, validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Any, Optional

from .data import create_data_loaders
from .models.backbones import get_backbone
from .models.head import CSIRegressionHead
from .utils import EarlyStopping, MetricsTracker
from .config import Config


class CSIModel(nn.Module):
    """Complete CSI prediction model."""
    
    def __init__(self, backbone_arch: str, num_zones: int = 6, pretrained: bool = True):
        """
        Initialize CSI model.
        
        Args:
            backbone_arch: Backbone architecture name
            num_zones: Number of CSI zones
            pretrained: Use pretrained backbone
        """
        super().__init__()
        
        # Get backbone
        self.backbone = get_backbone(backbone_arch, pretrained)
        
        # Get regression head
        self.head = CSIRegressionHead(
            input_dim=self.backbone.feature_dim,
            num_zones=num_zones
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions


def create_model(config) -> CSIModel:
    """
    Create CSI model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        CSI model
    """
    model = CSIModel(
        backbone_arch=config.model_arch,
        num_zones=6,
        pretrained=True
    )
    return model


def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Optimizer
    """
    optimizer_name = config.optimizer.lower()
    learning_rate = config.learning_rate
    
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Training metrics
    """
    model.train()
    metrics = MetricsTracker()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        metrics.update("loss", loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return metrics.get_averages()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Validation metrics
    """
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Track metrics
            metrics.update("loss", loss.item())
    
    return metrics.get_averages()


def train_model(config: Config) -> None:
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(config)
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    
    # Create model
    model = create_model(config)
    model.to(device)
    logger.info(f"Created model: {config.model_arch}")
    
    # Create optimizer and criterion
    optimizer = create_optimizer(model, config)
    criterion = nn.MSELoss()
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Initialize wandb (optional)
    use_wandb = False
    try:
        wandb.init(project="csi-predictor", config=vars(config))
        use_wandb = True
        logger.info("Initialized Weights & Biases logging")
    except Exception as e:
        logger.warning(f"Could not initialize wandb: {e}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics["loss"])
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "learning_rate": optimizer.param_groups[0]["lr"]
            })
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = Path(config.get_model_path("best_model"))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'config': config
            }, save_path)
            logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    logger.info("Training completed!")
    
    if use_wandb:
        wandb.finish() 