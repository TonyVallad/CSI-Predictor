"""
Utility functions and classes for CSI-Predictor.
Contains helper functions for training, evaluation, and general utilities.
"""

import torch
import numpy as np
import random
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Import and configure Loguru
from loguru import logger

# Configure Loguru with rotating file handler
def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    Setup Loguru logging with rotating file handler.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add rotating file handler
    logger.add(
        Path(log_dir) / "csi_predictor_{time:YYYY-MM-DD}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    logger.info(f"Logging setup complete. Log files will be stored in: {log_dir}")

# Setup logging on import
setup_logging()

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track and compute running averages of metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
        
    def update(self, metric_name: str, value: float) -> None:
        """
        Update metric with new value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)
        
    def get_average(self, metric_name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return np.mean(self.metrics[metric_name])
    
    def get_averages(self) -> Dict[str, float]:
        """
        Get averages for all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {name: self.get_average(name) for name in self.metrics.keys()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize average meter.
        
        Args:
            name: Name of the meter
            fmt: Format string for display
        """
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self) -> None:
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self) -> str:
        """String representation."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_model_summary(model: torch.nn.Module, input_size: tuple) -> None:
    """
    Print model summary with parameter counts.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        total_params = count_parameters(model)
        print(f"Model has {total_params:,} trainable parameters")
        print("Install torchsummary for detailed model summary: pip install torchsummary")


def create_dirs(*paths: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Paths to create
    """
    from pathlib import Path
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def show_batch(
    data_loader: DataLoader,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Show a batch of images with their CSI zone predictions side-by-side.
    
    This utility randomly samples a batch and displays the 6 predicted zone masks
    in a single matplotlib figure for debugging and visualization purposes.
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to display
        figsize: Figure size for matplotlib
        save_path: Path to save the figure (optional)
        
    # Unit test stub:
    # def test_show_batch():
    #     # Create mock data loader
    #     mock_dataset = MockCSIDataset()
    #     loader = DataLoader(mock_dataset, batch_size=4)
    #     show_batch(loader, num_samples=4)
    #     # Verify plot was created without errors
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch
    import numpy as np
    from torchvision.utils import make_grid
    
    # Get a batch from the data loader
    data_iter = iter(data_loader)
    try:
        images, labels = next(data_iter)
    except StopIteration:
        print("Data loader is empty")
        return
    
    # Limit to num_samples
    if images.size(0) > num_samples:
        indices = torch.randperm(images.size(0))[:num_samples]
        images = images[indices]
        labels = labels[indices]
    
    batch_size = images.size(0)
    
    # CSI zone names for labeling
    zone_names = ['Right Superior', 'Left Superior', 'Right Middle', 
                  'Left Middle', 'Right Inferior', 'Left Inferior']
    
    # CSI class names
    class_names = ['Normal (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Unknown (4)']
    
    # Create figure with subplots
    fig, axes = plt.subplots(batch_size, 7, figsize=figsize)
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm_images = images * std + mean
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    for i in range(batch_size):
        # Convert image to numpy for display
        img_np = denorm_images[i].permute(1, 2, 0).numpy()
        
        # Display original image in first column
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {i+1}')
        axes[i, 0].axis('off')
        
        # Display CSI zones predictions in remaining columns
        for j, zone_name in enumerate(zone_names):
            col_idx = j + 1
            csi_score = labels[i, j].item()
            
            # Create zone visualization (simplified lung diagram)
            axes[i, col_idx].imshow(img_np)
            
            # Add zone overlay (approximate lung regions)
            h, w = img_np.shape[:2]
            zone_color = ['green', 'yellow', 'orange', 'red', 'gray'][csi_score]
            zone_alpha = 0.3
            
            # Define approximate zone regions (simplified)
            if 'Superior' in zone_name:
                y_start, y_end = 0, h//3
            elif 'Middle' in zone_name:
                y_start, y_end = h//3, 2*h//3
            else:  # Inferior
                y_start, y_end = 2*h//3, h
            
            if 'Right' in zone_name:
                x_start, x_end = w//2, w
            else:  # Left
                x_start, x_end = 0, w//2
            
            # Add colored rectangle for the zone
            rect = patches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor=zone_color, facecolor=zone_color, alpha=zone_alpha
            )
            axes[i, col_idx].add_patch(rect)
            
            # Set title with zone name and score
            axes[i, col_idx].set_title(f'{zone_name}\n{class_names[csi_score]}', fontsize=8)
            axes[i, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    plt.show()


def visualize_data_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the distribution of CSI scores across train/val/test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # CSI zone columns
    from .data import CSI_COLUMNS
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Unknown']
    
    for i, zone in enumerate(CSI_COLUMNS):
        # Combine data from all splits
        train_counts = train_df[zone].value_counts().sort_index()
        val_counts = val_df[zone].value_counts().sort_index()
        test_counts = test_df[zone].value_counts().sort_index()
        
        # Ensure all classes are represented
        for class_idx in range(5):
            if class_idx not in train_counts.index:
                train_counts[class_idx] = 0
            if class_idx not in val_counts.index:
                val_counts[class_idx] = 0
            if class_idx not in test_counts.index:
                test_counts[class_idx] = 0
        
        train_counts = train_counts.sort_index()
        val_counts = val_counts.sort_index()
        test_counts = test_counts.sort_index()
        
        # Create stacked bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[i].bar(x - width, train_counts, width, label='Train', alpha=0.8)
        axes[i].bar(x, val_counts, width, label='Val', alpha=0.8)
        axes[i].bar(x + width, test_counts, width, label='Test', alpha=0.8)
        
        axes[i].set_title(f'{zone.replace("_", " ").title()} Zone')
        axes[i].set_xlabel('CSI Score')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('CSI Score Distribution Across Train/Val/Test Sets', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()


def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing data patterns in the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with missing data statistics
    """
    from .data import CSI_COLUMNS, CSI_UNKNOWN_CLASS
    
    analysis = {
        'total_samples': len(df),
        'zone_missing_counts': {},
        'missing_patterns': {},
        'completely_missing_samples': 0
    }
    
    # Count missing values per zone (class 4 = unknown)
    for zone in CSI_COLUMNS:
        if zone in df.columns:
            missing_count = (df[zone] == CSI_UNKNOWN_CLASS).sum()
            analysis['zone_missing_counts'][zone] = missing_count
    
    # Analyze missing patterns
    missing_matrix = df[CSI_COLUMNS] == CSI_UNKNOWN_CLASS
    patterns = missing_matrix.apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
    analysis['missing_patterns'] = patterns.value_counts().to_dict()
    
    # Count samples with all zones missing
    analysis['completely_missing_samples'] = (missing_matrix.all(axis=1)).sum()
    
    return analysis


def create_debug_dataset(
    num_samples: int = 100,
    image_size: Tuple[int, int] = (224, 224),
    output_dir: str = "./debug_data"
) -> None:
    """
    Create a small debug dataset for testing purposes.
    
    Args:
        num_samples: Number of samples to create
        image_size: Size of synthetic images
        output_dir: Output directory for debug data
    """
    import numpy as np
    import pandas as pd
    from PIL import Image
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Creating {num_samples} synthetic images...")
    
    # Generate CSV data
    data = []
    for i in tqdm(range(num_samples), desc="Generating debug data"):
        # Create synthetic image (random lung-like pattern)
        img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        
        # Add some lung-like structure
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        for x in range(image_size[1]):
            for y in range(image_size[0]):
                # Create lung-like oval shapes
                dist_left = ((x - center_x//2)**2 + (y - center_y)**2) ** 0.5
                dist_right = ((x - 3*center_x//2)**2 + (y - center_y)**2) ** 0.5
                
                if dist_left < center_x//2 or dist_right < center_x//2:
                    img_array[y, x] = img_array[y, x] * 0.7 + 30  # Darker lung regions
        
        # Save image
        filename = f"debug_img_{i:04d}.png"
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(images_dir / filename)
        
        # Generate CSI scores (with some missing values)
        csi_scores = []
        for zone in range(6):
            if np.random.random() < 0.1:  # 10% missing
                csi_scores.append(4)  # Unknown
            else:
                csi_scores.append(np.random.randint(0, 4))  # Normal scores
        
        data.append({
            'FileID': filename,
            'right_sup': csi_scores[0],
            'left_sup': csi_scores[1], 
            'right_mid': csi_scores[2],
            'left_mid': csi_scores[3],
            'right_inf': csi_scores[4],
            'left_inf': csi_scores[5]
        })
    
    # Save CSV
    df = pd.DataFrame(data)
    csv_path = output_path / "debug_labels.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Debug dataset created:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {csv_path}")
    print(f"  Samples: {num_samples}")


def make_model_name(cfg, task_tag: str = "Tr", extra_info: str = "") -> str:
    """
    Create a structured model name with the format:
    [YYYYMMDD_HHMMSS]_[ModelName]_[TaskTag]_[ExtraInfo]
    
    Args:
        cfg: Configuration object
        task_tag: Task identifier (Tr=Training, Va=Validation, Te=Test, Eval=Evaluation, Infer=Inference)
        extra_info: Optional extra information (dataset slice, augmentation tag, resolution, hyperparam ID)
        
    Returns:
        Formatted model name
        
    Examples:
        20250611_093054_ResNet50_Tr
        20250611_093054_RadDINO_Eval_batch64
        20250611_093054_ViT-B_Infer_518x518
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean up model architecture name for consistency
    model_arch = cfg.model_arch.replace("_", "-").replace(" ", "-")
    
    # Build the name components
    name_parts = [timestamp, model_arch, task_tag]
    
    # Add extra info if provided
    if extra_info:
        name_parts.append(extra_info)
    
    return "_".join(name_parts)


def make_run_name(cfg) -> str:
    """
    Create a timestamped run name for experiments.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Formatted run name with timestamp
        
    Example:
        20250611_093054_ResNet50_Tr
    """
    return make_model_name(cfg, task_tag="Tr")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for all libraries for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed} for all libraries")


def pretty_print_config(cfg) -> str:
    """
    Create a pretty-printed string representation of the configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Formatted configuration string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CSI-Predictor Configuration")
    lines.append("=" * 60)
    
    # Environment and Device Settings
    lines.append("\n🖥️  Environment & Device:")
    lines.append(f"  Device: {cfg.device}")
    lines.append(f"  Load data to memory: {cfg.load_data_to_memory}")
    lines.append(f"  Data source: {cfg.data_source}")
    
    # Data Paths
    lines.append("\n📁 Data Paths:")
    lines.append(f"  Data directory: {cfg.data_dir}")
    lines.append(f"  Models directory: {cfg.models_dir}")
    lines.append(f"  CSV directory: {cfg.csv_dir}")
    lines.append(f"  INI directory: {cfg.ini_dir}")
    lines.append(f"  Labels CSV: {cfg.labels_csv}")
    lines.append(f"  CSV separator: '{cfg.labels_csv_separator}'")
    lines.append(f"  Computed CSV path: {cfg.csv_path}")
    
    # Training Hyperparameters
    lines.append("\n🏋️  Training Settings:")
    lines.append(f"  Batch size: {cfg.batch_size}")
    lines.append(f"  Epochs: {cfg.n_epochs}")
    lines.append(f"  Patience: {cfg.patience}")
    lines.append(f"  Learning rate: {cfg.learning_rate}")
    lines.append(f"  Optimizer: {cfg.optimizer}")
    
    # Model Configuration
    lines.append("\n🤖 Model Configuration:")
    lines.append(f"  Architecture: {cfg.model_arch}")
    
    # Configuration Sources
    if hasattr(cfg, '_env_vars') and hasattr(cfg, '_ini_vars') and hasattr(cfg, '_missing_keys'):
        lines.append("\n📋 Configuration Sources:")
        lines.append(f"  Environment variables: {len(cfg._env_vars)}")
        lines.append(f"  INI file variables: {len(cfg._ini_vars)}")
        if cfg._missing_keys:
            lines.append(f"  Missing keys (using defaults): {len(cfg._missing_keys)}")
            lines.append(f"    {', '.join(cfg._missing_keys)}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def print_config(cfg) -> None:
    """
    Print configuration in a pretty format.
    
    Args:
        cfg: Configuration object
    """
    print(pretty_print_config(cfg))


def log_config(cfg) -> None:
    """
    Log configuration using Loguru.
    
    Args:
        cfg: Configuration object
    """
    config_str = pretty_print_config(cfg)
    logger.info(f"Configuration loaded:\n{config_str}")


def create_roc_curves(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> Dict[str, Dict]:
    """
    Create ROC curves for multi-class, multi-zone CSI predictions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation (default: 4 for ungradable)
        
    Returns:
        Dictionary with ROC metrics per zone and class
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.colors as mcolors
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create subfolder for zone-specific ROC curves
    zone_roc_dir = save_path / "zones" / "roc_curves"
    zone_roc_dir.mkdir(parents=True, exist_ok=True)
    
    roc_metrics = {}
    
    # Color palette for classes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for zone_idx, zone_name in enumerate(zone_names):
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            logger.warning(f"No valid samples for zone {zone_name}")
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        # Create binary labels for each class (One-vs-Rest)
        n_classes = len(class_names) - 1  # Exclude ungradable class
        zone_targets_binary = label_binarize(zone_targets_valid, classes=list(range(n_classes)))
        
        # Handle case where only one class is present
        if zone_targets_binary.shape[1] == 1:
            # Add a dummy column for the missing classes
            missing_classes = n_classes - zone_targets_binary.shape[1]
            dummy_cols = np.zeros((zone_targets_binary.shape[0], missing_classes))
            zone_targets_binary = np.hstack([zone_targets_binary, dummy_cols])
        
        # Create ROC curves
        plt.figure(figsize=(10, 8))
        
        roc_auc = {}
        for class_idx in range(n_classes):
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Get probabilities for this class
            class_proba = zone_proba_valid[:, class_idx]
            
            # Get binary targets for this class
            if zone_targets_binary.shape[1] > class_idx:
                class_targets = zone_targets_binary[:, class_idx]
            else:
                # This class wasn't present in the data
                continue
            
            # Skip if no positive samples for this class
            if class_targets.sum() == 0:
                continue
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(class_targets, class_proba)
            roc_auc[class_idx] = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                color=colors[class_idx % len(colors)],
                lw=2,
                label=f'{class_names[class_idx]} (AUC = {roc_auc[class_idx]:.3f})'
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {zone_name.replace("_", " ").title()}\n({split_name.title()} Set)', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot in zone subfolder
        filename = f"{split_name}_{zone_name}_roc_curves.png"
        zone_save_path = zone_roc_dir / filename
        plt.savefig(zone_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        roc_metrics[zone_name] = roc_auc
        logger.info(f"Saved ROC curves for {zone_name}: {zone_save_path}")
    
    # Create overall ROC curves (macro-averaged across zones)
    plt.figure(figsize=(12, 8))
    
    all_fpr = []
    all_tpr = []
    all_auc_scores = []
    
    for class_idx in range(n_classes):
        class_fpr_list = []
        class_tpr_list = []
        class_auc_list = []
        
        for zone_idx, zone_name in enumerate(zone_names):
            zone_targets = targets[:, zone_idx]
            zone_proba = predictions_proba[:, zone_idx, :]
            
            # Filter out ignore_class samples
            valid_mask = zone_targets != ignore_class
            if not valid_mask.any():
                continue
                
            zone_targets_valid = zone_targets[valid_mask]
            zone_proba_valid = zone_proba[valid_mask]
            
            # Create binary labels for this class
            class_targets = (zone_targets_valid == class_idx).astype(int)
            
            # Skip if no positive samples
            if class_targets.sum() == 0:
                continue
                
            if class_idx < zone_proba_valid.shape[1]:
                class_proba = zone_proba_valid[:, class_idx]
                fpr, tpr, _ = roc_curve(class_targets, class_proba)
                class_fpr_list.append(fpr)
                class_tpr_list.append(tpr)
                class_auc_list.append(auc(fpr, tpr))
        
        if class_auc_list:
            # Compute macro-averaged ROC for this class
            mean_auc = np.mean(class_auc_list)
            
            # Interpolate all ROC curves to common FPR points
            mean_fpr = np.linspace(0, 1, 100)
            interp_tpr_list = []
            
            for fpr, tpr in zip(class_fpr_list, class_tpr_list):
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0  # Ensure starts at 0
                interp_tpr_list.append(interp_tpr)
            
            if interp_tpr_list:
                mean_tpr = np.mean(interp_tpr_list, axis=0)
                mean_tpr[-1] = 1.0  # Ensure ends at 1
                
                plt.plot(
                    mean_fpr, mean_tpr,
                    color=colors[class_idx % len(colors)],
                    lw=3,
                    label=f'{class_names[class_idx]} (Macro AUC = {mean_auc:.3f})'
                )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Macro-Averaged ROC Curves Across All Zones\n({split_name.title()} Set)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save overall plot
    filename = f"{split_name}_overall_roc_curves.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overall ROC curves: {save_path / filename}")
    return roc_metrics


def create_precision_recall_curves(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> Dict[str, Dict]:
    """
    Create Precision-Recall curves for multi-class, multi-zone CSI predictions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation (default: 4 for ungradable)
        
    Returns:
        Dictionary with PR metrics per zone and class
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.colors as mcolors
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create subfolder for zone-specific PR curves
    zone_pr_dir = save_path / "zones" / "pr_curves"
    zone_pr_dir.mkdir(parents=True, exist_ok=True)
    
    pr_metrics = {}
    
    # Color palette for classes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for zone_idx, zone_name in enumerate(zone_names):
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            logger.warning(f"No valid samples for zone {zone_name}")
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        # Create binary labels for each class (One-vs-Rest)
        n_classes = len(class_names) - 1  # Exclude ungradable class
        zone_targets_binary = label_binarize(zone_targets_valid, classes=list(range(n_classes)))
        
        # Handle case where only one class is present
        if zone_targets_binary.shape[1] == 1:
            # Add a dummy column for the missing classes
            missing_classes = n_classes - zone_targets_binary.shape[1]
            dummy_cols = np.zeros((zone_targets_binary.shape[0], missing_classes))
            zone_targets_binary = np.hstack([zone_targets_binary, dummy_cols])
        
        # Create PR curves
        plt.figure(figsize=(10, 8))
        
        pr_auc = {}
        for class_idx in range(n_classes):
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Get probabilities for this class
            class_proba = zone_proba_valid[:, class_idx]
            
            # Get binary targets for this class
            if zone_targets_binary.shape[1] > class_idx:
                class_targets = zone_targets_binary[:, class_idx]
            else:
                # This class wasn't present in the data
                continue
            
            # Skip if no positive samples for this class
            if class_targets.sum() == 0:
                continue
            
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(class_targets, class_proba)
            pr_auc[class_idx] = average_precision_score(class_targets, class_proba)
            
            # Plot PR curve
            plt.plot(
                recall, precision,
                color=colors[class_idx % len(colors)],
                lw=2,
                label=f'{class_names[class_idx]} (AP = {pr_auc[class_idx]:.3f})'
            )
        
        # Plot baseline (random classifier)
        baseline = (zone_targets_valid >= 0).mean()  # Proportion of positive samples
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline = {baseline:.3f}')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {zone_name.replace("_", " ").title()}\n({split_name.title()} Set)', fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot in zone subfolder
        filename = f"{split_name}_{zone_name}_pr_curves.png"
        zone_save_path = zone_pr_dir / filename
        plt.savefig(zone_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        pr_metrics[zone_name] = pr_auc
        logger.info(f"Saved PR curves for {zone_name}: {zone_save_path}")
    
    # Create overall PR curves (macro-averaged across zones)
    plt.figure(figsize=(12, 8))
    
    for class_idx in range(n_classes):
        class_precision_list = []
        class_recall_list = []
        class_ap_list = []
        
        for zone_idx, zone_name in enumerate(zone_names):
            zone_targets = targets[:, zone_idx]
            zone_proba = predictions_proba[:, zone_idx, :]
            
            # Filter out ignore_class samples
            valid_mask = zone_targets != ignore_class
            if not valid_mask.any():
                continue
                
            zone_targets_valid = zone_targets[valid_mask]
            zone_proba_valid = zone_proba[valid_mask]
            
            # Create binary labels for this class
            class_targets = (zone_targets_valid == class_idx).astype(int)
            
            # Skip if no positive samples
            if class_targets.sum() == 0:
                continue
                
            if class_idx < zone_proba_valid.shape[1]:
                class_proba = zone_proba_valid[:, class_idx]
                precision, recall, _ = precision_recall_curve(class_targets, class_proba)
                class_precision_list.append(precision)
                class_recall_list.append(recall)
                class_ap_list.append(average_precision_score(class_targets, class_proba))
        
        if class_ap_list:
            # Compute macro-averaged PR for this class
            mean_ap = np.mean(class_ap_list)
            
            # Interpolate all PR curves to common recall points
            mean_recall = np.linspace(0, 1, 100)
            interp_precision_list = []
            
            for precision, recall in zip(class_precision_list, class_recall_list):
                # Reverse arrays for interpolation (recall should be decreasing)
                precision_rev = precision[::-1]
                recall_rev = recall[::-1]
                interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                interp_precision_list.append(interp_precision)
            
            if interp_precision_list:
                mean_precision = np.mean(interp_precision_list, axis=0)
                
                plt.plot(
                    mean_recall, mean_precision,
                    color=colors[class_idx % len(colors)],
                    lw=3,
                    label=f'{class_names[class_idx]} (Macro AP = {mean_ap:.3f})'
                )
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Macro-Averaged Precision-Recall Curves Across All Zones\n({split_name.title()} Set)', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save overall plot
    filename = f"{split_name}_overall_pr_curves.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overall PR curves: {save_path / filename}")
    return pr_metrics


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    save_dir: str,
    run_name: str
) -> None:
    """
    Plot training curves for loss, accuracy, and F1 score.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        train_f1_scores: Training F1 scores per epoch
        val_f1_scores: Validation F1 scores per epoch
        save_dir: Directory to save plots
        run_name: Name of the training run
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[2].plot(epochs, train_f1_scores, 'b-', label='Training F1 Score', linewidth=2)
    axes[2].plot(epochs, val_f1_scores, 'r-', label='Validation F1 Score', linewidth=2)
    axes[2].set_title('Model F1 Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Overall formatting
    plt.suptitle(f'Training Curves - {run_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{run_name}_training_curves.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves: {save_path / filename}")


def create_confusion_matrix_grid(
    confusion_matrices: Dict[str, np.ndarray],
    save_dir: str,
    split_name: str = "validation",
    run_name: str = "model"
) -> None:
    """
    Create a grid of confusion matrices matching anatomical zone positions.
    
    Grid layout matches chest X-ray anatomy:
    [Right Superior] [Left Superior]
    [Right Middle]   [Left Middle] 
    [Right Inferior] [Left Inferior]
    
    Args:
        confusion_matrices: Dictionary of confusion matrices per zone
        save_dir: Directory to save the grid plot
        split_name: Name of the data split
        run_name: Name of the model/run
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Zone mapping to grid positions (matches anatomical layout)
    zone_positions = {
        "right_sup": (0, 0),    # Top-left (patient's right superior)
        "left_sup": (0, 1),     # Top-right (patient's left superior)
        "right_mid": (1, 0),    # Middle-left (patient's right middle)
        "left_mid": (1, 1),     # Middle-right (patient's left middle)
        "right_inf": (2, 0),    # Bottom-left (patient's right inferior)
        "left_inf": (2, 1)      # Bottom-right (patient's left inferior)
    }
    
    # Zone display names
    zone_display_names = {
        "right_sup": "Right Superior",
        "left_sup": "Left Superior",
        "right_mid": "Right Middle", 
        "left_mid": "Left Middle",
        "right_inf": "Right Inferior",
        "left_inf": "Left Inferior"
    }
    
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle(f'CSI Confusion Matrices - Anatomical Grid Layout\n({split_name.title()} Set)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for zone_name, (row, col) in zone_positions.items():
        ax = axes[row, col]
        
        if zone_name in confusion_matrices and confusion_matrices[zone_name].sum() > 0:
            cm = confusion_matrices[zone_name].astype(int)
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False,  # Disable individual colorbars
                ax=ax,
                square=True
            )
            
            # Calculate accuracy for this zone
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            total_samples = np.sum(cm)
            
            ax.set_title(f'{zone_display_names[zone_name]}\n'
                        f'Accuracy: {accuracy:.3f} (n={total_samples})', 
                        fontweight='bold', fontsize=12)
        else:
            # No data for this zone
            ax.text(0.5, 0.5, f'{zone_display_names[zone_name]}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Set labels only for bottom and left edges
        if row == 2:  # Bottom row
            ax.set_xlabel('Predicted CSI Score', fontweight='bold')
        else:
            ax.set_xlabel('')
            
        if col == 0:  # Left column
            ax.set_ylabel('True CSI Score', fontweight='bold')
        else:
            ax.set_ylabel('')
    
    # Adjust layout first to make room for colorbar
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, right=0.75)
    
    # Add a single colorbar for the entire figure
    if any(cm.sum() > 0 for cm in confusion_matrices.values()):
        # Find the maximum value across all confusion matrices for consistent colorbar
        vmax = max(cm.max() for cm in confusion_matrices.values() if cm.sum() > 0)
        
        # Create a mappable object for the colorbar without using a temporary axis
        import matplotlib.cm as mpl_cm
        import matplotlib.colors as mpl_colors
        
        norm = mpl_colors.Normalize(vmin=0, vmax=vmax)
        sm = mpl_cm.ScalarMappable(norm=norm, cmap='Blues')
        sm.set_array([])
        
        # Create colorbar with explicit positioning
        cbar_ax = fig.add_axes([0.78, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Sample Count', fontweight='bold', rotation=270, labelpad=20)
    
    # Save plot
    filename = f"{split_name}_confusion_matrices_grid.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix grid: {save_path / filename}")


def create_roc_curves_grid(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> None:
    """
    Create a grid of ROC curves matching anatomical zone positions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Zone mapping to grid positions
    zone_positions = {
        "right_sup": (0, 0), "left_sup": (0, 1),
        "right_mid": (1, 0), "left_mid": (1, 1),
        "right_inf": (2, 0), "left_inf": (2, 1)
    }
    
    zone_display_names = {
        "right_sup": "Right Superior", "left_sup": "Left Superior",
        "right_mid": "Right Middle", "left_mid": "Left Middle",
        "right_inf": "Right Inferior", "left_inf": "Left Inferior"
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f'ROC Curves - Anatomical Grid Layout\n({split_name.title()} Set)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for zone_idx, zone_name in enumerate(zone_names):
        if zone_name not in zone_positions:
            continue
            
        row, col = zone_positions[zone_name]
        ax = axes[row, col]
        
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            ax.text(0.5, 0.5, f'{zone_display_names[zone_name]}\nNo Valid Data', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        n_classes = len(class_names) - 1  # Exclude ungradable class
        
        # Plot ROC curves for each class
        auc_scores = []
        for class_idx in range(n_classes):
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Create binary targets for this class
            class_targets = (zone_targets_valid == class_idx).astype(int)
            
            if class_targets.sum() == 0:
                continue
                
            class_proba = zone_proba_valid[:, class_idx]
            fpr, tpr, _ = roc_curve(class_targets, class_proba)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            
            ax.plot(fpr, tpr, color=colors[class_idx % len(colors)], lw=2,
                   label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0
        ax.set_title(f'{zone_display_names[zone_name]}\nMean AUC: {mean_auc:.3f}', 
                    fontweight='bold', fontsize=11)
        
        if row == 2:  # Bottom row
            ax.set_xlabel('False Positive Rate', fontweight='bold')
        if col == 0:  # Left column
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save plot
    filename = f"{split_name}_roc_curves_grid.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curves grid: {save_path / filename}")


def create_precision_recall_curves_grid(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> None:
    """
    Create a grid of Precision-Recall curves matching anatomical zone positions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Zone mapping to grid positions
    zone_positions = {
        "right_sup": (0, 0), "left_sup": (0, 1),
        "right_mid": (1, 0), "left_mid": (1, 1),
        "right_inf": (2, 0), "left_inf": (2, 1)
    }
    
    zone_display_names = {
        "right_sup": "Right Superior", "left_sup": "Left Superior",
        "right_mid": "Right Middle", "left_mid": "Left Middle",
        "right_inf": "Right Inferior", "left_inf": "Left Inferior"
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f'Precision-Recall Curves - Anatomical Grid Layout\n({split_name.title()} Set)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for zone_idx, zone_name in enumerate(zone_names):
        if zone_name not in zone_positions:
            continue
            
        row, col = zone_positions[zone_name]
        ax = axes[row, col]
        
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            ax.text(0.5, 0.5, f'{zone_display_names[zone_name]}\nNo Valid Data', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        n_classes = len(class_names) - 1  # Exclude ungradable class
        
        # Plot PR curves for each class
        ap_scores = []
        for class_idx in range(n_classes):
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Create binary targets for this class
            class_targets = (zone_targets_valid == class_idx).astype(int)
            
            if class_targets.sum() == 0:
                continue
                
            class_proba = zone_proba_valid[:, class_idx]
            precision, recall, _ = precision_recall_curve(class_targets, class_proba)
            ap_score = average_precision_score(class_targets, class_proba)
            ap_scores.append(ap_score)
            
            ax.plot(recall, precision, color=colors[class_idx % len(colors)], lw=2,
                   label=f'{class_names[class_idx]} (AP = {ap_score:.3f})')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        mean_ap = np.mean(ap_scores) if ap_scores else 0
        ax.set_title(f'{zone_display_names[zone_name]}\nMean AP: {mean_ap:.3f}', 
                    fontweight='bold', fontsize=11)
        
        if row == 2:  # Bottom row
            ax.set_xlabel('Recall', fontweight='bold')
        if col == 0:  # Left column
            ax.set_ylabel('Precision', fontweight='bold')
            
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save plot
    filename = f"{split_name}_pr_curves_grid.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR curves grid: {save_path / filename}")


# Export main utilities
__all__ = [
    'EarlyStopping', 'MetricsTracker', 'AverageMeter',
    'set_seed', 'seed_everything', 'count_parameters', 'get_learning_rate',
    'save_checkpoint', 'load_checkpoint', 'calculate_class_weights',
    'format_time', 'print_model_summary', 'create_dirs',
    'show_batch', 'visualize_data_distribution', 'analyze_missing_data',
    'create_debug_dataset', 'make_run_name', 'make_model_name', 'pretty_print_config',
    'print_config', 'log_config', 'setup_logging', 'logger',
    'create_roc_curves', 'create_precision_recall_curves', 'plot_training_curves',
    'create_confusion_matrix_grid', 'create_roc_curves_grid', 'create_precision_recall_curves_grid'
] 