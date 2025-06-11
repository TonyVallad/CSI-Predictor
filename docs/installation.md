# Installation Guide

This guide covers the complete installation process for CSI-Predictor, including system requirements, dependencies, and setup instructions.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM (16GB+ recommended for RadDINO)
- **Storage**: 5GB free space (additional space for datasets and models)
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional but recommended)

### Recommended Specifications
- **Python**: 3.9 or 3.10
- **Memory**: 16GB+ RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, V100, A100, etc.)
- **Storage**: SSD with 20GB+ free space

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd CSI-Predictor
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv (recommended)
   python -m venv .venv
   
   # Activate the environment
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Core Dependencies**
   ```bash
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```

4. **Install Optional Dependencies**
   
   **For RadDINO Support** (Recommended):
   ```bash
   pip install transformers>=4.30.0
   ```
   
   **For Development**:
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```

### Method 2: Conda Installation

1. **Create Conda Environment**
   ```bash
   conda create -n csi-predictor python=3.9
   conda activate csi-predictor
   ```

2. **Install PyTorch**
   ```bash
   # For CUDA 11.8 (check your CUDA version)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CPU only
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. **Install Remaining Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Method 3: Docker Installation

1. **Build Docker Image**
   ```bash
   docker build -t csi-predictor .
   ```

2. **Run Container**
   ```bash
   # With GPU support
   docker run --gpus all -it -v $(pwd):/workspace csi-predictor
   
   # CPU only
   docker run -it -v $(pwd):/workspace csi-predictor
   ```

## Dependency Details

### Core Dependencies

- **PyTorch**: Deep learning framework (≥1.12.0)
- **torchvision**: Computer vision library for PyTorch
- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis
- **Pillow**: Image processing library
- **scikit-learn**: Machine learning utilities (for metrics)
- **loguru**: Modern logging library
- **python-dotenv**: Environment variable management

### Optional Dependencies

- **transformers**: For RadDINO model support (≥4.30.0)
- **wandb**: Experiment tracking and visualization
- **optuna**: Hyperparameter optimization
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **tqdm**: Progress bars
- **plotly**: Interactive visualizations

### Development Dependencies

- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Type checking

## Verification

### Quick Verification

1. **Test Python Import**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import src.config; print('CSI-Predictor import successful')"
   ```

2. **Check GPU Availability**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
   ```

3. **Test RadDINO Support** (Optional)
   ```bash
   python -c "from src.models.backbones import RADDINO_AVAILABLE; print(f'RadDINO available: {RADDINO_AVAILABLE}')"
   ```

### Full Verification

Run the built-in verification script:
```bash
python -m src.verify_installation
```

This will check:
- All dependencies are installed
- GPU/CUDA setup (if available)
- Model architectures can be loaded
- Configuration system works
- Sample data can be processed

## Configuration

### Environment Setup

1. **Create Environment File**
   ```bash
   cp .env.example .env  # If template exists
   # Or create manually:
   ```

2. **Edit `.env` File**
   ```bash
   # Device configuration
   DEVICE=cuda  # or 'cpu' if no GPU
   
   # Data paths (update these for your setup)
   DATA_DIR=/path/to/your/Paradise_Images
   CSV_DIR=/path/to/your/Paradise_CSV
   MODELS_DIR=./models
   
   # Data loading
   LOAD_DATA_TO_MEMORY=True
   
   # Labels configuration
   LABELS_CSV=Labeled_Data_RAW.csv
   LABELS_CSV_SEPARATOR=;
   ```

3. **Create Configuration File**
   ```bash
   # Create config.ini
   cat > config.ini << EOF
   [TRAINING]
   BATCH_SIZE = 32
   N_EPOCHS = 100
   PATIENCE = 10
   LEARNING_RATE = 0.001
   OPTIMIZER = adam
   
   [MODEL]
   MODEL_ARCH = resnet50
   EOF
   ```

### Directory Structure

Create the necessary directories:
```bash
mkdir -p models logs docs/evaluation
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.ini
   BATCH_SIZE = 16  # or even 8
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

3. **Permission Issues**
   ```bash
   # On Unix systems, ensure proper permissions
   chmod +x scripts/*.sh  # If script files exist
   ```

4. **RadDINO Download Issues**
   ```bash
   # Pre-download RadDINO model
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/rad-dino')"
   ```

### Platform-Specific Issues

**Windows:**
- Use `python` instead of `python3`
- Use backslashes in paths or raw strings
- Ensure Visual C++ redistributables are installed

**macOS:**
- May need to install Xcode Command Line Tools
- Use `python3` explicitly if multiple Python versions

**Linux:**
- Ensure CUDA drivers are properly installed
- May need to install additional system packages

### Getting Help

If you encounter issues:

1. **Check the logs**: Look in `./logs/` for error messages
2. **Verify dependencies**: Run the verification script
3. **Update packages**: Ensure you have the latest versions
4. **Check GPU setup**: Verify CUDA installation
5. **Search issues**: Check GitHub Issues for similar problems
6. **Ask for help**: Create a new issue with:
   - Error messages
   - System information
   - Installation steps attempted

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick-start.md)**
2. **Configure your [Data Format](data-format.md)**
3. **Try [Training a Model](training.md)**
4. **Explore [Model Architectures](model-architectures.md)**

## Updates and Maintenance

### Updating CSI-Predictor

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Keeping Dependencies Updated

```bash
# Update all packages
pip list --outdated
pip install --upgrade package_name

# Or update all at once (be careful)
pip install --upgrade -r requirements.txt
```

### Version Management

Use virtual environments to manage different versions:
```bash
# Create environment for specific version
python -m venv .venv-v1.0
source .venv-v1.0/bin/activate
pip install -r requirements-v1.0.txt
``` 