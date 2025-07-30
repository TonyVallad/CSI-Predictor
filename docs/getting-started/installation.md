<div align="center">

# Installation Guide

Complete installation instructions for CSI-Predictor.

</div>

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