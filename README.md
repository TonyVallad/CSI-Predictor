# CSI-Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A modular PyTorch project for predicting 6-zone CSI (Congestion Score Index) scores on chest X-ray images using state-of-the-art deep learning models.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd CSI-Predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train a model
python -m src.train

# Evaluate model
python -m src.evaluate
```

## 📖 Documentation

### Getting Started
- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Quick Start Guide](docs/quick-start.md)** - Get up and running in minutes
- **[Configuration](docs/config-guide.md)** - Configure the project for your needs

### Usage
- **[Training Models](docs/training.md)** - Train CSI prediction models
- **[Model Evaluation](docs/evaluation.md)** - Evaluate and analyze model performance
- **[Hyperparameter Optimization](docs/hyperparameter_optimization.md)** - Optimize model hyperparameters

### Architecture & Models
- **[Model Architectures](docs/model-architectures.md)** - Available model architectures
- **[Data Format](docs/data-format.md)** - Expected data structure and format
- **[Model Naming Format](docs/model_naming_format.md)** - Model naming conventions

### Advanced Features
- **[Legacy Entry Point](docs/legacy-entry-point.md)** - Backward compatibility features
- **[Project Structure](docs/project-structure.md)** - Complete project organization guide

### Development
- **[Contributing](docs/contributing.md)** - How to contribute to the project
- **[API Reference](docs/api-reference.md)** - Detailed API documentation

### Technical References
- **[Image Preprocessing](docs/Image%20Preprocessing.md)** - Data preprocessing pipeline details
- **[ResNet50 Implementation](docs/ResNet50%20Implementation%20in%20this%20project.md)** - ResNet50 adaptation specifics
- **[ArchiMed Python Connector](docs/ArchiMed%20Python%20Connector.md)** - ArchiMed integration details

## 🏗️ Key Features

- **🎯 Multi-Architecture Support**: ResNet, CheXNet, RadDINO, Vision Transformers
- **⚡ Pure PyTorch**: No scikit-learn dependencies, optimized for performance
- **📊 Comprehensive Evaluation**: Per-zone metrics, confusion matrices, classification reports
- **🔧 Hyperparameter Optimization**: W&B Sweeps and Optuna integration
- **📈 Experiment Tracking**: Weights & Biases integration with rich visualizations
- **🔄 Flexible Configuration**: Environment variables, INI files, and command-line options
- **🚀 Production Ready**: Structured model naming, logging, and checkpointing

## 🏥 Medical Context

CSI-Predictor analyzes chest X-rays to predict congestion scores for 6 anatomical zones:
- **Right/Left Superior zones**
- **Right/Left Middle zones** 
- **Right/Left Inferior zones**

Each zone is classified into congestion levels (0-3) or marked as ungradable (4).

## 🛠️ Supported Models

| Model | Use Case | Strengths |
|-------|----------|-----------|
| **ResNet50** | General baseline | Fast, proven, low memory |
| **CheXNet** | Medical imaging | Domain-adapted, moderate size |
| **RadDINO** | Production accuracy | SOTA for chest X-rays, specialized |
| **Custom_1** | Quick testing | Very fast, lightweight |

## 📈 Performance Tracking

- **Real-time metrics**: Loss, F1-score, precision, recall per zone
- **Visualization**: Confusion matrices, learning curves, parameter importance
- **Artifacts**: Model checkpoints, predictions, evaluation reports
- **Integration**: Weights & Biases for experiment management

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the [docs](docs/) folder for detailed guides
- **Issues**: Report bugs or request features via GitHub Issues
- **Questions**: Start a discussion in GitHub Discussions

---

**⭐ Star this repository if you find it useful!**