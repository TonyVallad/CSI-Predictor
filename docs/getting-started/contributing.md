<div align="center">

# Contributing Guidelines

How to contribute to the CSI-Predictor project.

</div>

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with PyTorch and medical imaging concepts

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/CSI-Predictor.git
   cd CSI-Predictor
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Install development dependencies if available
   pip install -r requirements-dev.txt
   ```

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a detailed issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU, etc.)
   - Error messages and logs

### Suggesting Enhancements

1. **Check existing feature requests**
2. **Create an enhancement issue** with:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation approach
   - Any relevant examples or references

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Testing information

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Write clear, descriptive function and variable names
- Add docstrings to all public functions and classes

### Testing

- Write tests for new functionality
- Ensure existing tests still pass
- Test with different model architectures when relevant
- Include integration tests for end-to-end workflows

### Documentation

- Update documentation for any new features
- Add examples to help users understand new functionality
- Keep README and guides up to date
- Use clear, concise language

## Areas for Contribution

### High Priority

- **Model Architectures**: Adding new backbone architectures
- **Data Augmentation**: Advanced augmentation techniques for medical images
- **Metrics**: Additional evaluation metrics for medical imaging
- **Performance Optimization**: Speed and memory optimizations

### Medium Priority

- **Visualization**: Better plotting and analysis tools
- **Configuration**: Enhanced configuration management
- **Documentation**: More comprehensive guides and tutorials
- **Testing**: Improved test coverage

### Good First Issues

- **Bug fixes**: Small, well-defined bugs
- **Documentation**: Fixing typos, improving clarity
- **Examples**: Adding usage examples
- **Utilities**: Helper functions and tools

## Pull Request Process

1. **Ensure your PR has a clear purpose** and description
2. **Reference related issues** using keywords like "Fixes #123"
3. **Keep changes focused** - one feature/fix per PR
4. **Include tests** for new functionality
5. **Update documentation** as needed
6. **Ensure CI passes** - all tests and checks must pass

### PR Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by project maintainers
3. **Testing** of new functionality
4. **Documentation review** for completeness
5. **Final approval** and merge

## Community Guidelines

### Be Respectful

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Focus on constructive feedback
- Help newcomers get started

### Be Professional

- Keep discussions focused on the project
- Provide helpful, actionable feedback
- Be patient with questions and learning processes

## Recognition

Contributors will be:
- Listed in the project contributors
- Mentioned in release notes for significant contributions
- Given credit in academic papers if applicable

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create a GitHub Issue
- **Feature requests**: Create a GitHub Issue with enhancement label
- **Security issues**: Contact maintainers directly

Thank you for contributing to CSI-Predictor! Your contributions help advance medical AI research and benefit the community. 