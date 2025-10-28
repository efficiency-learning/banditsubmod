# Contributing to Online Submodular Bandit Data Selection

Thank you for your interest in contributing to our project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA 11.8+ (for GPU support)
- Familiarity with PyTorch

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/banditsubmod.git
   cd banditsubmod
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/efficiency-learning/banditsubmod.git
   ```

## 💡 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, PyTorch version)
- **Error messages** or logs
- **Code snippets** if applicable

**Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

**Additional context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Motivation**: Why is this enhancement needed?
- **Use cases**: When would this be useful?
- **Proposed solution**: How should it work?
- **Alternatives**: What alternatives have you considered?

### Contributing Code

We welcome code contributions! You can contribute by:

- Fixing bugs
- Adding new features
- Improving documentation
- Adding tests
- Optimizing performance
- Adding new datasets or models
- Implementing new selection strategies

## 🛠️ Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

For vision experiments:
```bash
cd OnlineSubmod-vision
pip install -e .
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

For LLM experiments:
```bash
cd OnlineSubmod-LLM
pip install -r requirements.txt
pip install pytest black flake8
```

### 3. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 2. Make Your Changes

- Write clear, commented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

Run tests locally before submitting:

```bash
# For vision code
cd OnlineSubmod-vision
pytest tests/

# For LLM code
cd OnlineSubmod-LLM
python -m pytest
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add new submodular function for graph-based selection"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR template:

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe the tests you ran and their results.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass
- [ ] My changes generate no new warnings
```

### 7. Code Review

- Address reviewer comments
- Push additional commits to your branch
- Discuss and iterate until approval

### 8. Merge

Once approved, a maintainer will merge your PR. Thank you! 🎉

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Organize imports in three groups (standard library, third-party, local)

### Code Formatting

Use `black` for automatic formatting:

```bash
black OnlineSubmod-vision/ --line-length 100
black OnlineSubmod-LLM/ --line-length 100
```

### Linting

Use `flake8` for linting:

```bash
flake8 OnlineSubmod-vision/ --max-line-length 100
flake8 OnlineSubmod-LLM/ --max-line-length 100
```

### Type Hints

Add type hints to function signatures:

```python
def compute_gradients(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: str = "cuda"
) -> torch.Tensor:
    """Compute gradients for all samples in data_loader."""
    ...
```

### Documentation

All public functions, classes, and modules should have docstrings:

```python
def submodular_selection(
    gradients: torch.Tensor,
    budget: int,
    submod_function: str = "facilityLocation",
    **kwargs
) -> List[int]:
    """
    Select a subset of samples using submodular optimization.
    
    Args:
        gradients: Tensor of shape (N, D) containing per-sample gradients
        budget: Number of samples to select
        submod_function: Name of submodular function to use
        **kwargs: Additional arguments for submodular function
        
    Returns:
        List of selected sample indices
        
    Raises:
        ValueError: If budget > len(gradients)
        
    Example:
        >>> grads = torch.randn(1000, 512)
        >>> selected = submodular_selection(grads, budget=100)
        >>> len(selected)
        100
    """
    ...
```

## 🧪 Testing

### Writing Tests

Place tests in the `tests/` directory:

```python
import pytest
import torch
from cords.selectionstrategies.SL import OnlineSubmodStrategy

def test_online_submod_selection():
    """Test online submodular selection with dummy data."""
    # Setup
    num_samples = 1000
    embedding_dim = 512
    budget = 100
    
    # Create dummy data
    gradients = torch.randn(num_samples, embedding_dim)
    
    # Run selection
    strategy = OnlineSubmodStrategy(...)
    selected_indices = strategy.select(budget)
    
    # Assertions
    assert len(selected_indices) == budget
    assert len(set(selected_indices)) == budget  # No duplicates
    assert all(0 <= idx < num_samples for idx in selected_indices)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run with coverage
pytest --cov=cords tests/
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style or NumPy-style docstrings
- Include examples where helpful

### README Updates

If your changes affect usage:
- Update relevant README.md files
- Add examples showing new functionality
- Update configuration documentation

### Notebooks

For new features, consider adding:
- Jupyter notebook demonstrating usage
- Tutorial walking through the feature
- Benchmark comparing with baselines

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality
- **PATCH**: Backwards-compatible bug fixes

## 📧 Getting Help

If you need help:

- Check existing [issues](https://github.com/efficiency-learning/banditsubmod/issues)
- Read the [documentation](README.md)
- Ask questions in [discussions](https://github.com/efficiency-learning/banditsubmod/discussions)
- Contact maintainers

## 🎯 Good First Issues

Look for issues labeled `good-first-issue` for beginner-friendly contributions:

- Documentation improvements
- Adding examples
- Bug fixes with clear reproduction steps
- Adding unit tests
- Performance optimizations

## 🌟 Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Acknowledged in release notes
- Given credit in relevant documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! Your efforts help make this project better for everyone. 🚀
