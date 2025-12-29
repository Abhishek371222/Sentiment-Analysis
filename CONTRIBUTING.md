# 🤝 Contributing Guide

Thank you for your interest in contributing to the Stock Prediction project! Your contributions help make this project better for everyone.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please read and follow our code of conduct:

- Be respectful and inclusive
- Focus on what is best for the community
- Show empathy towards other community members
- Report unacceptable behavior to maintainers

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing opinions
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, intimidation, or discrimination
- Offensive comments or language
- Trolling or spam
- Unwelcome sexual attention
- Any form of abuse

---

## Getting Started

### 1. Fork the Repository

```bash
# Visit: https://github.com/Abhishek371222/Sentiment-Analysis
# Click "Fork" button
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b bugfix/your-bug-fix
# Or for documentation:
git checkout -b docs/your-docs-update
```

### 3. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Unix
# Or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
pip install pytest pylint black flake8
```

---

## Development Workflow

### 1. Make Your Changes

Create your feature or fix with clear, logical commits:

```bash
# Make changes
git add [modified_files]
git commit -m "feat: Add feature description"  # Use conventional commits

# Or for fixes:
git commit -m "fix: Bug description"

# Or for documentation:
git commit -m "docs: Updated README"
```

### 2. Keep Up With Main Branch

```bash
git fetch origin
git rebase origin/main
```

### 3. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 4. Create a Pull Request

1. Go to: https://github.com/YOUR_USERNAME/Sentiment-Analysis
2. Click "Compare & pull request"
3. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (fixes issue #...)
- [ ] New feature (relates to issue #...)
- [ ] Documentation update
- [ ] Performance improvement

## How to Test
Steps to test the changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) conventions:

```python
# Good ✅
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

# Bad ❌
def rsi(p,n=14):
    d=p.diff()
    g=d.where(d>0,0).rolling(n).mean()
    l=(-d.where(d<0,0)).rolling(n).mean()
    return 100-(100/(1+g/l))
```

### Naming Conventions

```python
# Variables and functions: lowercase_with_underscores
closing_price = 100.50
def calculate_moving_average():
    pass

# Classes: CapitalCase
class StockAnalyzer:
    pass

# Constants: UPPERCASE_WITH_UNDERSCORES
MAX_PREDICTION_DAYS = 365
DEFAULT_BATCH_SIZE = 32
```

### Code Quality Tools

```bash
# Format code
black . --line-length 100

# Check style
flake8 . --max-line-length 100

# Linting
pylint [your_file.py]

# Type checking
mypy . --ignore-missing-imports
```

### Docstring Format

```python
def forecast_stock_price(data, model_type='lstm', days=30):
    """
    Forecast stock prices using specified model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical stock data with OHLCV columns
    model_type : str, default='lstm'
        Type of model: 'lstm', 'arima', 'prophet'
    days : int, default=30
        Number of days to forecast
        
    Returns
    -------
    pd.DataFrame
        Forecasted prices with confidence intervals
        
    Raises
    ------
    ValueError
        If data is empty or model_type invalid
    TypeError
        If data is not DataFrame
        
    Examples
    --------
    >>> data = pd.read_csv('stock.csv')
    >>> forecast = forecast_stock_price(data, 'lstm', 30)
    """
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

```python
# tests/test_arima.py
import pytest
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

def test_arima_forecast():
    """Test ARIMA model forecasting."""
    # Arrange
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = ARIMA(data, order=(1, 1, 1))
    
    # Act
    result = model.fit()
    forecast = result.get_forecast(steps=5)
    
    # Assert
    assert len(forecast.predicted_mean) == 5
    assert forecast.predicted_mean.dtype == 'float64'

def test_arima_invalid_order():
    """Test ARIMA with invalid parameters."""
    data = pd.Series([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError):
        ARIMA(data, order=(-1, 1, 1)).fit()
```

---

## Documentation

### Update README

- Update [README.md](README.md) with feature description
- Add example code if applicable
- Update table of contents

### Update Docstrings

- Add docstring to every function/class
- Use standard format (NumPy/Google style)
- Include parameters, returns, and examples

### Notebook Documentation

For Jupyter notebooks:
- Add markdown cells explaining sections
- Include data loading, preprocessing, modeling
- Add visualizations and interpretations
- Document hyperparameters and assumptions

---

## Submitting Changes

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation
- **style**: Code style (no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **chore**: Dependencies, build tools

Examples:
```bash
git commit -m "feat(arima): Add auto ARIMA parameter selection"
git commit -m "fix(lstm): Correct shape mismatch in sequences"
git commit -m "docs(readme): Update installation instructions"
git commit -m "perf(sentiment): Optimize BERT inference"
```

### PR Title Format

```
[Type] Brief description

Example:
[Feature] Add XGBoost forecasting model
[Fix] Correct matplotlib seaborn compatibility
[Docs] Add comprehensive MODELS reference
```

---

## Review Process

### What Happens After You Submit

1. **Automated Checks**
   - GitHub Actions runs tests
   - Code quality checks (flake8, pylint)
   - Coverage reports

2. **Manual Review**
   - Maintainers review code
   - Check for style compliance
   - Verify functionality
   - Suggest improvements

3. **Discussion**
   - Address feedback
   - Make requested changes
   - Update PR with new commits

4. **Approval & Merge**
   - Once approved, PR is merged
   - Your contribution is live!

### Tips for Approval

✅ **Likely to be approved:**
- Clear, focused changes
- Good documentation
- Tests included
- Follows coding standards
- Solves a real problem

❌ **Likely to be rejected:**
- Large, unfocused changes
- No tests or documentation
- Violates style guide
- Breaking changes without discussion
- Incomplete implementation

---

## Types of Contributions

### 🎯 Features

Add new models, indicators, or functionality:

```python
# Example: Add new technical indicator
def calculate_atr(data, period=14):
    """Calculate Average True Range indicator."""
    pass
```

### 🐛 Bug Fixes

Fix issues in existing code:

```python
# Example: Fix data type issue
# Before: series.astype('M')  # Deprecated
# After: series.dt.to_period('M')  # Modern
```

### 📚 Documentation

Improve README, guides, and docstrings:

- Clear examples
- Troubleshooting tips
- Use case descriptions
- Link to resources

### ♻️ Refactoring

Improve code quality:

- Remove duplication
- Improve readability
- Better error handling
- Performance optimization

### ✅ Tests

Add or improve tests:

```python
def test_edge_case():
    """Test model with edge case data."""
    pass
```

---

## Common Contribution Areas

| Area | Priority | Difficulty |
|------|----------|-----------|
| Add new model | ⭐⭐⭐ | Medium |
| Fix bugs | ⭐⭐⭐ | Low-Medium |
| Improve docs | ⭐⭐ | Low |
| Add tests | ⭐⭐ | Low-Medium |
| Performance | ⭐⭐ | High |
| Refactor code | ⭐ | Medium |

---

## Questions?

- 📖 Read [README.md](README.md)
- 🔧 Check [SETUP.md](SETUP.md)
- 💬 Open GitHub Discussion
- 📧 Contact maintainers

---

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Commit history
- Release notes (for major contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🙏**

Together we make this project better! ⭐

---

**Last Updated:** December 30, 2025  
**Version:** 2.0
