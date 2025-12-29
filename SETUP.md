# 📖 Setup & Installation Guide

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [GPU Setup (Optional)](#gpu-setup-optional)

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python:** 3.8 or higher (Tested on 3.13.2)
- **RAM:** 8 GB (16 GB recommended for deep learning)
- **Disk Space:** 5 GB (including virtual environment)
- **Internet:** Required for data download & model weights

### Recommended Setup
- **OS:** Ubuntu 20.04 LTS or Windows 11 Professional
- **Python:** 3.11 or 3.13
- **RAM:** 16 GB
- **GPU:** NVIDIA (for TensorFlow acceleration)
- **Processor:** Intel i7/i9 or AMD Ryzen 7/9

---

## Installation Steps

### Step 1: Install Python

#### Windows
```bash
# Download from: https://www.python.org/downloads/
# Run installer and check "Add Python to PATH"
python --version  # Verify installation
```

#### macOS
```bash
brew install python3
python3 --version
```

#### Linux (Ubuntu)
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
python3 --version
```

### Step 2: Clone Repository

```bash
# Using Git
git clone https://github.com/Abhishek371222/Sentiment-Analysis.git
cd Sentiment-Analysis

# Or download ZIP
# Extract and navigate to folder
```

### Step 3: Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

**Expected Output:**
```
(venv) $ _
```

### Step 4: Upgrade Pip

```bash
python -m pip install --upgrade pip
pip --version
```

### Step 5: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Step 6: Install Jupyter (if not included)

```bash
pip install jupyter jupyterlab
jupyter --version
```

### Step 7: Download NLP Models (Optional but Recommended)

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
print('NLTK models downloaded successfully')
"
```

---

## Verification

### Test Installation

```bash
# Test all major libraries
python << EOF
import sys
print(f"Python: {sys.version}")

imports = {
    'pandas': 'pd',
    'numpy': 'np',
    'matplotlib': 'plt',
    'seaborn': 'sns',
    'yfinance': 'yf',
    'tensorflow': 'tf',
    'torch': 'torch',
    'sklearn': 'sklearn',
    'statsmodels': 'sm',
    'transformers': 'transformers'
}

for lib, alias in imports.items():
    try:
        exec(f"import {lib} as {alias}")
        print(f"✓ {lib}")
    except ImportError:
        print(f"✗ {lib} - NOT INSTALLED")
EOF
```

### Expected Output
```
Python: 3.13.2 (main, Dec 30 2025, ...)
✓ pandas
✓ numpy
✓ matplotlib
✓ seaborn
✓ yfinance
✓ tensorflow
✓ torch
✓ sklearn
✓ statsmodels
✓ transformers
```

### Launch Jupyter

```bash
jupyter notebook
```

Then navigate to a notebook and run test cell:

```python
import pandas as pd
import yfinance as yf

# Test data download
data = yf.download("AAPL", start="2025-01-01", end="2025-12-30", progress=False)
print(f"Downloaded {len(data)} records")
print(data.tail())
```

---

## Troubleshooting

### Issue: "Python command not found"

**Windows:**
```bash
# Check if in PATH
where python
# If not, reinstall Python and check "Add to PATH"
```

**macOS/Linux:**
```bash
# Try python3 instead
python3 --version
# Create alias
echo "alias python=python3" >> ~/.bash_profile
```

### Issue: Virtual Environment activation fails

```bash
# Re-create virtual environment
rm -rf venv
python -m venv venv

# Windows: venv\Scripts\activate
# Unix: source venv/bin/activate
```

### Issue: Module import errors

```bash
# Verify requirements installation
pip list | grep -E "tensorflow|torch|pandas"

# Reinstall specific package
pip install --force-reinstall tensorflow==2.18.0
```

### Issue: Jupyter notebook not found

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Start Jupyter
jupyter notebook --port 8888
```

### Issue: CUDA not available (GPU errors)

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# TensorFlow will use CPU by default - no action needed
# Models run slower but still functional
```

### Issue: Memory errors during model training

```python
# Reduce batch size in notebooks
# Instead of:
model.fit(X_train, y_train, batch_size=256)

# Try:
model.fit(X_train, y_train, batch_size=32)

# Or use data generators for batching
```

### Issue: Download timeout from yfinance

```python
# Add timeout and retry logic
import yfinance as yf

data = yf.download(
    "AZN.L",
    start="2010-01-01",
    end="2019-12-31",
    progress=False,
    timeout=30
)

# If still fails, use backup:
data = pd.read_pickle("ftse100_stocks.pkl")
```

---

## GPU Setup (Optional)

### For NVIDIA GPU with CUDA

#### Step 1: Install CUDA Toolkit
```bash
# https://developer.nvidia.com/cuda-downloads
# Download and install CUDA 12.3+

nvidia-smi  # Verify installation
```

#### Step 2: Install cuDNN
```bash
# https://developer.nvidia.com/cudnn
# Download and extract cuDNN
# Add to PATH
```

#### Step 3: Verify TensorFlow GPU Support

```python
import tensorflow as tf
print("GPU Available:", tf.test.is_built_with_cuda())
print("GPUs:", len(tf.config.list_physical_devices('GPU')))

# Test GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
print("GPU computation successful!")
```

### For M1/M2 Mac (ARM-based)

```bash
# Use conda for better compatibility
conda create -n stock-pred python=3.11
conda activate stock-pred
conda install -c apple tensorflow-macos

pip install -r requirements.txt
```

---

## Next Steps

After successful installation:

1. **Read Main README:** See project overview and structure
2. **Start with EDA:** Run `FTSE100_data_collection_and_EDA.ipynb`
3. **Explore Models:** Try `Time_Series/ARIMA.ipynb`
4. **Test Sentiment:** Run `Sentiment_Analysis/BERT_Long_Text_Classification.ipynb`

---

## Getting Help

| Resource | Link |
|----------|------|
| GitHub Issues | [Create Issue](https://github.com/Abhishek371222/Sentiment-Analysis/issues) |
| Python Docs | [python.org](https://docs.python.org/) |
| TensorFlow | [tensorflow.org](https://www.tensorflow.org) |
| PyTorch | [pytorch.org](https://pytorch.org) |
| Conda | [conda.io](https://conda.io) |

---

**Status:** ✅ Updated December 30, 2025
