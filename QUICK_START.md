# ⚡ Quick Reference Guide

## 🚀 30-Second Start

```bash
git clone https://github.com/Abhishek371222/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
jupyter notebook
```

---

## 📂 Notebook Navigation

### Start Here 👇
1. **FTSE100_data_collection_and_EDA.ipynb** (Technical_Analysis)
   - Time: 15 mins
   - What: Download & explore data
   - Output: Dataset + visualizations

### Then Choose Your Path:

#### Path A: Time Series Forecasting ⏱️
```
ARIMA.ipynb (45 mins)
  ↓
SARIMA.ipynb (30 mins)
  ↓
Facebook_Prophet.ipynb (20 mins)
  ↓
LSTM.ipynb (60 mins) ⭐ BEST ACCURACY
```

#### Path B: Sentiment Analysis 💬
```
Stock_news_data_collection.ipynb (30 mins)
  ↓
NLP_Text_Preprocessing_and_Classification.ipynb (45 mins)
  ↓
BERT_Long_Text_Classification.ipynb (60 mins) ⭐ STATE-OF-ART
```

#### Path C: Technical Analysis 📊
```
Chart_patterns_and_technical_indicators.ipynb (40 mins)
  ↓
Hypothesis_Testing.ipynb (20 mins)
  ↓
Trading_Dashboards.ipynb (30 mins)
```

---

## 🎯 Common Tasks

### Download Stock Data
```python
import yfinance as yf
data = yf.download("AZN.L", start="2010-01-01", end="2019-12-31")
```

### ARIMA Forecast
```python
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data, order=(3,1,3)).fit()
forecast = model.get_forecast(steps=30)
```

### LSTM Prediction
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
model = Sequential([LSTM(50, input_shape=(10, 1)), Dense(1)])
model.fit(X_train, y_train, epochs=50)
```

### BERT Sentiment
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="bert-base-uncased")
result = sentiment("Great stock performance today!")
```

### Calculate RSI
```python
def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

---

## 🔧 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Slow GPU usage | Add `CUDA_VISIBLE_DEVICES=0` before running |
| Memory error | Reduce batch_size from 256 to 32 |
| Seaborn style error | Use `sns.set_style('whitegrid')` |
| Jupyter not found | `pip install jupyter` |
| File not found | Check working directory: `os.getcwd()` |
| Timeout on download | Increase timeout: `yf.download(..., timeout=60)` |

---

## 📊 Model Cheat Sheet

```
╔══════════════════════════════════════════════╗
║ Model Selector                               ║
╠══════════════════════════════════════════════╣
║ Need Speed?          → Naive Bayes           ║
║ Need Accuracy?       → LSTM / BERT           ║
║ Need Interpretable?  → ARIMA / Logistic      ║
║ Have Seasonality?    → SARIMA / Prophet      ║
║ Complex Patterns?    → LSTM / GRU            ║
║ Text Data?           → BERT                  ║
║ Linear Data?         → Linear Regression     ║
║ Imbalanced Data?     → XGBoost               ║
║ Real-time?           → GRU                   ║
║ Production?          → Ensemble              ║
╚══════════════════════════════════════════════╝
```

---

## 📈 Performance Reference

```
TIME SERIES MODELS:
├─ ARIMA      RMSE: 150  │ Speed: Fast    │ Complex: Low
├─ SARIMA     RMSE: 140  │ Speed: Medium  │ Complex: Medium
├─ Prophet    RMSE: 160  │ Speed: Slow    │ Complex: Low
├─ LSTM       RMSE: 120  │ Speed: Slow    │ Complex: High ⭐
├─ GRU        RMSE: 125  │ Speed: Medium  │ Complex: High
└─ RNN        RMSE: 135  │ Speed: Fast    │ Complex: High

SENTIMENT MODELS:
├─ BERT       Acc: 87%   │ Speed: Slow    │ Best for Production ⭐
├─ Naive Bayes Acc: 78%  │ Speed: Fast    │ Baseline
├─ SVM        Acc: 81%   │ Speed: Medium  │ Balanced
└─ LogReg     Acc: 79%   │ Speed: Fast    │ Simple
```

---

## 🔑 Key Parameters

### ARIMA(p,d,q)
- `p`: 0-5 (test with ACF)
- `d`: Usually 0-2 (ADF test)
- `q`: 0-5 (test with PACF)

### LSTM Layers
- Hidden: 32-128 neurons
- Dropout: 0.2-0.3
- Batch: 16-64
- Epochs: 50-200

### BERT
- Batch size: 8-32
- Learning rate: 2e-5 to 5e-5
- Epochs: 3-5
- Max length: 512 tokens

---

## 📝 File Templates

### Simple Forecast
```python
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA

# Get data
data = yf.download("AZN.L", start="2010-01-01", end="2019-12-31")['Adj Close']

# Model
model = ARIMA(data, order=(3,1,3))
result = model.fit()

# Forecast
forecast = result.get_forecast(steps=30)
print(forecast.summary_frame())
```

### Deep Learning Forecast
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequences
X_train = create_sequences(data, 10)
y_train = data[10:]

# Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(25, activation='relu'),
    Dense(1)
])

# Train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100)

# Predict
predictions = model.predict(X_test)
```

### Sentiment Analysis
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="bert-base-uncased")

texts = [
    "Stock prices soaring today!",
    "Market crash expected",
    "Neutral market conditions"
]

for text in texts:
    result = sentiment(text)
    print(f"{text}: {result}")
```

---

## 🎓 Learning Resources

| Topic | Resource |
|-------|----------|
| ARIMA | [Statsmodels Docs](https://www.statsmodels.org) |
| LSTM | [Keras Guide](https://keras.io) |
| BERT | [Hugging Face](https://huggingface.co) |
| NLP | [Fast.ai NLP](https://course.fast.ai) |
| Forecasting | [Prophet Docs](https://facebook.github.io/prophet) |

---

## 🐛 Debug Tips

1. **Check data shape:**
   ```python
   print(X_train.shape)  # Should be (samples, timesteps, features)
   ```

2. **Verify stationarity:**
   ```python
   from statsmodels.tsa.stattools import adfuller
   adfuller(data)  # p-value < 0.05 = stationary
   ```

3. **Monitor GPU:**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

4. **Check predictions:**
   ```python
   print(f"Min: {pred.min()}, Max: {pred.max()}, Mean: {pred.mean()}")
   ```

5. **Verify imports:**
   ```python
   import pandas as pd; print(pd.__version__)
   ```

---

## 💡 Pro Tips

1. **Always check stationarity first** - Use ADF test
2. **Normalize features** - Prevents gradient explosion
3. **Use train/test split** - Avoid data leakage
4. **Monitor validation loss** - Detect overfitting
5. **Save best model** - Use checkpoints
6. **Cross-validate** - Use TimeSeriesSplit
7. **Start simple** - Baseline before complex models
8. **Log results** - Track experiments
9. **Use ensemble** - Combine predictions
10. **Backtest strategy** - Test on historical data

---

## 📞 Support

- 📖 Full docs: See `README.md`
- 🔧 Setup help: See `SETUP.md`
- 🤖 Model details: See `MODELS.md`
- 🐛 Issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

**Last Updated:** December 30, 2025 | **Version:** 2.0
