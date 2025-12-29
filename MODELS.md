# 🤖 Model Documentation

## Overview

This document provides detailed explanations of all machine learning and deep learning models implemented in the Stock Prediction project.

---

## 📑 Table of Contents

1. [Time Series Models](#time-series-models)
2. [Sentiment Analysis Models](#sentiment-analysis-models)
3. [Technical Indicators](#technical-indicators)
4. [Ensemble Methods](#ensemble-methods)
5. [Performance Metrics](#performance-metrics)

---

## Time Series Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)

**Mathematical Formula:**
$$ARIMA(p,d,q): \Delta^d y_t = \mu + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{i=1}^{q} \theta_i \epsilon_{t-i}$$

**Components:**
- **p (AR):** Autoregressive order - dependency on past values
- **d (I):** Integration order - degree of differencing for stationarity
- **q (MA):** Moving Average order - dependency on past errors

**Implementation:**
```python
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(data, order=(3, 1, 3))
results = model.fit(disp=0)
forecast = results.get_forecast(steps=30)
```

**Advantages:**
- ✅ Interpretable parameters
- ✅ Fast computation
- ✅ Well-suited for stationary data
- ✅ No GPU required

**Disadvantages:**
- ❌ Assumes linear relationships
- ❌ Requires stationarity
- ❌ Poor with non-linear patterns
- ❌ Sensitive to outliers

**Best For:** Stable, non-trending data

**Status:** ✅ Fixed & Working (v2.0)

---

### 2. SARIMA (Seasonal ARIMA)

**Formula:**
$$SARIMA(p,d,q)(P,D,Q,s) = ARIMA(p,d,q) \times Seasonal(P,D,Q,s)$$

**Key Difference:** Includes seasonal components for quarterly/yearly patterns

**Implementation:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit(disp=0)
```

**Advantages:**
- ✅ Captures seasonality
- ✅ Better for periodic data
- ✅ Handles yearly patterns

**Disadvantages:**
- ❌ More parameters to tune
- ❌ Computational overhead
- ❌ Risk of overfitting

**Best For:** Data with clear seasonal patterns (months, quarters)

**Status:** ✅ Working

---

### 3. Facebook Prophet

**Decomposition:**
$$y_t = g(t) + s(t) + h(t) + \epsilon_t$$

Where:
- **g(t):** Trend (piecewise linear or logistic)
- **s(t):** Seasonality (Fourier series)
- **h(t):** Holiday effects
- **ε_t:** Error term

**Implementation:**
```python
from prophet import Prophet

df = pd.DataFrame({'ds': dates, 'y': values})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

**Advantages:**
- ✅ Handles missing data
- ✅ Robust to outliers
- ✅ Holiday effects
- ✅ User-friendly

**Disadvantages:**
- ❌ Less flexible for complex patterns
- ❌ Slower training
- ❌ Black-box model

**Best For:** Business forecasting with holidays/events

**Status:** ✅ Fixed & Working (v2.0)

---

### 4. LSTM (Long Short-Term Memory)

**Architecture:**
```
Input → [Forget Gate | Input Gate | Output Gate] → Hidden State → Output
```

**Mathematical Gates:**
```
Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:      C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t ⊙ tanh(C_t)
```

**Implementation:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(timesteps, features)),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

**Advantages:**
- ✅ Captures long-term dependencies
- ✅ Excellent on non-linear data
- ✅ Best accuracy (120 RMSE)
- ✅ Handles variable-length sequences

**Disadvantages:**
- ❌ Requires GPU for fast training
- ❌ Black-box (difficult interpretation)
- ❌ Prone to overfitting
- ❌ Needs large datasets

**Best For:** Complex non-linear patterns, high-frequency data

**Status:** ✅ Working

---

### 5. GRU (Gated Recurrent Unit)

**Simplified Gates:**
```
Reset Gate:    r_t = σ(W_r · [h_{t-1}, x_t])
Update Gate:   z_t = σ(W_z · [h_{t-1}, x_t])
Candidate:     h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
Hidden State:  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Advantages:**
- ✅ Faster than LSTM (30% quicker)
- ✅ Similar accuracy
- ✅ Fewer parameters

**Disadvantages:**
- ❌ Less powerful than LSTM
- ❌ Still requires GPU

**Best For:** Real-time predictions, resource-constrained environments

**Status:** ✅ Working

---

### 6. RNN (Vanilla Recurrent Neural Network)

**Formula:**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**Advantages:**
- ✅ Simple architecture
- ✅ Fast training
- ✅ Minimal parameters

**Disadvantages:**
- ❌ Vanishing gradient problem
- ❌ Poor long-term memory
- ❌ Lower accuracy than LSTM/GRU

**Best For:** Short sequences, baseline models

**Status:** ✅ Working

---

## Sentiment Analysis Models

### 1. BERT (Bidirectional Encoder Representations from Transformers)

**Architecture:** Transformer with 12 layers, 768 hidden units, 12 attention heads

**Pre-training Objectives:**
1. **MLM (Masked Language Model):** Predict masked tokens
2. **NSP (Next Sentence Prediction):** Predict sentence order

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=3  # Positive, Negative, Neutral
)

inputs = tokenizer("Great stock performance!", return_tensors="pt")
outputs = model(**inputs)
sentiment = torch.argmax(outputs.logits, dim=1)
```

**Performance:**
- **Accuracy:** 87%
- **F1-Score:** 0.85
- **Speed:** 2-3 sec per batch

**Advantages:**
- ✅ State-of-the-art performance
- ✅ Bidirectional context
- ✅ Transfer learning
- ✅ 85%+ accuracy

**Disadvantages:**
- ❌ Slow inference
- ❌ Large model (440 MB)
- ❌ Requires GPU for speed

**Best For:** Production sentiment analysis, complex text

**Status:** ✅ Working

---

### 2. Naive Bayes

**Formula:**
$$P(Sentiment|Text) = \frac{P(Text|Sentiment) \cdot P(Sentiment)}{P(Text)}$$

**Implementation:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

clf = MultinomialNB()
clf.fit(X, y)
predictions = clf.predict(X_test)
```

**Performance:**
- **Accuracy:** 78%
- **Speed:** <1ms per prediction

**Advantages:**
- ✅ Fast & simple
- ✅ Low memory
- ✅ Interpretable

**Disadvantages:**
- ❌ Lower accuracy
- ❌ Assumes feature independence
- ❌ Struggles with context

**Best For:** Fast baseline, resource-limited systems

**Status:** ✅ Working

---

### 3. Support Vector Machine (SVM)

**Formula:**
$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(x_i, x) + b\right)$$

**Kernels:** Linear, RBF (Radial Basis Function), Polynomial

**Implementation:**
```python
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

X = vectorizer.fit_transform(texts)
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X, y)
```

**Performance:**
- **Accuracy:** 81%
- **Speed:** 50ms per batch

**Advantages:**
- ✅ Good accuracy (81%)
- ✅ Works with high dimensions
- ✅ Memory efficient

**Disadvantages:**
- ❌ Slower than Naive Bayes
- ❌ Hyperparameter tuning needed
- ❌ Binary classification focus

**Best For:** Balanced accuracy/speed, structured data

**Status:** ✅ Working

---

### 4. Logistic Regression

**Formula:**
$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
```

**Performance:**
- **Accuracy:** 79%
- **Speed:** <1ms

**Advantages:**
- ✅ Fast
- ✅ Interpretable coefficients
- ✅ Probabilistic output

**Disadvantages:**
- ❌ Assumes linear separability
- ❌ Lower accuracy

**Best For:** Linear relationships, baseline models

**Status:** ✅ Working

---

## Technical Indicators

### 1. Simple Moving Average (SMA)

**Formula:**
$$SMA_n = \frac{P_1 + P_2 + ... + P_n}{n}$$

**Signal:** Trend direction, support/resistance

**Implementation:**
```python
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
```

---

### 2. Moving Average Convergence Divergence (MACD)

**Formula:**
$$MACD = EMA_{12} - EMA_{26}$$
$$Signal = EMA_9(MACD)$$
$$Histogram = MACD - Signal$$

**Signals:**
- MACD > Signal: Bullish
- MACD < Signal: Bearish
- Histogram crossover: Momentum change

---

### 3. Relative Strength Index (RSI)

**Formula:**
$$RSI = 100 - \frac{100}{1 + RS}$$
$$RS = \frac{\text{Average Gain}}{\text{Average Loss}}$$

**Thresholds:**
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)

---

### 4. Bollinger Bands

**Formula:**
$$Middle = SMA_{20}$$
$$Upper = SMA_{20} + 2 \times \sigma$$
$$Lower = SMA_{20} - 2 \times \sigma$$

**Signals:**
- Price > Upper: Overbought
- Price < Lower: Oversold
- Squeeze: Breakout coming

---

### 5. Rate of Change (ROC)

**Formula:**
$$ROC = \frac{Price_t - Price_{t-n}}{Price_{t-n}} \times 100$$

**Signal:** Momentum strength

---

## Ensemble Methods

### Hybrid Approach (Combined Models)

```python
# Ensemble prediction
lstm_pred = lstm_model.predict(X_test)
arima_pred = arima_model.forecast(steps=len(X_test))
prophet_pred = prophet_forecast['yhat'].values

# Weighted average
ensemble_pred = (0.5 * lstm_pred + 
                 0.3 * arima_pred + 
                 0.2 * prophet_pred)
```

**Advantages:**
- Reduces model risk
- Better generalization
- Captures different patterns

---

## Performance Metrics

### Regression Metrics

**Mean Squared Error (MSE):**
$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE):**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE):**
$$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

**R-Squared:**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

---

### Classification Metrics

**Accuracy:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**
$$Precision = \frac{TP}{TP + FP}$$

**Recall:**
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score:**
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Confusion Matrix:** TP, TN, FP, FN visualization

---

## Model Selection Guide

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Stationary data | ARIMA | Interpretable, fast |
| Seasonal patterns | SARIMA/Prophet | Handles seasonality |
| Complex non-linear | LSTM | Best accuracy |
| Real-time inference | GRU | Speed priority |
| Sentiment (text) | BERT | State-of-the-art |
| Baseline/fast | Naive Bayes | Speed priority |

---

**Status:** ✅ Updated December 30, 2025

For implementation examples, see respective `.ipynb` files in each folder.
