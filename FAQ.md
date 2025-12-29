# ❓ Frequently Asked Questions (FAQ)

## Installation & Setup

### Q1: What are the system requirements?
**A:** Minimum: Python 3.8+, 8GB RAM, 5GB disk space. Recommended: Python 3.11+, 16GB RAM, SSD, NVIDIA GPU.

### Q2: Can I use Windows/Mac/Linux?
**A:** Yes! This project works on all platforms. Use `venv` on Unix and `venv\Scripts\activate` on Windows.

### Q3: Do I need a GPU?
**A:** No, GPU is optional. All models run on CPU, but LSTM/GRU training will be slower (5-10x).

### Q4: How do I install GPU support?
**A:** Follow [SETUP.md](SETUP.md) GPU section for CUDA 12.3+ and cuDNN installation.

### Q5: Which Python version should I use?
**A:** Python 3.13.2 (tested), or 3.10, 3.11, 3.12 work well. Avoid Python 3.8 (deprecated libraries).

---

## Data & Notebooks

### Q6: Where does the stock data come from?
**A:** Yahoo Finance via `yfinance` library. Covers 2010-2019 FTSE 100 data.

### Q7: Can I use different stocks or dates?
**A:** Yes! Modify the `yfinance` download:
```python
yf.download("YOUR_TICKER", start="2015-01-01", end="2024-12-31")
```

### Q8: What if yfinance download fails?
**A:** Code has automatic pickle file fallback. Check internet connection or increase timeout:
```python
yf.download(..., timeout=60)
```

### Q9: How long does each notebook take to run?
**A:** FTSE100_EDA: 15 mins | ARIMA: 45 mins | LSTM: 90 mins | BERT: 120 mins

### Q10: Can I run notebooks in different order?
**A:** Not recommended. Run `FTSE100_data_collection_and_EDA.ipynb` first to get data, then choose your path.

---

## Models & Predictions

### Q11: Which model is best for forecasting?
**A:** LSTM (RMSE: 120) is most accurate. For interpretability, use ARIMA. For robustness, use Prophet.

### Q12: How do I choose ARIMA parameters (p,d,q)?
**A:** Run `ndiffs()` for d, use ACF plot for q, use PACF plot for p. Or use auto_arima:
```python
from pmdarima import auto_arima
auto_arima(data, seasonal=False)
```

### Q13: What's the difference between LSTM and GRU?
**A:** GRU is faster (30% quicker), LSTM is more powerful. Use GRU for real-time, LSTM for accuracy.

### Q14: How do I improve model accuracy?
**A:** 
1. Feature engineering (add indicators)
2. Hyperparameter tuning (grid search)
3. Ensemble multiple models
4. Use more training data
5. Normalize/standardize inputs

### Q15: Why is my LSTM overfitting?
**A:** Add regularization:
```python
model.add(Dropout(0.2))  # 20% dropout
model.add(Dense(50, kernel_regularizer=L2(0.001)))
```

---

## Sentiment Analysis

### Q16: How accurate is BERT sentiment analysis?
**A:** 87% accuracy on financial news. Works well for binary classification (positive/negative).

### Q17: Can I fine-tune BERT on custom data?
**A:** Yes! See [BERT_Long_Text_Classification.ipynb](Sentiment_Analysis/BERT_Long_Text_Classification.ipynb) for fine-tuning.

### Q18: What if my text is very long?
**A:** BERT has 512 token limit. Split long text:
```python
def split_text(text, max_length=512):
    tokens = text.split()
    return [' '.join(tokens[i:i+max_length]) 
            for i in range(0, len(tokens), max_length)]
```

### Q19: How do I use models offline?
**A:** Download BERT locally:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.save_pretrained("./my_bert")
# Later use: from_pretrained("./my_bert")
```

### Q20: Can I combine sentiment with price prediction?
**A:** Yes! Use sentiment as feature:
```python
features = np.column_stack([price_features, sentiment_scores])
```

---

## Technical Issues

### Q21: I get "ModuleNotFoundError: No module named X"
**A:** Install missing package:
```bash
pip install [package_name]
# Or reinstall all:
pip install -r requirements.txt
```

### Q22: Memory issues during training
**A:** Reduce batch size:
```python
# From: batch_size=256
# To:
model.fit(X_train, y_train, batch_size=32)
```

### Q23: Training very slow (CPU mode)
**A:** Enable GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Should show GPU
# If empty, TensorFlow using CPU (still works, slower)
```

### Q24: Plots not showing in Jupyter
**A:** Add magic command:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

### Q25: Seaborn style error "seaborn not found"
**A:** Use correct syntax (already fixed in v2.0):
```python
import seaborn as sns
sns.set_style('whitegrid')  # Not: plt.style.use('seaborn')
```

---

## Performance & Optimization

### Q26: How do I make predictions faster?
**A:** 
- Use GRU instead of LSTM
- Reduce model size (fewer layers)
- Use quantization/pruning
- Deploy on GPU

### Q27: Can I parallelize model training?
**A:** Yes, with data parallelism:
```python
from tensorflow.keras.utils import multi_gpu_model
if len(tf.config.list_physical_devices('GPU')) > 1:
    model = multi_gpu_model(model, gpus=2)
```

### Q28: How do I save/load trained models?
**A:** 
```python
# Save
model.save('my_model.h5')  # Or model.save_weights('weights.h5')

# Load
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
```

### Q29: What about model monitoring during training?
**A:** Use TensorBoard:
```python
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs')
model.fit(..., callbacks=[tensorboard])
# Then: tensorboard --logdir=./logs
```

### Q30: How do I do cross-validation on time series?
**A:** Use TimeSeriesSplit:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(X):
    model.fit(X[train], y[train])
```

---

## Forecasting Questions

### Q31: How far ahead can I forecast?
**A:** Generally reliable 5-30 days. Beyond 30 days, uncertainty grows exponentially.

### Q32: Should I use log returns or prices?
**A:** Use log returns for:
- Volatility analysis
- Normalization
- Better statistical properties

Use prices for:
- Interpretability
- Direct predictions

### Q33: How do I backtest a trading strategy?
**A:** See [Trading_Dashboards.ipynb](Technical_Analysis/Trading_Dashboards.ipynb) for backtest examples.

### Q34: What's the best way to validate models?
**A:** Use time-based split:
```python
split_point = int(0.8 * len(data))
train, test = data[:split_point], data[split_point:]
```

### Q35: How do I handle missing data?
**A:** Methods:
```python
df.fillna(method='ffill')  # Forward fill
df.fillna(df.mean())  # Mean fill
df.interpolate()  # Interpolation
```

---

## Business & Usage Questions

### Q36: Can I use this for real trading?
**A:** **NOT recommended.** This is for educational purposes. Actual trading requires:
- Professional risk management
- Capital preservation strategies
- Regulatory compliance
- Licensed financial advisor consultation

### Q37: What are typical prediction errors?
**A:** 
- ARIMA RMSE: ~150 (2-3% of price)
- LSTM RMSE: ~120 (1.5-2% of price)
- Prophet RMSE: ~160 (2.5% of price)

### Q38: How often should I retrain models?
**A:** Weekly recommended. Daily if data changes significantly.

### Q39: Can I share modified code?
**A:** Yes! Please maintain attribution and follow MIT license.

### Q40: Is this production-ready?
**A:** Good for learning. For production:
- Add API layer (Flask/FastAPI)
- Database integration
- Error monitoring
- Scheduled retraining
- Model versioning

---

## Advanced Questions

### Q41: How do I add new indicators?
**A:** Create a function and add to DataFrame:
```python
def my_indicator(data, period=20):
    return data.rolling(period).mean()

df['my_ind'] = my_indicator(df['Close'])
```

### Q42: Can I combine multiple timeframes?
**A:** Yes! Use multi-timeframe analysis:
```python
daily = yf.download(ticker, interval='1d')
weekly = yf.download(ticker, interval='1wk')
```

### Q43: How do I handle Black Swan events?
**A:** 
- Use robust loss functions (Huber)
- Outlier detection
- Stop-loss orders
- Ensemble models

### Q44: What about market anomalies?
**A:** Documented in project:
- January Effect
- Monday Effect
- Sell in May
- Volatility clustering

### Q45: Can I use deep learning for high-frequency trading?
**A:** Difficult due to:
- Data latency issues
- Computational constraints
- Noise in high-frequency data
- Better for lower frequencies (hourly+)

---

## Community & Support

### Q46: How do I report a bug?
**A:** Create GitHub issue with:
- Python version
- Error message
- Reproducible code
- Platform (Windows/Mac/Linux)

### Q47: How can I contribute?
**A:** 
1. Fork repository
2. Create feature branch
3. Make improvements
4. Submit pull request

### Q48: Where can I learn more?
**A:** See references in [MODELS.md](MODELS.md) and [README.md](README.md)

### Q49: Is there a Discord/Slack community?
**A:** Not yet, but GitHub Discussions available.

### Q50: What's the roadmap?
**A:** Future plans:
- Real-time data API
- Web dashboard
- Mobile app
- AutoML integration
- Reinforcement learning models

---

## Still Have Questions?

1. **Check documentation:** [README.md](README.md), [SETUP.md](SETUP.md), [QUICK_START.md](QUICK_START.md)
2. **Search GitHub issues:** https://github.com/Abhishek371222/Sentiment-Analysis/issues
3. **Create an issue:** Include error details and steps to reproduce
4. **Check notebook comments:** Many notebooks have detailed explanations

---

**Last Updated:** December 30, 2025  
**Version:** 2.0  
**Helpful?** Star ⭐ the repository!
