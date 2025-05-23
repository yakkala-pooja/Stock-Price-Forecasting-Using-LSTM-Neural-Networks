
# LSTM-Based Stock Price Forecasting: GE and Apple Inc.

## Objective
This project develops deep learning models using Long Short-Term Memory (LSTM) networks to forecast stock prices. It started with General Electric (GE) stock data to build foundational models and compare loss functions and optimizers. The focus then shifted to Apple Inc. (AAPL) for more robust, long-term forecasting, leveraging its rich historical data and relatively stable volatility.

---

## Phase 1: GE Stock – Baseline Forecasting and Loss/Optimizer Comparison

- **Data Source:** Historical GE stock data via `yfinance`.
- **Methods:**  
  - Data preprocessing, feature scaling, and sequence windowing.  
  - LSTM model to forecast future stock prices.  
  - Experimented with:
    - Loss Functions: Mean Squared Error (MSE) vs. Mean Absolute Error (MAE)  
    - Optimizers: Adam vs. RMSprop  
- **Outcome:** Established baseline performance and insights into the effect of different optimizers and loss functions on forecasting accuracy.

---

## Phase 2: Apple Inc. (AAPL) – Short and Long-Term Forecasting

### Data and Features
- Daily stock data from 2010 to 2025 scraped using `yfinance`.
- Engineered features include:
  - Log returns  
  - 50-day and 200-day Simple Moving Averages (SMA)  
  - Relative Strength Index (RSI)  
- Preprocessing steps:
  - Normalization with MinMaxScaler  
  - Forward-fill to handle missing values from indicator calculations  
  - 180-day lookback window to create input sequences  

### Forecasting Approaches

#### 1. Multi-Step Direct Prediction
- **Goal:** Predict the full future sequence (30 or 365 days) in a single forward pass.
- **Architecture:**  
  - LSTM (64 units, `tanh` activation) → Dense (64 units, `ReLU`) → Output layer  
- **Deeper Variant (Overfit):**  
  - 3 stacked LSTMs (128, 128, 64 units) with LeakyReLU and Dropout  
  - Additional Dense layer (128 units)  
- **Other experiments:**  
  - Bidirectional LSTM (discarded due to future data leakage)  
  - Attention mechanisms (minimal improvement likely due to noisy financial data)  

#### 2. Recursive Prediction
- **Goal:** Predict one day at a time; feed predictions back into the input sequence for subsequent predictions.
- **Architecture:**  
  - LSTM (100 units) → LSTM (50 units) → Dense output layer  
- **Pros:** Better for short-term accuracy  
- **Cons:** Error accumulation causes volatility over longer horizons (365 days)  

---

## Training and Tuning

- **Optimizers:**  
  - RMSprop (preferred for noisy GE data and long-term AAPL forecasts)  
  - Adam (faster convergence for recursive short-term forecasting)  
- **Loss Functions:** MSE and MAE compared.  
- **Hyperparameters:**  
  - Learning rate: 0.0005  
  - Batch size: 32 (better performance than 64)  
  - Epochs: 50 initially, then 100 for better convergence  
  - EarlyStopping on validation loss to avoid overfitting  

---

## Evaluation Metrics

- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- Mean Absolute Percentage Error (MAPE)  
- Visual plots comparing actual vs. predicted prices for both short and long horizons.

---

## Key Takeaways

- Recursive models offer flexibility and strong short-term predictions but suffer from error drift long-term.  
- Multi-step direct prediction models provide more stable long-range forecasts but require careful temporal pattern learning.  
- Shallower LSTM architectures generalize better in noisy financial data than deeper, more complex networks.  
- The choice of optimizer and loss function significantly impacts training stability and prediction accuracy.

---

## Files Included

- `Stock_Prediction.ipynb` — Jupyter notebook containing the full implementation and experiments.
- `Model.h5` — Saved trained LSTM model weights.
- `ge_stock.txt` — Raw GE stock data used for initial experiments.
- `model.png` — Architecture visualization of the LSTM model.
- `stock_training.png` — Training loss and accuracy plots during model training.

---

## How to Run

1. Install required Python packages (e.g., tensorflow, numpy, pandas, yfinance, matplotlib, scikit-learn).
2. Open `Stock_Prediction.ipynb` and run the cells sequentially.
3. Use `Model.h5` to load and make predictions with the saved model.
4. Visualize training results and predictions using provided images or by generating new plots from the notebook.

---
