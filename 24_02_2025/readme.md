**Stock Price Prediction Using LSTM - Learning Summary**

**Date:** February 25, 2025

---

## **1. Data Preprocessing and Feature Engineering**
- Used **moving average technique** to smooth out fluctuations and handle missing values.
- Avoided **IQR-based outlier removal** as stock prices follow a sequential trend.
- Extracted key financial indicators such as:
  - **Moving Averages** (7-day, 10-day, 50-day) to analyze price trends.
  - **Relative Strength Index (RSI)** to assess overbought/oversold conditions.
  - **Correlation Matrices and Heatmaps** to study relationships between stocks.

---

## **2. Handling Missing Data**
- Stocks not traded daily create missing values.
- Used a **rolling window (7-day moving average)** instead of forward filling to avoid synthetic data issues.
- Increased dataset size to improve model robustness against missing values.

---

## **3. Normalization for LSTM Input**
- Applied **Min-Max Scaling** to normalize stock prices between **[0,1]**.
- Ensured all stocks started and ended on the same time window for meaningful correlations.

---

## **4. LSTM Input and Architecture Understanding**
### **LSTM Input Structure:**
- LSTM processes sequential stock price data in **sliding window** format.
- Example:
  - Given **10 days of stock prices**, the model predicts the next day's price.
  - Each input sequence consists of **past n days' prices (features) mapped to one output (next day price).**

### **LSTM Memory System:**
- Maintains two types of memory:
  1. **Short-Term Memory (ℎ_t)**: Hidden state passed between time steps.
  2. **Long-Term Memory (ℂ_t)**: Stores historical patterns for trend retention.

### **LSTM Workflow (Time Step Processing):**
1. **Forget Gate:** Decides how much of the past information to keep/discard.
2. **Input Gate:** Determines what new information to store in memory.
3. **Cell State Update:** Combines past memory and new input to update **ℂ_t**.
4. **Output Gate:** Generates the next hidden state **ℎ_t** to pass forward.

---

## **5. LSTM Prediction Strategy**
- Used **step-by-step forecasting** instead of predicting 7 days in one pass:
  1. Predict **day 1**.
  2. Use **day 1 prediction** as input to predict **day 2**, and so on.
- Used **Stochastic Gradient Descent (SGD)** for optimization to maintain interpretability.

---

## **6. Handling Model Errors**
- If **predicted trend** differs from the actual stock movement:
  - Checked for **outliers in training data**.
  - Increased dataset size to enhance generalization.
  - Considered external factors (e.g., company announcements) causing sudden price spikes.

---

## **Conclusion**
- Understood **how LSTM processes stock price sequences** by maintaining long-term and short-term dependencies.
- **Feature engineering** and **proper normalization** are crucial for effective time-series forecasting.
- **Sliding window forecasting** with iterative predictions provides better accuracy than multi-step direct prediction.

---

**Next Steps:**
- Implement **bi-directional LSTM** for improved trend analysis.
- Experiment with **GRU models** to compare efficiency with LSTM.
- Introduce **attention mechanisms** to focus on crucial time points in stock movement.

---

