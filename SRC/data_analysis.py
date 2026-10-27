
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model

# Step 1 Volatility analysis engine. It downloads real market data and compares 
# four different ways of estimating how volatile the stock market is.

# 1. Download 10 years of S&P 500 ETF price data
prices = yf.download(
    "SPY",
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False,
)

if prices.empty:
    raise RuntimeError("No market data downloaded. Check your internet connection.")

# 2. Extract daily closing prices
close = prices["Close"]

# Handle different pandas/yfinance column formats
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.dropna().sort_index()

# 3. Calculate daily log returns
returns = np.log(close / close.shift(1)).dropna()

# 4a. Calculate 20-day and 60-day rolling volatility
volatility_20d = returns.rolling(20).std() * np.sqrt(252)
volatility_60d = returns.rolling(60).std() * np.sqrt(252)


# 4b. EWMA volatility model - gives more weight to recent squared returns 
# and gradually less weight to older ones. When market is  turbulent recent price movements
# should have more effect than oldones

lambda_ = 0.94

# Estimate daily variance from squared returns
ewma_variance = returns.pow(2).ewm(
    alpha=1 - lambda_,
    adjust=False
).mean()

# Convert daily variance into annualised volatility
ewma_volatility = np.sqrt(ewma_variance * 252)



# 4c. GARCH(1,1) volatility model

# Scale daily returns to percentage points for model estimation
returns_pct = returns * 100

# Specify and estimate the GARCH(1,1) model
model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_result = model.fit(disp="off")

# Convert conditional volatility back to decimal returns
garch_volatility = garch_result.conditional_volatility / 100

# Annualise the volatility
garch_volatility = garch_volatility * np.sqrt(252)

print("\nGARCH(1,1) model results:")
print(garch_result.summary())

# Inspect the dates with the highest GARCH volatility
print("\nTop 10 GARCH volatility observations:")
print(
    (garch_volatility * 100)
    .nlargest(10)
    .round(2)
)


# 5. Plot and compare volatility estimates
plt.figure(figsize=(12, 6))

plt.plot(
    volatility_20d,
    label="20-day historical volatility",
    alpha=0.7
)
plt.plot(
    volatility_60d,
    label="60-day historical volatility",
    alpha=0.7
)
plt.plot(
    ewma_volatility,
    label="EWMA volatility",
    linewidth=1.5
)

plt.plot(
    garch_volatility,
    label="GARCH(1,1) volatility",
    linewidth=1.5
)

plt.title("SPY: Historical vs EWMA Annualised Volatility")
plt.xlabel("Date")
plt.ylabel("Annualised volatility")
plt.legend()
plt.tight_layout()
plt.show()

