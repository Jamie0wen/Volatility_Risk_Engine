
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import norm
from arch import arch_model


# ============================================================
# 1. Download SPY price data and calculate daily log returns
# ============================================================

prices = yf.download(
    "SPY",
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False,
)

if prices.empty:
    raise ValueError("No price data downloaded. Check your connection.")

# Handle yfinance's possible MultiIndex column format.
if isinstance(prices.columns, pd.MultiIndex):
    close = prices["Close"]["SPY"]
else:
    close = prices["Close"]

returns = np.log(close / close.shift(1)).dropna()

# Use returns in percentage points when fitting GARCH.
returns_pct = returns * 100


# ============================================================
# 2. Define the VaR confidence levels
# ============================================================

confidence_levels = [0.95, 0.99]

# Number of trading days used for Historical VaR.
historical_window = 252

# RiskMetrics EWMA decay factor.
lambda_ewma = 0.94


# ============================================================
# 3. Historical VaR
# ============================================================

# Shift by one day so today's VaR only uses previous returns.
historical_var = pd.DataFrame(index=returns.index)

for confidence in confidence_levels:
    tail_probability = 1 - confidence

    historical_var[f"Historical_VaR_{int(confidence * 100)}"] = (
        -returns.rolling(
            window=historical_window,
            min_periods=historical_window,
        )
        .quantile(tail_probability)
        .shift(1)
    )


# ============================================================
# 4. EWMA volatility and EWMA-based VaR
# ============================================================

# Initialise variance using the first 60 daily returns.
initial_window = 60

ewma_variance = pd.Series(
    np.nan,
    index=returns.index,
    dtype=float,
)

if len(returns) > initial_window:
    ewma_variance.iloc[initial_window] = (
        returns.iloc[:initial_window].var(ddof=1)
    )

    # Forecast each day's variance using information available
    # before that day's return occurs.
    for i in range(initial_window + 1, len(returns)):
        previous_variance = ewma_variance.iloc[i - 1]
        previous_return = returns.iloc[i - 1]

        ewma_variance.iloc[i] = (
            lambda_ewma * previous_variance
            + (1 - lambda_ewma) * previous_return**2
        )

ewma_volatility = np.sqrt(ewma_variance)

ewma_var = pd.DataFrame(index=returns.index)

for confidence in confidence_levels:
    z_score = norm.ppf(confidence)

    # Zero-mean, normal-distribution VaR.
    ewma_var[f"EWMA_VaR_{int(confidence * 100)}"] = (
        z_score * ewma_volatility
    )


# ============================================================
# 5. GARCH(1,1) volatility and GARCH-based VaR
# ============================================================

garch = arch_model(
    returns_pct,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
    rescale=False,
)

garch_results = garch.fit(disp="off")

# Conditional volatility is expressed in percentage points.
# Shift one day to align volatility with the following return.
garch_volatility = (
    garch_results.conditional_volatility.shift(1) / 100
)

garch_var = pd.DataFrame(index=returns.index)

for confidence in confidence_levels:
    z_score = norm.ppf(confidence)

    garch_var[f"GARCH_VaR_{int(confidence * 100)}"] = (
        z_score * garch_volatility
    )


# ============================================================
# 6. Combine results
# ============================================================

results = pd.concat(
    [
        returns.rename("Return"),
        historical_var,
        ewma_var,
        garch_var,
    ],
    axis=1,
)

# Remove dates where the models do not yet have enough history.
var_columns = [
    column for column in results.columns
    if "VaR" in column
]

results = results.dropna(subset=var_columns)

# A breach occurs when the actual loss exceeds the VaR forecast.
for column in var_columns:
    breach_column = column.replace("VaR", "Breach")

    results[breach_column] = (
        results["Return"] < -results[column]
    )


# ============================================================
# 7. Print summary statistics
# ============================================================

print("\nGARCH(1,1) model summary:")
print(garch_results.summary())

print("\nLatest VaR estimates (% of portfolio value):")

latest_var = results[var_columns].iloc[-1] * 100
print(latest_var.round(3).to_string())

print("\nVaR breach rates:")
for column in var_columns:
    breach_column = column.replace("VaR", "Breach")
    observed_rate = results[breach_column].mean()

    print(
        f"{column}: "
        f"{observed_rate:.2%} observed breaches"
    )


# ============================================================
# 8. Plot returns and VaR forecasts
# ============================================================

plot_columns = [
    "Historical_VaR_95",
    "EWMA_VaR_95",
    "GARCH_VaR_95",
]

plt.figure(figsize=(14, 7))

plt.plot(
    results.index,
    results["Return"] * 100,
    label="Daily log return",
    color="black",
    linewidth=0.7,
)

for column in plot_columns:
    plt.plot(
        results.index,
        -results[column] * 100,
        label=column.replace("_", " "),
        linewidth=1,
    )

plt.title("SPY Daily Returns and One-Day 95% VaR")
plt.xlabel("Date")
plt.ylabel("Return / VaR threshold (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
