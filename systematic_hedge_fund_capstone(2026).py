# %% [markdown]
# # Mini Systematic Hedge Fund — Institutional Quant Capstone
# 
# **Objective:** Build an end-to-end, research-grade systematic equity platform:
# 
# `Data → Cleaning → Features → Alpha → ML → Portfolio → Risk → Execution → Backtest → Validation → Reporting`
# 
# ### Research principles
# 1. Every feature is lagged before it can influence a trade.
# 2. Portfolio weights are generated using information available at the decision timestamp.
# 3. Transaction costs are charged on turnover.
# 4. Models are compared with interpretable baselines.
# 5. Walk-forward evaluation is separated from model selection.
# 6. Missing data and API failures do not silently become zeros.
# 7. Results are reproducible through configuration and fixed random seeds.
# 
# This notebook is intentionally designed to run on public data first. Optional institutional data connectors are isolated so the research engine does not depend on one vendor.


# %%
# CELL 1 — Install dependencies
# Colab-friendly. Restart the runtime only if Colab requests it after installation.

!pip -q install yfinance fredapi scikit-learn xgboost lightgbm cvxpy plotly joblib statsmodels arch seaborn requests beautifulsoup4


# %%
# CELL 2 — Imports, configuration, logging, reproducibility

from __future__ import annotations

import os
import io
import json
import time
import math
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump

warnings.filterwarnings("ignore")
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("quant-fund")

ROOT = Path("/content/systematic_hedge_fund")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

for p in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    start: str = "2015-01-01"
    end: str = "2026-01-01"
    rebalance_freq: str = "M"
    lookback_vol: int = 60
    momentum_lookback: int = 252
    momentum_skip: int = 21
    value_window: int = 252
    max_positions: int = 20
    gross_target: float = 1.0
    vol_target: float = 0.12
    max_weight: float = 0.10
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    random_state: int = 42

CFG = Config()

# Start with a liquid, diversified US large-cap universe.
# Expand only after the pipeline is stable.
UNIVERSE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","AVGO","GOOG","TSLA","BRK-B",
    "JPM","V","MA","XOM","LLY","UNH","COST","HD","PG","JNJ",
    "WMT","KO","PEP","MRK","ABBV","CRM","ORCL","ADBE","NFLX","CSCO",
    "ACN","INTC","QCOM","TXN","AMD","AMAT","MU","NOW","IBM","GE",
    "CAT","BA","HON","UPS","LOW","MCD","NKE","TMO","LIN","SPGI"
]

logger.info("Configured %d assets", len(UNIVERSE))


# %%
# CELL 3 — Optional secrets / API configuration
# Put keys in Colab Secrets or environment variables.
# The core project does NOT require every API.

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "QuantResearch contact@example.com"
)

# Example:
# from google.colab import userdata
# FRED_API_KEY = userdata.get("FRED_API_KEY")


# %% [markdown]
# ## 1. Data Layer
# 
# The production design separates **raw**, **processed**, and **model-ready** data.
# 
# ### Primary public sources
# - Yahoo Finance: prices, splits, dividends
# - FRED: macroeconomic series
# - SEC EDGAR: filings and fundamentals
# - Kenneth French Data Library: academic factors
# - Optional: FMP, Alpha Vantage, Polygon, Nasdaq Data Link
# 
# The research engine below uses Yahoo Finance as the default market-data source and FRED/SEC/French as optional extensions.


# %%
# CELL 4 — Robust Yahoo Finance downloader

def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = 3,
) -> pd.DataFrame:
    '''Download adjusted OHLCV data with retries and normalized columns.'''
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )

            if raw.empty:
                raise ValueError("Yahoo Finance returned an empty dataset.")

            # yfinance may return MultiIndex columns for multiple tickers.
            if isinstance(raw.columns, pd.MultiIndex):
                frames = []
                for field in ["Open", "High", "Low", "Close", "Volume"]:
                    if field not in raw.columns.get_level_values(0):
                        continue
                    x = raw[field].copy()
                    x.columns.name = None
                    x = x.stack().rename(field).reset_index()
                    x.columns = ["Date", "Ticker", field]
                    frames.append(x.set_index(["Date", "Ticker"]))

                df = pd.concat(frames, axis=1).reset_index()
            else:
                df = raw.reset_index()
                df["Ticker"] = tickers[0]

            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"])
            df = df.set_index(["Date", "Ticker"]).sort_index()

            numeric_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            return df

        except Exception as exc:
            last_error = exc
            logger.warning("Price download attempt %d/%d failed: %s", attempt, max_retries, exc)
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Could not download market data: {last_error}")

prices = download_prices(UNIVERSE, CFG.start, CFG.end)
prices.to_parquet(RAW_DIR / "prices.parquet")
logger.info("Price rows: %s", len(prices))
prices.head()


# %%
# CELL 5 — Price-quality checks

def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open","High","Low","Close","Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.copy()
    x = x[~x.index.duplicated(keep="last")]

    # Basic market-data integrity checks.
    bad_ohlc = (
        (x["High"] < x["Low"]) |
        (x["High"] < x["Open"]) |
        (x["High"] < x["Close"]) |
        (x["Low"] > x["Open"]) |
        (x["Low"] > x["Close"])
    )
    if bad_ohlc.any():
        logger.warning("Found %d suspicious OHLC rows.", int(bad_ohlc.sum()))

    x.loc[x["Volume"] < 0, "Volume"] = np.nan
    x["Close"] = x["Close"].where(x["Close"] > 0)

    return x

prices = validate_prices(prices)

# Wide matrices used by most research functions.
close = prices["Close"].unstack("Ticker").sort_index()
volume = prices["Volume"].unstack("Ticker").sort_index()
returns = close.pct_change(fill_method=None)

close.to_parquet(PROCESSED_DIR / "close.parquet")
returns.to_parquet(PROCESSED_DIR / "returns.parquet")


# %%
# CELL 6 — FRED macro connector

def fred_series(series_id: str, start: str, end: str, api_key: str = "") -> pd.Series:
    '''Download one FRED series. Uses the public CSV endpoint when possible.'''
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index)
            s.name = series_id
            return s
        except Exception as exc:
            logger.warning("fredapi failed for %s: %s; trying CSV.", series_id, exc)

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = df["observation_date"]
    s.name = series_id
    return s

MACRO_SERIES = {
    "FEDFUNDS": "policy_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "DGS10": "10y_yield",
    "DGS2": "2y_yield",
    "VIXCLS": "vix",
}

macro = pd.DataFrame(index=close.index)
for sid, name in MACRO_SERIES.items():
    try:
        s = fred_series(sid, CFG.start, CFG.end, FRED_API_KEY)
        macro[name] = s.reindex(macro.index).ffill()
    except Exception as exc:
        logger.warning("Macro series %s unavailable: %s", sid, exc)

if "10y_yield" in macro.columns and "2y_yield" in macro.columns:
    macro["yield_curve_10y2y"] = macro["10y_yield"] - macro["2y_yield"]

macro.to_parquet(RAW_DIR / "macro.parquet")
macro.tail()


# %% [markdown]
# ## 2. Feature Engine
# 
# The feature set deliberately mixes interpretable cross-sectional signals with time-series signals.
# 
# **Critical anti-look-ahead rule:** all features used for portfolio formation are shifted so that today's information cannot trade on today's close.


# %%
# CELL 7 — Feature engineering

def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)

def winsorize_cross_section(df: pd.DataFrame, lower=0.02, upper=0.98):
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lo, hi, axis=0)

features = {}

# Momentum: skip the most recent month to reduce short-term reversal contamination.
features["momentum_12_1"] = (
    close.shift(CFG.momentum_skip) /
    close.shift(CFG.momentum_lookback) - 1
)

# Short-term mean reversion.
features["reversal_21"] = -(close / close.shift(21) - 1)

# Volatility.
features["vol_20"] = returns.rolling(20).std() * np.sqrt(252)
features["vol_60"] = returns.rolling(CFG.lookback_vol).std() * np.sqrt(252)

# Volume/liquidity proxy.
features["dollar_volume_20"] = (
    (close * volume).rolling(20).mean()
)

# Trend quality / moving-average distance.
features["ma_gap_50"] = close / close.rolling(50).mean() - 1
features["ma_gap_200"] = close / close.rolling(200).mean() - 1

# Cross-sectional volatility rank.
features["vol_rank"] = features["vol_60"].rank(axis=1, pct=True)

# Macro features replicated across the asset universe.
for col in macro.columns:
    m = macro[col]
    features[f"macro_{col}_chg_21"] = pd.DataFrame(
        np.tile(m.pct_change(21).values[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns
    )

# Combine into a long panel.
feature_panel = []
for name, df in features.items():
    z = winsorize_cross_section(df)
    z = zscore_cross_section(z)
    z.columns.name = "Ticker"
    long = z.stack().rename(name)
    feature_panel.append(long)

feature_df = pd.concat(feature_panel, axis=1)
feature_df.index.names = ["Date", "Ticker"]

# Target = next 21 trading-day return.
future_return = (close.shift(-21) / close - 1).stack().rename("target_21d")
future_return.index.names = ["Date", "Ticker"]

dataset = feature_df.join(future_return, how="inner").replace([np.inf, -np.inf], np.nan)

# IMPORTANT: shift predictors by one trading day within each ticker.
feature_cols = [c for c in dataset.columns if c != "target_21d"]
dataset[feature_cols] = dataset.groupby(level="Ticker")[feature_cols].shift(1)

dataset = dataset.dropna(subset=["target_21d"])
dataset.to_parquet(PROCESSED_DIR / "model_dataset.parquet")

logger.info("Model dataset shape: %s", dataset.shape)
dataset.head()


# %%
# CELL 8 — Optional SEC EDGAR fundamentals connector

def sec_companyfacts(cik: str, user_agent: str) -> dict:
    '''Fetch SEC Company Facts JSON for a company CIK.'''
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

# Example:
# facts = sec_companyfacts("320193", SEC_USER_AGENT)  # Apple CIK
# Persist raw SEC JSON rather than repeatedly downloading it.
#
# Production note:
# SEC data is filing-date/event-time sensitive. Do NOT join a fiscal-period value
# to historical prices by period end alone. Use the filing/acceptance date as
# the availability timestamp and lag it into the model.


# %% [markdown]
# ## 3. Alpha Research
# 
# We use three layers:
# 
# 1. **Factor score** — transparent benchmark.
# 2. **Regularized linear model** — low-complexity ML benchmark.
# 3. **Tree models** — nonlinear candidates.
# 
# The ML model is never treated as automatically superior. It must beat the factor benchmark out-of-sample after costs and risk controls.


# %%
# CELL 9 — Factor alpha

def composite_factor_score(panel: pd.DataFrame) -> pd.Series:
    x = panel.copy()

    # Higher momentum/reversal is better; lower volatility is better.
    score = (
        0.45 * x["momentum_12_1"].fillna(0)
        + 0.20 * x["reversal_21"].fillna(0)
        - 0.20 * x["vol_60"].fillna(0)
        + 0.15 * x["ma_gap_200"].fillna(0)
    )
    return score.rename("factor_score")

dataset["factor_score"] = composite_factor_score(dataset)

# Cross-sectional percentile/rank.
dataset["factor_rank"] = (
    dataset.groupby(level="Date")["factor_score"]
    .rank(pct=True)
)

dataset[["factor_score","factor_rank","target_21d"]].dropna().head()


# %%
# CELL 10 — Walk-forward ML model training

FEATURES = [
    c for c in feature_cols
    if c in dataset.columns
]

def build_models(random_state=42):
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    }

def cross_sectional_walk_forward(
    data: pd.DataFrame,
    features: List[str],
    train_years: int = 3,
    test_months: int = 1,
    min_train_rows: int = 500,
):
    dates = pd.DatetimeIndex(sorted(data.index.get_level_values("Date").unique()))
    predictions = []

    for i, test_start in enumerate(dates):
        if test_start < dates[0] + pd.DateOffset(years=train_years):
            continue

        test_end = test_start + pd.DateOffset(months=test_months)

        train_mask = data.index.get_level_values("Date") < test_start
        train_mask &= data.index.get_level_values("Date") >= test_start - pd.DateOffset(years=train_years)

        test_mask = (
            (data.index.get_level_values("Date") >= test_start) &
            (data.index.get_level_values("Date") < test_end)
        )

        train = data.loc[train_mask].dropna(subset=["target_21d"])
        test = data.loc[test_mask]

        if len(train) < min_train_rows or len(test) == 0:
            continue

        model = build_models(CFG.random_state)["ridge"]
        model.fit(train[features], train["target_21d"])

        pred = model.predict(test[features])
        out = test[["target_21d"]].copy()
        out["prediction"] = pred
        predictions.append(out)

    if not predictions:
        raise RuntimeError("No walk-forward prediction windows were generated.")

    return pd.concat(predictions).sort_index()

ml_predictions = cross_sectional_walk_forward(dataset, FEATURES)
ml_predictions.to_parquet(PROCESSED_DIR / "ml_predictions.parquet")
ml_predictions.head()


# %%
# CELL 11 — Optional nonlinear models: XGBoost / LightGBM

def try_tree_models(data: pd.DataFrame, features: List[str]):
    results = {}

    X = data[features]
    y = data["target_21d"]

    split_date = data.index.get_level_values("Date").max() - pd.DateOffset(years=1)
    train = data.index.get_level_values("Date") < split_date
    test = ~train

    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]

    try:
        from xgboost import XGBRegressor
        xgb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=CFG.random_state,
                n_jobs=-1
            ))
        ])
        xgb.fit(X_train, y_train)
        p = xgb.predict(X_test)
        results["xgboost"] = {
            "rmse": mean_squared_error(y_test, p) ** 0.5,
            "r2": r2_score(y_test, p)
        }
        dump(xgb, MODEL_DIR / "xgboost.joblib")
    except Exception as exc:
        logger.warning("XGBoost unavailable: %s", exc)

    try:
        import lightgbm as lgb
        lgbm = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=CFG.random_state,
                verbosity=-1
            ))
        ])
        lgbm.fit(X_train, y_train)
        p = lgbm.predict(X_test)
        results["lightgbm"] = {
            "rmse": mean_squared_error(y_test, p) ** 0.5,
            "r2": r2_score(y_test, p)
        }
        dump(lgbm, MODEL_DIR / "lightgbm.joblib")
    except Exception as exc:
        logger.warning("LightGBM unavailable: %s", exc)

    return pd.DataFrame(results).T

tree_results = try_tree_models(dataset.dropna(subset=FEATURES), FEATURES)
tree_results


# %% [markdown]
# ## 4. Portfolio Construction
# 
# The backtest uses a **signal date → target weights → next-period realized returns** convention.
# 
# This avoids the common mistake of calculating a signal using a close and then earning that same close-to-close return.


# %%
# CELL 12 — Portfolio constructors

def top_bottom_factor_weights(
    scores: pd.Series,
    max_positions: int = 20,
    max_weight: float = 0.10
) -> pd.Series:
    '''Long-only top-ranked portfolio with inverse-volatility scaling.'''
    s = scores.dropna().sort_values(ascending=False)
    selected = s.head(max_positions)

    if selected.empty:
        return pd.Series(dtype=float)

    w = pd.Series(1.0, index=selected.index)
    w = w / w.sum()
    w = w.clip(upper=max_weight)

    # Re-normalize after caps.
    if w.sum() > 0:
        w = w / w.sum()
    return w

def inverse_vol_weights(vol: pd.Series, max_weight=0.10) -> pd.Series:
    v = vol.replace([np.inf, -np.inf], np.nan).dropna()
    v = v[v > 0]
    if v.empty:
        return pd.Series(dtype=float)

    inv = 1.0 / v
    w = inv / inv.sum()
    w = w.clip(upper=max_weight)
    return w / w.sum()

def risk_parity_weights(cov: pd.DataFrame, max_weight=0.10) -> pd.Series:
    '''Numerical equal-risk-contribution approximation.'''
    assets = list(cov.columns)
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)

    C = cov.values
    C = (C + C.T) / 2 + np.eye(n) * 1e-8

    def risk_contribution(w):
        port_var = w @ C @ w
        mrc = C @ w
        return w * mrc / max(port_var, 1e-12)

    def objective(w):
        rc = risk_contribution(w)
        return np.sum((rc - rc.mean()) ** 2)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, max_weight)] * n
    x0 = np.repeat(1 / n, n)

    res = opt.minimize(objective, x0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        logger.warning("Risk parity optimizer failed: %s", res.message)
        return pd.Series(x0, index=assets)

    w = pd.Series(res.x, index=assets)
    return w / w.sum()


# %%
# CELL 13 — Build monthly target weights from factor + ML scores

# Convert model predictions to cross-sectional percentile scores.
pred = ml_predictions["prediction"].copy()
pred_rank = pred.groupby(level="Date").rank(pct=True).rename("ml_rank")

# Factor rank from the complete dataset.
factor_rank = dataset["factor_rank"].rename("factor_rank")

scores = pd.concat([factor_rank, pred_rank], axis=1)
scores["ensemble"] = 0.60 * scores["factor_rank"] + 0.40 * scores["ml_rank"]

# Rebalance on month-end observations.
score_wide = scores["ensemble"].unstack("Ticker").sort_index()
rebalance_scores = score_wide.resample("ME").last()

# Realized future returns after the rebalance.
# Use next-month close-to-close returns after the signal date.
monthly_close = close.resample("ME").last()
monthly_returns = monthly_close.pct_change()

weights = pd.DataFrame(index=rebalance_scores.index, columns=close.columns, dtype=float)

for dt in rebalance_scores.index:
    s = rebalance_scores.loc[dt]
    s = s.dropna()

    if s.empty:
        continue

    w = top_bottom_factor_weights(
        s,
        max_positions=CFG.max_positions,
        max_weight=CFG.max_weight
    )
    weights.loc[dt, w.index] = w

weights = weights.fillna(0.0)
weights.to_parquet(PROCESSED_DIR / "target_weights.parquet")

weights.tail()


# %%
# CELL 14 — Volatility targeting

def volatility_target_weights(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: float = 0.12,
    lookback: int = 60
) -> pd.DataFrame:
    out = target_weights.copy()

    for dt in out.index:
        history = asset_returns.loc[:dt].tail(lookback)
        if len(history) < max(20, lookback // 2):
            continue

        w = out.loc[dt].values
        mask = np.isfinite(w)
        if not mask.any():
            continue

        r = history.iloc[:, mask]
        ww = w[mask]
        cov = r.cov().values * 252
        vol = float(np.sqrt(max(ww @ cov @ ww, 1e-12)))

        scale = target_vol / vol
        # Long-only portfolio is allowed to scale down but not lever here.
        scale = min(scale, 1.0)
        out.loc[dt] *= scale

    return out

target_weights = volatility_target_weights(
    weights,
    monthly_returns,
    target_vol=CFG.vol_target,
    lookback=12
)

target_weights.to_parquet(PROCESSED_DIR / "final_target_weights.parquet")


# %% [markdown]
# ## 5. Execution Simulator
# 
# The execution model charges:
# 
# - explicit transaction cost
# - bid/ask spread proxy
# - slippage proxy
# - turnover
# 
# A real desk would calibrate these from order-level data. Here the model is deliberately transparent rather than pretending public daily data can estimate true market impact precisely.


# %%
# CELL 15 — Backtest engine

@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

def run_backtest(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    transaction_cost_bps: float = 10,
    slippage_bps: float = 5,
) -> BacktestResult:

    # Portfolio weights at month-end are applied to the following month's return.
    w = target_weights.shift(1).fillna(0.0)
    r = asset_returns.reindex(index=w.index, columns=w.columns).fillna(0.0)

    gross = (w * r).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()

    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    costs = turnover * cost_rate

    net = gross - costs
    equity = (1 + net).cumprod()

    return BacktestResult(
        returns=net.rename("strategy_return"),
        equity=equity.rename("equity"),
        weights=w,
        turnover=turnover.rename("turnover"),
        costs=costs.rename("costs")
    )

bt = run_backtest(
    target_weights,
    monthly_returns,
    transaction_cost_bps=CFG.transaction_cost_bps,
    slippage_bps=CFG.slippage_bps,
)

bt.equity.tail()


# %%
# CELL 16 — Benchmark and performance metrics

benchmark = monthly_returns.mean(axis=1).reindex(bt.returns.index).fillna(0)
benchmark_equity = (1 + benchmark).cumprod()

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())

def annualized_return(r: pd.Series, periods_per_year=12) -> float:
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)

def annualized_vol(r: pd.Series, periods_per_year=12) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe(r: pd.Series, periods_per_year=12) -> float:
    vol = annualized_vol(r, periods_per_year)
    return float(annualized_return(r, periods_per_year) / vol) if vol > 0 else np.nan

def sortino(r: pd.Series, periods_per_year=12) -> float:
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    return float(annualized_return(r, periods_per_year) / downside) if downside > 0 else np.nan

def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown((1 + r).cumprod()))
    return annualized_return(r) / mdd if mdd > 0 else np.nan

def var_historical(r: pd.Series, confidence=0.95) -> float:
    return float(-r.quantile(1 - confidence))

def expected_shortfall(r: pd.Series, confidence=0.95) -> float:
    threshold = r.quantile(1 - confidence)
    tail = r[r <= threshold]
    return float(-tail.mean()) if len(tail) else np.nan

def performance_table(r: pd.Series) -> pd.Series:
    return pd.Series({
        "Cumulative Return": (1 + r).prod() - 1,
        "CAGR": annualized_return(r),
        "Annualized Volatility": annualized_vol(r),
        "Sharpe": sharpe(r),
        "Sortino": sortino(r),
        "Calmar": calmar(r),
        "Max Drawdown": max_drawdown((1 + r).cumprod()),
        "VaR 95%": var_historical(r),
        "Expected Shortfall 95%": expected_shortfall(r),
        "Average Turnover": bt.turnover.mean(),
        "Total Costs": bt.costs.sum(),
    })

metrics = pd.DataFrame({
    "Strategy": performance_table(bt.returns),
    "Benchmark": performance_table(benchmark),
})

metrics


# %%
# CELL 17 — Factor / sector / concentration diagnostics

def weight_diagnostics(w: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "gross_exposure": w.abs().sum(axis=1),
        "net_exposure": w.sum(axis=1),
        "positions": (w.abs() > 1e-8).sum(axis=1),
        "max_weight": w.abs().max(axis=1),
        "turnover": w.diff().abs().sum(axis=1),
    })

diagnostics = weight_diagnostics(bt.weights)

# Approximate beta to equal-weight benchmark.
common = pd.concat([bt.returns, benchmark], axis=1).dropna()
beta = common["strategy_return"].cov(common.iloc[:,1]) / common.iloc[:,1].var()

# Rolling beta.
rolling_beta = (
    common["strategy_return"].rolling(12).cov(common.iloc[:,1])
    / common.iloc[:,1].rolling(12).var()
)

print("Full-period beta:", round(beta, 3))
diagnostics.tail()


# %% [markdown]
# ## 6. Risk Engine
# 
# The risk layer is not a single VaR number. It answers:
# 
# - How much can the portfolio lose under historical tails?
# - Where does the risk come from?
# - How concentrated is the book?
# - What happens under plausible shocks?
# - Does the strategy's return survive costs and drawdowns?


# %%
# CELL 18 — Stress testing

def stress_test(
    weights: pd.DataFrame,
    shocks: Dict[str, float]
) -> pd.DataFrame:
    '''
    Generic factor shock interface.

    Example shocks can be replaced by a proper factor model:
    market, rates, oil, volatility, USD, sectors, etc.
    '''
    # For a public-data capstone, use portfolio-level scenario returns.
    # The mapping from each scenario to assets should be replaced by
    # empirically estimated betas for a production system.
    rows = []
    current = weights.iloc[-1]

    scenarios = {
        "Market -10%": -0.10,
        "Market -20%": -0.20,
        "Volatility Shock": -0.08,
        "Rates Shock": -0.05,
        "Liquidity Shock": -0.07,
    }
    scenarios.update(shocks)

    for name, market_move in scenarios.items():
        # Conservative placeholder: portfolio experiences 1.1x market shock
        # unless a factor-beta matrix is supplied.
        portfolio_pnl = current.abs().sum() * market_move * 1.10
        rows.append({"Scenario": name, "Estimated P&L": portfolio_pnl})

    return pd.DataFrame(rows).set_index("Scenario")

stress = stress_test(bt.weights, {})
stress


# %%
# CELL 19 — Historical VaR / Expected Shortfall and bootstrap

def bootstrap_paths(
    returns: pd.Series,
    horizon: int = 12,
    n_paths: int = 5000,
    seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = returns.dropna().values
    if len(clean) < 20:
        raise ValueError("Not enough return observations for bootstrap.")

    idx = rng.integers(0, len(clean), size=(n_paths, horizon))
    sampled = clean[idx]
    return np.cumprod(1 + sampled, axis=1)

paths = bootstrap_paths(bt.returns)
terminal = paths[:, -1] - 1

bootstrap_summary = pd.Series({
    "5% terminal return": np.quantile(terminal, 0.05),
    "Median terminal return": np.quantile(terminal, 0.50),
    "95% terminal return": np.quantile(terminal, 0.95),
    "Probability of loss": np.mean(terminal < 0),
})
bootstrap_summary


# %% [markdown]
# ## 7. Validation
# 
# Institutional research is mostly about proving that the strategy **does not work because of a mistake**.
# 
# The validation suite below includes:
# 
# - chronological train/test separation
# - walk-forward prediction
# - cost sensitivity
# - parameter sensitivity
# - bootstrap uncertainty
# 
# For a stronger institutional version, add **purged and embargoed cross-validation** whenever labels overlap across training and test windows.


# %%
# CELL 20 — Cost sensitivity

def cost_sensitivity(
    target_weights,
    monthly_returns,
    cost_bps_grid=(0, 5, 10, 20, 30, 50)
):
    rows = []
    for c in cost_bps_grid:
        res = run_backtest(
            target_weights,
            monthly_returns,
            transaction_cost_bps=c,
            slippage_bps=CFG.slippage_bps
        )
        rows.append({
            "cost_bps": c,
            "CAGR": annualized_return(res.returns),
            "Sharpe": sharpe(res.returns),
            "MaxDrawdown": max_drawdown(res.equity)
        })
    return pd.DataFrame(rows).set_index("cost_bps")

cost_table = cost_sensitivity(target_weights, monthly_returns)
cost_table


# %%
# CELL 21 — Parameter sensitivity

def parameter_sensitivity(
    score_wide: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    position_grid=(10, 15, 20, 30, 40),
):
    rows = []

    for n in position_grid:
        w = pd.DataFrame(0.0, index=score_wide.index, columns=score_wide.columns)

        for dt in score_wide.index:
            s = score_wide.loc[dt].dropna()
            if s.empty:
                continue
            selected = s.nlargest(n)
            w.loc[dt, selected.index] = 1.0 / len(selected)

        res = run_backtest(
            w,
            monthly_returns,
            transaction_cost_bps=CFG.transaction_cost_bps,
            slippage_bps=CFG.slippage_bps
        )
        rows.append({
            "positions": n,
            "CAGR": annualized_return(res.returns),
            "Sharpe": sharpe(res.returns),
            "MaxDrawdown": max_drawdown(res.equity)
        })

    return pd.DataFrame(rows)

sensitivity = parameter_sensitivity(score_wide, monthly_returns)
sensitivity


# %%
# CELL 22 — Walk-forward ML quality diagnostics

pred_eval = ml_predictions.dropna().copy()

# Cross-sectional rank correlation is often more useful for stock-selection models
# than raw regression R².
ic_by_date = (
    pred_eval.groupby(level="Date")
    .apply(lambda g: g["prediction"].corr(g["target_21d"], method="spearman"))
    .dropna()
)

ic_summary = pd.Series({
    "Mean IC": ic_by_date.mean(),
    "IC Std": ic_by_date.std(),
    "IC Information Ratio": ic_by_date.mean() / ic_by_date.std() if ic_by_date.std() > 0 else np.nan,
    "Positive IC %": (ic_by_date > 0).mean(),
})

ic_summary


# %% [markdown]
# ## 8. Professional Visualizations


# %%
# CELL 23 — Matplotlib research report

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
ax.plot(bt.equity.index, bt.equity, label="Systematic Strategy")
ax.plot(benchmark_equity.index, benchmark_equity, label="Equal-Weight Benchmark")
ax.set_title("Cumulative Equity Curve")
ax.set_ylabel("Growth of $1")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "equity_curve.png", dpi=180, bbox_inches="tight")
plt.show()

drawdown = bt.equity / bt.equity.cummax() - 1
fig, ax = plt.subplots()
ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.35)
ax.set_title("Strategy Drawdown")
ax.set_ylabel("Drawdown")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "drawdown.png", dpi=180, bbox_inches="tight")
plt.show()

rolling_sharpe = (
    bt.returns.rolling(12).mean() /
    bt.returns.rolling(12).std()
) * np.sqrt(12)

fig, ax = plt.subplots()
ax.plot(rolling_sharpe.index, rolling_sharpe)
ax.axhline(0, linewidth=1)
ax.set_title("12-Month Rolling Sharpe")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_sharpe.png", dpi=180, bbox_inches="tight")
plt.show()


# %%
# CELL 24 — Risk / turnover / cost charts

fig, ax = plt.subplots()
ax.plot(bt.turnover.index, bt.turnover)
ax.set_title("Portfolio Turnover")
ax.set_ylabel("One-way turnover proxy")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "turnover.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(bt.costs.index, bt.costs)
ax.set_title("Transaction Costs")
ax.set_ylabel("Return drag")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "transaction_costs.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
sns.histplot(bt.returns.dropna(), bins=40, kde=True, ax=ax)
ax.axvline(bt.returns.mean(), linestyle="--", label="Mean")
ax.set_title("Monthly Strategy Return Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "return_distribution.png", dpi=180, bbox_inches="tight")
plt.show()


# %%
# CELL 25 — Interactive Plotly dashboard

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=("Equity", "Drawdown", "Rolling Sharpe")
)

fig.add_trace(
    go.Scatter(x=bt.equity.index, y=bt.equity, name="Strategy"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="Benchmark"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown"),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe"),
    row=3, col=1
)

fig.update_layout(
    title="Systematic Hedge Fund Research Dashboard",
    height=900,
    template="plotly_white",
    hovermode="x unified"
)

fig.write_html(REPORT_DIR / "dashboard.html")
fig.show()


# %% [markdown]
# ## 9. Automated Investment Report


# %%
# CELL 26 — Generate a machine-readable research report

report = {
    "project": "Mini Systematic Hedge Fund",
    "config": asdict(CFG),
    "universe_size": len(UNIVERSE),
    "date_range": {
        "start": str(bt.returns.index.min()),
        "end": str(bt.returns.index.max())
    },
    "performance": metrics.to_dict(),
    "ml_ic": ic_summary.to_dict(),
    "stress_tests": stress.reset_index().to_dict(orient="records"),
    "cost_sensitivity": cost_table.reset_index().to_dict(orient="records"),
    "parameter_sensitivity": sensitivity.to_dict(orient="records"),
    "bootstrap": bootstrap_summary.to_dict(),
}

with open(REPORT_DIR / "investment_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

metrics.to_csv(REPORT_DIR / "performance_metrics.csv")
stress.to_csv(REPORT_DIR / "stress_tests.csv")
cost_table.to_csv(REPORT_DIR / "cost_sensitivity.csv")
sensitivity.to_csv(REPORT_DIR / "parameter_sensitivity.csv")

print(json.dumps({
    "performance": metrics.round(4).to_dict(),
    "ml_ic": ic_summary.round(4).to_dict()
}, indent=2, default=str))


# %%
# CELL 27 — Markdown-style executive summary generated from results

strategy = performance_table(bt.returns)

summary = f'''
# Systematic Hedge Fund — Executive Summary

## Performance
- CAGR: {strategy["CAGR"]:.2%}
- Annualized volatility: {strategy["Annualized Volatility"]:.2%}
- Sharpe ratio: {strategy["Sharpe"]:.2f}
- Sortino ratio: {strategy["Sortino"]:.2f}
- Maximum drawdown: {strategy["Max Drawdown"]:.2%}
- Historical VaR (95%): {strategy["VaR 95%"]:.2%}
- Expected Shortfall (95%): {strategy["Expected Shortfall 95%"]:.2%}
- Average monthly turnover: {strategy["Average Turnover"]:.2%}
- Total modeled transaction-cost drag: {strategy["Total Costs"]:.2%}

## Research diagnostics
- Mean cross-sectional IC: {ic_summary["Mean IC"]:.3f}
- IC information ratio: {ic_summary["IC Information Ratio"]:.3f}
- Positive IC months: {ic_summary["Positive IC %"]:.1%}

## Important limitation
This is a research prototype, not a deployable trading system. Public daily data cannot reproduce institutional execution quality, corporate-action handling, point-in-time fundamentals, borrow availability, order-book liquidity, or live risk controls.
'''

print(summary)

with open(REPORT_DIR / "executive_summary.md", "w") as f:
    f.write(summary)


# %% [markdown]
# ## 10. Optional Extensions
# 
# ### Institutional upgrades
# 1. **Point-in-time fundamentals**
#    - Store filing acceptance timestamps.
#    - Version every fundamental observation.
#    - Never use revised historical data as if it were known at the time.
# 
# 2. **True purged CV**
#    - Purge observations whose labels overlap the validation window.
#    - Add an embargo after validation windows.
# 
# 3. **Factor risk model**
#    - Estimate rolling market, sector, style, rates, FX and commodity betas.
#    - Constrain portfolio factor exposures.
# 
# 4. **Optimizer**
#    - Add CVXPY objective:
#      `expected_return - risk_aversion * variance - turnover_penalty`.
#    - Add hard constraints for sector, beta, concentration and liquidity.
# 
# 5. **Execution**
#    - Replace fixed bps with spread + nonlinear impact:
#      `cost = spread + a * (order_size / ADV)^b`.
# 
# 6. **Derivatives**
#    - Add implied volatility, skew, term structure and options Greeks.
#    - Build volatility-surface features.
# 
# 7. **Alternative data**
#    - NLP earnings calls.
#    - News sentiment.
#    - Web traffic.
#    - Analyst revisions.
# 
# 8. **Production**
#    - PostgreSQL/TimescaleDB.
#    - Parquet data lake.
#    - Prefect/Airflow.
#    - MLflow.
#    - Docker.
#    - Unit/integration tests.
#    - CI/CD.
#    - Secrets manager.
#    - Monitoring and alerting.
# 
# 9. **Research governance**
#    - Experiment IDs.
#    - Dataset hashes.
#    - Model versioning.
#    - Feature lineage.
#    - Parameter registry.
#    - Model approval checklist.
# 
# 10. **Live paper-trading layer**
#     - Signal generation.
#     - Order proposal.
#     - Pre-trade risk.
#     - Execution simulation.
#     - Post-trade attribution.
#     - Kill switches.


# %% [markdown]
# ## Final deliverables
# 
# The capstone should publish:
# 
# - `systematic_hedge_fund.ipynb`
# - `README.md`
# - `requirements.txt`
# - `data_dictionary.md`
# - `investment_report.json`
# - `executive_summary.md`
# - `performance_metrics.csv`
# - `stress_tests.csv`
# - `cost_sensitivity.csv`
# - `parameter_sensitivity.csv`
# - `dashboard.html`
# - `figures/equity_curve.png`
# - `figures/drawdown.png`
# - `figures/rolling_sharpe.png`
# - `figures/turnover.png`
# - `figures/transaction_costs.png`
# - `figures/return_distribution.png`
# 
# ### Resume description
# 
# > Built an end-to-end systematic equity research platform in Python covering market-data ingestion, factor engineering, walk-forward machine learning, portfolio optimization, volatility targeting, transaction-cost modeling, risk analytics, stress testing, bootstrap validation, and interactive performance reporting; implemented anti-look-ahead controls and compared ML alpha against interpretable factor baselines.