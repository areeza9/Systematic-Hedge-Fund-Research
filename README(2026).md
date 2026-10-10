# Mini Systematic Hedge Fund — Institutional Quant Capstone

End-to-end quantitative equity research platform:

**Data → Features → Alpha → ML → Portfolio → Risk → Execution → Backtest → Validation → Dashboard**

## Run
Open `systematic_hedge_fund_capstone.ipynb` in Google Colab and run cells sequentially.

## Data
Default market data uses Yahoo Finance. FRED is optional. SEC EDGAR, Kenneth French, FMP, Alpha Vantage, Polygon and Nasdaq Data Link can be added as connectors.

## Important research controls
- Predictor lagging
- Chronological walk-forward validation
- Transaction-cost modeling
- Cost and parameter sensitivity
- Bootstrap uncertainty
- Benchmark comparison

## Not production trading software
This is a research capstone. A live system requires point-in-time data, corporate-action/event-time correctness, real liquidity and execution models, borrow constraints, portfolio/factor risk models, monitoring, controls, and operational infrastructure.
