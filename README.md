# QUANTCONNECT

This repository contains the `MomentumTrendXgboostAlgorithm`, a QuantConnect LEAN algorithm example.

- Entry point: `momentum_xgboost_algorithm.py`
- Algorithm class: `MomentumTrendXgboostAlgorithm`
- To backtest, upload the file to QuantConnect and run a backtest covering the desired period (default 2020-01-01 to 2024-01-01).

The algorithm builds a liquid equity universe, engineers momentum features, trains an XGBoost regressor, and rebalances into the symbols with the strongest predicted next-day returns.
