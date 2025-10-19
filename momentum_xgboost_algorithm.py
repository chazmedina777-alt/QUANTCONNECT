from AlgorithmImports import *
import xgboost as xgb
import pandas as pd
from typing import Dict, List, Optional


class MomentumTrendXgboostAlgorithm(QCAlgorithm):
    """
    A cross-sectional momentum strategy that uses an XGBoost regressor to
    forecast the next-day return of the most liquid U.S. equities. The algorithm
    trains daily on rolling feature windows built from multiple look-back
    periods and rebalances into the symbols with the strongest predicted
    momentum.
    """

    def Initialize(self) -> None:
        """Initialize the backtest parameters and universe."""
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(10000)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted

        self.lookback = 120
        self.minimum_history = 40
        self.portfolio_size = 10

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_columns = [
            "ret_1",
            "ret_5",
            "ret_10",
            "ret_20",
            "ret_60",
            "volatility_20",
            "volatility_60",
            "volume_trend",
            "price_distance_20",
        ]

        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.AddUniverse(self.CoarseSelectionFunction)

        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketOpen(self.spy, 30),
            self.TrainModel,
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 1),
            self.RebalancePortfolio,
        )

        self.active_symbols: List[Symbol] = []
        self.predicted_returns: Dict[Symbol, float] = {}

    def CoarseSelectionFunction(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        """Select a liquid universe for downstream modeling."""
        filtered = [
            c for c in coarse if c.HasFundamentalData and c.Price is not None and c.Price > 5
        ]
        filtered = [c for c in filtered if c.DollarVolume is not None and c.DollarVolume > 1e7]
        top_by_liquidity = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)[:300]
        return [c.Symbol for c in top_by_liquidity]

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for added in changes.AddedSecurities:
            added.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        self.active_symbols = [security.Symbol for security in self.ActiveSecurities.Values]

    def TrainModel(self) -> None:
        """Pull recent history, engineer features, and fit the XGBoost model."""
        if not self.active_symbols:
            return

        history = self.History(self.active_symbols, self.lookback + 1, Resolution.Daily)
        if history.empty:
            return

        training_features: List[pd.DataFrame] = []
        training_labels: List[pd.Series] = []
        prediction_rows: Dict[Symbol, pd.Series] = {}

        for symbol in self.active_symbols:
            try:
                symbol_history = history.loc[symbol]
            except KeyError:
                continue

            if symbol_history.empty or len(symbol_history) < self.minimum_history:
                continue

            df = pd.DataFrame(index=symbol_history.index)
            df["close"] = symbol_history["close"]
            df["volume"] = symbol_history["volume"]
            df["ret_1"] = df["close"].pct_change()
            df["ret_5"] = df["close"].pct_change(5)
            df["ret_10"] = df["close"].pct_change(10)
            df["ret_20"] = df["close"].pct_change(20)
            df["ret_60"] = df["close"].pct_change(60)
            df["volatility_20"] = df["ret_1"].rolling(20).std()
            df["volatility_60"] = df["ret_1"].rolling(60).std()
            df["volume_trend"] = df["volume"].pct_change(20)
            rolling_mean_20 = df["close"].rolling(20).mean()
            df["price_distance_20"] = (df["close"] / rolling_mean_20) - 1
            df["forward_return"] = df["ret_1"].shift(-1)

            df = df.dropna(subset=self.feature_columns)
            if df.empty or len(df) < self.minimum_history:
                continue

            df["symbol"] = symbol
            train_rows = df[df["forward_return"].notna()]
            if len(train_rows) < self.minimum_history:
                continue

            training_features.append(train_rows[self.feature_columns])
            training_labels.append(train_rows["forward_return"])
            prediction_rows[symbol] = df.iloc[-1][self.feature_columns]

        if not training_features or not training_labels:
            return

        X = pd.concat(training_features)
        y = pd.concat(training_labels)

        if len(y) < 200:
            # Wait for additional samples to stabilize the model.
            return

        self.model = xgb.XGBRegressor(
            max_depth=3,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        self.model.fit(X, y)

        predict_frame = pd.DataFrame.from_dict(prediction_rows, orient="index")
        self.predicted_returns = {}
        if not predict_frame.empty:
            predictions = self.model.predict(predict_frame[self.feature_columns])
            for symbol, prediction in zip(predict_frame.index, predictions):
                self.predicted_returns[symbol] = float(prediction)

    def RebalancePortfolio(self) -> None:
        """Use the latest predictions to rebalance the portfolio."""
        if not self.predicted_returns:
            return

        sorted_predictions = sorted(
            self.predicted_returns.items(), key=lambda item: item[1], reverse=True
        )
        selected = [symbol for symbol, prediction in sorted_predictions if prediction > 0][: self.portfolio_size]

        if not selected:
            return

        weight = 1.0 / len(selected)
        targets = {symbol: weight for symbol in selected}

        invested_symbols = list(self.Portfolio.Keys)
        for symbol in invested_symbols:
            if symbol not in targets:
                self.Liquidate(symbol)

        for symbol, target_weight in targets.items():
            if symbol in self.Securities and self.Securities[symbol].IsTradable:
                self.SetHoldings(symbol, target_weight)

    def OnEndOfDay(self) -> None:
        self.Plot("Strategy", "Invested", int(self.Portfolio.Invested))
