import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import (
    N_ESTIMATORS, RANDOM_STATE, MIN_SIGNAL_THRESHOLD, FEATURE_COLS,
    META_CONFIDENCE, PURGE_EMBARGO_DAYS, CV_SPLITS, SWING_HORIZON,
)


class PrimaryModel:
    def generate(self, data):
        scores = data["CPP_score"].values
        p75 = np.percentile(scores[~np.isnan(scores)], 75)
        p25 = np.percentile(scores[~np.isnan(scores)], 25)
        return np.where(scores >= p75, 1, np.where(scores <= p25, -1, 0))


class MetaModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.fitted = False

    def fit(self, train_data, primary_signals):
        mask = primary_signals != 0
        if mask.sum() < 20:
            return
        X = train_data[FEATURE_COLS].values[mask]
        tb = train_data["Target"].values[mask]
        y = ((tb == 1) & (primary_signals[mask] == 1)) | \
            ((tb == -1) & (primary_signals[mask] == -1))
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.fitted = True

    def predict_proba(self, test_data):
        X = test_data[FEATURE_COLS].values
        if not self.fitted:
            return np.ones(len(X)) * 0.5
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return np.zeros(len(X))

    def feature_importances(self):
        if not self.fitted:
            return {f: 0.0 for f in FEATURE_COLS}
        return dict(zip(FEATURE_COLS, self.model.feature_importances_))


class MLStrategy:
    def __init__(self):
        self.primary = PrimaryModel()
        self.meta = MetaModel()

    def fit(self, train_data):
        primary_signals = self.primary.generate(train_data)
        self.meta.fit(train_data, primary_signals)

    def generate_signals(self, test_data, regime=None):
        primary_signals = self.primary.generate(test_data)
        confidence = self.meta.predict_proba(test_data)

        signals = np.where(
            (primary_signals != 0) & (confidence >= META_CONFIDENCE),
            primary_signals,
            0,
        )
        return signals, confidence


def purged_walk_forward(data, n_splits=CV_SPLITS, embargo=PURGE_EMBARGO_DAYS):
    n = len(data)
    test_size = n // (n_splits + 1)
    folds = []
    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = min(test_start + test_size, n)
        train_end = test_start - embargo - SWING_HORIZON
        if train_end < 20:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))
    return folds


def run_cv(data):
    folds = purged_walk_forward(data)
    fold_results = []
    all_importances = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        strat = MLStrategy()
        strat.fit(train)
        signals, confidence = strat.generate_signals(test)

        buy_signals = (signals == 1).sum()
        win_rate = 0.0
        if buy_signals > 0:
            buys_tb = test["Target"].values[signals == 1]
            win_rate = (buys_tb == 1).mean() * 100

        avg_ret = 0.0
        if buy_signals > 0:
            avg_ret = test["TB_return"].values[signals == 1].mean() * 100

        importances = strat.meta.feature_importances()
        all_importances.append(importances)

        fold_results.append({
            "fold": fold_i + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "buy_signals": int(buy_signals),
            "win_rate": win_rate,
            "avg_return": avg_ret,
        })

    avg_importance = {}
    if all_importances:
        for feat in FEATURE_COLS:
            avg_importance[feat] = np.mean([imp[feat] for imp in all_importances])

    return fold_results, avg_importance
