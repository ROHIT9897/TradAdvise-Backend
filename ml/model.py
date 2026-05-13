# ml/model.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Precision is ill-defined.*")
warnings.filterwarnings("ignore", message=".*Parameters.*not used.*")

import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class StockModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = None
        self.trained      = False

    def train(self, features_df: pd.DataFrame) -> dict:
        X = features_df.drop("target", axis=1)
        y = features_df["target"] + 1   # -1,0,1  →  0,1,2

        self.feature_cols = X.columns.tolist()

        counts = y.value_counts().sort_index()
        logger.info(
            f"Class distribution — "
            f"SELL:{counts.get(0, 0)} "
            f"HOLD:{counts.get(1, 0)} "
            f"BUY:{counts.get(2, 0)}"
        )

        tscv       = TimeSeriesSplit(n_splits=5)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx]
            X_val   = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val   = y.iloc[val_idx]

            # Fresh model each fold — avoids state leakage between folds
            fold_model = xgb.XGBClassifier(
                **{k: v for k, v in self.model.get_params().items()}
            )
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = fold_model.predict(X_val)
            score = accuracy_score(y_val, preds)
            val_scores.append(score)
            logger.info(f"Fold {fold + 1}: {score:.3f}")

        # Final model trained on ALL data for production use
        self.model.fit(X, y, verbose=False)
        self.trained = True

        avg     = np.mean(val_scores) * 100
        trimmed = np.mean(sorted(val_scores)[1:]) * 100
        logger.info(f"CV accuracy: {avg:.1f}% | Trimmed: {trimmed:.1f}%")

        return {
            "cv_accuracy":      round(avg, 1),
            "trimmed_accuracy": round(trimmed, 1),
            "fold_scores":      [round(s * 100, 1) for s in val_scores],
        }

    def predict(self, X: pd.DataFrame) -> dict:
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = X[self.feature_cols]

        proba      = self.model.predict_proba(X)
        pred_class = int(proba.argmax(axis=1)[0])
        confidence = float(proba.max(axis=1)[0]) * 100

        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

        return {
            "signal":     signal_map[pred_class],
            "confidence": round(confidence, 1),
            "probabilities": {
                "SELL": round(float(proba[0][0]) * 100, 1),
                "HOLD": round(float(proba[0][1]) * 100, 1),
                "BUY":  round(float(proba[0][2]) * 100, 1),
            },
        }

    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True
        )
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StockModel":
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model