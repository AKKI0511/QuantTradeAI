"""Machine learning models for trading signals.

Currently the package offers a single :class:`MomentumClassifier` which
combines several algorithms into a voting ensemble.

Key Components:
    - :class:`MomentumClassifier`: orchestrates training and evaluation

Typical Usage:
    ```python
    from quanttradeai.models import MomentumClassifier
    model = MomentumClassifier()
    X, y = model.prepare_data(df)
    model.train(X, y)
    predictions = model.predict(X)
    ```
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from quanttradeai.utils.metrics import classification_metrics
import xgboost as xgb
import optuna
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumClassifier:
    """Voting Classifier for momentum trading strategy."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the model with configuration."""
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction.

        Args:
            df: DataFrame with features and labels

        Returns:
            Tuple of features array and labels array
        """
        # Remove non-feature columns
        exclude_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "forward_returns",
            "label",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        X = df[feature_cols].values
        y = df["label"].values

        return X, y

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, n_trials: int = 100
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Labels array
            n_trials: Number of optimization trials

        Returns:
            Dictionary of best parameters
        """

        def objective(trial):
            # Logistic Regression parameters
            lr_params = {
                "C": trial.suggest_float("lr_C", 1e-5, 100, log=True),
                "max_iter": 1000,
                "class_weight": "balanced",
            }

            # Random Forest parameters
            rf_params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
                "max_depth": trial.suggest_int("rf_max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
                "class_weight": "balanced",
            }

            # XGBoost parameters
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "xgb_learning_rate", 1e-3, 0.1, log=True
                ),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "xgb_colsample_bytree", 0.6, 1.0
                ),
            }

            # Create models with trial parameters
            lr = LogisticRegression(**lr_params)
            rf = RandomForestClassifier(**rf_params)
            xgb_clf = xgb.XGBClassifier(**xgb_params)

            # Create voting classifier
            voting_clf = VotingClassifier(
                estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_clf)], voting="soft"
            )

            # Perform cross-validation
            scores = cross_val_score(voting_clf, X, y, cv=5, scoring="f1_weighted")
            return scores.mean()

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def train(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None
    ) -> None:
        """
        Train the voting classifier.

        Args:
            X: Feature matrix
            y: Labels array
            params: Optional hyperparameters
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Use default parameters if none provided
        if params is None:
            params = {
                "lr_C": 1.0,
                "rf_n_estimators": 100,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "xgb_n_estimators": 100,
                "xgb_max_depth": 6,
                "xgb_learning_rate": 0.1,
                "xgb_subsample": 0.8,
                "xgb_colsample_bytree": 0.8,
            }

        # Create base models with parameters
        lr = LogisticRegression(
            C=params["lr_C"], max_iter=1000, class_weight="balanced"
        )

        rf = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            class_weight="balanced",
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=params["xgb_n_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample_bytree"],
        )

        # Create and train voting classifier
        self.model = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_clf)], voting="soft"
        )

        self.model.fit(X_scaled, y)
        logger.info("Model training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(X)
        return classification_metrics(y, predictions)

    def save_model(self, path: str) -> None:
        """Save the trained model and scaler."""
        import joblib

        joblib.dump(self.model, f"{path}/voting_classifier.joblib")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        joblib.dump(self.feature_columns, f"{path}/feature_columns.joblib")
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained model and scaler."""
        import joblib

        self.model = joblib.load(f"{path}/voting_classifier.joblib")
        self.scaler = joblib.load(f"{path}/scaler.joblib")
        self.feature_columns = joblib.load(f"{path}/feature_columns.joblib")
        logger.info(f"Model loaded from {path}")
