# Models API

## Overview

The current model API centers on `MomentumClassifier`, a scikit-learn voting classifier that combines logistic regression, random forest, and XGBoost estimators. It is used by the research workflow and by model agents that load saved artifacts.

## Public Imports

```python
from quanttradeai import MomentumClassifier
from quanttradeai.models import MomentumClassifier
from quanttradeai.models.classifier import MomentumClassifier
```

`MomentumClassifier` is a lazy top-level export. If importing the model stack fails at the root package path, `quanttradeai.MomentumClassifier` resolves to `None`; direct imports from `quanttradeai.models.classifier` will raise the underlying import error.

## Main Classes and Methods

| API | Import Path | Purpose |
| --- | --- | --- |
| `MomentumClassifier` | `quanttradeai.models.classifier` | Train, tune, evaluate, save, and load a voting classifier |
| `prepare_data(df)` | method | Split feature columns from `label` |
| `optimize_hyperparameters(X, y, n_trials=100)` | method | Tune model parameters with Optuna and time-series CV |
| `train(X, y, params=None)` | method | Fit scaler and voting classifier |
| `predict(X)` | method | Predict labels with a trained model |
| `evaluate(X, y)` | method | Return classification metrics |
| `save_model(path)` | method | Write joblib artifacts |
| `load_model(path)` | method | Load joblib artifacts |

## `MomentumClassifier`

**Signature**

```python
class MomentumClassifier:
    def __init__(self, config_path: str = "config/model_config.yaml")
```

The constructor reads YAML from `config_path`, creates a `StandardScaler`, and initializes `model` and `feature_columns` to `None`.

```python
from quanttradeai import MomentumClassifier

classifier = MomentumClassifier("config/model_config.yaml")
```

**Configuration used directly**

| Config Path | Used By |
| --- | --- |
| `training.cv_folds` | `optimize_hyperparameters` for `TimeSeriesSplit` |

Other config values are primarily consumed upstream by data loading, feature generation, labels, and research orchestration.

## `prepare_data`

**Signature**

```python
def prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]
```

Builds `X` and `y` from a labeled feature DataFrame.

| Input Requirement | Behavior |
| --- | --- |
| `label` column | Required as the target vector |
| Feature columns | All columns except `Open`, `High`, `Low`, `Close`, `Volume`, `forward_returns`, and `label` |
| Return | `(X, y)` NumPy arrays |

`prepare_data` also sets `self.feature_columns` to the selected feature column names.

```python
X, y = classifier.prepare_data(labeled_features)
```

## `optimize_hyperparameters`

**Signature**

```python
def optimize_hyperparameters(
    self,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 100,
) -> dict
```

Runs an Optuna study using weighted F1 from `cross_val_score` with `TimeSeriesSplit`. It tunes:

| Estimator | Tuned Parameters |
| --- | --- |
| Logistic regression | `lr_C` |
| Random forest | `rf_n_estimators`, `rf_max_depth`, `rf_min_samples_split` |
| XGBoost | `xgb_n_estimators`, `xgb_max_depth`, `xgb_learning_rate`, `xgb_subsample`, `xgb_colsample_bytree` |

```python
params = classifier.optimize_hyperparameters(X, y, n_trials=25)
```

## `train`

**Signature**

```python
def train(
    self,
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
) -> None
```

Fits the internal `StandardScaler`, creates a soft-voting classifier, and trains it.

If `params` is omitted, defaults are used:

```python
{
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
```

```python
classifier.train(X, y, params=params)
```

## `predict`

**Signature**

```python
def predict(self, X: np.ndarray) -> np.ndarray
```

Transforms `X` with the fitted scaler and returns predicted labels.

```python
predictions = classifier.predict(X_test)
```

**Errors**

Raises `ValueError("Model not trained yet")` when `self.model` is `None`.

## `evaluate`

**Signature**

```python
def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]
```

Calls `predict(X)` and returns `classification_metrics(y, predictions)` with:

| Metric | Source |
| --- | --- |
| `accuracy` | `sklearn.metrics.accuracy_score` |
| `precision` | weighted precision |
| `recall` | weighted recall |
| `f1` | weighted F1 |

```python
metrics = classifier.evaluate(X_test, y_test)
```

## Persistence

### `save_model`

**Signature**

```python
def save_model(self, path: str) -> None
```

Writes three joblib files under `path`:

| File | Contents |
| --- | --- |
| `voting_classifier.joblib` | Fitted `VotingClassifier` |
| `scaler.joblib` | Fitted `StandardScaler` |
| `feature_columns.joblib` | List of selected feature columns |

`save_model` does not create `path`; create the directory before calling it.

```python
classifier.save_model("models/aapl_momentum")
```

### `load_model`

**Signature**

```python
def load_model(self, path: str) -> None
```

Loads the three joblib artifacts written by `save_model`.

```python
classifier = MomentumClassifier("config/model_config.yaml")
classifier.load_model("models/aapl_momentum")
```

## Minimal Example

```python
from pathlib import Path

from quanttradeai import DataProcessor, MomentumClassifier

processor = DataProcessor("config/features_config.yaml")
features = processor.generate_features(raw_bars)
labeled = processor.generate_labels(features)

model = MomentumClassifier("config/model_config.yaml")
X, y = model.prepare_data(labeled)
model.train(X, y)

Path("models/example").mkdir(parents=True, exist_ok=True)
model.save_model("models/example")
```

## Relationship to `quanttradeai research run`

The research CLI path handles the full sequence around `MomentumClassifier`:

1. Load raw bars with `DataLoader`.
2. Generate and preprocess features with `DataProcessor`.
3. Generate labels.
4. Split data in time order.
5. Train, evaluate, backtest, and persist artifacts.

Use the Python class directly when you need a custom notebook, a custom split, or an embedding in a larger Python research system.

## Optional Dependency Notes

The model module imports `scikit-learn`, `xgboost`, `optuna`, `numpy`, `pandas`, `yaml`, and `joblib`. These are package dependencies in the current project metadata. If a runtime environment omits one, importing or using `MomentumClassifier` will fail at the point that dependency is needed.

## Related CLI/YAML Docs

- [CLI docs](../cli/)
- [Config docs](../config/)
- [Artifacts](../artifacts.md)

## Common Mistakes

- Calling `prepare_data` before adding a `label` column.
- Passing raw OHLCV data directly to `train`; train on generated feature columns.
- Saving to a directory that does not exist.
- Predicting with columns in a different order from `feature_columns`.
- Using shuffled cross-validation for time series. `optimize_hyperparameters` uses `TimeSeriesSplit`; keep custom evaluation time-aware too.
