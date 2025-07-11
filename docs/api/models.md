# Machine Learning Models

API documentation for the MomentumClassifier and training utilities.

## MomentumClassifier Class

### `MomentumClassifier(config_path: str = "config/model_config.yaml")`

Voting Classifier for momentum trading strategy using Logistic Regression, Random Forest, and XGBoost.

**Parameters:**
- `config_path` (str): Path to model configuration file

**Example:**
```python
from quanttradeai.models.classifier import MomentumClassifier

# Initialize classifier
classifier = MomentumClassifier("config/model_config.yaml")
```

### `prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`

Prepares data for training/prediction.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features and labels

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Tuple of features array and labels array

**Example:**
```python
# Prepare data for training
X, y = classifier.prepare_data(labeled_df)
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### `optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict`

Optimizes hyperparameters using Optuna.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Labels array
- `n_trials` (int): Number of optimization trials

**Returns:**
- `Dict`: Dictionary of best parameters

**Example:**
```python
# Optimize hyperparameters
best_params = classifier.optimize_hyperparameters(X_train, y_train, n_trials=50)
print(f"Best parameters: {best_params}")
```

### `train(X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None)`

Trains the voting classifier.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Labels array
- `params` (Dict[str, Any], optional): Hyperparameters

**Example:**
```python
# Train with optimized parameters
classifier.train(X_train, y_train, params=best_params)

# Train with default parameters
classifier.train(X_train, y_train)
```

### `predict(X: np.ndarray) -> np.ndarray`

Makes predictions using the trained model.

**Parameters:**
- `X` (np.ndarray): Feature matrix

**Returns:**
- `np.ndarray`: Array of predictions

**Example:**
```python
# Make predictions
predictions = classifier.predict(X_test)
print(f"Predictions: {predictions}")
```

### `evaluate(X: np.ndarray, y: np.ndarray) -> Dict[str, float]`

Evaluates model performance.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): True labels

**Returns:**
- `Dict[str, float]`: Dictionary of performance metrics

**Example:**
```python
# Evaluate model
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### `save_model(path: str)`

Saves the trained model and scaler.

**Parameters:**
- `path` (str): Directory path to save model

**Example:**
```python
# Save model
classifier.save_model("models/trained/AAPL")
```

### `load_model(path: str)`

Loads a trained model and scaler.

**Parameters:**
- `path` (str): Directory path containing saved model

**Example:**
```python
# Load model
classifier.load_model("models/trained/AAPL")
```

## Model Architecture

### Voting Classifier
The MomentumClassifier uses a voting ensemble with three base models:

1. **Logistic Regression** - Linear model for baseline performance
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting for complex patterns

### Hyperparameter Optimization
The framework optimizes hyperparameters for each base model:

```python
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
    "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.1, log=True),
    "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
}
```

## Training Workflow

### Complete Training Example
```python
from quanttradeai.models.classifier import MomentumClassifier
from sklearn.model_selection import train_test_split

# Initialize classifier
classifier = MomentumClassifier("config/model_config.yaml")

# Prepare data
X, y = classifier.prepare_data(df_labeled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Optimize hyperparameters
best_params = classifier.optimize_hyperparameters(X_train, y_train, n_trials=50)

# Train model
classifier.train(X_train, y_train, params=best_params)

# Evaluate performance
train_metrics = classifier.evaluate(X_train, y_train)
test_metrics = classifier.evaluate(X_test, y_test)

# Save model
classifier.save_model("models/trained/AAPL")

print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
```

### Model Persistence
```python
# Save model with metadata
classifier.save_model("models/trained/AAPL")

# Load model for inference
classifier = MomentumClassifier("config/model_config.yaml")
classifier.load_model("models/trained/AAPL")

# Make predictions
predictions = classifier.predict(X_new)
```

## Configuration

### Model Configuration Example
```yaml
models:
  voting_classifier:
    voting: 'soft'
    weights: [1, 2, 2]
  
  logistic_regression:
    C: 1.0
    max_iter: 1000
    class_weight: 'balanced'
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    class_weight: 'balanced'
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

## Error Handling

### Training Issues
```python
try:
    # Train model
    classifier.train(X_train, y_train)
except ValueError as e:
    print(f"Training error: {e}")
    # Check data shapes and class distribution
    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
```

### Prediction Issues
```python
try:
    # Make predictions
    predictions = classifier.predict(X_test)
except Exception as e:
    print(f"Prediction error: {e}")
    # Ensure model is trained
    if classifier.model is None:
        print("Model not trained yet")
```

## Performance Tips

### Memory Management
```python
# Use smaller data for hyperparameter optimization
X_sample = X_train[:1000]
y_sample = y_train[:1000]
best_params = classifier.optimize_hyperparameters(X_sample, y_sample, n_trials=20)
```

### Parallel Processing
```python
# Use multiple workers for hyperparameter optimization
import optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=4)
```

## Related Documentation

- **[Data Loading](data.md)** - Data fetching and processing
- **[Feature Engineering](features.md)** - Technical indicators and features
- **[Backtesting](backtesting.md)** - Trade simulation and evaluation
- **[Configuration](../configuration.md)** - Configuration guide
- **[Quick Reference](../quick-reference.md)** - Common patterns