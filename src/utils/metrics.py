import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return basic classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown.min()
