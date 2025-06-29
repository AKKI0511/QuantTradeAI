# QuantTradeAI

A comprehensive machine learning framework for quantitative trading strategies, with current focus on the Quantinsti Momentum Trading Strategy Competition.

## Current Competition Focus
- **Competition**: Quantinsti Quantitative Research Challenge
- **Objective**: ML-based momentum trading strategy using Voting Classifier
- **Target Assets**: AAPL, META, TSLA, JPM, AMZN
- **Data Period**: 
  - Training: 2015-2024 (Daily OHLCV)
  - Testing: January 2025

## Key Features
- Robust data pipeline for financial instruments
- Advanced feature engineering toolkit:
  - Momentum indicators (SMA, EMA, RSI, MACD, Stochastic, Bollinger Bands)
  - Volume indicators (On-Balance Volume and volume MA ratios)
  - Return-based features
  - Custom feature engineering
  - Reusable indicator helpers in `features/technical.py`
  - Strategy-specific features in `features/custom.py`
- Ensemble ML models:
  - Voting Classifier with Logistic Regression, Random Forest, and XGBoost
  - Hyperparameter optimization
  - Cross-validation framework
  - Comprehensive performance analytics and visualization
  - Utility metrics in `utils/metrics.py`
  - Charting helpers in `utils/visualization.py`

## Tech Stack
- Python 3.9+
- yfinance (data acquisition)
- scikit-learn (ML models)
- XGBoost (gradient boosting)
- pandas-ta (technical analysis)
- Optuna (hyperparameter optimization)

## Project Structure
```
QuantTradeAI/
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py        # Data fetching (yfinance integration)
│   │   └── processor.py     # Data preprocessing
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical.py     # Technical indicators
│   │   └── custom.py        # Custom features
│   ├── models/              # ML models
│   │   ├── __init__.py
│   │   ├── classifier.py    # Voting classifier implementation
│   │   └── optimization.py  # Hyperparameter tuning
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── metrics.py       # Performance metrics
│       └── visualization.py # Plotting functions
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── config/                  # Configuration files
│   ├── model_config.yaml    # Model parameters
│   └── features_config.yaml # Feature engineering settings
├── data/                    # Data storage
│   ├── raw/                 # Raw OHLCV data
│   ├── processed/          # Processed data
│   └── features/           # Feature engineered data
├── models/                  # Saved model artifacts
│   ├── trained/            # Production models
│   └── experiments/        # Experimental results
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── reports/                # Analysis reports
│   └── figures/           # Visualizations
└── research/              # Research and experimentation
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run initial setup:
```bash
python setup.py develop
```

## Development

Install pre-commit hooks to run the same checks as the CI pipeline:

```bash
pre-commit install
pre-commit run --all-files
```

The hooks will run `black --check`, `flake8`, and `pytest` before every commit.

## Data Caching

The `DataLoader` caches downloaded OHLCV data in parquet format to reduce
network usage. Configure caching in `config/model_config.yaml`:

```yaml
data:
  cache_dir: 'data/raw'   # Storage location for cached files
  use_cache: true         # Toggle cache usage
 refresh: false          # Force fresh download when true
```

## Command Line Interface

Use the CLI in `src/main.py` to run common tasks:

```bash
python -m src.main fetch-data -c config/model_config.yaml       # download data
python -m src.main train -c config/model_config.yaml             # run pipeline
python -m src.main evaluate -m models/trained/MY_MODEL           # evaluate
```

`fetch-data` saves raw data to the configured cache directory. `train` executes
the full training pipeline. `evaluate` loads an existing model directory and
generates evaluation metrics.

## Competition Evaluation Metrics

### Trade-Level Analytics
- Prediction Accuracy (buy/sell/hold)
- Precision, Recall, F1-score
- Win/Loss Ratio

### Performance Metrics
- Total Return (%)
- Annualized Sharpe Ratio
- Maximum Drawdown (MDD)
- Transaction Cost Impact

## Future Development
- Integration with additional data sources
- Support for more asset classes
- Real-time trading capabilities
- Advanced risk management features
- API development for strategy deployment

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Project Status
🚧 Under active development
