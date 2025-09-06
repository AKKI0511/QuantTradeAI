# QuantTradeAI Documentation

Welcome to the QuantTradeAI documentation! This comprehensive machine learning framework for quantitative trading strategies provides a complete pipeline from data acquisition to model training and backtesting.

## 📚 Documentation Index

### Core Documentation
- **[API Documentation](api/)** - Comprehensive reference for all public APIs, functions, and components
- **[Quick Reference Guide](quick-reference.md)** - Common usage patterns and examples
- **[LLM Sentiment Analysis](llm-sentiment.md)** - Configure LLM-based sentiment scoring
- **[Main README](../README.md)** - Project overview and getting started guide

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI

# Install dependencies
pip install poetry
poetry install
```

### Basic Usage
```bash
# Fetch data for all symbols
poetry run quanttradeai fetch-data -c config/model_config.yaml

# Run complete training pipeline
poetry run quanttradeai train -c config/model_config.yaml

# Evaluate a saved model
poetry run quanttradeai evaluate -c config/model_config.yaml -m models/trained/AAPL
```

## 📖 Documentation Structure

### 1. API Documentation
The [API Documentation](api/) provides comprehensive coverage of:

- **Command Line Interface** - CLI commands and entry points
- **Data Loading and Processing** - DataLoader, DataProcessor, and DataSource classes
- **Feature Engineering** - Technical indicators and custom features
- **Machine Learning Models** - MomentumClassifier and training utilities
- **Backtesting Framework** - Trade simulation and performance metrics
- **Trading Utilities** - Risk management, position sizing, real-time position control
- **Utility Functions** - Metrics, visualization, and configuration schemas
- **Configuration** - Model and feature configuration examples

### 2. Quick Reference Guide
The [Quick Reference Guide](quick-reference.md) includes:

- **Quick Start** - Basic setup and usage
- **Common Patterns** - Data loading, feature engineering, model training, backtesting
- **Technical Indicators** - Usage examples for all technical indicators
- **Custom Features** - Momentum score and volatility breakout functions
- **Risk Management** - Stop-loss, take-profit, drawdown guard, and position sizing
- **Performance Metrics** - Classification and trading metrics
- **Visualization** - Price charts and performance plots
- **Configuration Examples** - Model and feature configuration templates
- **Error Handling** - Common error patterns and solutions
- **Best Practices** - Recommended approaches for each component
- **Troubleshooting** - Solutions for common issues

## 🏗️ Architecture Overview

### Core Components

```
QuantTradeAI/
├── quanttradeai/
│   ├── main.py              # CLI entry point
│   ├── data/                # Data processing
│   │   ├── loader.py        # Data fetching and caching
│   │   ├── processor.py     # Feature engineering
│   │   └── datasource.py    # Data source abstractions
│   ├── features/            # Feature engineering
│   │   ├── technical.py     # Technical indicators
│   │   └── custom.py        # Custom features
│   ├── models/              # Machine learning
│   │   └── classifier.py    # Voting classifier
│   ├── backtest/            # Backtesting
│   │   └── backtester.py    # Trade simulation
│   ├── trading/             # Trading utilities
│   │   ├── drawdown_guard.py  # Drawdown protection
│   │   ├── portfolio.py      # Portfolio operations
│   │   ├── position_manager.py # Real-time position tracking
│   │   └── risk_manager.py   # Risk coordination
│   └── utils/               # Utilities
│       ├── metrics.py       # Performance metrics
│       ├── visualization.py # Plotting functions
│       └── config_schemas.py # Configuration validation
├── config/                  # Configuration files
│   ├── model_config.yaml    # Model parameters
│   ├── features_config.yaml # Feature engineering settings
│   └── risk_config.yaml     # Drawdown and turnover limits
└── docs/                    # Documentation
    ├── api/                 # API documentation
    ├── quick-reference.md   # Quick reference guide
    └── README.md            # This file
```

### Data Flow

1. **Data Acquisition** - Fetch OHLCV data from multiple sources
2. **Feature Engineering** - Generate technical indicators and custom features
3. **Label Generation** - Create trading signals based on forward returns
4. **Model Training** - Train ensemble models with hyperparameter optimization
5. **Backtesting** - Simulate trades with risk management
6. **Performance Analysis** - Calculate metrics and generate reports

## 🔧 Key Features

### Data Management
- **Multi-source data fetching** (YFinance, AlphaVantage)
- **Intelligent caching** with expiration
- **Data validation** and quality checks
- **Parallel processing** for multiple symbols

### Feature Engineering
- **Technical indicators** (SMA, EMA, RSI, MACD, Stochastic, Bollinger Bands)
- **Volume indicators** (OBV, volume ratios)
- **Custom features** (momentum score, volatility breakout)
- **Feature preprocessing** (scaling, outlier handling)

### Machine Learning
- **Ensemble models** (Voting Classifier with Logistic Regression, Random Forest, XGBoost)
- **Hyperparameter optimization** using Optuna
- **Cross-validation** for robust evaluation
- **Model persistence** and loading

### Backtesting
- **Trade simulation** with limit/stop orders and intrabar tick fills
- **Adaptive market impact modeling** with dynamic spreads and asymmetric coefficients
- **Borrow fee accounting** for short positions
- **Risk management** (stop-loss, take-profit)
- **Performance metrics** (Sharpe ratio, max drawdown)
- **Position sizing** based on risk parameters

### Risk Management
- **Stop-loss and take-profit** rules
- **Position sizing** calculations
- **Risk-adjusted returns** analysis
- **Drawdown and turnover guards**

## 📊 Supported Assets

The framework is currently configured for:
- **AAPL** (Apple Inc.)
- **META** (Meta Platforms)
- **TSLA** (Tesla Inc.)
- **JPM** (JPMorgan Chase)
- **AMZN** (Amazon.com)

## 🎯 Use Cases

### 1. Research and Development
- Test new trading strategies
- Experiment with different feature combinations
- Optimize model hyperparameters
- Analyze strategy performance

### 2. Model Training
- Train models for multiple assets
- Cross-validate model performance
- Save and load trained models
- Evaluate out-of-sample performance

### 3. Backtesting
- Simulate historical trading
- Test risk management rules
- Calculate performance metrics
- Generate equity curves

### 4. Production Deployment
- Load trained models
- Make real-time predictions
- Implement risk management
- Monitor performance

## 🔍 Finding Information

### By Component
- **Data Loading**: See [DataLoader API](api/data.md#dataloader-class)
- **Feature Engineering**: See [Feature Engineering API](api/features.md)
- **Model Training**: See [Machine Learning Models API](api/models.md)
- **Backtesting**: See [Backtesting Framework API](api/backtesting.md)
- **Risk Management**: See [Trading Utilities API](api/trading.md)

### By Task
- **Getting Started**: See [Quick Reference Guide](quick-reference.md#-cli-commands)
- **Configuration**: See [Configuration Examples](quick-reference.md#-configuration-examples)
- **Error Handling**: See [Troubleshooting](quick-reference.md#-troubleshooting)

### By Function
- **Technical Indicators**: See [Technical Indicators API](api/features.md#technical-indicators)
- **Performance Metrics**: See [Data Loading API](api/data.md)
- **Visualization**: See [Data Loading API](api/data.md)

## 🤝 Contributing

When contributing to the documentation:

1. **Update API Documentation** - Add new functions and classes to the appropriate files in `api/`
2. **Add Examples** - Include usage examples in `quick-reference.md`
3. **Update Configuration** - Document new configuration options
4. **Add Troubleshooting** - Include solutions for common issues

## 📝 Documentation Standards

### Code Examples
- Include complete, runnable examples
- Show both simple and advanced usage
- Include error handling where appropriate
- Use consistent formatting and naming

### API Documentation
- Document all public functions and classes
- Include parameter types and descriptions
- Provide return value descriptions
- Include usage examples

### Configuration
- Document all configuration options
- Provide example configurations
- Explain parameter effects
- Include validation rules

## 🔗 External Resources

- **Project Repository**: [GitHub](https://github.com/AKKI0511/QuantTradeAI)
- **Poetry Documentation**: [poetry.dev](https://python-poetry.org/docs/)
- **YFinance Documentation**: [yfinance](https://pypi.org/project/yfinance/)
- **Scikit-learn Documentation**: [scikit-learn](https://scikit-learn.org/)
- **XGBoost Documentation**: [XGBoost](https://xgboost.readthedocs.io/)

## 📞 Support

For questions and support:

1. **Check the documentation** - Most questions are answered in the API docs or quick reference
2. **Review examples** - See the complete workflow examples
3. **Check configuration** - Ensure your configuration files are correct
4. **Review troubleshooting** - Common issues and solutions are documented

---

This documentation provides comprehensive coverage of the QuantTradeAI framework. Start with the [Quick Reference Guide](quick-reference.md) for common usage patterns, then refer to the [API Documentation](api/) for detailed function references.
