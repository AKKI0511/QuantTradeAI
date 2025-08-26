# QuantTradeAI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-blue.svg)](https://python-poetry.org/)

> A comprehensive machine learning framework for quantitative trading strategies with focus on momentum trading using ensemble models.

## 🚀 Quick Start

```bash
# Install dependencies
poetry install

# Fetch data for all symbols
poetry run quanttradeai fetch-data

# Run complete training pipeline
poetry run quanttradeai train

# Evaluate a saved model
poetry run quanttradeai evaluate -m models/trained/AAPL
```

## ✨ Features

- **📊 Multi-source Data Fetching** - YFinance, AlphaVantage with intelligent caching
- **🔧 Advanced Feature Engineering** - 20+ technical indicators and custom features
- **🤖 Ensemble ML Models** - Voting Classifier with LR, RF, XGBoost
- **🗞️ LLM Sentiment Analysis** - Swapable provider/model via LiteLLM
- **⚡ Hyperparameter Optimization** - Optuna-based automatic tuning
- **📈 Comprehensive Backtesting** - Risk management and performance metrics
- **🎯 Production Ready** - Model persistence, CLI interface, configuration management

## 📋 Supported Assets

| Symbol | Company | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Technology |
| META | Meta Platforms | Technology |
| TSLA | Tesla Inc. | Automotive |
| JPM | JPMorgan Chase | Financial |
| AMZN | Amazon.com | Consumer |

## 🏗️ Architecture

```
QuantTradeAI/
├── quanttradeai/           # Core framework
│   ├── data/              # Data loading & processing
│   ├── features/          # Technical indicators
│   ├── models/            # ML models & training
│   ├── backtest/          # Backtesting engine
│   ├── trading/           # Risk management
│   └── utils/             # Utilities & metrics
├── config/                # Configuration files
├── docs/                  # 📚 Documentation
└── tests/                 # Unit tests
```

## 📚 Documentation

- **[📖 Getting Started](docs/getting-started.md)** - Installation and first steps
- **[🔧 API Reference](docs/api/)** - Complete API documentation
- **[📊 Examples](docs/examples/)** - Usage examples and tutorials
- **[⚙️ Configuration](docs/configuration.md)** - Configuration guide
- **[🚀 Quick Reference](docs/quick-reference.md)** - Common patterns and commands

## 🎯 Use Cases

- **Research & Development** - Test new trading strategies
- **Model Training** - Train and optimize ML models
- **Backtesting** - Simulate historical trading
- **Production Deployment** - Real-time trading systems

## 🔧 Installation

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip

### Install with Poetry
```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install
```

### Install with pip
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Command Line Interface

```bash
# Show available commands
poetry run quanttradeai --help

# Fetch and cache data
poetry run quanttradeai fetch-data

# Run complete training pipeline
poetry run quanttradeai train

# Evaluate a trained model
poetry run quanttradeai evaluate -m models/trained/AAPL
```

### Python API

```python
from quanttradeai import DataLoader, DataProcessor, MomentumClassifier

# Initialize components
loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")
classifier = MomentumClassifier("config/model_config.yaml")

# Fetch and process data
data_dict = loader.fetch_data()
df = data_dict['AAPL']
df_processed = processor.process_data(df)
df_labeled = processor.generate_labels(df_processed)

# Train model
X, y = classifier.prepare_data(df_labeled)
classifier.train(X, y)
```

### Sentiment Analysis

LiteLLM powers provider-agnostic sentiment scoring. Configure it in
`config/features_config.yaml` and set the API key in your environment:

```yaml
sentiment:
  enabled: true
  provider: openai
  model: gpt-3.5-turbo
  api_key_env_var: OPENAI_API_KEY
```

```bash
export OPENAI_API_KEY="sk-..."
```

Switching providers is as simple as updating the YAML config to point to a
different `provider`/`model` pair supported by LiteLLM.

## 📊 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Classification accuracy |
| **F1 Score** | Balanced precision/recall |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Max Drawdown** | Maximum portfolio decline |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black quanttradeai/

# Lint code
poetry run flake8 quanttradeai/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YFinance](https://github.com/ranaroussi/yfinance) for data fetching
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [Optuna](https://optuna.org/) for hyperparameter optimization

## �️ Roadmap & Future Enhancements

### 🚀 Phase 1: Core Infrastructure
- **Real-time Data Streaming** - WebSocket integration for live market data
- **Advanced Risk Management** - Portfolio-level risk controls and position sizing
- **Multi-timeframe Support** - Intraday, daily, weekly, monthly analysis
- **Enhanced Backtesting** - Transaction costs, slippage, market impact modeling

### 🤖 Phase 2: AI & LLM Integration
- **LLM-Powered Analysis** - OpenAI, Anthropic, Gemini integration for market sentiment
- **Natural Language Processing** - News sentiment analysis and earnings call transcripts
- **AI-Driven Feature Engineering** - Automated feature selection and generation
- **Intelligent Portfolio Allocation** - LLM-based asset allocation strategies

### ⚡ Phase 3: Performance & Latency
- **C++ Core Components** - Low-latency data processing and signal generation
- **GPU Acceleration** - CUDA/OpenCL for parallel model training
- **Real-time Trading** - Direct exchange connectivity and order execution
- **Microservices Architecture** - Scalable, containerized deployment

### 🌐 Phase 4: Advanced AI & Cloud
- **Multi-Modal AI** - Vision models for chart pattern recognition
- **Reinforcement Learning** - RL agents for dynamic strategy adaptation
- **Cloud-Native Deployment** - Kubernetes orchestration and auto-scaling
- **Federated Learning** - Privacy-preserving model training across institutions

### 🔮 Phase 5: Next-Generation Features
- **Quantum Computing Integration** - Quantum algorithms for optimization
- **Blockchain Integration** - DeFi protocol connectivity and token trading
- **Advanced NLP** - Real-time news analysis and social sentiment
- **Autonomous Trading** - Self-optimizing strategies with minimal human intervention

## 📞 Support

- **📚 Documentation**: [docs/](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/AKKI0511/QuantTradeAI/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/AKKI0511/QuantTradeAI/discussions)

---

**Made with ❤️ for quantitative trading**
