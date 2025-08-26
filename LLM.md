# QuantTradeAI - LLM Agent Guide

## Project Overview

QuantTradeAI is a comprehensive machine learning framework for quantitative trading strategies. The codebase implements momentum trading using ensemble models (Logistic Regression, Random Forest, XGBoost) with advanced feature engineering and backtesting capabilities.

## Core Architecture

### Key Components
- **Data Layer**: `quanttradeai/data/` - Data fetching, caching, validation
- **Feature Engineering**: `quanttradeai/features/` - Technical indicators, custom features
- **ML Models**: `quanttradeai/models/` - Ensemble classifiers, hyperparameter optimization
- **Backtesting**: `quanttradeai/backtest/` - Trade simulation, performance metrics
- **Risk Management**: `quanttradeai/trading/` - Stop-loss, position sizing
- **Utilities**: `quanttradeai/utils/` - Metrics, visualization, configuration

### Data Flow
1. Fetch OHLCV data (YFinance/AlphaVantage)
2. Generate technical indicators (SMA, EMA, RSI, MACD, etc.)
3. Create custom features (momentum score, volatility breakout)
4. Generate trading labels (forward returns)
5. Train ensemble models with hyperparameter optimization
6. Backtest with risk management
7. Evaluate performance metrics

## Development Guidelines

### Code Quality Standards
- **Testing**: All new code MUST have unit tests
- **Formatting**: Use Black for code formatting
- **Linting**: Use flake8 for code quality
- **Type Hints**: Include type annotations for all functions

### Pre-commit Requirements
```bash
# Run all quality checks
make format   # Black formatting
make lint     # flake8 linting
make test     # pytest testing
```

### Dependency Management
- **CRITICAL**: Use Poetry CLI for ALL dependency changes
- **NEVER** manually edit `pyproject.toml` dependencies
- **ALWAYS** use: `poetry add package-name` or `poetry add --group dev package-name`
- **REMOVE** dependencies with: `poetry remove package-name`

### Testing Requirements
- Unit tests for all new functions/classes
- Integration tests for data pipelines
- Performance tests for critical paths
- Test coverage > 80%

## Key Technologies

### Core Dependencies
- **Python 3.11+** - Main language
- **Poetry** - Dependency management
- **pandas/numpy** - Data manipulation
- **scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Optuna** - Hyperparameter optimization
- **yfinance** - Market data
- **pandas-ta** - Technical indicators

### Configuration
- **YAML** - Configuration files
- **Pydantic** - Configuration validation
- **joblib** - Model persistence

## API Structure

### Data Loading
```python
from quanttradeai import DataLoader, DataProcessor

# Initialize components
loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")

# Fetch and process data
data_dict = loader.fetch_data()
df_processed = processor.process_data(df)
df_labeled = processor.generate_labels(df_processed)
```

### Model Training
```python
from quanttradeai import MomentumClassifier

# Initialize and train
classifier = MomentumClassifier("config/model_config.yaml")
X, y = classifier.prepare_data(df_labeled)
classifier.train(X, y)
```

### Backtesting
```python
from quanttradeai import simulate_trades, compute_metrics

# Simulate trades
df_trades = simulate_trades(df_labeled)
metrics = compute_metrics(df_trades)
```

## Configuration Files

### Model Configuration (`config/model_config.yaml`)
- Data parameters (symbols, date ranges, caching)
- Model hyperparameters (LR, RF, XGBoost)
- Training settings (test size, CV folds)
- Trading parameters (position sizing, risk)

### Feature Configuration (`config/features_config.yaml`)
- Technical indicator parameters
- Feature preprocessing settings
- Feature selection methods
- Pipeline steps

## Error Handling Patterns

### Data Validation
```python
# Validate data quality
is_valid = loader.validate_data(data_dict)
if not is_valid:
    raise ValueError("Data validation failed")
```

### Model Training
```python
try:
    classifier.train(X, y)
except ValueError as e:
    logger.error(f"Training error: {e}")
    # Check data shapes and class distribution
```

### Configuration Validation
```python
from quanttradeai.utils.config_schemas import ModelConfigSchema
ModelConfigSchema(**config)  # Validates configuration
```

## Performance Considerations

### Memory Management
- Use smaller data types for large datasets
- Process data in batches for memory efficiency
- Cache intermediate results appropriately

### Computational Optimization
- Vectorized operations over loops
- Parallel processing for multiple assets
- GPU acceleration for model training (future)

## Testing Patterns

### Unit Tests
```python
def test_data_loader():
    loader = DataLoader("config/model_config.yaml")
    data = loader.fetch_data()
    assert len(data) > 0
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
```

### Integration Tests
```python
def test_complete_pipeline():
    # Test end-to-end workflow
    loader = DataLoader()
    processor = DataProcessor()
    classifier = MomentumClassifier()
    
    data = loader.fetch_data()
    df = processor.process_data(data['AAPL'])
    df_labeled = processor.generate_labels(df)
    
    X, y = classifier.prepare_data(df_labeled)
    classifier.train(X, y)
    
    predictions = classifier.predict(X)
    assert len(predictions) == len(y)
```

## Documentation Standards

### Code Documentation
- Docstrings for all public functions
- Type hints for all parameters
- Usage examples in docstrings
- Clear parameter descriptions

### API Documentation
- Update `docs/api/` files for new functions
- Include parameter types and return values
- Provide usage examples
- Document error conditions

## Common Patterns

### Feature Engineering
```python
from quanttradeai.features import technical as ta

# Generate technical indicators
df['sma_20'] = ta.sma(df['Close'], 20)
df['rsi'] = ta.rsi(df['Close'], 14)
macd_df = ta.macd(df['Close'])
```

### Risk Management
```python
from quanttradeai import apply_stop_loss_take_profit

# Apply risk rules
df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02)
```

### Performance Metrics
```python
from quanttradeai.utils.metrics import classification_metrics, sharpe_ratio

# Calculate metrics
metrics = classification_metrics(y_true, y_pred)
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
```

## Troubleshooting

### Common Issues
1. **Data Loading Failures**: Check network connectivity, API limits
2. **Memory Issues**: Reduce data size, use batching
3. **Model Training Errors**: Check data quality, class balance
4. **Configuration Errors**: Validate YAML syntax, required fields

### Debugging Steps
1. Check logs for error messages
2. Validate input data quality
3. Test individual components
4. Verify configuration parameters

## LLM Sentiment Analysis

LiteLLM provides a unified interface for scoring text sentiment. Enable it through `config/features_config.yaml`:

```yaml
sentiment:
  enabled: true
  provider: openai
  model: gpt-3.5-turbo
  api_key_env_var: OPENAI_API_KEY
```

Set the corresponding API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Switch providers by editing `provider`, `model`, and `api_key_env_var`. The data pipeline automatically adds a `sentiment_score` column during the `generate_sentiment` step.

For CLI and Python examples see [docs/llm-sentiment.md](docs/llm-sentiment.md).

## Future Development Areas

### High Priority
- Real-time data streaming
- Advanced risk management
- Multi-timeframe support

### Medium Priority
- GPU acceleration
- Microservices architecture
- Advanced NLP features
- Reinforcement learning

### Low Priority
- Quantum computing integration
- Blockchain connectivity
- Multi-modal AI
- Federated learning

## Resources

### Documentation
- [API Reference](docs/api/)
- [Configuration Guide](docs/configuration.md)
- [Quick Reference](docs/quick-reference.md)

### External Libraries
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Optuna](https://optuna.org/)
- [pandas-ta](https://twopirllc.github.io/pandas-ta/)

### Testing & Quality
- [pytest](https://docs.pytest.org/)
- [Black](https://black.readthedocs.io/)
- [flake8](https://flake8.pycqa.org/)

## Commit Guidelines

### Before Committing
1. Run `make format` - Format code with Black
2. Run `make lint` - Check code quality with flake8
3. Run `make test` - Execute all tests
4. Update documentation if needed
5. Add/update tests for new functionality

### Commit Messages
- Use conventional commit format
- Be descriptive and concise
- Reference issues when applicable
- Example: `feat: add new technical indicator for momentum`

### Pull Request Requirements
- All tests must pass
- Code coverage > 80%
- Documentation updated
- No linting errors
- Clear description of changes

## Emergency Procedures

### Breaking Changes
- Maintain backward compatibility when possible
- Use deprecation warnings for removed features
- Update documentation immediately
- Notify team of breaking changes

### Critical Bugs
- Create hotfix branch immediately
- Add regression tests
- Deploy fix as soon as possible
- Document the issue and solution

---

**Remember**: Always test thoroughly, follow coding standards, and maintain documentation. This codebase is used for financial applications - accuracy and reliability are paramount.