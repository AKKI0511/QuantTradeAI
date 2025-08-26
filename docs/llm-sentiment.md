# LLM-Powered Sentiment Analysis

QuantTradeAI can attach numeric sentiment scores to any dataset that includes a text column. The feature uses [LiteLLM](https://github.com/BerriAI/litellm) so you can swap providers and models without changing code.

## Setup

1. **Install QuantTradeAI (if not already installed)**
   ```bash
   poetry install
   ```
2. **Configure `features_config.yaml`**
   ```yaml
   sentiment:
     enabled: true
     provider: openai        # e.g. openai, anthropic, huggingface, ollama
     model: gpt-3.5-turbo    # model name for the chosen provider
     api_key_env_var: OPENAI_API_KEY
     extra: {}               # optional LiteLLM params
   ```
3. **Set the API key**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

### Switching Providers
Change `provider`, `model`, and `api_key_env_var` in the YAML config. Example for Anthropic:
```yaml
sentiment:
  enabled: true
  provider: anthropic
  model: claude-instant-1
  api_key_env_var: ANTHROPIC_API_KEY
```
No code changes are required—update the YAML and corresponding environment variable.

## Usage

### Command Line
```bash
# with sentiment enabled in features_config.yaml
export OPENAI_API_KEY="sk-..."
poetry run quanttradeai train
```
The training pipeline loads sentiment settings automatically and adds a `sentiment_score` column.

### Python
```python
import pandas as pd
from quanttradeai.data import DataProcessor

# DataFrame must contain a 'text' column
raw = pd.DataFrame({"text": ["Stock surges on earnings", "Lawsuit worries investors"]})
processor = DataProcessor("config/features_config.yaml")
features = processor.process_data(raw)
print(features[["text", "sentiment_score"]])
```

## Input & Output
- **Input**: DataFrame with a `text` column
- **Output**: New `sentiment_score` column with values in `[-1, 1]`
- Added via the `generate_sentiment` step in the feature pipeline

## Troubleshooting
| Problem | Fix |
|--------|-----|
| `API key environment variable ... is not set` | Ensure the variable from `api_key_env_var` is exported |
| Unsupported provider/model | Verify the provider/model pair is supported by LiteLLM |
| `sentiment enabled but 'text' column not found` | Provide a `text` column in your input DataFrame |
| Network/API errors | Check connectivity, credentials, or provider status |
