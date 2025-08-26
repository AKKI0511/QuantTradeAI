import os
from unittest.mock import patch

import pandas as pd
import pytest

from quanttradeai.features.sentiment import SentimentAnalyzer
from quanttradeai.data.processor import DataProcessor


def test_sentiment_score_success(monkeypatch):
    os.environ["TEST_API_KEY"] = "key"

    def mock_completion(**kwargs):
        return {"choices": [{"message": {"content": "0.5"}}]}

    monkeypatch.setattr("quanttradeai.features.sentiment.completion", mock_completion)
    analyzer = SentimentAnalyzer(
        provider="openai", model="gpt-3.5-turbo", api_key_env_var="TEST_API_KEY"
    )
    assert analyzer.score("great day") == 0.5


def test_missing_api_key():
    with pytest.raises(ValueError):
        SentimentAnalyzer("openai", "gpt-3.5-turbo", "MISSING_KEY")


def test_completion_failure(monkeypatch):
    os.environ["TEST_API_KEY"] = "key"

    def mock_completion(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("quanttradeai.features.sentiment.completion", mock_completion)
    analyzer = SentimentAnalyzer(
        provider="openai", model="gpt-3.5-turbo", api_key_env_var="TEST_API_KEY"
    )
    with pytest.raises(RuntimeError):
        analyzer.score("bad")


def test_data_processor_sentiment_integration(tmp_path, monkeypatch):
    os.environ["TEST_API_KEY"] = "key"
    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        """
        pipeline:
          steps: [generate_sentiment]
        sentiment:
          enabled: true
          provider: openai
          model: gpt-3.5-turbo
          api_key_env_var: TEST_API_KEY
        """
    )

    def mock_completion(**kwargs):
        return {"choices": [{"message": {"content": "0.2"}}]}

    monkeypatch.setattr("quanttradeai.features.sentiment.completion", mock_completion)

    data = pd.DataFrame(
        {
            "Open": [1] * 205,
            "High": [1] * 205,
            "Low": [1] * 205,
            "Close": [1] * 205,
            "Volume": [1] * 205,
            "text": ["news"] * 205,
        }
    )

    processor = DataProcessor(str(config_path))
    result = processor.process_data(data)
    assert "sentiment_score" in result.columns
    assert result["sentiment_score"].iloc[0] == 0.2
