import logging
import os

import pandas as pd

from quanttradeai.data.datasource import DataSource, NewsDataSource
from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor


class DummyPriceSource(DataSource):
    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=205, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0] * len(index),
                "High": [1.0] * len(index),
                "Low": [1.0] * len(index),
                "Close": [1.0] * len(index),
                "Volume": [1.0] * len(index),
            },
            index=index,
        )


class DummyNewsSource(NewsDataSource):
    def __init__(self):
        super().__init__(provider="dummy")

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"Datetime": [pd.Timestamp("2024-07-20 12:00")], "text": ["optimistic"]}
        )


def test_sentiment_pipeline_consumes_news(tmp_path, monkeypatch, caplog):
    os.environ["TEST_API_KEY"] = "key"

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        """
        data:
          symbols: ['TEST']
          start_date: '2024-01-01'
          end_date: '2024-07-24'
          timeframe: '1d'
          cache_path: '{cache}'
          use_cache: false
          refresh: true
        news:
          enabled: true
          provider: dummy
        """.format(
            cache=tmp_path / "cache"
        )
    )

    features_config = tmp_path / "features.yaml"
    features_config.write_text(
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
        return {"choices": [{"message": {"content": "0.8"}}]}

    monkeypatch.setattr("quanttradeai.features.sentiment.completion", mock_completion)

    loader = DataLoader(
        str(model_config),
        data_source=DummyPriceSource(),
        news_data_source=DummyNewsSource(),
    )
    caplog.set_level(logging.WARNING, logger="quanttradeai.data.processor")

    data = loader.fetch_data()["TEST"]
    processor = DataProcessor(str(features_config))
    result = processor.process_data(data)

    assert "sentiment_score" in result.columns
    assert result["sentiment_score"].iloc[0] == 0.8
    assert "text column not found" not in caplog.text
