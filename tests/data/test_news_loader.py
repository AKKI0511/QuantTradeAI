import pandas as pd

from quanttradeai.data.datasource import DataSource, NewsDataSource
from quanttradeai.data.loader import DataLoader


class DummyPriceSource(DataSource):
    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.2, 2.2, 3.2],
                "Volume": [100, 200, 300],
            },
            index=index,
        )


class DummyNewsSource(NewsDataSource):
    def __init__(self, records: list[dict]):
        super().__init__(provider="dummy")
        self.records = records

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def test_loader_merges_news_into_price_data(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
        data:
          symbols: ['TEST']
          start_date: '2024-01-01'
          end_date: '2024-01-03'
          timeframe: '1d'
          cache_path: '{cache}'
          use_cache: false
          refresh: true
        news:
          enabled: true
          provider: dummy
          lookback_days: 2
        """.format(
            cache=tmp_path / "cache"
        )
    )

    headlines = [
        {"Datetime": pd.Timestamp("2023-12-31 20:00"), "text": "pre-start news"},
        {"Datetime": pd.Timestamp("2024-01-02 12:00"), "text": "headline"},
    ]
    loader = DataLoader(
        str(config_path),
        data_source=DummyPriceSource(),
        news_data_source=DummyNewsSource(headlines),
    )

    data = loader.fetch_data()
    df = data["TEST"]

    assert "text" in df.columns
    assert df.loc[pd.Timestamp("2024-01-02"), "text"] == "headline"
