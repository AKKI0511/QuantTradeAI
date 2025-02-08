import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class for fetching and validating stock data."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.symbols = self.config['data']['symbols']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        
    def fetch_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for specified symbols.
        
        Args:
            symbols: List of stock symbols. If None, uses symbols from config.
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol.
        """
        symbols = symbols or self.symbols
        data_dict = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if df.empty:
                    logger.error(f"No data found for {symbol}")
                    continue
                    
                # Validate data completeness
                missing_dates = self._check_missing_dates(df)
                if missing_dates:
                    logger.warning(f"Missing dates for {symbol}: {len(missing_dates)} days")
                
                data_dict[symbol] = df
                logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return data_dict
    
    def _check_missing_dates(self, df: pd.DataFrame) -> List[datetime]:
        """Check for missing trading days in the data."""
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        missing_dates = all_dates.difference(df.index)
        return list(missing_dates)
    
    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate the fetched data meets requirements.
        
        Args:
            data_dict: Dictionary of DataFrames with OHLCV data.
            
        Returns:
            bool: True if data is valid, False otherwise.
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for symbol, df in data_dict.items():
            # Check required columns
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return False
            
            # Check data range
            date_range = (df.index.max() - df.index.min()).days
            if date_range < 365:  # At least one year of data
                logger.error(f"Insufficient data range for {symbol}")
                return False
            
            # Check for excessive missing values
            if df.isnull().sum().max() > len(df) * 0.01:  # Max 1% missing values
                logger.error(f"Too many missing values for {symbol}")
                return False
        
        return True
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "data/raw") -> None:
        """Save the fetched data to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        
        for symbol, df in data_dict.items():
            file_path = f"{path}/{symbol}_data.parquet"
            df.to_parquet(file_path)
            logger.info(f"Saved data for {symbol} to {file_path}") 