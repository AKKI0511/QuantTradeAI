import sys
import os

# Add project root to import path for direct execution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from quanttradeai.data.processor import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)  # Ensure logging is configured to see output
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Attempting to initialize DataProcessor...")
    sys.stdout.flush()  # Try to flush output
    try:
        processor = DataProcessor()
        logger.info("DataProcessor initialized successfully.")
        sys.stdout.flush()  # Try to flush output
    except Exception as e:
        logger.error(f"Error initializing DataProcessor: {e}", exc_info=True)
        sys.stderr.flush()  # Try to flush error
    finally:
        sys.stdout.flush()  # Final flush
        sys.stderr.flush()
