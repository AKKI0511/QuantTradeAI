import sys
import os

# Add src to Python path to allow direct import of DataProcessor
# This assumes the script is run from the root of the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "./src")))

from data.processor import DataProcessor  # noqa: E402
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
