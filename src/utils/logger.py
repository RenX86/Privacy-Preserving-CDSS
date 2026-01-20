"""
Logger Module - Centralized Logging

PURPOSE:
Instead of using print() everywhere, we use a logger that:
  - Shows timestamps so you know when things happened
  - Shows log levels (INFO, WARNING, ERROR) so you can filter
  - Can write to both console AND a file
  - Can be turned on/off or filtered without changing code

USAGE:
    from src.utils.logger import logger
    
    logger.info("Starting document processing...")
    logger.warning("File might be too large")
    logger.error("Failed to connect to database!")
    logger.debug("Detailed info for debugging")  # Only shows if DEBUG level set
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file with today's date
LOG_FILE = LOG_DIR / f"cdss_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logger(name: str = "cdss") -> logging.Logger:
    """
    Create and configure a logger instance.
    
    The logger outputs to:
    1. Console (stdout) - for immediate feedback
    2. File (logs/cdss_YYYYMMDD.log) - for keeping records
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # === Console Handler ===
    # Shows INFO and above in the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # === File Handler ===
    # Saves DEBUG and above to file (more detailed)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Create the main logger instance for the app
logger = setup_logger("cdss")


# Quick test when running this file directly
if __name__ == "__main__":
    print(f"Log file location: {LOG_FILE}\n")
    
    logger.debug("This is a DEBUG message (only in file)")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print(f"\nCheck the log file at: {LOG_FILE}")
