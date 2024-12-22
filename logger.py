import logging
from config import Config

def setup_logger(name=__name__, level=logging.INFO):
    """
    Set up a logger for the given module name with hierarchical logging.

    Args:
        name (str): Name of the logger, typically `__name__`.
        level (int): Logging level for the specific logger (default: INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)  # Use hierarchical logger names
    logger.setLevel(level)

    # Configure the root logger once to avoid redundant handlers
    if not logging.getLogger().hasHandlers():
        # Add handlers to the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Default to INFO for all loggers

        # Stream handler for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        root_logger.addHandler(ch)

        # File handler for logging to a file
        fh = logging.FileHandler(Config.LOG_FILE)
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        root_logger.addHandler(fh)

        # Log the log file path for reference
        root_logger.info(f"Logging to file: {Config.LOG_FILE}")

    return logger
