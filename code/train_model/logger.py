import logging
from config import Config

def setup_logger(name=__name__, level=logging.INFO):
    """
    Set up a logger for the given module name with hierarchical logging.

    Args:
        name (str): Name of the logger, typically `__name__`.
        level (int): Logging level for the specific logger (default: DEBUG).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Configure the root logger only once to avoid duplicate handlers
    if not logging.getLogger().hasHandlers():
        # Add handlers to the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Set root logger to DEBUG for capturing all logs

        # Stream handler for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Set stream handler to DEBUG
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        root_logger.addHandler(ch)

        # File handler for logging to a file
        fh = logging.FileHandler(Config.LOG_FILE)
        fh.setLevel(logging.DEBUG)  # Set file handler to DEBUG
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        root_logger.addHandler(fh)

        # Log the log file path for reference
        root_logger.info(f"Logging to file: {Config.LOG_FILE}")

    return logger
