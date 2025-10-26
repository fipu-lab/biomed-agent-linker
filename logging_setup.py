import logging
import sys
from typing import Optional


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, use_colors: bool = True
) -> logging.Logger:
    """
    Set up logging with colored output.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to also log to file
        use_colors: Whether to use colored output (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors:

        class SimpleColoredFormatter(logging.Formatter):
            def format(self, record):
                if record.levelno == logging.INFO:
                    record.msg = f"{Colors.GREEN}{record.msg}{Colors.RESET}"
                elif record.levelno == logging.WARNING:
                    record.msg = f"{Colors.YELLOW}{record.msg}{Colors.RESET}"
                elif record.levelno >= logging.ERROR:
                    record.msg = f"{Colors.RED}{record.msg}{Colors.RESET}"
                return super().format(record)

        formatter = SimpleColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    logger = get_logger("example")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
