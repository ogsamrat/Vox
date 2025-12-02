import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, format=log_format, level=log_level, rotation=rotation, retention=retention, compression="zip")
