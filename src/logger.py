# src/logger.py
import logging
from logging.handlers import RotatingFileHandler
from src.config import load_config

cfg = load_config()  # default
LOG_PATH = f"{cfg.log_dir}/app.log"

logger = logging.getLogger("onion_classifier")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=3)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
