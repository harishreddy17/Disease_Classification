import logging
from logging.handlers import RotatingFileHandler
from config import LOG_DIR

logger = logging.getLogger("onion_classifier")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(f"{LOG_DIR}/app.log", maxBytes=5*1024*1024, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
