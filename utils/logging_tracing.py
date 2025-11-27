# utils/logging_tracing.py
import logging
import time
import json

logger = logging.getLogger("agrisense")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def log_event(message: str, **meta):
    data = {"ts": time.time(), "message": message}
    data.update(meta)
    logger.info(json.dumps(data))
