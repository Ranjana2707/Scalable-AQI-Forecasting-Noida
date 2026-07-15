"""
src/utils/logger.py  —  lightweight shim (no loguru dependency in this environment).
Falls back to Python stdlib logging.
"""
import logging, sys
from pathlib import Path

_CONFIGURED = False

def configure_logging(log_dir=None, station_id="default", level="INFO", **kwargs):
    global _CONFIGURED
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True

class _Logger:
    def __init__(self, name):
        self._log = logging.getLogger(name)
    def info(self, msg):    self._log.info(msg)
    def debug(self, msg):   self._log.debug(msg)
    def warning(self, msg): self._log.warning(msg)
    def error(self, msg):   self._log.error(msg)
    def bind(self, **kw):   return self

def get_logger(name=None):
    return _Logger(name or "aqi")
