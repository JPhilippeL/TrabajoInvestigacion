#logger.py
import logging
from PySide6.QtCore import QObject, Signal

class QtHandler(logging.Handler):
    def __init__(self, signal_callback):
        super().__init__()
        self.signal_callback = signal_callback

    def emit(self, record):
        msg = self.format(record)
        self.signal_callback(msg)

