import json
import sys
import traceback

from PySide6.QtCore import QThread, Signal

from Planet.data_pipeline.build_planet_data import build_data


class _ThreadLoggerProxy:
    def __init__(self, emit_func):
        self._emit = emit_func

    def info(self, *args):
        if not args:
            return
        message = " ".join(str(arg) for arg in args)
        self._emit(message)

    def error(self, *args):
        if not args:
            return
        message = " ".join(str(arg) for arg in args)
        self._emit(f"ERROR: {message}")

    def debug(self, *args):
        if not args:
            return
        message = " ".join(str(arg) for arg in args)
        self._emit(f"DEBUG: {message}")


class DBGenerationThread(QThread):
    finished_success = Signal()
    finished_error = Signal(str)
    log_message = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)
            build_data(**self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)
