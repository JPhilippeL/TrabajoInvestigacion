import traceback

from PySide6.QtCore import QThread, Signal

from GraphDTA.model.generate_graph import generate_all_dta_dta
from GraphDTA.model.predict import predict
from GraphDTA.model.train import train


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
            generate_all_dta_dta(**self.params)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)


class TrainDTAThread(QThread):
    finished_success = Signal()
    finished_error = Signal(str)
    log_message = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)
            train(**self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)


class PredictDTAThread(QThread):
    finished_success = Signal()
    finished_error = Signal(str)
    log_message = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)
            predict(**self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)
