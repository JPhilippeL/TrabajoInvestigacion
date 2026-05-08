import traceback

from PySide6.QtCore import QThread, Signal

from GIGN_GUI.controller.hyperparameter_controller import launch_ray_tune_hyperparameter_search
from GIGN_GUI.model.graph_generation import generate_all_graphs
from GIGN_GUI.model.predict_gign import predict
from GIGN_GUI.model.train_gign import train_gign


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
            generate_all_graphs(**self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)


class TrainGIGNThread(QThread):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)

            train_gign(**self.params, log_callback=thread_logger)

            self.finished_success.emit()

        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)


class PredictThread(QThread):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

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


class HPGIGNThread(QThread):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)
            launch_ray_tune_hyperparameter_search(self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as _:
            traceback_error = str(traceback.format_exc())
            self.finished_error.emit(traceback_error)
