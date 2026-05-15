import json
import logging
import sys
import traceback

from PySide6.QtCore import QThread, Signal, QObject, QProcess

from CheapNet_GUI.model.graph_generation import generate_all_graphs
from CheapNet_GUI.model.predict import predict
from CheapNet_GUI.model.train import train_cheapnet

logger = logging.getLogger(__name__)


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
            error_detailed = str(traceback.format_exc())
            self.finished_error.emit(error_detailed)


class TrainCheapNetThread(QThread):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)

            train_cheapnet(**self.params, log_callback=thread_logger)

            self.finished_success.emit()

        except Exception as _:
            error_detailed = str(traceback.format_exc())
            self.finished_error.emit(error_detailed)


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
            error_detailed = str(traceback.format_exc())
            self.finished_error.emit(error_detailed)


class HyperparameterTuningProcess(QObject):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.process = QProcess()

        self.process.readyReadStandardOutput.connect(self.__read_stdout)
        self.process.readyReadStandardError.connect(self.__read_stderr)

        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.process.stateChanged.connect(self._on_state_changed)

    def start(self, payload):
        params = json.dumps(payload)
        self.process.start(sys.executable, ["-u", "CheapNet_GUI/adapter/hyperparameter_adapter.py", params])

    def __read_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf-8")
        self.log_message.emit(stdout)

    def __read_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf-8")
        self.log_message.emit(stderr)

    def _on_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.finished_success.emit()
        else:
            self.finished_error.emit(
                f"Hyperparameter tuning failed with exit code {exit_code} and code status {exit_status}.")

    def _on_state_changed(self, state):
        self.log_message.emit(f"Hyperparameter tuning state changed to {state}")

    def _on_error(self, error):
        error_message = f"Hyperparameter tuning failed with exit code {error}"
        self.finished_error.emit(f"[HPTUNING]{error_message}")
