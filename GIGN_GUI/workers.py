from PySide6.QtCore import QThread, Signal, QObject, QProcess
import json
import sys

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


class DBGenerationThread(QThread):
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            generate_all_graphs(**self.params)
            self.finished_success.emit()
        except Exception as e:
            self.finished_error.emit(str(e))


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

        except Exception as e:
            self.finished_error.emit(str(e))


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
        except Exception as e:
            self.finished_error.emit(str(e))


"""
Thread for hyperparameter tuning process, We use Ray Tune who is a framework who operate on top Ray.
This worker will execute the script to run Ray properly. Ray Tune uses his own process management.
"""


class HyperparameterTuningProcess(QObject):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, script_path, parent=None):
        super().__init__(parent)
        self.params = params
        self.script_path = script_path
        self.process = QProcess(self)

        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

    def start(self):
        payload = json.dumps(self.params)
        self.process.start(sys.executable, [self.script_path, payload])

    def _read_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.log_message.emit(line)

    def _read_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.log_message.emit(line)

    def _on_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.finished_success.emit()
        else:
            self.finished_error.emit(
                f"Hyperparameter tuning failed with exit code {exit_code} and code status {exit_status}.")

    def _on_error(self, process_error):
        self.finished_error.emit(f"QProcess error: {process_error}")
