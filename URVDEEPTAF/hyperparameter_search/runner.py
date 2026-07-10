import json
import sys

from PySide6.QtCore import QObject, QProcess, Signal


class HyperparameterTuningProcess(QObject):
    log_message = Signal(str)
    finished_success = Signal()
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.script_path = "URVDEEPTAF/hyperparameter_search/adapter.py"
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
                f"Hyperparameter tuning failed with exit code {exit_code} and code status {exit_status}."
            )

    def _on_error(self, process_error):
        self.finished_error.emit(f"QProcess error: {process_error}")
