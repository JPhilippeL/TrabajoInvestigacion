import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal


class HyperparameterTuningProcess(QObject):
    log_message = Signal(str)
    ready = Signal()
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, script_path, parent=None):
        super().__init__(parent)

        self.script_path = str(script_path)
        self.process = QProcess(self)

        self.process.readyReadStandardOutput.connect(self.__read_stdout)
        self.process.readyReadStandardError.connect(self.__read_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.process.stateChanged.connect(self._on_state_changed)

    def start(self, payload):
        if not Path(self.script_path).exists():
            self.error.emit(f"Script not found: {self.script_path}")
            return

        payload = self._make_json_serializable(payload)
        params = json.dumps(payload)

        self.process.start(
            sys.executable,
            ["-u", self.script_path, params],
        )

    def _make_json_serializable(self, obj):
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}

        if isinstance(obj, list):
            return [self._make_json_serializable(value) for value in obj]

        if isinstance(obj, tuple):
            return [self._make_json_serializable(value) for value in obj]

        if isinstance(obj, Path):
            return str(obj)

        return obj

    def __read_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf-8", errors="replace")

        if stdout.strip():
            self.log_message.emit(stdout)

    def __read_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf-8", errors="replace")

        if stderr.strip():
            self.log_message.emit(stderr)

    def _on_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.ready.emit()
        else:
            self.error.emit(
                f"Hyperparameter tuning failed with exit code {exit_code} "
                f"and exit status {exit_status}."
            )

    def _on_state_changed(self, state):
        self.log_message.emit(f"Hyperparameter tuning state changed to {state}")

    def _on_error(self, error):
        self.error.emit(f"[HPTUNING] QProcess error: {error}")
