"""Background QThreads for DEAttentionDTA GUI operations."""

from __future__ import annotations

import contextlib
import traceback
from typing import Any, Callable, Mapping

from PySide6.QtCore import QThread, Signal


class _SignalWriter:
    """Minimal text stream forwarding complete lines to a Qt signal."""

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._emit(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emit(self._buffer.rstrip())
        self._buffer = ""


class _BaseDEAttentionDTAThread(QThread):
    log_line = Signal(str)
    finished_success = Signal(dict)
    finished_error = Signal(str)

    workflow_name = "DEAttentionDTA operation"

    def __init__(self, params: Mapping[str, Any], parent=None):
        super().__init__(parent)
        self.params = dict(params)

    def execute(self) -> dict[str, Any]:
        raise NotImplementedError

    def run(self) -> None:
        writer = _SignalWriter(self.log_line.emit)
        try:
            self.log_line.emit(f"Starting {self.workflow_name}...")
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                summary = self.execute()
            writer.flush()
            self.finished_success.emit(summary)
        except Exception:
            writer.flush()
            error = traceback.format_exc()
            self.log_line.emit(error)
            self.finished_error.emit(error)


class PrepareURVDatasetThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA URV dataset preparation"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import prepare_urv_dataset

        return prepare_urv_dataset(self.params)


class DebugOfficialDatasetThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA prepared-dataset validation"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import debug_prepared_dataset

        return debug_prepared_dataset(self.params)


class DebugPretrainedCheckpointThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA pretrained-checkpoint validation"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import debug_pretrained_checkpoint

        return debug_pretrained_checkpoint(self.params)


class TrainOfficialSplitsThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA official-split training"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import train_official_splits

        return train_official_splits(self.params)


class EvaluateCheckpointThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA checkpoint evaluation"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import evaluate_checkpoint

        return evaluate_checkpoint(self.params)


class PretrainedVsFinetunedThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA pretrained fine-tuning"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import finetune_pretrained_checkpoint

        return finetune_pretrained_checkpoint(self.params)


class HyperparameterSearchDEAttentionDTAThread(_BaseDEAttentionDTAThread):
    workflow_name = "DEAttentionDTA hyperparameter search"

    def execute(self) -> dict[str, Any]:
        from DEAttentionDTA.core import run_hyperparameter_search

        return run_hyperparameter_search(self.params)
