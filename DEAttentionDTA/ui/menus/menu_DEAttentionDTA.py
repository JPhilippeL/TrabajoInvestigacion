"""Menu integration for the DEAttentionDTA module."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from DEAttentionDTA.ui.dialogs.hyperparameter_search_deattentiondta_dialog import HyperparameterSearchDEAttentionDTADialog
from DEAttentionDTA.ui.dialogs.prepare_deattentiondta_dataset_dialog import PrepareDEAttentionDTADatasetDialog
from DEAttentionDTA.ui.dialogs.test_deattentiondta_dialog import TestDEAttentionDTADialog
from DEAttentionDTA.ui.dialogs.train_deattentiondta_dialog import TrainDEAttentionDTADialog
from DEAttentionDTA.workers import (
    EvaluateCheckpointThread,
    HyperparameterSearchDEAttentionDTAThread,
    PrepareURVDatasetThread,
    TrainOfficialSplitsThread,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MenuDEAttentionDTA(QMenu):
    """DEAttentionDTA menu for the main GUI."""

    def __init__(self, parent_window):
        super().__init__("DEAttentionDTA", parent_window)
        self.parent_window = parent_window
        self.worker_thread = None
        self.init_actions()

    def init_actions(self) -> None:
        self.generate_data_action = QAction("Generate Data", self)
        self.generate_data_action.triggered.connect(self.open_prepare_dialog)
        self.addAction(self.generate_data_action)

        self.train_action = QAction("Train", self)
        self.train_action.triggered.connect(self.open_train_dialog)
        self.addAction(self.train_action)

        self.hpo_action = QAction("Search", self)
        self.hpo_action.triggered.connect(self.open_hpo_dialog)
        self.addAction(self.hpo_action)

        self.test_action = QAction("Evaluate", self)
        self.test_action.triggered.connect(self.open_test_dialog)
        self.addAction(self.test_action)

    def _set_actions_enabled(self, enabled: bool) -> None:
        for action in (self.generate_data_action, self.train_action, self.hpo_action, self.test_action):
            action.setEnabled(enabled)

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute(): path = PROJECT_ROOT / path
        return path.resolve()

    def _require_directory(self, value: str, label: str) -> None:
        path = self._resolve(value)
        if not path.is_dir(): raise ValueError(f"{label} directory does not exist:\n{path}")

    def _require_file(self, value: str, label: str) -> None:
        path = self._resolve(value)
        if not path.is_file(): raise ValueError(f"{label} file does not exist:\n{path}")

    @staticmethod
    def _require_text(value: str, label: str) -> None:
        if not str(value).strip(): raise ValueError(f"{label} is required.")

    def _log(self, message: str) -> None:
        if hasattr(self.parent_window, "log"): self.parent_window.log(message)
        else: LOGGER.info(message)

    def _start_worker(self, worker) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self.parent_window, "DEAttentionDTA Busy", "A DEAttentionDTA operation is already running.")
            return
        self.worker_thread = worker
        self._set_actions_enabled(False)
        worker.log_line.connect(self._log)
        worker.finished_success.connect(self.on_success)
        worker.finished_error.connect(self.on_error)
        worker.finished.connect(self.on_thread_finished)
        worker.start()

    def open_prepare_dialog(self) -> None:
        dialog = PrepareDEAttentionDTADatasetDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        try:
            params = dialog.get_inputs()
            self._require_directory(params["raw_root"], "Raw MPro-v2-like dataset")
            self._require_text(params["output_root"], "Prepared output root")
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DEAttentionDTA Generate Data Configuration Error", str(exc)); return
        self._start_worker(PrepareURVDatasetThread(params))

    def open_train_dialog(self) -> None:
        dialog = TrainDEAttentionDTADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        try:
            params = dialog.get_inputs(); self._require_directory(params["prepared_dir"], "Prepared dataset"); self._require_text(params["output_dir"], "Output directory"); self._require_text(params["splits"], "Fold index")
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DEAttentionDTA Training Configuration Error", str(exc)); return
        self._start_worker(TrainOfficialSplitsThread(params))

    def open_hpo_dialog(self) -> None:
        dialog = HyperparameterSearchDEAttentionDTADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        try:
            params = dialog.get_inputs(); self._require_directory(params["prepared_dir"], "Prepared dataset"); self._require_text(params["output_dir"], "Output root"); self._require_text(params["splits"], "Fold index")
            if not params["lr_values"]: raise ValueError("At least one learning-rate value is required.")
            if not params["batch_size_values"]: raise ValueError("At least one batch-size value is required.")
            if not params["weight_decay_values"]: raise ValueError("At least one weight-decay value is required.")
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DEAttentionDTA Search Configuration Error", str(exc)); return
        self._start_worker(HyperparameterSearchDEAttentionDTAThread(params))

    def open_test_dialog(self) -> None:
        dialog = TestDEAttentionDTADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        try:
            params = dialog.get_inputs(); self._require_file(params["checkpoint"], "Checkpoint"); self._require_directory(params["prepared_dir"], "Prepared dataset"); self._require_text(params["results_dir"], "Output directory"); self._require_text(params["splits"], "Fold index")
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DEAttentionDTA Evaluation Configuration Error", str(exc)); return
        self._start_worker(EvaluateCheckpointThread(params))

    def on_success(self, summary: dict) -> None:
        artifacts = summary.get("artifacts", {})
        details = summary.get("summary", {}) if isinstance(summary.get("summary", {}), dict) else {}
        if summary.get("operation") == "generate_deattentiondta_dataset_from_mpro_v2":
            split_sizes = details.get("split_sizes", {})
            split_text = ", ".join(f"{key}={value}" for key, value in list(split_sizes.items())[:6])
            if len(split_sizes) > 6:
                split_text += ", ..."
            message = (
                "Status: success\n\n"
                f"Output path: {artifacts.get('prepared_dataset', '')}\n"
                f"Samples kept: {details.get('samples_kept', '')}\n"
                f"Samples skipped: {details.get('samples_skipped', '')}\n"
                f"Split sizes: {split_text or 'See report'}\n"
                f"Report path: {artifacts.get('generate_data_report_json', '')}\n"
                f"Warnings: {details.get('warnings_count', 0)}"
            )
            QMessageBox.information(self.parent_window, "DEAttentionDTA Generate Data Finished", message)
            return
        artifact_text = "\n".join(f"{key}: {value}" for key, value in artifacts.items())
        QMessageBox.information(self.parent_window, "DEAttentionDTA Operation Finished", "Status: success\n\nGenerated artifacts:\n" + (artifact_text or "See application logs."))

    def on_error(self, error_message: str) -> None:
        self._log(error_message)
        QMessageBox.critical(self.parent_window, "DEAttentionDTA Error", "The DEAttentionDTA operation failed. The complete traceback is available in the application logs.")

    def on_thread_finished(self) -> None:
        self._set_actions_enabled(True)
        self.worker_thread = None
