"""
@file menu_CAPLA.py
@author Mohamed EL BOUKHIARI
@brief Menu integration for the CAPLA module.
"""


from __future__ import annotations

import logging
import json
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from CAPLA.ui.dialogs.hyperparameter_search_capla_dialog import HyperparameterSearchCAPLADialog
from CAPLA.ui.dialogs.prepare_capla_dataset_dialog import PrepareCAPLADatasetDialog
from CAPLA.ui.dialogs.test_capla_dialog import TestCAPLADialog
from CAPLA.ui.dialogs.train_capla_dialog import TrainCAPLADialog
from CAPLA.workers import (
    GenerateDataThread,
    HyperparameterSearchCAPLAThread,
    PredictPreparedDatasetThread,
    TrainOfficialSplitsThread,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MenuCAPLA(QMenu):
    """CAPLA menu for the main GUI."""

    def __init__(self, parent_window):
        super().__init__("CAPLA", parent_window)
        self.parent_window = parent_window
        self.worker_thread = None
        self.init_actions()

    def init_actions(self) -> None:

        self.generate_data_action = QAction("Generate Data", self)
        self.generate_data_action.triggered.connect(self.open_generate_data_dialog)
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
        for action in (
            self.generate_data_action,
            self.train_action,
            self.hpo_action,
            self.test_action,
        ):
            action.setEnabled(enabled)

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def _require_directory(self, value: str, label: str) -> None:
        path = self._resolve(value)
        if not path.is_dir():
            raise ValueError(f"{label} directory does not exist:\n{path}")

    def _require_file(self, value: str, label: str) -> None:
        path = self._resolve(value)
        if not path.is_file():
            raise ValueError(f"{label} file does not exist:\n{path}")

    @staticmethod
    def _require_text(value: str, label: str) -> None:
        if not str(value).strip():
            raise ValueError(f"{label} is required.")

    def _log(self, message: str) -> None:
        if hasattr(self.parent_window, "log"):
            self.parent_window.log(message)
        else:
            LOGGER.info(message)

    def _start_worker(self, worker) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(
                self.parent_window,
                "CAPLA Busy",
                "A CAPLA operation is already running.",
            )
            return

        self.worker_thread = worker
        self._set_actions_enabled(False)
        worker.log_line.connect(self._log)
        worker.finished_success.connect(self.on_success)
        worker.finished_error.connect(self.on_error)
        worker.finished.connect(self.on_thread_finished)
        worker.start()

    def open_generate_data_dialog(self) -> None:
        dialog = PrepareCAPLADatasetDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            self._require_directory(params["raw_root"], "Raw MPro-URV_Version2 root")
            self._require_text(params["output_root"], "Output prepared dataset root")
            if params.get("feature_mode") != "generate":
                self._require_directory(params["feature_source_root"], "Existing CAPLA feature source root")
        except Exception as exc:
            QMessageBox.critical(
                self.parent_window,
                "CAPLA Generate Data Configuration Error",
                str(exc),
            )
            return
        self._start_worker(GenerateDataThread(params))

    def open_train_dialog(self) -> None:
        dialog = TrainCAPLADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            self._require_directory(params["dataset_dir"], "Prepared dataset")
            self._require_text(params["output_dir"], "Results directory")
            self._require_text(params["models_dir"], "Models directory")
            self._require_text(params["splits"], "Official splits")
        except Exception as exc:
            QMessageBox.critical(
                self.parent_window,
                "CAPLA Training Configuration Error",
                str(exc),
            )
            return
        self._start_worker(TrainOfficialSplitsThread(params))

    def open_hpo_dialog(self) -> None:
        dialog = HyperparameterSearchCAPLADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            self._require_directory(params["dataset_dir"], "Prepared dataset")
            self._require_text(params["models_root"], "HPO models directory")
            self._require_text(params["results_root"], "HPO results directory")
            self._require_text(params["splits"], "Tuning splits")
            if not params["lr_values"]:
                raise ValueError("At least one learning-rate value is required.")
            if not params["batch_size_values"]:
                raise ValueError("At least one batch-size value is required.")
            if not params["weight_decay_values"]:
                raise ValueError("At least one weight-decay value is required.")
        except Exception as exc:
            QMessageBox.critical(
                self.parent_window,
                "CAPLA Search Configuration Error",
                str(exc),
            )
            return
        self._start_worker(HyperparameterSearchCAPLAThread(params))

    def open_test_dialog(self) -> None:
        dialog = TestCAPLADialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            self._require_file(params["model_pt"], "CAPLA checkpoint")
            self._require_directory(params["dataset_dir"], "Prepared dataset")
            self._require_text(params["output_dir"], "Output directory")
        except Exception as exc:
            QMessageBox.critical(
                self.parent_window,
                "CAPLA Evaluation Configuration Error",
                str(exc),
            )
            return
        self._start_worker(PredictPreparedDatasetThread(params))

    def on_success(self, summary: dict) -> None:
        if "samples_kept" in summary:
            split_sizes = json.dumps(summary.get("split_sizes", {}), indent=2)
            QMessageBox.information(
                self.parent_window,
                "CAPLA Generate Data Finished",
                "Status: success\n\n"
                f"Output path: {summary.get('output_root')}\n"
                f"Samples kept: {summary.get('samples_kept')}\n"
                f"Samples skipped: {summary.get('samples_skipped')}\n"
                f"Mode: {summary.get('mode')}\n"
                f"Warnings: {len(summary.get('warnings', []))}\n"
                f"Report: {summary.get('report_json')}\n\n"
                f"Split sizes:\n{split_sizes}",
            )
            return
        artifacts = summary.get("artifacts", {})
        artifact_text = "\n".join(
            f"{key}: {value}" for key, value in artifacts.items()
        )
        QMessageBox.information(
            self.parent_window,
            "CAPLA Operation Finished",
            "Status: success\n\nGenerated artifacts:\n"
            + (artifact_text or "See application logs."),
        )

    def on_error(self, error_message: str) -> None:
        self._log(error_message)
        lines = [line.strip() for line in str(error_message).splitlines() if line.strip()]
        detail = lines[-1] if lines else str(error_message)
        QMessageBox.critical(
            self.parent_window,
            "CAPLA Error",
            "The CAPLA operation failed.\n\n"
            f"{detail}\n\n"
            "The complete traceback is available in the application logs.",
        )

    def on_thread_finished(self) -> None:
        self._set_actions_enabled(True)
        self.worker_thread = None
