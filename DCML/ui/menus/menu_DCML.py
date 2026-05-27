"""
@file menu_DCML.py
@author Mohamed EL BOUKHIARI
@brief Menu integration for the DCML module.
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from DCML.ui.dialogs.hyperparameter_search_dcml_dialog import HyperparameterSearchDCMLDialog
from DCML.ui.dialogs.test_dcml_dialog import TestDCMLDialog
from DCML.ui.dialogs.train_dcml_dialog import TrainDCMLDialog
from DCML.workers import HyperparameterSearchThread, TestThread, TrainThread


class MenuDCML(QMenu):
    """DCML menu for the main GUI."""

    def __init__(self, parent_window):
        super().__init__("DCML", parent_window)
        self.parent_window = parent_window
        self.train_thread = None
        self.test_thread = None
        self.hyperparameter_search_thread = None
        self.init_actions()

    def init_actions(self):
        self.train_action = QAction("Train Model", self)
        self.train_action.triggered.connect(self.open_train_dialog)
        self.addAction(self.train_action)

        self.hyperparameter_search_action = QAction("Hyperparameter Search", self)
        self.hyperparameter_search_action.triggered.connect(self.open_hyperparameter_search_dialog)
        self.addAction(self.hyperparameter_search_action)

        self.test_action = QAction("Evaluate Model", self)
        self.test_action.triggered.connect(self.open_test_dialog)
        self.addAction(self.test_action)

    def _set_actions_enabled(self, enabled: bool):
        self.train_action.setEnabled(enabled)
        self.test_action.setEnabled(enabled)
        self.hyperparameter_search_action.setEnabled(enabled)
        if hasattr(self.parent_window, "setEnabled"):
            self.parent_window.setEnabled(enabled)

    @staticmethod
    def _require_paths(params: dict, keys: list[str]) -> tuple[bool, str]:
        missing = [key for key in keys if not str(params.get(key, "")).strip()]
        if missing:
            return False, "Missing required fields: " + ", ".join(missing)
        return True, ""

    def open_train_dialog(self):
        dialog = TrainDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(params, ["train_feature_zip", "train_label_npy", "output_model", "output_dir"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Training Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self.train_thread = TrainThread(params)
        self.train_thread.finished_success.connect(self.on_train_success)
        self.train_thread.finished_error.connect(self.on_error)
        self.train_thread.finished.connect(self.on_thread_finished)
        self.train_thread.start()

    def open_test_dialog(self):
        dialog = TestDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(params, ["model_pt", "feature_zip", "label_npy", "output_dir"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Evaluation Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self.test_thread = TestThread(params)
        self.test_thread.finished_success.connect(self.on_test_success)
        self.test_thread.finished_error.connect(self.on_error)
        self.test_thread.finished.connect(self.on_thread_finished)
        self.test_thread.start()

    def open_hyperparameter_search_dialog(self):
        dialog = HyperparameterSearchDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(
                params,
                [
                    "train_feature_zip",
                    "train_label_npy",
                    "validation_feature_zip",
                    "validation_label_npy",
                    "models_root",
                    "results_root",
                ],
            )
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Hyperparameter Search Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self.hyperparameter_search_thread = HyperparameterSearchThread(params)
        self.hyperparameter_search_thread.finished_success.connect(self.on_hyperparameter_search_success)
        self.hyperparameter_search_thread.finished_error.connect(self.on_error)
        self.hyperparameter_search_thread.finished.connect(self.on_thread_finished)
        self.hyperparameter_search_thread.start()

    def on_train_success(self, summary: dict):
        artifact = summary.get("artifact", {})
        training = summary.get("training", {})
        train_metrics = training.get("train_metrics", {})
        outputs = summary.get("outputs", {})
        text = (
            "Status: success\n\n"
            f"Model bundle:\n{artifact.get('output_model', '')}\n\n"
            f"Training summary:\n{outputs.get('training_summary_json', '')}\n\n"
            f"Train RMSE: {train_metrics.get('RMSE', 0.0):.6f}\n"
            f"Train Pearson: {train_metrics.get('Pearson', 0.0):.6f}\n"
            f"Train MAE: {train_metrics.get('MAE', 0.0):.6f}"
        )
        QMessageBox.information(self.parent_window, "DCML Training Finished", text)

    def on_test_success(self, summary: dict):
        metrics = summary.get("metrics", {})
        outputs = summary.get("outputs", {})
        text = (
            "Status: success\n\n"
            f"RMSE: {metrics.get('RMSE', 0.0):.6f}\n"
            f"Pearson: {metrics.get('Pearson', 0.0):.6f}\n"
            f"MAE: {metrics.get('MAE', 0.0):.6f}\n\n"
            f"Predictions CSV:\n{outputs.get('predictions_csv', '')}\n\n"
            f"Metrics CSV:\n{outputs.get('metrics_csv', '')}\n\n"
            f"Scatter PNG:\n{outputs.get('scatter_png', '')}"
        )
        QMessageBox.information(self.parent_window, "DCML Evaluation Finished", text)

    def on_hyperparameter_search_success(self, results: dict):
        metrics = results.get("best_metrics", {})
        text = (
            f"Status: {results.get('status', 'success')}\n\n"
            f"{results.get('message', '')}\n\n"
            f"Best trial: {results.get('best_trial', '')}\n"
            f"Best RMSE: {metrics.get('RMSE', 0.0):.6f}\n"
            f"Best Pearson: {metrics.get('Pearson', 0.0):.6f}\n"
            f"Best MAE: {metrics.get('MAE', 0.0):.6f}\n"
            f"Elapsed time: {results.get('elapsed_time', '')}\n\n"
            f"Run directory:\n{results.get('run_dir', '')}\n\n"
            f"Trials CSV:\n{results.get('trials_csv', '')}\n\n"
            f"Best config:\n{results.get('best_config_yaml', '')}\n\n"
            f"Best model:\n{results.get('best_model_path', '')}"
        )
        QMessageBox.information(self.parent_window, "DCML Hyperparameter Search Finished", text)

    def on_error(self, error_message: str):
        QMessageBox.critical(self.parent_window, "DCML Error", error_message)

    def on_thread_finished(self):
        self._set_actions_enabled(True)
        self.train_thread = None
        self.test_thread = None
        self.hyperparameter_search_thread = None
