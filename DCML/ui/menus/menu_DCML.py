"""
@file menu_DCML.py
@brief Menu integration for the DCML module.
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox, QPlainTextEdit, QVBoxLayout

from DCML.ui.dialogs.generate_data_dcml_dialog import GenerateDataDCMLDialog
from DCML.ui.dialogs.hyperparameter_search_dcml_dialog import HyperparameterSearchDCMLDialog
from DCML.ui.dialogs.test_dcml_dialog import TestDCMLDialog
from DCML.ui.dialogs.train_dcml_dialog import TrainDCMLDialog
from DCML.workers import GenerateDataThread, HyperparameterSearchThread, TestThread, TrainThread


class MenuDCML(QMenu):
    """DCML menu for the main GUI."""

    def __init__(self, parent_window):
        super().__init__("DCML", parent_window)
        self.parent_window = parent_window
        self.generate_data_thread = None
        self.train_thread = None
        self.test_thread = None
        self.hyperparameter_search_thread = None
        self.progress_dialog = None
        self.progress_log = None
        self.init_actions()

    def init_actions(self):
        self.generate_data_action = QAction("Generate Data", self)
        self.generate_data_action.triggered.connect(self.open_generate_data_dialog)
        self.addAction(self.generate_data_action)

        self.train_action = QAction("Train", self)
        self.train_action.triggered.connect(self.open_train_dialog)
        self.addAction(self.train_action)

        self.search_action = QAction("Search", self)
        self.search_action.triggered.connect(self.open_search_dialog)
        self.addAction(self.search_action)

        self.evaluate_action = QAction("Evaluate", self)
        self.evaluate_action.triggered.connect(self.open_test_dialog)
        self.addAction(self.evaluate_action)

    def _set_actions_enabled(self, enabled: bool):
        for action in (self.generate_data_action, self.train_action, self.search_action, self.evaluate_action):
            action.setEnabled(enabled)

    def _open_progress_log(self, title: str):
        self.progress_dialog = QDialog(self.parent_window)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.resize(900, 520)
        layout = QVBoxLayout(self.progress_dialog)
        self.progress_log = QPlainTextEdit(self.progress_dialog)
        self.progress_log.setReadOnly(True)
        layout.addWidget(self.progress_log)
        self.progress_dialog.show()
        self._append_progress(f"{title} started.")

    def _append_progress(self, message: str):
        if self.progress_log is not None:
            self.progress_log.appendPlainText(str(message))
            self.progress_log.verticalScrollBar().setValue(self.progress_log.verticalScrollBar().maximum())

    @staticmethod
    def _require_paths(params: dict, keys: list[str]) -> tuple[bool, str]:
        missing = [key for key in keys if not str(params.get(key, "")).strip()]
        if missing:
            return False, "Missing required fields: " + ", ".join(missing)
        return True, ""

    def open_generate_data_dialog(self):
        dialog = GenerateDataDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(params, ["raw_root", "output_root", "variant"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Generate Data Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self._open_progress_log("DCML Generate Data Log")
        self.generate_data_thread = GenerateDataThread(params)
        self.generate_data_thread.progress.connect(self._append_progress)
        self.generate_data_thread.finished_success.connect(self.on_generate_data_success)
        self.generate_data_thread.finished_error.connect(self.on_error)
        self.generate_data_thread.finished.connect(self.on_thread_finished)
        self.generate_data_thread.start()

    def open_train_dialog(self):
        dialog = TrainDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(params, ["prepared_feature_root", "output_dir", "variant"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Train Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self._open_progress_log("DCML Train Log")
        self.train_thread = TrainThread(params)
        self.train_thread.progress.connect(self._append_progress)
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
            ok, error = self._require_paths(params, ["prepared_feature_root", "model_pt", "output_dir", "variant", "split"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Evaluate Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self._open_progress_log("DCML Evaluate Log")
        self.test_thread = TestThread(params)
        self.test_thread.progress.connect(self._append_progress)
        self.test_thread.finished_success.connect(self.on_test_success)
        self.test_thread.finished_error.connect(self.on_error)
        self.test_thread.finished.connect(self.on_thread_finished)
        self.test_thread.start()

    def open_search_dialog(self):
        dialog = HyperparameterSearchDCMLDialog(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            params = dialog.get_inputs()
            ok, error = self._require_paths(params, ["prepared_feature_root", "output_root", "variant"])
            if not ok:
                raise ValueError(error)
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "DCML Search Configuration Error", str(exc))
            return

        self._set_actions_enabled(False)
        self._open_progress_log("DCML Search Log")
        self.hyperparameter_search_thread = HyperparameterSearchThread(params)
        self.hyperparameter_search_thread.progress.connect(self._append_progress)
        self.hyperparameter_search_thread.finished_success.connect(self.on_search_success)
        self.hyperparameter_search_thread.finished_error.connect(self.on_error)
        self.hyperparameter_search_thread.finished.connect(self.on_thread_finished)
        self.hyperparameter_search_thread.start()

    def on_generate_data_success(self, summary: dict):
        self._append_progress("DCML Generate Data finished successfully.")
        outputs = summary.get("outputs", {})
        shape = summary.get("matrix_shape_summary", {})
        split_sizes = summary.get("split_sizes", {})
        split_text = ", ".join(f"{key}: {value}" for key, value in split_sizes.items())
        text = (
            "Status: success\n\n"
            f"Variant: {summary.get('variant', '')}\n"
            f"Samples kept: {summary.get('samples_kept', 0)}\n"
            f"Samples skipped: {summary.get('samples_skipped', 0)}\n"
            f"Split sizes: {split_text}\n"
            f"Feature shape: {shape.get('shape', '')}\n"
            f"Warnings: {summary.get('warnings_count', 0)}\n\n"
            f"Output path:\n{outputs.get('output_feature_dir', summary.get('output_root', ''))}\n\n"
            f"Report:\n{outputs.get('report_json', summary.get('reports', {}).get('generate_data_report_json', ''))}"
        )
        QMessageBox.information(self.parent_window, "DCML Generate Data Finished", text)

    def on_train_success(self, summary: dict):
        self._append_progress("DCML Train finished successfully.")
        artifact = summary.get("artifact", {})
        training = summary.get("training", {})
        train_metrics = training.get("train_metrics", {})
        outputs = summary.get("outputs", {})
        gui = summary.get("gui", {})
        text = (
            "Status: success\n\n"
            f"Variant: {gui.get('variant', '')}\n"
            f"Model bundle:\n{artifact.get('output_model', '')}\n\n"
            f"Training summary:\n{outputs.get('training_summary_json', '')}\n\n"
            f"Train RMSE: {train_metrics.get('RMSE', 0.0):.6f}\n"
            f"Train Pearson: {train_metrics.get('Pearson', 0.0):.6f}\n"
            f"Train MAE: {train_metrics.get('MAE', 0.0):.6f}"
        )
        QMessageBox.information(self.parent_window, "DCML Train Finished", text)

    def on_test_success(self, summary: dict):
        self._append_progress("DCML Evaluate finished successfully.")
        metrics = summary.get("metrics", {})
        outputs = summary.get("outputs", {})
        gui = summary.get("gui", {})
        text = (
            "Status: success\n\n"
            f"Variant: {gui.get('variant', '')}\n"
            f"Split: {gui.get('resolved_split', gui.get('split', ''))}\n"
            f"Samples: {summary.get('dataset', {}).get('n_samples', 0)}\n\n"
            f"RMSE: {metrics.get('RMSE', 0.0):.6f}\n"
            f"MSE: {metrics.get('MSE', 0.0):.6f}\n"
            f"Pearson: {metrics.get('Pearson', 0.0):.6f}\n"
            f"MAE: {metrics.get('MAE', 0.0):.6f}\n\n"
            f"Predictions CSV:\n{outputs.get('predictions_csv', '')}\n\n"
            f"Metrics CSV:\n{outputs.get('metrics_csv', '')}\n\n"
            f"Scatter PNG:\n{outputs.get('scatter_png', '')}"
        )
        QMessageBox.information(self.parent_window, "DCML Evaluate Finished", text)

    def on_search_success(self, results: dict):
        self._append_progress("DCML Search finished successfully.")
        metrics = results.get("best_metrics", {})
        gui = results.get("gui", {})
        text = (
            f"Status: {results.get('status', 'success')}\n\n"
            f"Variant: {gui.get('variant', '')}\n"
            f"{results.get('message', '')}\n\n"
            f"Best trial: {results.get('best_trial', '')}\n"
            f"Best validation RMSE: {metrics.get('RMSE', 0.0):.6f}\n"
            f"Best validation Pearson: {metrics.get('Pearson', 0.0):.6f}\n"
            f"Best validation MAE: {metrics.get('MAE', 0.0):.6f}\n"
            f"Elapsed time: {results.get('elapsed_time', '')}\n\n"
            f"Run directory:\n{results.get('run_dir', '')}\n\n"
            f"Trials CSV:\n{results.get('trials_csv', '')}\n\n"
            f"Best config:\n{results.get('best_config_yaml', '')}\n\n"
            f"Best model:\n{results.get('best_model_path', '')}"
        )
        QMessageBox.information(self.parent_window, "DCML Search Finished", text)

    def on_error(self, error_message: str):
        self._append_progress("DCML action failed.")
        self._append_progress(error_message)
        QMessageBox.critical(self.parent_window, "DCML Error", error_message)

    def on_thread_finished(self):
        self._append_progress("DCML worker finished.")
        self._set_actions_enabled(True)
        self.generate_data_thread = None
        self.train_thread = None
        self.test_thread = None
        self.hyperparameter_search_thread = None
