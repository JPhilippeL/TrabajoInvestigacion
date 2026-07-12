"""
@file menu_WideDTA.py
@brief Menu integration for the WideDTA module.
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from WideDTA.ui.dialogs.evaluate_widedta_dialog import EvaluateWideDTADialog
from WideDTA.ui.dialogs.generate_data_widedta_dialog import GenerateDataWideDTADialog
from WideDTA.ui.dialogs.hyperparameter_search_widedta_dialog import HyperparameterSearchWideDTADialog
from WideDTA.ui.dialogs.train_widedta_dialog import TrainWideDTADialog
from WideDTA.workers import EvaluateThread, GenerateDataThread, TrainAllModelsThread, TrainThread


class MenuWideDTA(QMenu):
    def __init__(self, parent_window):
        super().__init__("WideDTA", parent_window)
        self.parent_window = parent_window
        self.generate_thread = None
        self.train_thread = None
        self.evaluate_thread = None
        self.hyperparameter_search_thread = None

        self.generate_action = QAction("Generate Data", self)
        self.train_action = QAction("Train", self)
        self.hyperparameter_search_action = QAction("Search", self)
        self.evaluate_action = QAction("Evaluate", self)


        self.generate_action.triggered.connect(self.open_generate_data_dialog)
        self.train_action.triggered.connect(self.open_train_dialog)
        self.evaluate_action.triggered.connect(self.open_evaluate_dialog)
        self.hyperparameter_search_action.triggered.connect(self.open_hyperparameter_search_dialog)

        self.addAction(self.generate_action)
        self.addAction(self.train_action)
        self.addAction(self.hyperparameter_search_action)
        self.addAction(self.evaluate_action)


    def _run_dialog(self, dialog_class, thread_class, action, success_slot, title):
        dialog = dialog_class(self.parent_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        try:
            params = dialog.get_inputs()
        except Exception as exc:
            QMessageBox.critical(self.parent_window, f"{title} Configuration Error", str(exc))
            return None
        action.setEnabled(False)
        thread = thread_class(params)
        thread.finished_success.connect(success_slot)
        thread.finished_error.connect(lambda message, t=title: QMessageBox.critical(self.parent_window, f"{t} Error", message))
        thread.finished.connect(lambda a=action: a.setEnabled(True))
        thread.finished.connect(self._clear_finished_threads)
        thread.start()
        return thread

    def _clear_finished_threads(self):
        for name in ("generate_thread", "train_thread", "evaluate_thread", "hyperparameter_search_thread"):
            thread = getattr(self, name)
            if thread is not None and thread.isFinished():
                setattr(self, name, None)

    def open_generate_data_dialog(self):
        self.generate_thread = self._run_dialog(GenerateDataWideDTADialog, GenerateDataThread, self.generate_action, self.on_generate_success, "WideDTA Generate Data")

    def open_train_dialog(self):
        self.train_thread = self._run_dialog(TrainWideDTADialog, TrainThread, self.train_action, self.on_train_success, "WideDTA Train Model")

    def open_evaluate_dialog(self):
        self.evaluate_thread = self._run_dialog(EvaluateWideDTADialog, EvaluateThread, self.evaluate_action, self.on_evaluate_success, "WideDTA Evaluate/Predict Model")

    def open_hyperparameter_search_dialog(self):
        self.hyperparameter_search_thread = self._run_dialog(HyperparameterSearchWideDTADialog, TrainAllModelsThread, self.hyperparameter_search_action, self.on_hyperparameter_search_success, "WideDTA Hyperparameter Search")

    def on_generate_success(self, results: dict):
        metadata = results.get("metadata", {})
        audit = results.get("audit", {})
        warnings = "\n".join(audit.get("warnings", []))
        text = (
            f"Output folder:\n{results.get('output_root', '')}\n\n"
            f"Ligands kept: {metadata.get('num_ligands', metadata.get('kept_count', 'unknown'))}\n"
            f"Ligands removed/skipped: {metadata.get('skipped_smiles_longer_than_50', metadata.get('removed_count', 'unknown'))}\n"
            f"Y shape: {metadata.get('y_shape', metadata.get('Y_shape', 'unknown'))}\n"
            f"Fold sizes: {audit.get('split_sizes', {})}\n"
            f"Audit: {audit.get('audit_path', '')}\n\n"
            f"Warnings:\n{warnings}"
        )
        QMessageBox.information(self.parent_window, "WideDTA Data Generated", text)

    def on_train_success(self, results: dict):
        warning_text = "\n".join(results.get("warnings", []))
        text = (
            f"Checkpoint:\n{results.get('checkpoint_path', '')}\n\n"
            f"Train RMSE/Pearson: {results.get('train_rmse')} / {results.get('train_pearson')}\n"
            f"Validation RMSE/Pearson: {results.get('val_rmse')} / {results.get('val_pearson')}\n"
            f"Test RMSE/Pearson: {results.get('test_rmse')} / {results.get('test_pearson')}\n"
            f"Split mode: {results.get('split_mode')}\n"
            f"Fold index: {results.get('fold_index')}\n"
            f"Split audit: {results.get('split_audit_path', '')}\n\n"
            f"{warning_text}"
        )
        QMessageBox.information(self.parent_window, "WideDTA Training Finished", text)

    def on_evaluate_success(self, results: dict):
        warnings = "\n".join(results.get("warnings", []))
        text = (
            f"Output directory:\n{results.get('output_dir', '')}\n\n"
            f"Predictions CSV:\n{results.get('predictions_csv', '')}\n"
            f"Metrics JSON:\n{results.get('metrics_json', '')}\n"
            f"Split audit:\n{results.get('split_audit_json', '')}\n\n"
            f"Metrics:\n{results.get('metrics', {})}\n\n"
            f"Warnings:\n{warnings}"
        )
        QMessageBox.information(self.parent_window, "WideDTA Evaluation Finished", text)

    def on_hyperparameter_search_success(self, results: dict):
        text = (
            f"Status: {results.get('status', 'unknown')}\n\n"
            f"{results.get('message', '')}\n\n"
            f"Best trial: {results.get('best_trial', '')}\n"
            f"Elapsed time: {results.get('elapsed_time', 'unknown')}\n"
            f"Debug run: {results.get('debug_run', False)}\n\n"
            f"Run directory:\n{results.get('run_dir', '')}\n\n"
            f"Trials CSV:\n{results.get('trials_csv', '')}\n\n"
            f"Best config:\n{results.get('best_config_yaml', '')}\n\n"
            f"Split audit:\n{results.get('split_audit_json', '')}"
        )
        QMessageBox.information(self.parent_window, "WideDTA Hyperparameter Search Finished", text)
