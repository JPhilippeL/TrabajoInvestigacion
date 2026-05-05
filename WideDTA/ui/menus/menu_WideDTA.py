"""
@file menu_WideDTA.py
@author Mohamed EL BOUKHIARI
@brief Menu integration for the WideDTA module.
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from WideDTA.ui.dialogs.hyperparameter_search_widedta_dialog import (
    HyperparameterSearchWideDTADialog,
)
from WideDTA.workers import TrainAllModelsThread


class MenuWideDTA(QMenu):
    """
    @brief WideDTA menu for the main GUI.
    """

    def __init__(self, parent_window):
        super().__init__("WideDTA", parent_window)

        self.parent_window = parent_window
        self.hyperparameter_search_thread = None

        self.hyperparameter_search_action = QAction("Hyperparameter Search", self)
        self.hyperparameter_search_action.triggered.connect(self.open_hyperparameter_search_dialog)

        self.addAction(self.hyperparameter_search_action)

    def open_hyperparameter_search_dialog(self):
        dialog = HyperparameterSearchWideDTADialog(self.parent_window)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            params = dialog.get_inputs()
        except Exception as exc:
            QMessageBox.critical(self.parent_window, "WideDTA Configuration Error", str(exc))
            return

        self.hyperparameter_search_action.setEnabled(False)

        self.hyperparameter_search_thread = TrainAllModelsThread(params)
        self.hyperparameter_search_thread.finished_success.connect(self.on_hyperparameter_search_success)
        self.hyperparameter_search_thread.finished_error.connect(self.on_hyperparameter_search_error)
        self.hyperparameter_search_thread.finished.connect(self.on_hyperparameter_search_finished)

        self.hyperparameter_search_thread.start()

    def on_hyperparameter_search_success(self, results: dict):
        status = results.get("status", "unknown")
        message = results.get("message", "")

        best_trial = results.get("best_trial", "")
        elapsed_time = results.get("elapsed_time", "unknown")
        run_dir = results.get("run_dir", "")
        trials_csv = results.get("trials_csv", "")
        best_config_yaml = results.get("best_config_yaml", "")

        text = (
            f"Status: {status}\n\n"
            f"{message}\n\n"
            f"Best trial: {best_trial}\n"
            f"Elapsed time: {elapsed_time}\n\n"
            f"Run directory:\n{run_dir}\n\n"
            f"Trials CSV:\n{trials_csv}\n\n"
            f"Best config:\n{best_config_yaml}"
        )

        QMessageBox.information(self.parent_window, "WideDTA Hyperparameter Search Finished", text)

    def on_hyperparameter_search_error(self, error_message: str):
        QMessageBox.critical(self.parent_window, "WideDTA Hyperparameter Search Error", error_message)

    def on_hyperparameter_search_finished(self):
        self.hyperparameter_search_action.setEnabled(True)
        self.hyperparameter_search_thread = None
