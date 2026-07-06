from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import logging

from ui.dialogs.hyperparameter_search_dialog import HyperparameterSearchDialog

logger = logging.getLogger(__name__)


class MenuHyperparameterSearchGNN(QMenu):
    def __init__(self, parent_window):
        super().__init__("Hyperparameter Search GNN", parent_window)
        self.main_window = parent_window

        self.init_actions()

    def init_actions(self):
        search_action = QAction("Run Hyperparameter Search", self)
        search_action.triggered.connect(self.run_hyperparameter_search)
        self.addAction(search_action)

    def run_hyperparameter_search(self):
        dialog = HyperparameterSearchDialog(self.main_window)

        if dialog.exec():
            config = dialog.get_config()

            try:
                self.main_window.hyperparameter_search_controller.launch_search(config)
            except Exception as e:
                logger.exception(
                    "Error launching GNN hyperparameter search: " + str(e),
                    exc_info=True,
                )
