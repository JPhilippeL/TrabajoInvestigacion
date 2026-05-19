import logging

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from GraphDTA.view.dialog.generate_data_dialog import DBGenerationDialog
from GraphDTA.worker import DBGenerationThread

logger = logging.getLogger(__name__)


class MenuDTA(QMenu):
    def __init__(self, parent_window):
        super().__init__("GraphDTA", parent_window)
        self.main_window = parent_window

        self.init_actions()

    def init_actions(self):
        gendata_action = QAction("DB Generation", self)
        gendata_action.triggered.connect(self.generate_db)
        self.addAction(gendata_action)

    def generate_db(self):
        dialog = DBGenerationDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            if (
                    not params["pic50_file"]
                    or not params["sdf_dir"]
                    or not params["pdb_dir"]
            ):
                logger.error(
                    "Some required directories are missing for data generation."
                )
                return

            logger.info("Init DB generation...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.db_thread = DBGenerationThread(params, self)
            self.db_thread.log_message.connect(logger.info)

            self.db_thread.finished_success.connect(
                self.on_db_generation_success
            )
            self.db_thread.finished_error.connect(self.on_db_generation_error)
            self.db_thread.start()

    def on_db_generation_success(self):
        logger.info("DB generation finished successfully.")
        self.main_window.setEnabled(True)
        self.db_thread = None

    def on_db_generation_error(self, error_message):
        logger.error(f"DB generation failed: {error_message}")
        self.main_window.setEnabled(True)
        self.db_thread = None
