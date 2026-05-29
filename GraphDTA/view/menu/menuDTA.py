import logging

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from GraphDTA.view.dialog.generate_data_dialog import DBGenerationDialog
from GraphDTA.view.dialog.hyperparameter_search_dialog import GraphDTAHyperparameterSearchDialog
from GraphDTA.view.dialog.predict_dialog import PredictDialog
from GraphDTA.view.dialog.train_dialog import TrainGraphDTADialog
from GraphDTA.worker import DBGenerationThread, TrainDTAThread, PredictDTAThread, HPProcess

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

        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self.train_dta)
        self.addAction(train_action)

        predict_action = QAction("Predict Model", self)
        predict_action.triggered.connect(self.predict_dta)
        self.addAction(predict_action)

        hp_action = QAction("Hyperparameter Tuning", self)
        hp_action.triggered.connect(self.hp_dta)
        self.addAction(hp_action)

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

    def train_dta(self):
        dialog = TrainGraphDTADialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("[INIT TRAIN DTA] WAIT PLEASE")
            self.main_window.setEnabled(False)
            self.train_thread = TrainDTAThread(params, self)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_train_error)
            self.train_thread.log_message.connect(logger.info)
            self.train_thread.start()

    def on_train_success(self):
        logger.info("[TRAIN DTA] finished successfully.")
        self.main_window.setEnabled(True)
        self.train_thread = None

    def on_train_error(self, error_message):
        logger.error(f"[INIT TRAIN DTA] ERROR: {error_message}")
        self.main_window.setEnabled(True)
        self.train_thread = None

    def predict_dta(self):
        dialog = PredictDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("[INIT PREDICT DTA] WAIT PLEASE")
            self.main_window.setEnabled(False)
            self.predict_thread = PredictDTAThread(params, self)
            self.predict_thread.finished_success.connect(self.on_predict_success)
            self.predict_thread.finished_error.connect(self.on_predict_error)
            self.predict_thread.log_message.connect(logger.info)
            self.predict_thread.start()

    def on_predict_success(self):
        logger.info("[PREDICT DTA] finished successfully.")
        self.main_window.setEnabled(True)
        self.predict_thread = None

    def on_predict_error(self, error_message):
        logger.error(f"[INIT PREDICT DTA] ERROR: {error_message}")
        self.main_window.setEnabled(True)
        self.predict_thread = None

    def hp_dta(self):
        dialog = GraphDTAHyperparameterSearchDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_values()

            logger.info("[INIT HP DTA] WAIT PLEASE")
            self.main_window.setEnabled(False)
            self.hp_process = HPProcess(params)
            self.hp_process.log_message.connect(logger.info)
            self.hp_process.finished_error.connect(self.on_hp_error)
            self.hp_process.finished_success.connect(self.on_hp_success)
            self.hp_process.start()

    def on_hp_success(self):
        logger.info("[HP DTA] finished successfully.")
        self.main_window.setEnabled(True)
        self.hp_process = None

    def on_hp_error(self, error_message):
        logger.error(f"[INIT HP DTA] ERROR: {error_message}")
        self.main_window.setEnabled(True)
        self.hp_process = None
