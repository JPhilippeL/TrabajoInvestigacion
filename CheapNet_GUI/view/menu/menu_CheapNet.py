import logging

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from CheapNet_GUI.view.dialogs.generate_graph_dialog import DBGenerationDialog
from CheapNet_GUI.view.dialogs.hyperparameter_tuning_dialog import HyperparameterSearchDialog
from CheapNet_GUI.view.dialogs.prediction_dialog import PredictDialog
from CheapNet_GUI.view.dialogs.train_dialog import TrainDialog
from CheapNet_GUI.workers import DBGenerationThread, HyperparameterTuningProcess
from CheapNet_GUI.workers import PredictThread
from CheapNet_GUI.workers import TrainCheapNetThread

logger = logging.getLogger(__name__)


class MenuCheapNet(QMenu):
    def __init__(self, parent_window):
        super().__init__("CheapNet", parent_window)
        self.main_window = parent_window

        self.init_actions()

    def init_actions(self):
        gendata_action = QAction("DB Generation", self)
        gendata_action.triggered.connect(self.generate_db)
        self.addAction(gendata_action)

        train_action = QAction("Train", self)
        train_action.triggered.connect(self.train_cheapnet)
        self.addAction(train_action)

        prediction_action = QAction("Prediction", self)
        prediction_action.triggered.connect(self.predict_cheapnet)
        self.addAction(prediction_action)

        hp_action = QAction("Hyperparameter Tuning", self)
        hp_action.triggered.connect(self.tune_cheapnet)
        self.addAction(hp_action)

    def generate_db(self):
        dialog = DBGenerationDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            if (
                    not params["pic50_file"]
                    or not params["ligand_sdf_dir"]
                    or not params["proteine_pdb_dir"]
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

    def train_cheapnet(self):

        dialog = TrainDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("Init training...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.train_thread = TrainCheapNetThread(params, self)
            self.train_thread.log_message.connect(logger.info)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_train_error)
            self.train_thread.start()

    def on_train_success(self):
        logger.info("Training finished successfully.")
        self.main_window.setEnabled(True)
        self.train_thread = None

    def on_train_error(self, error_message):
        logger.error(f"Training failed: {error_message}")
        self.main_window.setEnabled(True)
        self.train_thread = None

    def predict_cheapnet(self):
        dialog = PredictDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("Init prediction...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.predict_thread = PredictThread(params, self)
            self.predict_thread.log_message.connect(logger.info)
            self.predict_thread.finished_success.connect(
                self.on_predict_success
            )
            self.predict_thread.finished_error.connect(self.on_predict_error)
            self.predict_thread.start()

    def on_predict_success(self):
        logger.info("Prediction finished successfully.")
        self.main_window.setEnabled(True)
        self.predict_thread = None

    def on_predict_error(self, error_message):
        logger.error(f"Prediction failed: {error_message}")
        self.main_window.setEnabled(True)
        self.predict_thread = None

    def tune_cheapnet(self):
        dialog = HyperparameterSearchDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_values()
            logger.info("[HP Tuning]Init hyperparameter tuning...")
            logger.info("[HP Tuning]Please wait...")

            self.main_window.setEnabled(False)

            self.hp_process = HyperparameterTuningProcess(params, self)
            self.hp_process.log_message.connect(logger.info)
            self.hp_process.finished_success.connect(self.on_success_hp)
            self.hp_process.finished_error.connect(self.on_error_hp)
            self.hp_process.start(payload=params)

    def on_success_hp(self):
        logger.info("Hyperparameter tuning finished successfully.")
        self.main_window.setEnabled(True)
        self.hp_process = None

    def on_error_hp(self):
        logger.info("Hyperparameter tuning failed.")
        self.main_window.setEnabled(True)
        self.hp_process = None
