import os

from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from pathlib import Path
import logging

from GIGN_GUI.view.dialogs.generate_db_dialog import DBGenerationDialog
from GIGN_GUI.view.dialogs.predict_dialog import PredictDialog
from GIGN_GUI.view.dialogs.train_dialog import TrainDialog
from GIGN_GUI.workers import DBGenerationThread, TrainGIGNThread, PredictThread, HyperparameterTuningProcess
from GIGN_GUI.view.dialogs.hyperparameter_tuning_dialog import HyperParameterTuningDialog

logger = logging.getLogger(__name__)
# this script will launch ray tune hyperparameter search separately.
SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "GIGN_GUI", "model",
    "run_hp_tuning.py", )


class MenuGIGN(QMenu):
    def __init__(self, parent_window):
        super().__init__("GIGN", parent_window)
        self.main_window = parent_window

        self.init_actions()

    def init_actions(self):
        # 1. Generate Data
        gendata_action = QAction("DB Generation", self)
        gendata_action.triggered.connect(self.generate_db)
        self.addAction(gendata_action)

        # 2. Train
        train_action = QAction("Train", self)
        train_action.triggered.connect(self.train_gign)
        self.addAction(train_action)

        # 3. Predict
        predict_action = QAction("Predict", self)
        predict_action.triggered.connect(self.predict_gign)
        self.addAction(predict_action)

        # 4. Hyperparameter tuning
        tuning_action = QAction("Hyperparameter Tuning", self)
        tuning_action.triggered.connect(self.hptuning_gign)
        self.addAction(tuning_action)

    # DB generation (function and slots)
    def generate_db(self):
        dialog = DBGenerationDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            if not params["pic50_file"] or not params["lig_dir"] or not params["pdb_dir"]:
                logger.error("Some required directories are missing for data generation.")
                return

            logger.info("Init DB generation...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.db_thread = DBGenerationThread(params, self)
            self.db_thread.finished_success.connect(self.on_db_generation_success)
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

    # train  (function and slots)

    def train_gign(self):
        dialog = TrainDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("Init training...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.train_thread = TrainGIGNThread(params, self)
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

    # predict

    def predict_gign(self):
        dialog = PredictDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("Init prediction...")
            logger.info("Please wait...")

            self.main_window.setEnabled(False)

            self.predict_thread = PredictThread(params, self)
            self.predict_thread.log_message.connect(logger.info)
            self.predict_thread.finished_success.connect(self.on_predict_success)
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

    # Hyperparameter tuning (function + slots)

    def hptuning_gign(self):
        dialog = HyperParameterTuningDialog(self.main_window)

        if dialog.exec():
            params = dialog.get_inputs()

            logger.info("\t[INIT] Hyperparameter Tuning")
            logger.info("\tPlease wait...")

            self.main_window.setEnabled(False)

            self.hptuning_process = HyperparameterTuningProcess(
                params=params,
                script_path=SCRIPT,
                parent=self,
            )
            self.hptuning_process.log_message.connect(logger.info)
            self.hptuning_process.finished_success.connect(self.on_hptuning_success)
            self.hptuning_process.finished_error.connect(self.on_hptuning_error)
            self.hptuning_process.start()

    def on_hptuning_success(self):
        logger.info("Hyperparameter tuning finished successfully.")
        self.main_window.setEnabled(True)
        self.hptuning_process = None

    def on_hptuning_error(self, error_message):
        logger.error(error_message)
        self.main_window.setEnabled(True)
        self.hptuning_process = None
