import logging
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from core.factory.job_factory import JobFactory
from core.workers.runner import HyperparameterTuningProcess
from core.workers.worker import Worker
from job_config.config_save import ConfigSave

logger = logging.getLogger(__name__)


class Menu(QMenu):
    def __init__(
        self,
        parent_window,
        model_name,
        db_dialog,
        train_dialog,
        predict_dialog,
        script_path,
        hp_dialog=None,
    ):
        super().__init__(model_name, parent_window)

        self.main_window = parent_window
        self.model_name = model_name

        self.db_dialog = db_dialog
        self.train_dialog = train_dialog
        self.predict_dialog = predict_dialog
        self.hp_dialog = hp_dialog

        self.db_thread = None
        self.train_thread = None
        self.predict_thread = None
        self.hp_process = None
        self.script_path = script_path

        self.init_actions()

    def init_actions(self):
        generate_data_action = QAction("Generate graph", self)
        generate_data_action.triggered.connect(self.generate_db)
        self.addAction(generate_data_action)

        train_action = QAction("Train", self)
        train_action.triggered.connect(self.train_model)
        self.addAction(train_action)

        predict_action = QAction("Predict", self)
        predict_action.triggered.connect(self.predict_model)
        self.addAction(predict_action)

        hp_action = QAction("Hyperparameter tuning", self)
        hp_action.triggered.connect(self.hyperparameter_tuning)
        self.addAction(hp_action)

    def generate_db(self):
        dialog = self.db_dialog(self.main_window)

        if not dialog.exec():
            return

        config = dialog.get_inputs()

        logger.info("Generating data compatible with model %s...", self.model_name)
        logger.info("Please wait...")

        self.main_window.setEnabled(False)

        strategy = JobFactory.create_strategy(
            model_name=self.model_name,
            task_name="generate_data",
        )

        self.db_thread = Worker(config=config, strategy=strategy)

        self.db_thread.log.connect(logger.info)
        self.db_thread.progress.connect(lambda value: logger.info("Progress: %s", value))
        self.db_thread.result_ready.connect(self.on_db_generation_success)
        self.db_thread.error.connect(self.on_db_generation_error)

        self.db_thread.start()

    def on_db_generation_success(self, result):
        logger.info("Data generation finished successfully.")
        logger.info("Result: %s", result)

        self.main_window.setEnabled(True)
        self.db_thread = None

    def on_db_generation_error(self, error_message):
        logger.error("Data generation failed: %s", error_message)

        self.main_window.setEnabled(True)
        self.db_thread = None

    def train_model(self):
        dialog = self.train_dialog(self.main_window)

        if not dialog.exec():
            return

        config = dialog.get_inputs()

        logger.info("Starting training for model %s...", self.model_name)
        logger.info("Please wait...")

        self.main_window.setEnabled(False)

        strategy = JobFactory.create_strategy(
            model_name=self.model_name,
            task_name="train",
        )
        config_saver = ConfigSave(config=config, output_path=config.output_path)
        config_saver.save_config()

        self.train_thread = Worker(config=config, strategy=strategy)

        self.train_thread.log.connect(logger.info)
        self.train_thread.progress.connect(lambda value: logger.info("Progress: %s", value))
        self.train_thread.result_ready.connect(self.on_train_success)
        self.train_thread.error.connect(self.on_train_error)

        self.train_thread.start()

    def on_train_success(self, result):
        logger.info("Training finished successfully.")
        logger.info("Result: %s", result)

        self.main_window.setEnabled(True)
        self.train_thread = None

    def on_train_error(self, error_message):
        logger.error("Training failed: %s", error_message)

        self.main_window.setEnabled(True)
        self.train_thread = None

    def predict_model(self):
        dialog = self.predict_dialog(self.main_window)

        if not dialog.exec():
            return

        config = dialog.get_inputs()

        logger.info("Starting prediction for model %s...", self.model_name)
        logger.info("Please wait...")

        self.main_window.setEnabled(False)

        strategy = JobFactory.create_strategy(
            model_name=self.model_name,
            task_name="predict",
        )

        self.predict_thread = Worker(config=config, strategy=strategy)

        self.predict_thread.log.connect(logger.info)
        self.predict_thread.progress.connect(lambda value: logger.info("Progress: %s", value))
        self.predict_thread.result_ready.connect(self.on_predict_success)
        self.predict_thread.error.connect(self.on_predict_error)

        self.predict_thread.start()

    def on_predict_success(self, result):
        logger.info("Prediction finished successfully.")
        logger.info("Result: %s", result)

        self.main_window.setEnabled(True)
        self.predict_thread = None

    def on_predict_error(self, error_message):
        logger.error("Prediction failed: %s", error_message)

        self.main_window.setEnabled(True)
        self.predict_thread = None

    def hyperparameter_tuning(self):
        if self.hp_dialog is None:
            logger.error("No hyperparameter tuning dialog provided for model %s.", self.model_name)
            return

        dialog = self.hp_dialog(self.main_window)

        if not dialog.exec():
            return

        payload = dialog.get_inputs()

        if isinstance(payload, dict):
            payload["model_name"] = self.model_name
        else:
            logger.error(
                "Hyperparameter tuning dialog must return a dict payload, got %s.",
                type(payload).__name__,
            )
            return

        logger.info("Starting hyperparameter tuning for model %s...", self.model_name)
        logger.info("Please wait...")

        self.main_window.setEnabled(False)

        self.hp_process = HyperparameterTuningProcess(
            script_path=self.script_path,
            parent=self,
        )

        self.hp_process.log_message.connect(logger.info)
        self.hp_process.progress.connect(
            lambda value: logger.info("Hyperparameter tuning progress: %s", value)
        )
        self.hp_process.ready.connect(self.on_hp_success)
        self.hp_process.error.connect(self.on_hp_error)

        self.hp_process.start(payload)

    def on_hp_success(self):
        logger.info("Hyperparameter tuning finished successfully.")

        self.main_window.setEnabled(True)
        self.hp_process = None

    def on_hp_error(self, error_message):
        logger.error("Hyperparameter tuning failed: %s", error_message)

        self.main_window.setEnabled(True)
        self.hp_process = None
