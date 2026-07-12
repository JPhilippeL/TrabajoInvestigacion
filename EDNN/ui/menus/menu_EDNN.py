"""
@file menu_EDNN.py
@author Mohamed EL BOUKHIARI
@brief Menu integration for the EDNN module.
"""

from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import logging

from EDNN.ui.dialogs.generate_data_dialog import DBGenerationDialog
from EDNN.ui.dialogs.train_ednn_dialog import TrainDialog
from EDNN.ui.dialogs.batch_train_ednn_dialog import BatchTrainDialog
from EDNN.ui.dialogs.test_ednn_dialog import TestDialog

from EDNN.workers import (
    DBGenerationThread,
    TrainThread,
    TrainAllModelsThread,
    TestThread,
)

logger = logging.getLogger(__name__)


class MenuEDNN(QMenu):
    def __init__(self, parent_window):
        super().__init__("EDNN", parent_window)
        self.main_window = parent_window
        self.init_actions()

    def init_actions(self):
        generate_data_action = QAction("Generate Data", self)
        generate_data_action.triggered.connect(self.generate_data_ednn)
        self.addAction(generate_data_action)

        train_action = QAction("Train", self)
        train_action.triggered.connect(self.train_model_ednn)
        self.addAction(train_action)

        batch_train_action = QAction("Search", self)
        batch_train_action.triggered.connect(self.train_all_models_ednn)
        self.addAction(batch_train_action)

        test_action = QAction("Evaluate", self)
        test_action.triggered.connect(self.test_model_ednn)
        self.addAction(test_action)


    def generate_data_ednn(self):
        dialog = DBGenerationDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if not params["pic50_file"] or not params["ligand_sdf_dir"] or not params["protein_pdb_dir"]:
                logger.warning("Missing required files or directories for EDNN graph generation.")
                return

            logger.info("Starting EDNN graph generation in background...")
            self.main_window.setEnabled(False)

            self.generation_thread = DBGenerationThread(params)
            self.generation_thread.finished_success.connect(self.on_generation_success)
            self.generation_thread.finished_error.connect(self.on_thread_error)
            self.generation_thread.start()

    def train_model_ednn(self):
        dialog = TrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if (
                not params["graphs_dir"]
                or not params["train_split_file"]
                or not params["val_split_file"]
                or not params["test_split_file"]
            ):
                logger.warning("Missing required paths for EDNN training.")
                return

            logger.info("Starting EDNN training in background...")
            self.main_window.setEnabled(False)

            self.train_thread = TrainThread(params)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_thread_error)
            self.train_thread.start()

    def train_all_models_ednn(self):
        dialog = BatchTrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if (
                not params["graphs_dir"]
                or not params["train_split_file"]
                or not params["val_split_file"]
                or not params["test_split_file"]
            ):
                logger.warning("Missing required paths for EDNN batch training / hyperparameter search.")
                return

            logger.info("Starting EDNN batch training / hyperparameter search...")
            self.main_window.setEnabled(False)

            self.train_batch_thread = TrainAllModelsThread(params)
            self.train_batch_thread.finished_success.connect(self.on_batch_train_success)
            self.train_batch_thread.finished_error.connect(self.on_thread_error)
            self.train_batch_thread.start()

    def test_model_ednn(self):
        dialog = TestDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if not params["graphs_dir"] or not params["test_split_file"] or not params["checkpoint_or_run"]:
                logger.warning("Missing required paths for EDNN evaluation.")
                return

            logger.info("Starting EDNN evaluation in background...")
            self.main_window.setEnabled(False)

            self.test_thread = TestThread(params)
            self.test_thread.finished_success.connect(self.on_test_success)
            self.test_thread.finished_error.connect(self.on_thread_error)
            self.test_thread.start()

    def on_thread_error(self, error_msg):
        self.main_window.setEnabled(True)
        logger.exception(f"Background process failed: {error_msg}")

    def on_generation_success(self, results):
        self.main_window.setEnabled(True)
        logger.info(f"EDNN data generation completed: {results}")

    def on_train_success(self, run_dir):
        self.main_window.setEnabled(True)
        logger.info(f"EDNN training completed. Results saved in: {run_dir}")

    def on_batch_train_success(self, results):
        self.main_window.setEnabled(True)
        logger.info(f"EDNN batch training / hyperparameter search completed: {results}")

    def on_test_success(self, metrics):
        self.main_window.setEnabled(True)
        metrics_log = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics.items()]
        )
        logger.info(f"EDNN evaluation completed. Metrics: {metrics_log}")

    def on_batch_test_model_success(self, model_name, metrics):
        metrics_log = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics.items()]
        )
        logger.info(f"[EDNN Evaluation] {model_name} completed. Metrics: {metrics_log}")

    def on_batch_test_all_finished(self, csv_path):
        self.main_window.setEnabled(True)
        if csv_path:
            logger.info(f"EDNN batch evaluation finished. Summary saved in: {csv_path}")
        else:
            logger.warning("EDNN batch evaluation finished without a summary CSV.")
