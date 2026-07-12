"""
@file menu_EGNN.py
@author Mohamed EL BOUKHIARI
@brief Menu integration for the EGNN module.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from EGNN.ui.dialogs.batch_train_egnn_dialog import BatchTrainDialog
from EGNN.ui.dialogs.generate_data_dialog import DBGenerationDialog
from EGNN.ui.dialogs.test_egnn_dialog import TestDialog
from EGNN.ui.dialogs.train_egnn_dialog import TrainDialog
from EGNN.workers import (
    DBGenerationThread,
    TestThread,
    TrainAllModelsThread,
    TrainThread,
)

logger = logging.getLogger(__name__)


class MenuEGNN(QMenu):
    def __init__(self, parent_window):
        super().__init__("EGNN", parent_window)
        self.main_window = parent_window
        self.init_actions()

    def init_actions(self):
        generate_data_action = QAction("Generate Data", self)
        generate_data_action.triggered.connect(self.generate_data_egnn)
        self.addAction(generate_data_action)

        train_action = QAction("Train", self)
        train_action.triggered.connect(self.train_model_egnn)
        self.addAction(train_action)

        batch_train_action = QAction("Search", self)
        batch_train_action.triggered.connect(self.train_all_models_egnn)
        self.addAction(batch_train_action)

        test_action = QAction("Evaluate", self)
        test_action.triggered.connect(self.test_model_egnn)
        self.addAction(test_action)

    def generate_data_egnn(self):
        dialog = DBGenerationDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if not params["pic50_file"] or not params["ligand_sdf_dir"] or not params["protein_pdb_dir"]:
                logger.warning("Missing required files or directories for EGNN graph generation.")
                return

            logger.info("Starting EGNN graph generation in the background.")
            self.main_window.setEnabled(False)

            self.generation_thread = DBGenerationThread(params)
            self.generation_thread.finished_success.connect(self.on_generation_success)
            self.generation_thread.finished_error.connect(self.on_thread_error)
            self.generation_thread.start()

    def train_model_egnn(self):
        dialog = TrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if (
                not params["graphs_dir"]
                or not params["train_split_file"]
                or not params["val_split_file"]
                or not params["test_split_file"]
            ):
                logger.warning("Missing required paths for EGNN training.")
                return

            logger.info("Starting EGNN training in the background.")
            self.main_window.setEnabled(False)

            self.train_thread = TrainThread(params)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_thread_error)
            self.train_thread.start()

    def train_all_models_egnn(self):
        dialog = BatchTrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if (
                not params["graphs_dir"]
                or not params["train_split_file"]
                or not params["val_split_file"]
                or not params["test_split_file"]
            ):
                logger.warning("Missing required paths for EGNN hyperparameter search.")
                return

            logger.info("Starting EGNN hyperparameter search.")
            self.main_window.setEnabled(False)

            self.train_batch_thread = TrainAllModelsThread(params)
            self.train_batch_thread.finished_success.connect(self.on_batch_train_success)
            self.train_batch_thread.finished_error.connect(self.on_thread_error)
            self.train_batch_thread.start()

    def test_model_egnn(self):
        dialog = TestDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()

            if not params["graphs_dir"] or not params["test_split_file"] or not params["checkpoint_or_run"]:
                logger.warning("Missing required paths for EGNN evaluation.")
                return

            logger.info("Starting EGNN evaluation in the background.")
            self.main_window.setEnabled(False)

            self.test_thread = TestThread(params)
            self.test_thread.finished_success.connect(self.on_test_success)
            self.test_thread.finished_error.connect(self.on_thread_error)
            self.test_thread.start()

    def on_thread_error(self, error_msg):
        self.main_window.setEnabled(True)
        logger.error("Background process failed:\n%s", error_msg)

    def on_generation_success(self, results):
        self.main_window.setEnabled(True)
        logger.info("EGNN data generation completed: %s", results)

    def on_train_success(self, run_dir):
        self.main_window.setEnabled(True)
        logger.info("EGNN training completed. Models saved in: %s", run_dir)

    def on_batch_train_success(self, results):
        self.main_window.setEnabled(True)
        logger.info("EGNN hyperparameter search completed: %s", results)

    def on_test_success(self, metrics):
        self.main_window.setEnabled(True)
        metrics_log = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics.items()]
        )
        logger.info("EGNN evaluation completed. Metrics: %s", metrics_log)

    def on_batch_test_model_success(self, model_name, metrics):
        metrics_log = " | ".join(
            [f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics.items()]
        )
        logger.info("[EGNN Evaluation] %s completed. Metrics: %s", model_name, metrics_log)

    def on_batch_test_all_finished(self, csv_path):
        self.main_window.setEnabled(True)
        if csv_path:
            logger.info("EGNN batch evaluation finished. Summary saved in: %s", csv_path)
        else:
            logger.warning("EGNN batch evaluation finished without a summary CSV.")
