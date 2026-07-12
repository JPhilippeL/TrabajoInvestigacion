"""
@file workers.py
@author Mohamed EL BOUKHIARI
@brief Background worker threads for the EDNN module.
"""

from PySide6.QtCore import QThread, Signal
import traceback

from EDNN.Core.ednn_generate_data import generate_data
from EDNN.Core.ednn_trainer import train
from EDNN.Core.ednn_tester import evaluate_checkpoint_or_run, test_all_models_in_folder
from EDNN.Core.ednn_hyperparameter_search import run_hyperparameter_search


class DBGenerationThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            results = generate_data(**self.params)
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TrainThread(QThread):
    finished_success = Signal(str)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            run_dir = train(**self.params)
            self.finished_success.emit(run_dir)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TrainAllModelsThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            results = run_hyperparameter_search(**self.params)
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TestThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            metrics = evaluate_checkpoint_or_run(**self.params)
            self.finished_success.emit(metrics)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TestAllModelsThread(QThread):
    finished_success = Signal(str, dict)
    finished_error = Signal(str)
    all_finished = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            csv_path, all_metrics = test_all_models_in_folder(**self.params)
            for model_name, metrics in all_metrics.items():
                self.finished_success.emit(model_name, metrics)
            self.all_finished.emit(csv_path)
        except Exception:
            self.finished_error.emit(traceback.format_exc())
