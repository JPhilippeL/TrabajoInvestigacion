"""
@file workers.py
@author Mohamed EL BOUKHIARI
@brief Background worker threads for the DCML module.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from DCML.Core.dcml_hyperparameter_search import run_hyperparameter_search
from DCML.Core.dcml_tester import test_dcml
from DCML.Core.dcml_trainer import train_dcml


class TrainThread(QThread):
    """Run DCML training without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            summary = train_dcml(**self.params)
            self.finished_success.emit(summary)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TestThread(QThread):
    """Run DCML evaluation without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            summary = test_dcml(**self.params)
            self.finished_success.emit(summary)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class HyperparameterSearchThread(QThread):
    """Run DCML hyperparameter search without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            results = run_hyperparameter_search(**self.params)
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(traceback.format_exc())
