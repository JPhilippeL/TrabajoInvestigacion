"""
@file workers.py
@author Mohamed EL BOUKHIARI
@brief Background worker threads for the WideDTA module.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from WideDTA.Core.widedta_hyperparameter_search import run_hyperparameter_search
from WideDTA.Core.widedta_trainer import train


class TrainThread(QThread):
    """
    @brief Background thread for a single WideDTA training run.
    """

    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            results = train(**self.params)
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TrainAllModelsThread(QThread):
    """
    @brief Background thread for WideDTA hyperparameter search.
    """

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
