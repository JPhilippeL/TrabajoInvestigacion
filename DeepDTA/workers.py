"""
@file workers.py
@brief Background worker threads for the DeepDTA module.
"""

from __future__ import annotations

import json
import os
import shutil
import traceback

from PySide6.QtCore import QThread, Signal

from DeepDTA.Core.deepdta_audit import audit_dataset_splits
from DeepDTA.Core.deepdta_evaluator import evaluate_checkpoint
from DeepDTA.Core.deepdta_hyperparameter_search import run_hyperparameter_search
from DeepDTA.Core.deepdta_mpro_urv_converter import convert_mpro_urv_to_deepdta
from DeepDTA.Core.deepdta_trainer import train


class GenerateDataThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            output_root = self.params["output_root"]
            overwrite = self.params.get("overwrite", False)
            if os.path.exists(output_root) and os.listdir(output_root):
                if not overwrite:
                    raise FileExistsError(f"Output folder is not empty: {output_root}")
                shutil.rmtree(output_root)
            convert_mpro_urv_to_deepdta(
                source_root=self.params["source_root"],
                output_root=output_root,
                protein_pdb_path=self.params.get("protein_pdb_path") or None,
            )
            metadata_path = os.path.join(output_root, "metadata.json")
            metadata = json.load(open(metadata_path, encoding="utf-8")) if os.path.exists(metadata_path) else {}
            audit = audit_dataset_splits(dataset_name="mpro_urv", output_dir=output_root)
            self.finished_success.emit({"output_root": output_root, "metadata": metadata, "audit": audit})
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class TrainThread(QThread):
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


class EvaluateThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            results = evaluate_checkpoint(**self.params)
            self.finished_success.emit(results)
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
