"""
@file workers.py
@brief Background worker threads for the WideDTA module.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback

from PySide6.QtCore import QThread, Signal

from DeepDTA.Core.deepdta_mpro_urv_converter import convert_mpro_urv_to_deepdta
from WideDTA.Core.widedta_audit import audit_dataset_splits
from WideDTA.Core.widedta_evaluator import evaluate_checkpoint
from WideDTA.Core.widedta_hyperparameter_search import run_hyperparameter_search
from WideDTA.Core.widedta_mpro_urv_converter import convert_from_deepdta_format
from WideDTA.Core.widedta_trainer import train


class GenerateDataThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            source_root = self.params["source_root"]
            output_root = self.params["output_root"]
            overwrite = self.params.get("overwrite", False)
            if os.path.exists(output_root) and os.listdir(output_root):
                if not overwrite:
                    raise FileExistsError(f"Output folder is not empty: {output_root}")
                shutil.rmtree(output_root)

            with tempfile.TemporaryDirectory(prefix="widedta_mpro_") as temp_dir:
                if all(os.path.exists(os.path.join(source_root, name)) for name in ("ligands_can.txt", "proteins.txt", "Y")):
                    deepdta_source = source_root
                else:
                    deepdta_source = os.path.join(temp_dir, "deepdta_mpro_urv")
                    convert_mpro_urv_to_deepdta(source_root=source_root, output_root=deepdta_source)
                result = convert_from_deepdta_format(
                    source_deepdta_dir=deepdta_source,
                    output_dir=output_root,
                    use_protein_as_motif=True,
                )
                folds_src = os.path.join(deepdta_source, "folds")
                folds_dst = os.path.join(output_root, "folds")
                if os.path.exists(folds_src):
                    shutil.copytree(folds_src, folds_dst, dirs_exist_ok=True)
                metadata_src = os.path.join(deepdta_source, "metadata.json")
                metadata = json.load(open(metadata_src, encoding="utf-8")) if os.path.exists(metadata_src) else {}
                metadata.update({
                    "dataset": "mpro_urv",
                    "output_dir": output_root,
                    "motif_mode": "technical_motif_baseline",
                    "motif_source": "protein_sequence",
                    "motif_warning": "motif2.txt duplicates proteins.txt; this is not biological motif extraction.",
                })
                with open(os.path.join(output_root, "metadata.json"), "w", encoding="utf-8") as file:
                    json.dump(metadata, file, indent=4)

            audit = audit_dataset_splits(dataset_name="mpro_urv", output_dir=output_root)
            self.finished_success.emit({"output_root": output_root, "metadata": metadata, "converter_result": result, "audit": audit})
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
