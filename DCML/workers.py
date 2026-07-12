"""
@file workers.py
@author Mohamed EL BOUKHIARI
@brief Background worker threads for the DCML module.
"""

from __future__ import annotations

import logging
from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Signal

from DCML.Core.generate_dcml_from_mpro_v2 import generate_dcml_dataset_from_mpro_v2
from DCML.Core.dcml_gui_workflows import (
    evaluate_dcml_from_prepared_root,
    search_dcml_from_prepared_root,
    train_dcml_from_prepared_root,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _log_path_for(action: str, params: dict) -> Path | None:
    if action == "generate_data":
        root = params.get("output_root")
        return Path(root).expanduser().resolve() / "generate_data.log" if root else None
    if action == "search":
        root = params.get("output_root")
        return Path(root).expanduser().resolve() / "search.log" if root else None
    if action == "train":
        root = params.get("output_dir")
        return Path(root).expanduser().resolve() / "train.log" if root else None
    if action == "evaluate":
        root = params.get("output_dir")
        return Path(root).expanduser().resolve() / "evaluate.log" if root else None
    return None


class _ProgressThread(QThread):
    progress = Signal(str)

    action_name = "dcml"

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        self._log_path = _log_path_for(self.action_name, params)

    def _emit_progress(self, message: str):
        text = str(message)
        LOGGER.info(text)
        if self._log_path is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(text.rstrip() + "\n")
            except Exception:
                LOGGER.exception("Failed to write DCML log file: %s", self._log_path)
        self.progress.emit(text)

    def _emit_traceback(self) -> str:
        tb = traceback.format_exc()
        self._emit_progress("ERROR: " + tb)
        return tb


class TrainThread(_ProgressThread):
    """Run DCML training without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)
    action_name = "train"

    def run(self):
        try:
            self._emit_progress("DCML Train started.")
            self._emit_progress(f"Prepared feature root: {self.params.get('prepared_feature_root', '')}")
            self._emit_progress(f"Output directory: {self.params.get('output_dir', '')}")
            self._emit_progress(f"Variant: {self.params.get('variant', '')}")
            summary = train_dcml_from_prepared_root(**self.params, progress_callback=self._emit_progress)
            self._emit_progress("DCML Train completed.")
            self.finished_success.emit(summary)
        except Exception:
            self.finished_error.emit(self._emit_traceback())


class TestThread(_ProgressThread):
    """Run DCML evaluation without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)
    action_name = "evaluate"

    def run(self):
        try:
            self._emit_progress("DCML Evaluate started.")
            self._emit_progress(f"Prepared feature root: {self.params.get('prepared_feature_root', '')}")
            self._emit_progress(f"Checkpoint path: {self.params.get('model_pt', '')}")
            self._emit_progress(f"Output directory: {self.params.get('output_dir', '')}")
            self._emit_progress(f"Variant: {self.params.get('variant', '')}")
            summary = evaluate_dcml_from_prepared_root(**self.params, progress_callback=self._emit_progress)
            self._emit_progress("DCML Evaluate completed.")
            self.finished_success.emit(summary)
        except Exception:
            self.finished_error.emit(self._emit_traceback())


class HyperparameterSearchThread(_ProgressThread):
    """Run DCML hyperparameter search without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)
    action_name = "search"

    def run(self):
        try:
            self._emit_progress("DCML Search started.")
            self._emit_progress(f"Prepared feature root: {self.params.get('prepared_feature_root', '')}")
            self._emit_progress(f"Output root: {self.params.get('output_root', '')}")
            self._emit_progress(f"Variant: {self.params.get('variant', '')}")
            self._emit_progress(f"Seed: {self.params.get('seed', '')}")
            self._emit_progress(f"Fold index: {self.params.get('fold_index', '')}")
            self._emit_progress(f"Use dataset folds: {self.params.get('use_dataset_folds', '')}")
            self._emit_progress(f"Cast float32: {self.params.get('cast_float32', '')}")
            results = search_dcml_from_prepared_root(**self.params, progress_callback=self._emit_progress)
            self._emit_progress("DCML Search completed.")
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(self._emit_traceback())


class GenerateDataThread(_ProgressThread):
    """Generate DCML feature matrices from raw MPro-v2-like data without blocking the GUI."""

    finished_success = Signal(dict)
    finished_error = Signal(str)
    action_name = "generate_data"

    def run(self):
        try:
            self._emit_progress("DCML Generate Data started.")
            self._emit_progress(f"Raw dataset root: {self.params.get('raw_root', '')}")
            self._emit_progress(f"Output prepared root: {self.params.get('output_root', '')}")
            self._emit_progress(f"Variant: {self.params.get('variant', '')}")
            summary = generate_dcml_dataset_from_mpro_v2(**self.params, progress_callback=self._emit_progress)
            self._emit_progress("DCML Generate Data completed.")
            self.finished_success.emit(summary)
        except Exception:
            self.finished_error.emit(self._emit_traceback())
