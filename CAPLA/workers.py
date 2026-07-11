"""
@file workers.py
@author Mohamed EL BOUKHIARI
@brief Background subprocess workers used by the CAPLA GUI module.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from CAPLA.core.generate_capla_from_mpro_v2 import generate_capla_dataset_from_mpro_v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute_project_path(value: str | Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


class CAPLASubprocessThread(QThread):
    """Execute one CAPLA CLI command without blocking the GUI."""

    log_line = Signal(str)
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = dict(params)
        self._process: subprocess.Popen[str] | None = None

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def artifact_paths(self) -> dict[str, str]:
        return {}

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def run(self) -> None:
        try:
            command = self.build_command()
            printable_command = shlex.join(command)
            self.log_line.emit(f"[CAPLA] Starting command:\n{printable_command}")

            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("MPLBACKEND", "Agg")

            self._process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            captured_lines: list[str] = []
            assert self._process.stdout is not None
            for raw_line in self._process.stdout:
                line = raw_line.rstrip("\n")
                captured_lines.append(line)
                self.log_line.emit(line)

            return_code = self._process.wait()
            output = "\n".join(captured_lines)
            if return_code != 0:
                self.finished_error.emit(
                    "CAPLA subprocess failed.\n\n"
                    f"Return code: {return_code}\n\n"
                    f"Command:\n{printable_command}\n\n"
                    f"Output:\n{output}"
                )
                return

            self.finished_success.emit(
                {
                    "status": "success",
                    "command": printable_command,
                    "cwd": str(PROJECT_ROOT),
                    "artifacts": self.artifact_paths(),
                    "output": output,
                }
            )
        except Exception:
            self.finished_error.emit(traceback.format_exc())
        finally:
            self._process = None


class GenerateDataThread(QThread):
    """Generate a CAPLA prepared dataset from raw MPro-URV_Version2."""

    log_line = Signal(str)
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = dict(params)

    def run(self) -> None:
        try:
            raw_root = _absolute_project_path(self.params["raw_root"])
            output_root = _absolute_project_path(self.params["output_root"])
            feature_source_root = self.params.get("feature_source_root") or None
            if feature_source_root:
                feature_source_root = _absolute_project_path(feature_source_root)

            self.log_line.emit(f"[CAPLA] Generating prepared dataset from raw MPro root: {raw_root}")
            report = generate_capla_dataset_from_mpro_v2(
                raw_root=raw_root,
                output_root=output_root,
                overwrite=bool(self.params.get("overwrite", False)),
                pocket_cutoff=float(self.params.get("pocket_cutoff", 4.5)),
                secondary_structure_mode=self.params.get("secondary_structure_mode", "dssp"),
                feature_mode=self.params.get("feature_mode", "generate"),
                feature_source_root=feature_source_root,
            )
            self.finished_success.emit(report)
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class PrepareOfficialDatasetThread(CAPLASubprocessThread):
    """Prepare URV v3b CSV files and official splits from existing CAPLA features."""

    def build_command(self) -> list[str]:
        command = [sys.executable, "-m", "CAPLA.core.Prepare_URV_V3B_CAPLA_Dataset"]
        _append_option(command, "--urv-v3b-dir", _absolute_project_path(self.params["urv_v3b_dir"]))
        _append_option(command, "--source-dataset-dir", _absolute_project_path(self.params["source_dataset_dir"]))
        _append_option(command, "--out-dir", _absolute_project_path(self.params["out_dir"]))
        _append_option(command, "--feature-mode", self.params["feature_mode"])
        return command

    def artifact_paths(self) -> dict[str, str]:
        out_dir = Path(_absolute_project_path(self.params["out_dir"]))
        return {
            "prepared_dataset_dir": str(out_dir),
            "prepare_report_json": str(out_dir / "reports" / "prepare_report.json"),
            "split_manifest_csv": str(out_dir / "split_manifest.csv"),
        }


class DebugOfficialDatasetThread(CAPLASubprocessThread):
    """Run the lightweight official-split CAPLA debug mode."""

    def build_command(self) -> list[str]:
        command = [sys.executable, "-m", "CAPLA.core.Run_URV_V3B_5Splits_CAPLA", "--mode", "debug"]
        _append_option(command, "--dataset-dir", _absolute_project_path(self.params["dataset_dir"]))
        _append_option(command, "--output-dir", _absolute_project_path(self.params["output_dir"]))
        _append_option(command, "--device", self.params["device"])
        _append_option(command, "--batch-size", self.params["batch_size"])
        return command

    def artifact_paths(self) -> dict[str, str]:
        return {
            "debug_report_json": str(
                Path(_absolute_project_path(self.params["output_dir"])) / "debug" / "debug_report.json"
            )
        }


class TrainOfficialSplitsThread(CAPLASubprocessThread):
    """Train fresh CAPLA models on one or more official splits."""

    def build_command(self) -> list[str]:
        command = [sys.executable, "-m", "CAPLA.core.Run_URV_V3B_5Splits_CAPLA", "--mode", "all"]
        for flag, key in (
            ("--dataset-dir", "dataset_dir"),
            ("--output-dir", "output_dir"),
            ("--models-dir", "models_dir"),
        ):
            _append_option(command, flag, _absolute_project_path(self.params[key]))
        for flag, key in (
            ("--splits", "splits"),
            ("--device", "device"),
            ("--epochs", "epochs"),
            ("--batch-size", "batch_size"),
            ("--lr", "lr"),
            ("--weight-decay", "weight_decay"),
            ("--early-stopping-rounds", "early_stopping_rounds"),
            ("--min-epochs-before-stopping", "min_epochs_before_stopping"),
            ("--min-delta", "min_delta"),
            ("--seed", "seed"),
            ("--num-workers", "num_workers"),
        ):
            _append_option(command, flag, self.params[key])
        if self.params.get("disable_amp", False):
            command.append("--disable-amp")
        return command

    def artifact_paths(self) -> dict[str, str]:
        output_dir = Path(_absolute_project_path(self.params["output_dir"]))
        return {
            "summary_csv": str(output_dir / "Summary_5splits.csv"),
            "aggregate_json": str(output_dir / "Aggregate_metrics.json"),
            "models_dir": _absolute_project_path(self.params["models_dir"]),
        }


class HyperparameterSearchCAPLAThread(CAPLASubprocessThread):
    """Run CAPLA HPO on official train/validation subsets only."""

    def build_command(self) -> list[str]:
        command = [sys.executable, "-m", "CAPLA.core.capla_hyperparameter_search"]
        _append_option(command, "--dataset-dir", _absolute_project_path(self.params["dataset_dir"]))
        _append_option(command, "--models-root", _absolute_project_path(self.params["models_root"]))
        _append_option(command, "--results-root", _absolute_project_path(self.params["results_root"]))
        for flag, key in (
            ("--splits", "splits"),
            ("--device", "device"),
            ("--epochs", "epochs"),
            ("--early-stopping-rounds", "early_stopping_rounds"),
            ("--min-epochs-before-stopping", "min_epochs_before_stopping"),
            ("--min-delta", "min_delta"),
            ("--seed", "seed"),
            ("--num-workers", "num_workers"),
        ):
            _append_option(command, flag, self.params[key])
        command.extend(["--lr-values", ",".join(str(v) for v in self.params["lr_values"])])
        command.extend(["--batch-size-values", ",".join(str(v) for v in self.params["batch_size_values"])])
        command.extend(["--weight-decay-values", ",".join(str(v) for v in self.params["weight_decay_values"])])
        if self.params.get("disable_amp", False):
            command.append("--disable-amp")
        return command

    def artifact_paths(self) -> dict[str, str]:
        results_root = Path(_absolute_project_path(self.params["results_root"]))
        return {
            "latest_run_json": str(results_root / "latest_run.json"),
            "results_root": str(results_root),
            "models_root": _absolute_project_path(self.params["models_root"]),
        }


class PredictPreparedDatasetThread(CAPLASubprocessThread):
    """Evaluate one CAPLA checkpoint on the prepared dataset."""

    def build_command(self) -> list[str]:
        dataset_dir = Path(_absolute_project_path(self.params["dataset_dir"]))
        command = [sys.executable, "-m", "CAPLA.core.Predict_CAPLA"]
        _append_option(command, "--model-pt", _absolute_project_path(self.params["model_pt"]))
        _append_option(command, "--affinity-csv", str(dataset_dir / "affinity_data.csv"))
        _append_option(command, "--smi-csv", str(dataset_dir / "urv_v3b_smi.csv"))
        _append_option(command, "--global-dir", str(dataset_dir / "global"))
        _append_option(command, "--pocket-dir", str(dataset_dir / "pocket"))
        _append_option(command, "--output-dir", _absolute_project_path(self.params["output_dir"]))
        _append_option(command, "--device", self.params["device"])
        _append_option(command, "--batch-size", self.params["batch_size"])
        _append_option(command, "--num-workers", self.params["num_workers"])
        _append_option(command, "--dataset-name", self.params.get("dataset_name"))
        _append_option(command, "--split-id", self.params.get("split_id"))
        return command

    def artifact_paths(self) -> dict[str, str]:
        output_dir = Path(_absolute_project_path(self.params["output_dir"]))
        suffix = "" if not self.params.get("split_id") else f"_split{self.params['split_id']}"
        return {
            "predictions_csv": str(output_dir / f"Predictions_CAPLA{suffix}.csv"),
            "metrics_csv": str(output_dir / f"Metrics_CAPLA{suffix}.csv"),
            "scatter_png": str(output_dir / f"Scatter_CAPLA{suffix}.png"),
        }


class PretrainedVsFinetunedThread(CAPLASubprocessThread):
    """Compare original pretrained CAPLA against split-specific fine-tuning."""

    def build_command(self) -> list[str]:
        command = [sys.executable, "-m", "CAPLA.core.Run_CAPLA_Pretrained_vs_Finetuned_URV_V3B", "--mode", "all"]
        for flag, key in (
            ("--dataset-dir", "dataset_dir"),
            ("--checkpoint-original", "checkpoint_original"),
            ("--results-dir", "results_dir"),
            ("--models-dir", "models_dir"),
        ):
            _append_option(command, flag, _absolute_project_path(self.params[key]))
        for flag, key in (
            ("--splits", "splits"),
            ("--device", "device"),
            ("--epochs", "epochs"),
            ("--batch-size", "batch_size"),
            ("--lr", "lr"),
            ("--weight-decay", "weight_decay"),
            ("--early-stopping-rounds", "early_stopping_rounds"),
            ("--min-epochs-before-stopping", "min_epochs_before_stopping"),
            ("--min-delta", "min_delta"),
            ("--grad-clip-norm", "grad_clip_norm"),
            ("--seed", "seed"),
            ("--num-workers", "num_workers"),
        ):
            _append_option(command, flag, self.params[key])
        return command

    def artifact_paths(self) -> dict[str, str]:
        results_dir = Path(_absolute_project_path(self.params["results_dir"]))
        return {
            "comparison_csv": str(results_dir / "Comparison_per_split.csv"),
            "aggregate_json": str(results_dir / "Comparison_aggregate.json"),
            "report_md": str(results_dir / "Comparison_report.md"),
            "models_dir": _absolute_project_path(self.params["models_dir"]),
        }
