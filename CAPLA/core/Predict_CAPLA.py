"""Run CAPLA inference on an external dataset already prepared in CAPLA format.

This script loads a trained ``CAPLA.pt`` bundle (or, for backwards
compatibility, the original ``best_model.pt`` state dict), rebuilds the clean
CAPLA model, runs batched inference, and generates the expected academic
artifacts:

- ``Predictions_CAPLA.csv``
- ``Metrics_CAPLA.csv``
- ``Scatter_CAPLA.png``

Examples
--------
Predict on an explicit external dataset::

    python TFM_Implementation/CAPLA/Predict_CAPLA.py \
        --model-pt TFM_Implementation/CAPLA/models/CAPLA.pt \
        --affinity-csv CAPLA/CAPLA/data/affinity_data.csv \
        --smi-csv CAPLA/CAPLA/data/Test2016_290_smi.csv \
        --global-dir CAPLA/CAPLA/data/Test2016_290/global \
        --pocket-dir CAPLA/CAPLA/data/Test2016_290/pocket \
        --device auto \
        --output-dir TFM_Implementation/CAPLA/outputs/predict_test2016_290

Predict and label the outputs as Split 00::

    python TFM_Implementation/CAPLA/Predict_CAPLA.py \
        --model-pt TFM_Implementation/CAPLA/models/CAPLA.pt \
        --affinity-csv CAPLA/CAPLA/data/affinity_data.csv \
        --smi-csv CAPLA/CAPLA/data/CSAR-HiQ_36_smi.csv \
        --global-dir CAPLA/CAPLA/data/CSAR-HiQ_36/global \
        --pocket-dir CAPLA/CAPLA/data/CSAR-HiQ_36/pocket \
        --split-id 00
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_HINT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
if str(_REPO_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_HINT))

from CAPLA.core.common import (  # noqa: E402
    CAPLAImplementationError,
    choose_device,
    ensure_dir,
    find_capla_repo_root,
    get_logger,
    resolve_path,
)
from CAPLA.core.data_utils import (  # noqa: E402
    CAPLADataError,
    CAPLADataset,
    DatasetPaths,
    ValidationReport,
    build_dataset_index,
)
from CAPLA.core.metrics_utils import (  # noqa: E402
    MetricsError,
    compute_regression_metrics,
    save_metrics_csv,
    save_predictions_csv,
    save_scatter_plot,
)
from CAPLA.core.model_adapter import build_capla_model, load_capla_checkpoint  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_OUTPUT_DIR = "TFM_Implementation/CAPLA/outputs/predict"


class PredictCAPLAError(CAPLAImplementationError):
    """Raised when prediction cannot complete safely."""


class PredictionArtifacts(object):
    def __init__(self, predictions_csv, metrics_csv, scatter_png):
        self.predictions_csv = predictions_csv
        self.metrics_csv = metrics_csv
        self.scatter_png = scatter_png


class PredictionSummary(object):
    def __init__(
        self,
        repo_root,
        device,
        model_path,
        dataset_name,
        split_id,
        dataset_paths,
        dataset_size,
        model_metadata_present,
        model_metadata_keys,
        validation_warnings,
        metrics,
        artifacts,
        inference_seconds,
        max_seq_len,
        max_pkt_len,
        max_smi_len,
    ):
        self.repo_root = repo_root
        self.device = device
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.split_id = split_id
        self.dataset_paths = dataset_paths
        self.dataset_size = dataset_size
        self.model_metadata_present = model_metadata_present
        self.model_metadata_keys = model_metadata_keys
        self.validation_warnings = validation_warnings
        self.metrics = metrics
        self.artifacts = artifacts
        self.inference_seconds = inference_seconds
        self.max_seq_len = max_seq_len
        self.max_pkt_len = max_pkt_len
        self.max_smi_len = max_smi_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAPLA inference on an external CAPLA-format dataset.")
    parser.add_argument("--model-pt", required=True, help="Path to CAPLA.pt or the original best_model.pt")
    parser.add_argument("--affinity-csv", default=None, help="Path to affinity_data.csv")
    parser.add_argument("--smi-csv", default=None, help="Path to the CAPLA *_smi.csv file")
    parser.add_argument("--global-dir", default=None, help="Directory containing global/*.csv files")
    parser.add_argument("--pocket-dir", default=None, help="Directory containing pocket/*.csv files")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--split-id", default=None, help="Optional split identifier, e.g. 00")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset label used in logs and outputs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for prediction artifacts")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers")
    return parser.parse_args()


def validate_cli_config(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise PredictCAPLAError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise PredictCAPLAError("--num-workers must be >= 0.")


def _get_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        class _NullContext:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _NullContext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _run_or_raise_oom(fn):
    try:
        return fn()
    except RuntimeError as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise PredictCAPLAError(
                "CUDA out of memory during prediction. Reduce --batch-size or use --device cpu."
            ) from exc
        raise


def infer_lengths_from_metadata(metadata: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Infer tensor lengths from the bundle metadata, with conservative fallbacks."""
    max_seq_len = int(metadata.get("max_seq_len", DEFAULT_MAX_SEQ_LEN))
    max_pkt_len = int(metadata.get("max_pkt_len", DEFAULT_MAX_PKT_LEN))
    max_smi_len = int(metadata.get("max_smi_len", DEFAULT_MAX_SMI_LEN))
    if max_seq_len < 1 or max_pkt_len < 1 or max_smi_len < 1:
        raise PredictCAPLAError("The checkpoint metadata contains invalid maximum lengths.")
    return max_seq_len, max_pkt_len, max_smi_len


def _metadata_dataset_paths(metadata: Mapping[str, Any]) -> Optional[DatasetPaths]:
    dataset_meta = metadata.get("dataset")
    if not isinstance(dataset_meta, ABCMapping):
        return None
    keys = ["affinity_csv", "smi_csv", "global_dir", "pocket_dir"]
    if not all(key in dataset_meta for key in keys):
        return None
    return DatasetPaths(
        affinity_csv=Path(str(dataset_meta["affinity_csv"])),
        smi_csv=Path(str(dataset_meta["smi_csv"])),
        global_dir=Path(str(dataset_meta["global_dir"])),
        pocket_dir=Path(str(dataset_meta["pocket_dir"])),
    )


def resolve_dataset_paths(
    args: argparse.Namespace,
    repo_root: Path,
    metadata: Mapping[str, Any],
) -> DatasetPaths:
    explicit = [args.affinity_csv, args.smi_csv, args.global_dir, args.pocket_dir]
    if all(value is not None for value in explicit):
        return DatasetPaths(
            affinity_csv=resolve_path(args.affinity_csv, base_dir=repo_root, must_exist=True),
            smi_csv=resolve_path(args.smi_csv, base_dir=repo_root, must_exist=True),
            global_dir=resolve_path(args.global_dir, base_dir=repo_root, must_exist=True),
            pocket_dir=resolve_path(args.pocket_dir, base_dir=repo_root, must_exist=True),
        )

    if any(value is not None for value in explicit):
        raise PredictCAPLAError(
            "Provide either all explicit dataset paths (--affinity-csv, --smi-csv, --global-dir, --pocket-dir) "
            "or none of them."
        )

    metadata_paths = _metadata_dataset_paths(metadata)
    if metadata_paths is None:
        raise PredictCAPLAError(
            "No dataset paths were provided and the checkpoint does not contain reusable dataset metadata."
        )
    return DatasetPaths(
        affinity_csv=resolve_path(metadata_paths.affinity_csv, base_dir=repo_root, must_exist=True),
        smi_csv=resolve_path(metadata_paths.smi_csv, base_dir=repo_root, must_exist=True),
        global_dir=resolve_path(metadata_paths.global_dir, base_dir=repo_root, must_exist=True),
        pocket_dir=resolve_path(metadata_paths.pocket_dir, base_dir=repo_root, must_exist=True),
    )


def infer_dataset_name(args: argparse.Namespace, dataset_paths: DatasetPaths) -> str:
    if args.dataset_name:
        return str(args.dataset_name)
    smi_name = dataset_paths.smi_csv.stem
    if smi_name.endswith("_smi"):
        smi_name = smi_name[: -len("_smi")]
    return smi_name or "dataset"


def _validation_warnings(report: ValidationReport) -> List[str]:
    warnings: List[str] = []
    if report.duplicated_affinity_ids:
        warnings.append(
            f"Duplicate affinity ids detected globally ({len(report.duplicated_affinity_ids)}); "
            "the first occurrence per pdbid was kept for prediction."
        )
    if report.duplicated_smi_ids:
        warnings.append(
            f"Duplicate SMILES ids detected ({len(report.duplicated_smi_ids)}); "
            "the first occurrence per pdbid was kept for prediction."
        )
    if report.affinity_null_rows or report.smi_null_rows:
        warnings.append(
            f"Null rows detected in metadata tables (affinity={report.affinity_null_rows}, smiles={report.smi_null_rows})."
        )
    if report.missing_global_ids or report.missing_pocket_ids:
        warnings.append("Some ids were missing global/pocket feature files and were excluded from prediction.")
    if report.invalid_global_files or report.invalid_pocket_files:
        warnings.append("Some feature files had an invalid schema and were excluded from prediction.")
    if report.unknown_smiles:
        warnings.append("Some ids contain unsupported SMILES characters and were excluded from prediction.")
    return warnings


def load_model_and_metadata(model_pt: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    try:
        model, metadata = load_capla_checkpoint(
            model_pt,
            model=build_capla_model(),
            map_location=device,
            strict=True,
        )
    except FileNotFoundError:
        raise
    except RuntimeError as exc:
        raise PredictCAPLAError(
            "The checkpoint is incompatible with the clean CAPLA architecture or is corrupted."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise PredictCAPLAError(f"Failed to load CAPLA checkpoint: {exc}") from exc

    if metadata and "model_name" in metadata and str(metadata["model_name"]).upper() != "CAPLA":
        raise PredictCAPLAError(
            f"Checkpoint model_name={metadata['model_name']!r} is not compatible with CAPLA prediction."
        )
    return model, dict(metadata)


@torch.no_grad()
def run_inference(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[float], List[float]]:
    model.eval()
    amp_enabled = device.type == "cuda"
    pdbids_all: List[str] = []
    y_true: List[float] = []
    y_pred: List[float] = []

    progress = tqdm(loader, desc="Predict", leave=False)
    for batch in progress:
        pdbids, seq_tensor, pkt_tensor, smi_tensor, affinity = batch
        seq_tensor = seq_tensor.to(device)
        pkt_tensor = pkt_tensor.to(device)
        smi_tensor = smi_tensor.to(device)
        affinity = affinity.to(device)

        def _forward():
            with _get_autocast_context(device=device, enabled=amp_enabled):
                return model(seq_tensor, pkt_tensor, smi_tensor).view(-1)

        output = _run_or_raise_oom(_forward)
        if not torch.isfinite(output).all():
            raise PredictCAPLAError("The model produced NaN or Inf values during inference.")
        if output.numel() != affinity.numel():
            raise PredictCAPLAError(
                f"Prediction shape mismatch: got {tuple(output.shape)} for targets with shape {tuple(affinity.shape)}."
            )

        pdbids_all.extend([str(item) for item in pdbids])
        y_true.extend(affinity.detach().cpu().view(-1).tolist())
        y_pred.extend(output.detach().cpu().view(-1).tolist())

    if not pdbids_all:
        raise PredictCAPLAError("The prediction loader produced zero samples.")
    return pdbids_all, y_true, y_pred


def build_output_paths(output_dir: Path, split_id: Optional[str]) -> Tuple[Path, Path, Path]:
    suffix = "" if not split_id else f"_split{split_id}"
    predictions_csv = output_dir / f"Predictions_CAPLA{suffix}.csv"
    metrics_csv = output_dir / f"Metrics_CAPLA{suffix}.csv"
    scatter_png = output_dir / f"Scatter_CAPLA{suffix}.png"
    return predictions_csv, metrics_csv, scatter_png


def run_prediction(args: argparse.Namespace) -> PredictionSummary:
    logger = get_logger("TFM_CAPLA_PREDICT")
    repo_root = find_capla_repo_root(__file__)
    validate_cli_config(args)

    device = choose_device(args.device)
    logger.info("Using device: %s", device)

    model_path = resolve_path(args.model_pt, base_dir=repo_root, must_exist=True)
    model, metadata = load_model_and_metadata(model_path, device)
    max_seq_len, max_pkt_len, max_smi_len = infer_lengths_from_metadata(metadata)
    dataset_paths = resolve_dataset_paths(args, repo_root, metadata)
    dataset_name = infer_dataset_name(args, dataset_paths)
    output_dir = ensure_dir(resolve_path(args.output_dir, base_dir=repo_root, must_exist=False))

    logger.info(
        "Dataset paths resolved: %s",
        json.dumps({key: str(value) for key, value in dataset_paths.__dict__.items()}, ensure_ascii=False),
    )

    index_df, validation_report = build_dataset_index(
        dataset_paths.affinity_csv,
        dataset_paths.smi_csv,
        dataset_paths.global_dir,
        dataset_paths.pocket_dir,
        strict=False,
    )
    validation_warnings = _validation_warnings(validation_report)
    for warning in validation_warnings:
        logger.warning(warning)

    if len(index_df) == 0:
        raise PredictCAPLAError("No valid samples remain after dataset validation.")

    dataset = CAPLADataset(
        index_df=index_df,
        max_seq_len=max_seq_len,
        max_pkt_len=max_pkt_len,
        max_smi_len=max_smi_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    started_at = time.time()
    pdbids, y_true, y_pred = run_inference(model=model, loader=loader, device=device)
    metrics = compute_regression_metrics(y_true, y_pred)
    predictions_csv, metrics_csv, scatter_png = build_output_paths(output_dir, args.split_id)

    save_predictions_csv(pdbids, y_true, y_pred, predictions_csv)
    save_metrics_csv(metrics, metrics_csv, std_overrides={name: 0.0 for name in metrics})
    save_scatter_plot(
        y_true,
        y_pred,
        scatter_png,
        model_name="CAPLA",
        split_id=args.split_id,
        metrics=metrics,
    )

    return PredictionSummary(
        repo_root=str(repo_root),
        device=str(device),
        model_path=str(model_path),
        dataset_name=dataset_name,
        split_id=args.split_id,
        dataset_paths={key: str(value) for key, value in dataset_paths.__dict__.items()},
        dataset_size=len(index_df),
        model_metadata_present=bool(metadata),
        model_metadata_keys=sorted(str(key) for key in metadata.keys()),
        validation_warnings=validation_warnings,
        metrics={name: float(value) for name, value in metrics.items()},
        artifacts=PredictionArtifacts(
            predictions_csv=str(predictions_csv),
            metrics_csv=str(metrics_csv),
            scatter_png=str(scatter_png),
        ),
        inference_seconds=float(time.time() - started_at),
        max_seq_len=max_seq_len,
        max_pkt_len=max_pkt_len,
        max_smi_len=max_smi_len,
    )


def print_summary(summary: PredictionSummary) -> None:
    print("CAPLA prediction finished")
    print("=" * 80)
    print(f"Device               : {summary.device}")
    print(f"Model                : {summary.model_path}")
    print(f"Dataset              : {summary.dataset_name}")
    print(f"Split                : {summary.split_id if summary.split_id is not None else 'None'}")
    print(f"Samples predicted    : {summary.dataset_size}")
    print(
        "Lengths              : "
        f"max_seq_len={summary.max_seq_len} | "
        f"max_pkt_len={summary.max_pkt_len} | "
        f"max_smi_len={summary.max_smi_len}"
    )
    print(
        "Metrics              : "
        f"RMSE={summary.metrics['RMSE']:.4f} | "
        f"Pearson={summary.metrics['Pearson']:.4f} | "
        f"MAE={summary.metrics['MAE']:.4f} | "
        f"SD={summary.metrics['SD']:.4f}"
    )
    print(f"Inference seconds    : {summary.inference_seconds:.2f}")
    if summary.validation_warnings:
        print("Validation warnings  :")
        for warning in summary.validation_warnings:
            print(f"  - {warning}")
    print("Artifacts            :")
    print(f"  - predictions_csv: {summary.artifacts.predictions_csv}")
    print(f"  - metrics_csv    : {summary.artifacts.metrics_csv}")
    print(f"  - scatter_png    : {summary.artifacts.scatter_png}")


def main() -> int:
    args = parse_args()
    logger = get_logger("TFM_CAPLA_PREDICT")
    try:
        summary = run_prediction(args)
        print_summary(summary)
        return 0
    except (CAPLAImplementationError, CAPLADataError, MetricsError, FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.error("Prediction interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
