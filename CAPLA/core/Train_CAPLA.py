"""Train the clean CAPLA implementation and save a self-contained ``CAPLA.pt`` bundle.

This script replaces the original phase-based CAPLA training entrypoint with an
explicit-path interface that works on any dataset already prepared in CAPLA
format.

Examples
--------
Train with explicit dataset paths::

    python TFM_Implementation/CAPLA/Train_CAPLA.py \
        --affinity-csv CAPLA/CAPLA/data/affinity_data.csv \
        --smi-csv CAPLA/CAPLA/data/CSAR-HiQ_36_smi.csv \
        --global-dir CAPLA/CAPLA/data/CSAR-HiQ_36/global \
        --pocket-dir CAPLA/CAPLA/data/CSAR-HiQ_36/pocket \
        --device auto \
        --epochs 20 \
        --batch-size 32

Train from repository defaults::

    python TFM_Implementation/CAPLA/Train_CAPLA.py \
        --output-dir TFM_Implementation/CAPLA/outputs/train_run_01 \
        --output-model TFM_Implementation/CAPLA/models/CAPLA.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
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
    save_json,
    seed_everything,
)
from CAPLA.core.data_utils import (  # noqa: E402
    CHAR_SMI_SET,
    CAPLADataError,
    CAPLADataset,
    DatasetPaths,
    ValidationReport,
    build_dataset_index,
)
from CAPLA.core.metrics_utils import compute_regression_metrics  # noqa: E402
from CAPLA.core.model_adapter import build_capla_model, save_capla_bundle  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_NUM_WORKERS = 0
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_EARLY_STOPPING_ROUNDS = 8


class TrainCAPLAError(CAPLAImplementationError):
    """Raised when training cannot complete safely."""


class SplitSummary(object):
    def __init__(self, train_count, val_count, holdout_count, train_ratio_effective, val_ratio_effective, holdout_ratio_effective):
        self.train_count = train_count
        self.val_count = val_count
        self.holdout_count = holdout_count
        self.train_ratio_effective = train_ratio_effective
        self.val_ratio_effective = val_ratio_effective
        self.holdout_ratio_effective = holdout_ratio_effective


class EpochRecord(object):
    def __init__(
        self,
        epoch,
        train_loss,
        train_rmse,
        train_pearson,
        train_mae,
        train_sd,
        val_loss,
        val_rmse,
        val_pearson,
        val_mae,
        val_sd,
        epoch_seconds,
        learning_rate,
        is_best,
    ):
        self.epoch = epoch
        self.train_loss = train_loss
        self.train_rmse = train_rmse
        self.train_pearson = train_pearson
        self.train_mae = train_mae
        self.train_sd = train_sd
        self.val_loss = val_loss
        self.val_rmse = val_rmse
        self.val_pearson = val_pearson
        self.val_mae = val_mae
        self.val_sd = val_sd
        self.epoch_seconds = epoch_seconds
        self.learning_rate = learning_rate
        self.is_best = is_best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAPLA and save a clean CAPLA.pt bundle.")
    parser.add_argument("--affinity-csv", default=None, help="Path to affinity_data.csv")
    parser.add_argument("--smi-csv", default=None, help="Path to the CAPLA *_smi.csv file")
    parser.add_argument("--global-dir", default=None, help="Directory containing global/*.csv files")
    parser.add_argument("--pocket-dir", default=None, help="Directory containing pocket/*.csv files")
    parser.add_argument(
        "--output-model",
        default="TFM_Implementation/CAPLA/models/CAPLA.pt",
        help="Final model bundle path",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Stop after this many non-improving validation epochs",
    )
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Maximum sequence length")
    parser.add_argument("--max-pkt-len", type=int, default=DEFAULT_MAX_PKT_LEN, help="Maximum pocket length")
    parser.add_argument("--max-smi-len", type=int, default=DEFAULT_MAX_SMI_LEN, help="Maximum SMILES length")
    parser.add_argument(
        "--output-dir",
        default="TFM_Implementation/CAPLA/outputs/train",
        help="Directory for training artifacts",
    )
    return parser.parse_args()


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


def _build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _candidate_dataset_names(data_dir: Path) -> List[str]:
    names: List[str] = []
    for smi_csv in sorted(data_dir.glob("*_smi.csv")):
        name = smi_csv.name[: -len("_smi.csv")]
        if (data_dir / name / "global").is_dir() and (data_dir / name / "pocket").is_dir():
            names.append(name)
    return names


def infer_default_dataset_paths(repo_root: Path) -> Tuple[DatasetPaths, Dict[str, Any]]:
    """Infer a sensible default training dataset from the repository.

    Preference order:
    1. Original `training` interface if present.
    2. Largest available `CSAR-HiQ_*` dataset.
    3. Largest available non-`Test*` dataset.
    4. Largest available dataset overall.
    """
    data_dir = repo_root / "CAPLA" / "CAPLA" / "data"
    if not data_dir.exists():
        raise TrainCAPLAError(
            f"Default CAPLA data directory not found: {data_dir}. Provide explicit dataset paths."
        )

    training_candidate = DatasetPaths(
        affinity_csv=data_dir / "affinity_data.csv",
        smi_csv=data_dir / "training_smi.csv",
        global_dir=data_dir / "training" / "global",
        pocket_dir=data_dir / "training" / "pocket",
    )
    if (
        training_candidate.affinity_csv.exists()
        and training_candidate.smi_csv.exists()
        and training_candidate.global_dir.is_dir()
        and training_candidate.pocket_dir.is_dir()
    ):
        return training_candidate, {"strategy": "explicit_training_interface", "dataset_name": "training"}

    candidates = _candidate_dataset_names(data_dir)
    if not candidates:
        raise TrainCAPLAError(
            f"No default CAPLA datasets were found in {data_dir}. Provide explicit dataset paths."
        )

    sizes: Dict[str, int] = {}
    for name in candidates:
        try:
            sizes[name] = int(len(pd.read_csv(data_dir / f"{name}_smi.csv")))
        except Exception:
            sizes[name] = 0

    ranked: List[str] = []
    ranked.extend(sorted([n for n in candidates if n.startswith("CSAR-HiQ_")], key=lambda n: (-sizes[n], n)))
    ranked.extend(sorted([n for n in candidates if not n.startswith("Test") and n not in ranked], key=lambda n: (-sizes[n], n)))
    ranked.extend(sorted([n for n in candidates if n not in ranked], key=lambda n: (-sizes[n], n)))
    chosen = ranked[0]

    return (
        DatasetPaths(
            affinity_csv=data_dir / "affinity_data.csv",
            smi_csv=data_dir / f"{chosen}_smi.csv",
            global_dir=data_dir / chosen / "global",
            pocket_dir=data_dir / chosen / "pocket",
        ),
        {
            "strategy": "fallback_to_available_dataset",
            "dataset_name": chosen,
            "candidate_sizes": sizes,
        },
    )


def resolve_dataset_paths(args: argparse.Namespace, repo_root: Path) -> Tuple[DatasetPaths, Dict[str, Any]]:
    defaults, default_meta = infer_default_dataset_paths(repo_root)
    return (
        DatasetPaths(
            affinity_csv=resolve_path(args.affinity_csv or defaults.affinity_csv, base_dir=repo_root, must_exist=True),
            smi_csv=resolve_path(args.smi_csv or defaults.smi_csv, base_dir=repo_root, must_exist=True),
            global_dir=resolve_path(args.global_dir or defaults.global_dir, base_dir=repo_root, must_exist=True),
            pocket_dir=resolve_path(args.pocket_dir or defaults.pocket_dir, base_dir=repo_root, must_exist=True),
        ),
        default_meta,
    )


def validate_cli_config(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise TrainCAPLAError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise TrainCAPLAError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise TrainCAPLAError("--num-workers must be >= 0.")
    if args.early_stopping_rounds < 1:
        raise TrainCAPLAError("--early-stopping-rounds must be >= 1.")
    if args.max_seq_len < 1 or args.max_pkt_len < 1 or args.max_smi_len < 1:
        raise TrainCAPLAError("All maximum length arguments must be >= 1.")
    if not (0.0 < args.train_ratio < 1.0):
        raise TrainCAPLAError("--train-ratio must be in the open interval (0, 1).")
    if not (0.0 < args.val_ratio < 1.0):
        raise TrainCAPLAError("--val-ratio must be in the open interval (0, 1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise TrainCAPLAError("--train-ratio + --val-ratio must be strictly smaller than 1.0.")


def split_indices(num_items: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SplitSummary]:
    if num_items < 2:
        raise TrainCAPLAError("At least two valid samples are required to create a train/validation split.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_items)
    train_count = max(1, int(round(num_items * train_ratio)))
    val_count = max(1, int(round(num_items * val_ratio)))

    if train_count + val_count >= num_items:
        overflow = train_count + val_count - (num_items - 1)
        if overflow > 0 and val_count > 1:
            reduce_val = min(overflow, val_count - 1)
            val_count -= reduce_val
            overflow -= reduce_val
        if overflow > 0 and train_count > 1:
            train_count -= min(overflow, train_count - 1)
        if train_count + val_count >= num_items:
            raise TrainCAPLAError("Dataset is too small for the requested split ratios.")

    holdout_count = num_items - train_count - val_count
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    holdout_idx = indices[train_count + val_count :]
    summary = SplitSummary(
        train_count=train_count,
        val_count=val_count,
        holdout_count=holdout_count,
        train_ratio_effective=train_count / num_items,
        val_ratio_effective=val_count / num_items,
        holdout_ratio_effective=holdout_count / num_items,
    )
    return train_idx, val_idx, holdout_idx, summary


def _move_batch_to_device(
    batch: Tuple[Sequence[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pdbids, seq_tensor, pkt_tensor, smi_tensor, affinity = batch
    return list(pdbids), seq_tensor.to(device), pkt_tensor.to(device), smi_tensor.to(device), affinity.to(device)


def _run_or_raise_oom(fn):
    try:
        return fn()
    except RuntimeError as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise TrainCAPLAError(
                "CUDA out of memory during training. Reduce --batch-size, reduce the maximum lengths, or use --device cpu."
            ) from exc
        raise


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    y_true: List[float] = []
    y_pred: List[float] = []

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        _, seq_tensor, pkt_tensor, smi_tensor, affinity = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        def _forward():
            with _get_autocast_context(device=device, enabled=amp_enabled):
                output = model(seq_tensor, pkt_tensor, smi_tensor).view(-1)
                loss = loss_fn(output, affinity.view(-1))
            return output, loss

        output, loss = _run_or_raise_oom(_forward)
        _run_or_raise_oom(lambda: scaler.scale(loss).backward())
        _run_or_raise_oom(lambda: scaler.step(optimizer))
        scaler.update()

        batch_size = int(affinity.numel())
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        y_true.extend(affinity.detach().cpu().view(-1).tolist())
        y_pred.extend(output.detach().cpu().view(-1).tolist())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: List[float] = []
    y_pred: List[float] = []

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        _, seq_tensor, pkt_tensor, smi_tensor, affinity = _move_batch_to_device(batch, device)

        def _forward():
            with _get_autocast_context(device=device, enabled=amp_enabled):
                output = model(seq_tensor, pkt_tensor, smi_tensor).view(-1)
                loss = loss_fn(output, affinity.view(-1))
            return output, loss

        output, loss = _run_or_raise_oom(_forward)
        batch_size = int(affinity.numel())
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        y_true.extend(affinity.detach().cpu().view(-1).tolist())
        y_pred.extend(output.detach().cpu().view(-1).tolist())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def build_dataloaders(
    *,
    dataset: CAPLADataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _validation_warnings(report: ValidationReport) -> List[str]:
    warnings: List[str] = []
    if report.duplicated_affinity_ids:
        warnings.append(
            f"Duplicate affinity ids detected globally ({len(report.duplicated_affinity_ids)}); "
            "the first occurrence per pdbid was kept for training."
        )
    if report.duplicated_smi_ids:
        warnings.append(
            f"Duplicate SMILES ids detected ({len(report.duplicated_smi_ids)}); "
            "the first occurrence per pdbid was kept for training."
        )
    if report.missing_global_ids or report.missing_pocket_ids:
        warnings.append("Some ids were missing global/pocket feature files and were excluded from training.")
    if report.invalid_global_files or report.invalid_pocket_files:
        warnings.append("Some feature files had an invalid schema and were excluded from training.")
    if report.unknown_smiles:
        warnings.append("Some ids contain unsupported SMILES characters and were excluded from training.")
    return warnings


def write_training_outputs(
    *,
    output_dir: Path,
    history: Sequence[EpochRecord],
    best_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, str]:
    ensure_dir(output_dir)
    history_path = output_dir / "train_history.csv"
    best_path = output_dir / "best_val_metrics.json"
    config_path = output_dir / "training_config.json"

    pd.DataFrame([item.__dict__.copy() for item in history]).to_csv(history_path, index=False)
    save_json(dict(best_metrics), best_path)
    save_json(dict(config), config_path)
    return {
        "train_history_csv": str(history_path),
        "best_val_metrics_json": str(best_path),
        "training_config_json": str(config_path),
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    logger = get_logger("TFM_CAPLA_TRAIN")
    repo_root = find_capla_repo_root(__file__)
    validate_cli_config(args)
    seed_everything(args.seed)

    device = choose_device(args.device)
    logger.info("Using device: %s", device)

    output_dir = ensure_dir(resolve_path(args.output_dir, base_dir=repo_root, must_exist=False))
    output_model = resolve_path(args.output_model, base_dir=repo_root, must_exist=False)
    dataset_paths, default_resolution = resolve_dataset_paths(args, repo_root)
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
    for message in validation_warnings:
        logger.warning(message)

    if len(index_df) < 2:
        raise TrainCAPLAError("Need at least two valid complexes after validation to train CAPLA.")

    dataset = CAPLADataset(
        index_df=index_df,
        max_seq_len=args.max_seq_len,
        max_pkt_len=args.max_pkt_len,
        max_smi_len=args.max_smi_len,
    )
    train_idx, val_idx, holdout_idx, split_summary = split_indices(
        num_items=len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader, val_loader = build_dataloaders(
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_capla_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")
    amp_enabled = device.type == "cuda"
    scaler = _build_grad_scaler(enabled=amp_enabled)

    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[EpochRecord] = []
    train_started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            epoch=epoch,
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"Val {epoch}",
        )

        is_best = float(val_metrics["loss"]) < best_val_loss
        if is_best:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = EpochRecord(
            epoch=epoch,
            train_loss=float(train_metrics["loss"]),
            train_rmse=float(train_metrics["RMSE"]),
            train_pearson=float(train_metrics["Pearson"]),
            train_mae=float(train_metrics["MAE"]),
            train_sd=float(train_metrics["SD"]),
            val_loss=float(val_metrics["loss"]),
            val_rmse=float(val_metrics["RMSE"]),
            val_pearson=float(val_metrics["Pearson"]),
            val_mae=float(val_metrics["MAE"]),
            val_sd=float(val_metrics["SD"]),
            epoch_seconds=float(time.time() - epoch_started_at),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            is_best=is_best,
        )
        history.append(record)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_rmse=%.4f%s",
            epoch,
            args.epochs,
            record.train_loss,
            record.val_loss,
            record.val_rmse,
            " [best]" if is_best else "",
        )

        if epochs_without_improvement >= args.early_stopping_rounds:
            logger.info(
                "Early stopping triggered after %d non-improving validation epoch(s).",
                epochs_without_improvement,
            )
            break

    if best_state_dict is None:
        raise TrainCAPLAError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state_dict)
    best_record = next(item.__dict__.copy() for item in history if item.epoch == best_epoch)

    metadata = {
        "model_name": "CAPLA",
        "repo_variant": "CAPLA/CAPLA/src",
        "state_dict_format": "clean_capla_bundle",
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "max_pkt_len": args.max_pkt_len,
        "max_smi_len": args.max_smi_len,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "early_stopping_rounds": args.early_stopping_rounds,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "num_workers": args.num_workers,
            "optimizer": "AdamW",
            "loss": "MSELoss(mean)",
            "amp_enabled": amp_enabled,
        },
        "smiles_vocabulary": CHAR_SMI_SET,
        "dataset": {
            "affinity_csv": str(dataset_paths.affinity_csv),
            "smi_csv": str(dataset_paths.smi_csv),
            "global_dir": str(dataset_paths.global_dir),
            "pocket_dir": str(dataset_paths.pocket_dir),
            "num_valid_samples": int(len(index_df)),
            "affinity_column": validation_report.affinity_column,
            "default_resolution": default_resolution,
            "validation_report": validation_report.to_dict(),
            "validation_warnings": validation_warnings,
        },
        "split_summary": split_summary.__dict__.copy(),
        "best_epoch": best_epoch,
        "best_val_metrics": best_record,
        "compatibility": {
            "loadable_by": ["model_adapter.load_capla_checkpoint", "Predict_CAPLA.py"],
            "compatible_with_original_state_dict": True,
        },
        "inference_defaults": {
            "predictions_filename": "Predictions_CAPLA.csv",
            "metrics_filename": "Metrics_CAPLA.csv",
            "scatter_filename": "Scatter_CAPLA.png",
        },
        "timing": {
            "total_training_seconds": float(time.time() - train_started_at),
        },
    }

    ensure_dir(output_model.parent)
    saved_model = save_capla_bundle(model, output_model, metadata=metadata)

    training_config = {
        "repo_root": str(repo_root),
        "device_requested": args.device,
        "device_selected": str(device),
        "dataset_paths": {key: str(value) for key, value in dataset_paths.__dict__.items()},
        "default_dataset_resolution": default_resolution,
        "arguments": vars(args),
        "split_summary": split_summary.__dict__.copy(),
        "validation_warnings": validation_warnings,
        "holdout_count_unused": int(len(holdout_idx)),
        "output_model": str(saved_model),
    }
    auxiliary_outputs = write_training_outputs(
        output_dir=output_dir,
        history=history,
        best_metrics=best_record,
        config=training_config,
    )

    return {
        "repo_root": str(repo_root),
        "device": str(device),
        "output_model": str(saved_model),
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_metrics": best_record,
        "split_summary": split_summary.__dict__.copy(),
        "dataset_paths": {key: str(value) for key, value in dataset_paths.__dict__.items()},
        "default_dataset_resolution": default_resolution,
        "validation_warnings": validation_warnings,
        "auxiliary_outputs": auxiliary_outputs,
    }


def print_summary(result: Mapping[str, Any]) -> None:
    print("CAPLA training finished")
    print("=" * 80)
    print(f"Device               : {result['device']}")
    print(f"Output model         : {result['output_model']}")
    print(f"Output dir           : {result['output_dir']}")
    print(f"Best epoch           : {result['best_epoch']}")
    print(f"Epochs completed     : {result['epochs_completed']}")
    split_summary = result["split_summary"]
    print(
        "Split sizes          : "
        f"train={split_summary['train_count']} | "
        f"val={split_summary['val_count']} | "
        f"holdout={split_summary['holdout_count']}"
    )
    best = result["best_val_metrics"]
    print(
        "Best validation      : "
        f"loss={best['val_loss']:.4f} | "
        f"RMSE={best['val_rmse']:.4f} | "
        f"Pearson={best['val_pearson']:.4f} | "
        f"MAE={best['val_mae']:.4f} | "
        f"SD={best['val_sd']:.4f}"
    )
    if result["validation_warnings"]:
        print("Validation warnings  :")
        for warning in result["validation_warnings"]:
            print(f"  - {warning}")
    print("Auxiliary outputs    :")
    for key, value in result["auxiliary_outputs"].items():
        print(f"  - {key}: {value}")


def main() -> int:
    args = parse_args()
    logger = get_logger("TFM_CAPLA_TRAIN")
    try:
        result = run_training(args)
        print_summary(result)
        return 0
    except (CAPLAImplementationError, CAPLADataError, FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.error("Training interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
