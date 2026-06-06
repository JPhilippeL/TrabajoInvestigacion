#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train CAPLA on the official URV v3b 5 splits.

Python 3.6 compatible.

Default workflow:
  1) Read TFM_Implementation/CAPLA/urv_dataset_v3b_prepared.
  2) Train one fresh CAPLA model per official split.
  3) Save five .pt bundles, predictions, scatter plots, per-split metrics and
     aggregate metrics.

The defaults are intentionally patient to avoid stopping at epoch 1:
  --epochs 150
  --early-stopping-rounds 25
  --min-epochs-before-stopping 20
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = _THIS_FILE.parent
_REPO_HINT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
if str(_REPO_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_HINT))

from CAPLA.core.common import choose_device, ensure_dir, get_logger, save_json, seed_everything  # noqa: E402
from CAPLA.core.data_utils import CAPLADataset, DatasetPaths, build_dataset_index  # noqa: E402
from CAPLA.core.metrics_utils import compute_regression_metrics, save_metrics_csv, save_predictions_csv, save_scatter_plot  # noqa: E402
from CAPLA.core.model_adapter import build_capla_model, save_capla_bundle  # noqa: E402
from CAPLA.core.Predict_CAPLA import run_inference  # noqa: E402
from CAPLA.core.Train_CAPLA import EpochRecord, _build_grad_scaler, evaluate_one_epoch, train_one_epoch  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_EARLY_STOPPING_ROUNDS = 25
DEFAULT_MIN_EPOCHS_BEFORE_STOPPING = 20
DEFAULT_NUM_WORKERS = 0


def resolve_path(value, base=SCRIPT_DIR):
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def parse_splits(value: str) -> List[int]:
    if value.strip().lower() in {"all", "*"}:
        return [1, 2, 3, 4, 5]
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        split_id = int(part)
        if split_id < 1 or split_id > 5:
            raise ValueError("Split id must be between 1 and 5")
        out.append(split_id)
    return sorted(set(out))


def dataset_paths(dataset_dir: Path) -> DatasetPaths:
    paths = DatasetPaths(
        affinity_csv=dataset_dir / "affinity_data.csv",
        smi_csv=dataset_dir / "urv_v3b_smi.csv",
        global_dir=dataset_dir / "global",
        pocket_dir=dataset_dir / "pocket",
    )
    for required in [paths.affinity_csv, paths.smi_csv, paths.global_dir, paths.pocket_dir]:
        if not Path(required).exists():
            raise FileNotFoundError(f"Prepared CAPLA dataset is incomplete. Missing: {required}")
    return paths


def load_split_ids(dataset_dir: Path, split_id: int, role: str) -> List[str]:
    path = dataset_dir / "splits" / f"split_{split_id:02d}" / f"{role}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    df = pd.read_csv(path)
    if "pdbid" not in df.columns:
        raise ValueError(f"Split CSV {path} must contain a pdbid column")
    return df["pdbid"].astype(str).str.strip().str.lower().tolist()


def subset_index_df(index_df: pd.DataFrame, ids: Sequence[str], split_id: int, role: str) -> pd.DataFrame:
    wanted = [str(pid).strip().lower() for pid in ids]
    available = set(index_df["pdbid"].astype(str))
    missing = [pid for pid in wanted if pid not in available]
    if missing:
        raise ValueError(f"Split {split_id} role {role} contains ids not present in prepared dataset: {missing[:10]}")
    by_id = index_df.set_index("pdbid", drop=False)
    rows = [by_id.loc[pid] for pid in wanted]
    return pd.DataFrame(rows).reset_index(drop=True)


def make_loader(index_df: pd.DataFrame, args: argparse.Namespace, shuffle: bool, device: torch.device) -> DataLoader:
    dataset = CAPLADataset(
        index_df=index_df,
        max_seq_len=args.max_seq_len,
        max_pkt_len=args.max_pkt_len,
        max_smi_len=args.max_smi_len,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def save_subset_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_dir: Path, role: str, split_id: int) -> Dict[str, Any]:
    ensure_dir(output_dir)
    pdbids, y_true, y_pred = run_inference(model=model, loader=loader, device=device)
    metrics = compute_regression_metrics(y_true, y_pred)
    pred_csv = save_predictions_csv(pdbids, y_true, y_pred, output_dir / f"Predictions_CAPLA_{role}.csv")
    metrics_csv = save_metrics_csv(metrics, output_dir / f"Metrics_CAPLA_{role}.csv", std_overrides={k: 0.0 for k in metrics})
    scatter_png = save_scatter_plot(y_true, y_pred, output_dir / f"Scatter_CAPLA_{role}.png", model_name="CAPLA", split_id=f"{split_id:02d}_{role}", metrics=metrics)
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "predictions_csv": str(pred_csv),
        "metrics_csv": str(metrics_csv),
        "scatter_png": str(scatter_png),
        "n": int(len(y_true)),
    }


def write_history(history: Sequence[EpochRecord], path: Path) -> Path:
    ensure_dir(path.parent)
    pd.DataFrame([item.__dict__.copy() for item in history]).to_csv(path, index=False)
    return path


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(summaries)
    out: Dict[str, Any] = {"splits_completed": int(len(df))}
    for col in ["test_RMSE", "test_MAE", "test_Pearson", "test_SD"]:
        if col in df.columns:
            key = col.replace("test_", "test_")
            out[f"{key}_mean"] = float(df[col].mean())
            out[f"{key}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")
    return out


def run_debug(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = resolve_path(args.dataset_dir)
    paths = dataset_paths(dataset_dir)
    index_df, report = build_dataset_index(paths.affinity_csv, paths.smi_csv, paths.global_dir, paths.pocket_dir, strict=False)
    split_ids = load_split_ids(dataset_dir, 1, "train")
    train_df = subset_index_df(index_df, split_ids, 1, "train")
    device = choose_device(args.device)
    loader = make_loader(train_df.head(max(args.batch_size, 2)), args, shuffle=False, device=device)
    model = build_capla_model().to(device)
    model.eval()
    batch = next(iter(loader))
    pdbids, seq_tensor, pkt_tensor, smi_tensor, affinity = batch
    with torch.no_grad():
        output = model(seq_tensor.to(device), pkt_tensor.to(device), smi_tensor.to(device)).view(-1)
    report_dict = {
        "script": "Run_URV_V3B_5Splits_CAPLA",
        "mode": "debug",
        "device": str(device),
        "dataset_dir": str(dataset_dir),
        "n_valid_dataset_rows": int(len(index_df)),
        "train_split_1_rows": int(len(train_df)),
        "batch_size": int(args.batch_size),
        "output_shape": list(output.detach().cpu().shape),
        "target_shape": list(affinity.shape),
        "output_has_nan": bool(torch.isnan(output.detach().cpu()).any().item()),
        "output_has_inf": bool(torch.isinf(output.detach().cpu()).any().item()),
        "validation_report": report.to_dict(),
    }
    out_dir = ensure_dir(resolve_path(args.output_dir) / "debug")
    save_json(report_dict, out_dir / "debug_report.json")
    print("CAPLA debug OK")
    print("  device:", device)
    print("  valid rows:", len(index_df))
    print("  output_shape:", report_dict["output_shape"])
    print("  report:", out_dir / "debug_report.json")
    return report_dict


def train_one_split(args: argparse.Namespace, base_index_df: pd.DataFrame, validation_report: Any, split_id: int) -> Dict[str, Any]:
    logger = get_logger(f"TFM_CAPLA_URV_V3B_SPLIT_{split_id:02d}")
    dataset_dir = resolve_path(args.dataset_dir)
    output_root = ensure_dir(resolve_path(args.output_dir))
    models_root = ensure_dir(resolve_path(args.models_dir))
    split_out = ensure_dir(output_root / f"split_{split_id:02d}")
    split_model_dir = ensure_dir(models_root / f"split_{split_id:02d}")

    train_ids = load_split_ids(dataset_dir, split_id, "train")
    valid_ids = load_split_ids(dataset_dir, split_id, "valid")
    test_ids = load_split_ids(dataset_dir, split_id, "test")

    train_df = subset_index_df(base_index_df, train_ids, split_id, "train")
    valid_df = subset_index_df(base_index_df, valid_ids, split_id, "valid")
    test_df = subset_index_df(base_index_df, test_ids, split_id, "test")

    seed_everything(args.seed + split_id)
    device = choose_device(args.device)
    train_loader = make_loader(train_df, args, shuffle=True, device=device)
    valid_loader = make_loader(valid_df, args, shuffle=False, device=device)
    test_loader = make_loader(test_df, args, shuffle=False, device=device)

    model = build_capla_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = _build_grad_scaler(enabled=amp_enabled)

    best_state_dict = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []  # type: List[EpochRecord]
    started = time.time()

    logger.info("Training split %02d | train=%d valid=%d test=%d | device=%s", split_id, len(train_df), len(valid_df), len(test_df), device)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
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
            loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            desc=f"Val {split_id:02d}/{epoch}",
        )

        improved = float(val_metrics["loss"]) < (best_val_loss - args.min_delta)
        if improved:
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
            epoch_seconds=float(time.time() - epoch_started),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            is_best=bool(improved),
        )
        history.append(record)
        logger.info(
            "Split %02d epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | val_RMSE=%.4f%s | no_improve=%d",
            split_id,
            epoch,
            args.epochs,
            record.train_loss,
            record.val_loss,
            record.val_rmse,
            " [best]" if improved else "",
            epochs_without_improvement,
        )

        can_stop = epoch >= args.min_epochs_before_stopping
        if can_stop and epochs_without_improvement >= args.early_stopping_rounds:
            logger.info("Early stopping split %02d after %d non-improving epochs; best_epoch=%d", split_id, epochs_without_improvement, best_epoch)
            break

    if best_state_dict is None:
        raise RuntimeError(f"Split {split_id} did not produce a best checkpoint")

    model.load_state_dict(best_state_dict)
    model = model.to(device)

    validation_pred = save_subset_predictions(model, valid_loader, device, split_out / "validation", "validation", split_id)
    test_pred = save_subset_predictions(model, test_loader, device, split_out / "test", "test", split_id)
    history_csv = write_history(history, split_out / "train_history.csv")

    model_path = split_model_dir / f"CAPLA_URV_v3b_split{split_id:02d}.pt"
    best_record = next(item.__dict__.copy() for item in history if item.epoch == best_epoch)
    metadata = {
        "model_name": "CAPLA",
        "workflow": "URV_v3b_official_5splits",
        "split_id": int(split_id),
        "state_dict_format": "clean_capla_bundle",
        "dataset_dir": str(dataset_dir),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_record,
        "validation_metrics": validation_pred["metrics"],
        "test_metrics": test_pred["metrics"],
        "hyperparameters": {
            "epochs_requested": int(args.epochs),
            "epochs_completed": int(len(history)),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "min_epochs_before_stopping": int(args.min_epochs_before_stopping),
            "min_delta": float(args.min_delta),
            "optimizer": "AdamW",
            "loss": "MSELoss(mean)",
            "amp_enabled": bool(amp_enabled),
            "seed": int(args.seed + split_id),
        },
        "lengths": {
            "max_seq_len": int(args.max_seq_len),
            "max_pkt_len": int(args.max_pkt_len),
            "max_smi_len": int(args.max_smi_len),
        },
        "validation_report": validation_report.to_dict(),
        "timing": {"total_seconds": float(time.time() - started)},
    }
    save_capla_bundle(model, model_path, metadata=metadata)

    summary = {
        "split": int(split_id),
        "model_pt": str(model_path),
        "results_dir": str(split_out),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "val_RMSE": float(validation_pred["metrics"]["RMSE"]),
        "val_MAE": float(validation_pred["metrics"]["MAE"]),
        "val_Pearson": float(validation_pred["metrics"]["Pearson"]),
        "val_SD": float(validation_pred["metrics"]["SD"]),
        "test_RMSE": float(test_pred["metrics"]["RMSE"]),
        "test_MAE": float(test_pred["metrics"]["MAE"]),
        "test_Pearson": float(test_pred["metrics"]["Pearson"]),
        "test_SD": float(test_pred["metrics"]["SD"]),
        "history_csv": str(history_csv),
        "validation_predictions_csv": validation_pred["predictions_csv"],
        "test_predictions_csv": test_pred["predictions_csv"],
    }
    save_json(summary, split_out / "split_summary.json")
    save_json(metadata, split_out / "training_config.json")
    logger.info("Finished split %02d | test_RMSE=%.4f | test_Pearson=%.4f | model=%s", split_id, summary["test_RMSE"], summary["test_Pearson"], model_path)
    return summary


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = resolve_path(args.dataset_dir)
    paths = dataset_paths(dataset_dir)
    index_df, validation_report = build_dataset_index(paths.affinity_csv, paths.smi_csv, paths.global_dir, paths.pocket_dir, strict=False)
    requested_splits = parse_splits(args.splits)
    summaries = []
    for split_id in requested_splits:
        summaries.append(train_one_split(args, index_df, validation_report, split_id))

    output_root = ensure_dir(resolve_path(args.output_dir))
    summary_df = pd.DataFrame(summaries).sort_values("split")
    summary_path = output_root / "Summary_5splits.csv"
    summary_df.to_csv(summary_path, index=False)
    aggregate = aggregate_summaries(summaries)
    aggregate_path = output_root / "Aggregate_metrics.json"
    save_json(aggregate, aggregate_path)
    print("CAPLA official URV v3b splits completed")
    print("  splits:", requested_splits)
    print("  summary:", summary_path)
    print("  aggregate:", aggregate_path)
    if "test_RMSE_mean" in aggregate:
        print("  test RMSE: %.4f ± %.4f" % (aggregate["test_RMSE_mean"], aggregate["test_RMSE_std"]))
    return {"summaries": summaries, "aggregate": aggregate, "summary_csv": str(summary_path), "aggregate_json": str(aggregate_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CAPLA on URV v3b official 5 splits.")
    parser.add_argument("--mode", choices=["debug", "all"], default="all")
    parser.add_argument("--dataset-dir", default="../data/urv_dataset_v3b_prepared")
    parser.add_argument("--output-dir", default="../outputs/from_scratch")
    parser.add_argument("--models-dir", default="../models/from_scratch")
    parser.add_argument("--splits", default="all", help="all or comma-separated split ids, e.g. 1,3,5")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=DEFAULT_MIN_EPOCHS_BEFORE_STOPPING)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=990721)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-pkt-len", type=int, default=DEFAULT_MAX_PKT_LEN)
    parser.add_argument("--max-smi-len", type=int, default=DEFAULT_MAX_SMI_LEN)
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA AMP mixed precision")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.early_stopping_rounds < 1:
        raise ValueError("--early-stopping-rounds must be >= 1")
    if args.min_epochs_before_stopping < 1:
        raise ValueError("--min-epochs-before-stopping must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.mode == "debug":
        run_debug(args)
    else:
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
