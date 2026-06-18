"""
@file capla_hyperparameter_search.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search for CAPLA using official URV v3b splits.

The search uses official train and validation subsets only. Official test subsets
are intentionally excluded from HPO to avoid test leakage.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from CAPLA.core.common import choose_device, ensure_dir, get_logger, save_json, seed_everything
from CAPLA.core.data_utils import build_dataset_index
from CAPLA.core.model_adapter import build_capla_model, save_capla_bundle
from CAPLA.core.Run_URV_V3B_5Splits_CAPLA import dataset_paths, load_split_ids, make_loader, subset_index_df
from CAPLA.core.Train_CAPLA import EpochRecord, _build_grad_scaler, evaluate_one_epoch, train_one_epoch

LOGGER = get_logger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150


class CAPLAHPOError(RuntimeError):
    """Raised when CAPLA HPO cannot complete safely."""


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def parse_csv_values(raw: str, caster) -> list[Any]:
    values = [caster(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def parse_splits(value: str) -> list[int]:
    text = str(value).strip().lower()
    if text in {"all", "*"}:
        return [1, 2, 3, 4, 5]

    split_ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        split_id = int(part)
        if split_id < 1 or split_id > 5:
            raise ValueError("Split id must be between 1 and 5.")
        split_ids.append(split_id)

    split_ids = sorted(set(split_ids))
    if not split_ids:
        raise ValueError("At least one tuning split is required.")
    return split_ids


def finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        return fallback
    return number if math.isfinite(number) else fallback


def format_elapsed_time(seconds: float) -> str:
    """Convert elapsed seconds to HH:MM:SS format."""
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_better(candidate: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    if candidate.get("status") != "success":
        return False
    if current is None:
        return True

    candidate_rmse = finite_or(candidate.get("mean_val_RMSE"), float("inf"))
    current_rmse = finite_or(current.get("mean_val_RMSE"), float("inf"))
    if candidate_rmse < current_rmse:
        return True
    if candidate_rmse > current_rmse:
        return False

    candidate_pearson = finite_or(candidate.get("mean_val_Pearson"), float("-inf"))
    current_pearson = finite_or(current.get("mean_val_Pearson"), float("-inf"))
    return candidate_pearson > current_pearson


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_trials_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fieldnames = [
        "trial_id",
        "status",
        "lr",
        "batch_size",
        "weight_decay",
        "tuning_splits",
        "mean_val_RMSE",
        "mean_val_Pearson",
        "mean_val_MAE",
        "mean_val_SD",
        "mean_best_epoch",
        "elapsed_seconds",
        "elapsed_time",
        "models_dir",
        "trial_dir",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def write_history(history: Sequence[EpochRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.__dict__.copy() for record in history]).to_csv(path, index=False)
    return path


def build_loader_args(args: argparse.Namespace, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=int(batch_size),
        num_workers=int(args.num_workers),
        max_seq_len=int(args.max_seq_len),
        max_pkt_len=int(args.max_pkt_len),
        max_smi_len=int(args.max_smi_len),
    )


def build_search_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for lr, batch_size, weight_decay in itertools.product(
        args.lr_values,
        args.batch_size_values,
        args.weight_decay_values,
    ):
        if float(lr) <= 0:
            raise ValueError("Learning rates must be > 0.")
        if int(batch_size) < 1:
            raise ValueError("Batch sizes must be >= 1.")
        if float(weight_decay) < 0:
            raise ValueError("Weight decays must be >= 0.")
        grid.append(
            {
                "lr": float(lr),
                "batch_size": int(batch_size),
                "weight_decay": float(weight_decay),
            }
        )
    return grid


def run_one_split(
    *,
    args: argparse.Namespace,
    trial_id: str,
    split_id: int,
    hparams: Mapping[str, Any],
    dataset_dir: Path,
    base_index_df: pd.DataFrame,
    trial_result_dir: Path,
    trial_model_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    split_result_dir = ensure_dir(trial_result_dir / f"split_{split_id:02d}")
    split_model_dir = ensure_dir(trial_model_dir / f"split_{split_id:02d}")

    train_ids = load_split_ids(dataset_dir, split_id, "train")
    valid_ids = load_split_ids(dataset_dir, split_id, "valid")
    train_df = subset_index_df(base_index_df, train_ids, split_id, "train")
    valid_df = subset_index_df(base_index_df, valid_ids, split_id, "valid")

    loader_args = build_loader_args(args, int(hparams["batch_size"]))
    seed_everything(seed)
    train_loader = make_loader(train_df, loader_args, shuffle=True, device=device)
    valid_loader = make_loader(valid_df, loader_args, shuffle=False, device=device)

    model = build_capla_model().to(device)
    optimizer = AdamW(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))
    loss_fn = nn.MSELoss(reduction="mean")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = _build_grad_scaler(enabled=amp_enabled)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_val_rmse = float("inf")
    best_epoch = -1
    no_improve = 0
    history: list[EpochRecord] = []
    started_at = time.time()

    LOGGER.info(
        "CAPLA HPO %s split %02d | train=%d valid=%d | lr=%s batch=%s wd=%s device=%s",
        trial_id,
        split_id,
        len(train_df),
        len(valid_df),
        hparams["lr"],
        hparams["batch_size"],
        hparams["weight_decay"],
        device,
    )

    for epoch in range(1, int(args.epochs) + 1):
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
            desc=f"HPO {trial_id} split {split_id:02d} val {epoch}",
        )

        current_rmse = float(val_metrics["RMSE"])
        improved = current_rmse < best_val_rmse - float(args.min_delta)

        if improved:
            best_val_rmse = current_rmse
            best_metrics = {name: float(value) for name, value in val_metrics.items()}
            best_epoch = int(epoch)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        history.append(
            EpochRecord(
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
        )

        LOGGER.info(
            "%s split %02d epoch %03d/%03d | val_RMSE=%.6f | val_Pearson=%.6f%s",
            trial_id,
            split_id,
            epoch,
            args.epochs,
            float(val_metrics["RMSE"]),
            float(val_metrics["Pearson"]),
            " [best]" if improved else "",
        )

        if epoch >= int(args.min_epochs_before_stopping) and no_improve >= int(args.early_stopping_rounds):
            LOGGER.info("%s split %02d early stopping | best_epoch=%d", trial_id, split_id, best_epoch)
            break

    if best_state_dict is None or best_metrics is None:
        raise CAPLAHPOError(f"{trial_id} split {split_id} produced no valid checkpoint.")

    model.load_state_dict(best_state_dict)
    model = model.to(device)

    history_csv = write_history(history, split_result_dir / "train_history.csv")
    model_path = split_model_dir / f"CAPLA_HPO_{trial_id}_split{split_id:02d}.pt"

    save_capla_bundle(
        model,
        model_path,
        metadata={
            "model_name": "CAPLA",
            "workflow": "CAPLA_HPO_official_train_validation_only",
            "trial_id": trial_id,
            "split_id": int(split_id),
            "best_epoch": int(best_epoch),
            "best_validation_metrics": best_metrics,
            "hyperparameters": dict(hparams),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "test_rows_used": 0,
        },
    )

    elapsed_seconds = float(time.time() - started_at)

    summary = {
        "trial_id": trial_id,
        "split": int(split_id),
        "model_pt": str(model_path),
        "history_csv": str(history_csv),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "val_RMSE": float(best_metrics["RMSE"]),
        "val_Pearson": float(best_metrics["Pearson"]),
        "val_MAE": float(best_metrics["MAE"]),
        "val_SD": float(best_metrics["SD"]),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_time": format_elapsed_time(elapsed_seconds),
    }

    save_json(summary, split_result_dir / "split_summary.json")
    return summary


def aggregate_split_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        "mean_val_RMSE": float(frame["val_RMSE"].mean()),
        "mean_val_Pearson": float(frame["val_Pearson"].mean()),
        "mean_val_MAE": float(frame["val_MAE"].mean()),
        "mean_val_SD": float(frame["val_SD"].mean()),
        "mean_best_epoch": float(frame["best_epoch"].mean()),
    }


def run_hyperparameter_search(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = resolve_path(args.dataset_dir)
    models_root = ensure_dir(resolve_path(args.models_root))
    results_root = ensure_dir(resolve_path(args.results_root))
    tuning_splits = parse_splits(args.splits)
    device = choose_device(args.device)

    paths = dataset_paths(dataset_dir)
    base_index_df, validation_report = build_dataset_index(
        paths.affinity_csv,
        paths.smi_csv,
        paths.global_dir,
        paths.pocket_dir,
        strict=False,
    )

    if base_index_df.empty:
        raise CAPLAHPOError("Prepared CAPLA dataset contains zero valid rows.")

    search_grid = build_search_grid(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(results_root / f"capla_hpo_{timestamp}")
    trial_models_root = ensure_dir(models_root / f"capla_hpo_{timestamp}")
    trials_csv = run_dir / "capla_hpo_trials.csv"
    latest_run_json = results_root / "latest_run.json"

    save_json(
        {
            "script": "capla_hyperparameter_search.py",
            "methodology": "official train/validation subsets only; official test subsets excluded",
            "selection_rule": "min mean validation RMSE, then max mean validation Pearson",
            "dataset_dir": str(dataset_dir),
            "device": str(device),
            "tuning_splits": tuning_splits,
            "n_valid_dataset_rows": int(len(base_index_df)),
            "validation_report": validation_report.to_dict(),
            "search_grid": search_grid,
            "n_trials": int(len(search_grid)),
            "arguments": vars(args),
        },
        run_dir / "search_config.json",
    )

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    started_at = time.time()

    for trial_number, hparams in enumerate(search_grid, start=1):
        trial_id = f"trial_{trial_number:04d}"
        trial_result_dir = ensure_dir(run_dir / trial_id)
        trial_model_dir = ensure_dir(trial_models_root / trial_id)

        row: dict[str, Any] = {
            "trial_id": trial_id,
            "status": "failure",
            "lr": hparams["lr"],
            "batch_size": hparams["batch_size"],
            "weight_decay": hparams["weight_decay"],
            "tuning_splits": ",".join(str(item) for item in tuning_splits),
            "models_dir": str(trial_model_dir),
            "trial_dir": str(trial_result_dir),
            "error_message": "",
        }

        try:
            LOGGER.info("Starting CAPLA HPO %d/%d | %s", trial_number, len(search_grid), hparams)

            split_rows = []
            for split_id in tuning_splits:
                split_rows.append(
                    run_one_split(
                        args=args,
                        trial_id=trial_id,
                        split_id=split_id,
                        hparams=hparams,
                        dataset_dir=dataset_dir,
                        base_index_df=base_index_df,
                        trial_result_dir=trial_result_dir,
                        trial_model_dir=trial_model_dir,
                        device=device,
                        seed=int(args.seed) + trial_number * 1000 + split_id,
                    )
                )

            pd.DataFrame(split_rows).to_csv(trial_result_dir / "split_summaries.csv", index=False)

            row.update(aggregate_split_summaries(split_rows))
            row["status"] = "success"
            row["elapsed_seconds"] = float(sum(item["elapsed_seconds"] for item in split_rows))
            row["elapsed_time"] = format_elapsed_time(row["elapsed_seconds"])

            save_json(row, trial_result_dir / "trial_summary.json")

            if is_better(row, best_row):
                best_row = dict(row)

        except Exception as exc:
            row["status"] = "failure"
            row["error_message"] = str(exc)
            (trial_result_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            LOGGER.exception("CAPLA HPO trial failed: %s", trial_id)

        finally:
            rows.append(row)
            write_trials_csv(trials_csv, rows)

    if best_row is None:
        raise CAPLAHPOError("All CAPLA hyperparameter trials failed. Inspect capla_hpo_trials.csv.")

    best_models_dir = run_dir / "best_models"
    if best_models_dir.exists():
        shutil.rmtree(best_models_dir)
    shutil.copytree(Path(str(best_row["models_dir"])), best_models_dir)

    total_elapsed_seconds = float(time.time() - started_at)

    best_payload = {
        "model_name": "CAPLA",
        "status": "success",
        "methodology": "official train/validation subsets only; official test subsets excluded",
        "selection_rule": {
            "primary_metric": "mean validation RMSE",
            "primary_goal": "min",
            "secondary_metric": "mean validation Pearson",
            "secondary_goal": "max",
        },
        "best_trial": best_row["trial_id"],
        "best_hyperparameters": {
            "lr": float(best_row["lr"]),
            "batch_size": int(best_row["batch_size"]),
            "weight_decay": float(best_row["weight_decay"]),
        },
        "best_metrics": {
            "mean_val_RMSE": float(best_row["mean_val_RMSE"]),
            "mean_val_Pearson": float(best_row["mean_val_Pearson"]),
            "mean_val_MAE": float(best_row["mean_val_MAE"]),
            "mean_val_SD": float(best_row["mean_val_SD"]),
            "mean_best_epoch": float(best_row["mean_best_epoch"]),
        },
        "tuning_splits": tuning_splits,
        "n_trials": int(len(rows)),
        "n_success": int(sum(1 for row in rows if row.get("status") == "success")),
        "n_failures": int(sum(1 for row in rows if row.get("status") != "success")),
        "elapsed_seconds": total_elapsed_seconds,
        "elapsed_time": format_elapsed_time(total_elapsed_seconds),
        "paths": {
            "run_dir": str(run_dir),
            "trials_csv": str(trials_csv),
            "best_models_dir": str(best_models_dir),
        },
    }

    write_yaml(run_dir / "best_config_capla.yaml", best_payload)
    save_json(best_payload, run_dir / "best_config_capla.json")
    save_json(best_payload, latest_run_json)

    print("CAPLA hyperparameter search completed")
    print("  tuning splits       :", tuning_splits)
    print("  trials              :", len(rows))
    print("  best trial          :", best_payload["best_trial"])
    print("  best hyperparameters:", best_payload["best_hyperparameters"])
    print("  best mean val RMSE  :", f"{best_payload['best_metrics']['mean_val_RMSE']:.6f}")
    print("  best mean Pearson   :", f"{best_payload['best_metrics']['mean_val_Pearson']:.6f}")
    print("  elapsed time        :", best_payload["elapsed_time"])
    print("  run directory       :", run_dir)

    return best_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CAPLA HPO on official train/validation subsets only.")

    parser.add_argument("--dataset-dir", default="CAPLA/data/urv_dataset_v3b_prepared")
    parser.add_argument("--models-root", default="CAPLA/models/hpo")
    parser.add_argument("--results-root", default="CAPLA/outputs/hpo")
    parser.add_argument("--splits", default="1", help="Tuning split ids: 1, 1,2 or all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-pkt-len", type=int, default=DEFAULT_MAX_PKT_LEN)
    parser.add_argument("--max-smi-len", type=int, default=DEFAULT_MAX_SMI_LEN)
    parser.add_argument("--lr-values", type=lambda raw: parse_csv_values(raw, float), default=[5e-5, 1e-4])
    parser.add_argument("--batch-size-values", type=lambda raw: parse_csv_values(raw, int), default=[8, 16])
    parser.add_argument("--weight-decay-values", type=lambda raw: parse_csv_values(raw, float), default=[0.0, 0.01])

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_hyperparameter_search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
