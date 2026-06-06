#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare original pretrained CAPLA vs fine-tuned CAPLA on URV v3b official 5 splits.

Python 3.6 compatible.

This script performs two experiments on the same URV v3b prepared dataset:

A) Pretrained original CAPLA checkpoint -> predict each official test split.
B) Same original CAPLA checkpoint -> fine-tune on train/valid of each split -> predict the same test split.

Important methodological rule:
For every split, fine-tuning starts again from the original checkpoint. The model
fine-tuned on split 1 is never reused for split 2.

Expected working directory:
  .../CAPLA/CAPLA/TFM_Implementation/CAPLA

Typical command:
  python Run_CAPLA_Pretrained_vs_Finetuned_URV_V3B.py \
    --dataset-dir urv_dataset_v3b_prepared \
    --checkpoint-original ../../CAPLA/saveModel/CAPLA_bestModel/best_model.pt \
    --device cuda \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.00005 \
    --weight-decay 0.01 \
    --early-stopping-rounds 20 \
    --min-epochs-before-stopping 15 \
    --splits all
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

_THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = _THIS_FILE.parent
_REPO_HINT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
if str(_REPO_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_HINT))

from CAPLA.core.common import choose_device, ensure_dir, save_json, seed_everything  # noqa: E402
from CAPLA.core.data_utils import CAPLADataset, build_dataset_index  # noqa: E402
from CAPLA.core.metrics_utils import compute_regression_metrics, save_metrics_csv, save_predictions_csv, save_scatter_plot  # noqa: E402
from CAPLA.core.model_adapter import load_capla_model, save_capla_bundle  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_EARLY_STOPPING_ROUNDS = 20
DEFAULT_MIN_EPOCHS_BEFORE_STOPPING = 15
DEFAULT_NUM_WORKERS = 0


def resolve_path(value, base=SCRIPT_DIR, must_exist=False):
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise RuntimeError("Path does not exist: {0}".format(p))
    return p


def parse_splits(value):
    text = str(value).strip().lower()
    if text in ("all", "*"):
        return [1, 2, 3, 4, 5]
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        split_id = int(part)
        if split_id < 1 or split_id > 5:
            raise ValueError("Split id must be between 1 and 5")
        out.append(split_id)
    return sorted(set(out))


def dataset_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    smi_candidates = [dataset_dir / "urv_v3b_smi.csv", dataset_dir / "urv_dataset_smi.csv"]
    smi_csv = None
    for candidate in smi_candidates:
        if candidate.exists():
            smi_csv = candidate
            break
    if smi_csv is None:
        raise RuntimeError("Could not find SMILES CSV. Expected urv_v3b_smi.csv or urv_dataset_smi.csv in {0}".format(dataset_dir))
    return {
        "affinity_csv": dataset_dir / "affinity_data.csv",
        "smi_csv": smi_csv,
        "global_dir": dataset_dir / "global",
        "pocket_dir": dataset_dir / "pocket",
    }


def check_dataset_dir(dataset_dir):
    paths = dataset_paths(dataset_dir)
    for key, path in paths.items():
        if not Path(path).exists():
            raise RuntimeError("Prepared dataset is incomplete. Missing {0}: {1}".format(key, path))
    if not (Path(dataset_dir) / "splits").is_dir():
        raise RuntimeError("Prepared dataset is missing splits directory: {0}".format(Path(dataset_dir) / "splits"))
    return paths


def load_index(dataset_dir):
    paths = check_dataset_dir(dataset_dir)
    index_df, report = build_dataset_index(
        affinity_csv=paths["affinity_csv"],
        smi_csv=paths["smi_csv"],
        global_dir=paths["global_dir"],
        pocket_dir=paths["pocket_dir"],
        strict=True,
    )
    return index_df, report, paths


def read_role_ids(dataset_dir, split_id, role):
    path = Path(dataset_dir) / "splits" / "split_{0:02d}".format(split_id) / (role + ".csv")
    if not path.exists():
        raise RuntimeError("Missing split CSV: {0}".format(path))
    df = pd.read_csv(path)
    if "pdbid" not in df.columns:
        raise RuntimeError("Split CSV has no pdbid column: {0}".format(path))
    return [str(x).strip().lower() for x in df["pdbid"].tolist()]


def subset_index(index_df, ids, role, split_id):
    id_set = set(ids)
    sub = index_df[index_df["pdbid"].isin(id_set)].copy()
    missing = sorted(id_set - set(sub["pdbid"].tolist()))
    if missing:
        raise RuntimeError("Split {0} {1} has ids missing from valid dataset index: {2}".format(split_id, role, ",".join(missing[:20])))
    # Preserve official split order.
    order = {pdbid: i for i, pdbid in enumerate(ids)}
    sub["_order"] = sub["pdbid"].map(order)
    sub = sub.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return sub


def build_loader(index_df, args, shuffle):
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
        pin_memory=torch.cuda.is_available() and str(args.device).lower() != "cpu",
    )


def move_batch(batch, device):
    pdbids, seq, pkt, smi, y = batch
    return pdbids, seq.to(device), pkt.to(device), smi.to(device), y.to(device)


def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    all_ids = []
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            pdbids, seq, pkt, smi, y = move_batch(batch, device)
            pred = model(seq, pkt, smi).view(-1)
            target = y.view(-1)
            if criterion is not None:
                loss = criterion(pred, target)
                n = int(target.numel())
                total_loss += float(loss.item()) * n
                total_n += n
            all_ids.extend(list(pdbids))
            all_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
            all_pred.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    metrics = compute_regression_metrics(all_true, all_pred)
    if criterion is not None:
        metrics["loss"] = total_loss / float(total_n) if total_n else float("nan")
    return metrics, all_ids, all_true, all_pred


def train_one_epoch(model, loader, device, optimizer, criterion, epoch, split_id):
    model.train()
    total_loss = 0.0
    total_n = 0
    iterable = tqdm(loader, desc="split {0:02d} ft epoch {1:03d}".format(split_id, epoch))
    for batch in iterable:
        _, seq, pkt, smi, y = move_batch(batch, device)
        optimizer.zero_grad()
        pred = model(seq, pkt, smi).view(-1)
        target = y.view(-1)
        loss = criterion(pred, target)
        loss.backward()
        if getattr(args_global, "grad_clip_norm", 0.0) and args_global.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args_global.grad_clip_norm)
        optimizer.step()
        n = int(target.numel())
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / float(total_n) if total_n else float("nan")


# Used only to avoid threading many optional args through the Py3.6-compatible helper.
args_global = None


def save_prediction_outputs(out_dir, prefix, split_id, ids, y_true, y_pred, metrics):
    ensure_dir(out_dir)
    pred_path = out_dir / (prefix + "_Predictions_test.csv")
    metrics_path = out_dir / (prefix + "_Metrics_test.csv")
    scatter_path = out_dir / (prefix + "_Scatter_test.png")
    save_predictions_csv(ids, y_true, y_pred, pred_path)
    save_metrics_csv(metrics, metrics_path)
    try:
        save_scatter_plot(y_true, y_pred, scatter_path, model_name=prefix, split_id="{0:02d}".format(split_id), metrics=metrics)
    except Exception as exc:
        print("WARNING: scatter plot failed for {0} split {1}: {2}".format(prefix, split_id, exc))
    return pred_path, metrics_path, scatter_path


def load_original_model(checkpoint_path, device):
    model, metadata = load_capla_model(str(checkpoint_path), map_location=device, strict=True)
    model.to(device)
    return model, metadata


def run_debug(args):
    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR, must_exist=True)
    checkpoint = resolve_path(args.checkpoint_original, SCRIPT_DIR, must_exist=True)
    index_df, report, paths = load_index(dataset_dir)
    device = choose_device(args.device)
    model, metadata = load_original_model(checkpoint, device)
    test_ids = read_role_ids(dataset_dir, 1, "test")
    test_index = subset_index(index_df, test_ids, "test", 1).head(int(args.batch_size)).reset_index(drop=True)
    loader = build_loader(test_index, args, shuffle=False)
    metrics, ids, y_true, y_pred = evaluate_model(model, loader, device, criterion=nn.MSELoss())
    out_dir = ensure_dir(resolve_path(args.results_dir, SCRIPT_DIR, must_exist=False))
    report_obj = {
        "mode": "debug",
        "dataset_dir": str(dataset_dir),
        "checkpoint_original": str(checkpoint),
        "device": str(device),
        "valid_rows": int(len(index_df)),
        "batch_size": int(args.batch_size),
        "debug_rows": int(len(test_index)),
        "output_has_nan": bool(np.isnan(np.asarray(y_pred, dtype=float)).any()),
        "target_has_nan": bool(np.isnan(np.asarray(y_true, dtype=float)).any()),
        "metrics_on_debug_batch": metrics,
        "checkpoint_metadata_keys": sorted([str(k) for k in metadata.keys()]),
    }
    save_json(report_obj, out_dir / "debug_pretrained_vs_finetuned.json")
    print("CAPLA pretrained-vs-finetuned debug OK")
    print("  valid rows:       {0}".format(len(index_df)))
    print("  debug rows:       {0}".format(len(test_index)))
    print("  device:           {0}".format(device))
    print("  checkpoint:       {0}".format(checkpoint))
    print("  output has NaN:   {0}".format(report_obj["output_has_nan"]))
    return report_obj


def run_split(args, split_id, index_df, dataset_dir, checkpoint_path, device):
    split_name = "split_{0:02d}".format(split_id)
    split_out_root = ensure_dir(resolve_path(args.results_dir, SCRIPT_DIR, must_exist=False) / split_name)
    pre_out_dir = ensure_dir(split_out_root / "pretrained_original")
    ft_out_dir = ensure_dir(split_out_root / "finetuned_from_original")
    model_out_dir = ensure_dir(resolve_path(args.models_dir, SCRIPT_DIR, must_exist=False) / split_name)

    train_ids = read_role_ids(dataset_dir, split_id, "train")
    valid_ids = read_role_ids(dataset_dir, split_id, "valid")
    test_ids = read_role_ids(dataset_dir, split_id, "test")
    train_index = subset_index(index_df, train_ids, "train", split_id)
    valid_index = subset_index(index_df, valid_ids, "valid", split_id)
    test_index = subset_index(index_df, test_ids, "test", split_id)

    train_loader = build_loader(train_index, args, shuffle=True)
    valid_loader = build_loader(valid_index, args, shuffle=False)
    test_loader = build_loader(test_index, args, shuffle=False)
    criterion = nn.MSELoss()

    # A) Original pretrained checkpoint, no training, predict test.
    pre_model, pre_metadata = load_original_model(checkpoint_path, device)
    pre_metrics, pre_ids, pre_true, pre_pred = evaluate_model(pre_model, test_loader, device, criterion=criterion)
    save_prediction_outputs(pre_out_dir, "CAPLA_pretrained_original", split_id, pre_ids, pre_true, pre_pred, pre_metrics)
    del pre_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # B) Fine-tune from the same original checkpoint, using split train/valid only.
    seed_everything(args.seed + split_id)
    ft_model, ft_metadata = load_original_model(checkpoint_path, device)
    optimizer = AdamW(ft_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    history = []

    print("Split {0}: pretrained test RMSE={1:.6f}; fine-tuning train={2}, valid={3}, test={4}".format(
        split_id, pre_metrics["RMSE"], len(train_index), len(valid_index), len(test_index)
    ))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(ft_model, train_loader, device, optimizer, criterion, epoch, split_id)
        valid_metrics, _, _, _ = evaluate_model(ft_model, valid_loader, device, criterion=criterion)
        row = {
            "split": split_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics.get("loss", float("nan")),
            "valid_RMSE": valid_metrics["RMSE"],
            "valid_MAE": valid_metrics["MAE"],
            "valid_Pearson": valid_metrics["Pearson"],
            "valid_SD": valid_metrics["SD"],
        }
        history.append(row)
        current = valid_metrics["RMSE"]
        print("split {0:02d} epoch {1:03d} train_loss={2:.6f} val_RMSE={3:.6f} val_MAE={4:.6f} val_Pearson={5:.6f}".format(
            split_id, epoch, train_loss, valid_metrics["RMSE"], valid_metrics["MAE"], valid_metrics["Pearson"]
        ))
        if current < best_metric - args.min_delta:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ft_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= args.min_epochs_before_stopping and no_improve >= args.early_stopping_rounds:
            print("Early stopping split {0} at epoch {1}; best_epoch={2}; best_valid_RMSE={3:.6f}".format(
                split_id, epoch, best_epoch, best_metric
            ))
            break

    if best_state is None:
        raise RuntimeError("Fine-tuning did not capture a best state for split {0}".format(split_id))
    ft_model.load_state_dict(best_state)
    ft_model.to(device)
    ft_metrics, ft_ids, ft_true, ft_pred = evaluate_model(ft_model, test_loader, device, criterion=criterion)
    save_prediction_outputs(ft_out_dir, "CAPLA_finetuned_from_original", split_id, ft_ids, ft_true, ft_pred, ft_metrics)

    history_path = ft_out_dir / "train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    model_path = model_out_dir / "CAPLA_finetuned_from_original_URV_v3b_split{0:02d}.pt".format(split_id)
    save_capla_bundle(
        ft_model,
        model_path,
        metadata={
            "experiment": "pretrained_original_then_finetuned_urv_v3b",
            "split": split_id,
            "best_epoch": int(best_epoch),
            "source_checkpoint": str(checkpoint_path),
            "epochs_requested": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "min_epochs_before_stopping": int(args.min_epochs_before_stopping),
            "max_seq_len": int(args.max_seq_len),
            "max_pkt_len": int(args.max_pkt_len),
            "max_smi_len": int(args.max_smi_len),
            "pretrained_metrics_test": pre_metrics,
            "finetuned_metrics_test": ft_metrics,
        },
    )

    comparison = {
        "split": split_id,
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(valid_index)),
        "test_rows": int(len(test_index)),
        "pretrained_RMSE": pre_metrics["RMSE"],
        "pretrained_MAE": pre_metrics["MAE"],
        "pretrained_Pearson": pre_metrics["Pearson"],
        "pretrained_SD": pre_metrics["SD"],
        "finetuned_RMSE": ft_metrics["RMSE"],
        "finetuned_MAE": ft_metrics["MAE"],
        "finetuned_Pearson": ft_metrics["Pearson"],
        "finetuned_SD": ft_metrics["SD"],
        "delta_RMSE": ft_metrics["RMSE"] - pre_metrics["RMSE"],
        "delta_MAE": ft_metrics["MAE"] - pre_metrics["MAE"],
        "delta_Pearson": ft_metrics["Pearson"] - pre_metrics["Pearson"],
        "delta_SD": ft_metrics["SD"] - pre_metrics["SD"],
        "best_epoch": int(best_epoch),
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "finetuned_model_pt": str(model_path),
        "pretrained_results_dir": str(pre_out_dir),
        "finetuned_results_dir": str(ft_out_dir),
    }
    save_json(comparison, split_out_root / "comparison_split_{0:02d}.json".format(split_id))
    print("Finished split {0}: pre_RMSE={1:.6f}, ft_RMSE={2:.6f}, delta_RMSE={3:.6f}".format(
        split_id, pre_metrics["RMSE"], ft_metrics["RMSE"], comparison["delta_RMSE"]
    ))
    return comparison


def aggregate_comparisons(rows):
    df = pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
    out = {}
    for prefix in ["pretrained", "finetuned"]:
        for metric in ["RMSE", "MAE", "Pearson", "SD"]:
            col = prefix + "_" + metric
            out[col + "_mean"] = float(df[col].mean())
            out[col + "_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")
    for metric in ["RMSE", "MAE", "Pearson", "SD"]:
        col = "delta_" + metric
        out[col + "_mean"] = float(df[col].mean())
        out[col + "_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")
    out["splits_completed"] = int(len(df))
    return out



def dataframe_to_markdown(df):
    """Small pandas-to-markdown helper without tabulate dependency, Python 3.6 compatible."""
    columns = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append(["" if pd.isnull(row[c]) else str(row[c]) for c in df.columns])
    widths = []
    for idx, col in enumerate(columns):
        values = [r[idx] for r in rows]
        widths.append(max([len(col)] + [len(v) for v in values]))

    def fmt_row(values):
        cells = []
        for idx, value in enumerate(values):
            cells.append(" " + str(value).ljust(widths[idx]) + " ")
        return "|" + "|".join(cells) + "|"

    header = fmt_row(columns)
    sep = "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"
    body = [fmt_row(r) for r in rows]
    return "\n".join([header, sep] + body)


def write_markdown_report(path, aggregate, comparison_rows, args, checkpoint_path):
    df = pd.DataFrame(comparison_rows).sort_values("split")
    cols = [
        "split", "test_rows", "pretrained_RMSE", "finetuned_RMSE", "delta_RMSE",
        "pretrained_MAE", "finetuned_MAE", "delta_MAE",
        "pretrained_Pearson", "finetuned_Pearson", "delta_Pearson", "best_epoch", "epochs_completed",
    ]
    table = df[cols].copy()
    for col in table.columns:
        if table[col].dtype.kind in "fc":
            table[col] = table[col].map(lambda x: "{0:.4f}".format(x))
    md = []
    md.append("# CAPLA pretrained original vs fine-tuned on URV v3b\n")
    md.append("## Configuration\n")
    md.append("- Dataset: `{0}`".format(args.dataset_dir))
    md.append("- Original checkpoint: `{0}`".format(checkpoint_path))
    md.append("- Epochs requested: `{0}`".format(args.epochs))
    md.append("- Learning rate: `{0}`".format(args.lr))
    md.append("- Early stopping rounds: `{0}`".format(args.early_stopping_rounds))
    md.append("- Minimum epochs before stopping: `{0}`\n".format(args.min_epochs_before_stopping))
    md.append("## Aggregate metrics\n")
    md.append("| Experiment | RMSE | MAE | Pearson | SD |")
    md.append("|---|---:|---:|---:|---:|")
    md.append("| Pretrained original | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} ± {5:.4f} | {6:.4f} ± {7:.4f} |".format(
        aggregate["pretrained_RMSE_mean"], aggregate["pretrained_RMSE_std"],
        aggregate["pretrained_MAE_mean"], aggregate["pretrained_MAE_std"],
        aggregate["pretrained_Pearson_mean"], aggregate["pretrained_Pearson_std"],
        aggregate["pretrained_SD_mean"], aggregate["pretrained_SD_std"],
    ))
    md.append("| Fine-tuned from original | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} ± {5:.4f} | {6:.4f} ± {7:.4f} |".format(
        aggregate["finetuned_RMSE_mean"], aggregate["finetuned_RMSE_std"],
        aggregate["finetuned_MAE_mean"], aggregate["finetuned_MAE_std"],
        aggregate["finetuned_Pearson_mean"], aggregate["finetuned_Pearson_std"],
        aggregate["finetuned_SD_mean"], aggregate["finetuned_SD_std"],
    ))
    md.append("\n## Per-split comparison\n")
    md.append(dataframe_to_markdown(table))
    md.append("\n## Interpretation\n")
    md.append("- `delta_RMSE < 0` means fine-tuning improved RMSE.")
    md.append("- `delta_MAE < 0` means fine-tuning improved MAE.")
    md.append("- `delta_Pearson > 0` means fine-tuning improved correlation.")
    Path(path).write_text("\n".join(md), encoding="utf-8")



def run_report_only(args):
    results_dir = ensure_dir(resolve_path(args.results_dir, SCRIPT_DIR, must_exist=False))
    comparison_csv = results_dir / "Comparison_per_split.csv"
    aggregate_json = results_dir / "Comparison_aggregate.json"
    if not comparison_csv.is_file():
        raise RuntimeError("Missing Comparison_per_split.csv: {0}".format(comparison_csv))
    if not aggregate_json.is_file():
        raise RuntimeError("Missing Comparison_aggregate.json: {0}".format(aggregate_json))
    df = pd.read_csv(str(comparison_csv))
    with open(str(aggregate_json), "r") as handle:
        aggregate = json.load(handle)
    checkpoint = aggregate.get("checkpoint_original", args.checkpoint_original)
    rows = df.to_dict(orient="records")
    write_markdown_report(results_dir / "Comparison_report.md", aggregate, rows, args, checkpoint)
    print("Report regenerated")
    print("  report: {0}".format(results_dir / "Comparison_report.md"))
    return aggregate


def run_all(args):
    global args_global
    args_global = args
    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR, must_exist=True)
    checkpoint = resolve_path(args.checkpoint_original, SCRIPT_DIR, must_exist=True)
    results_dir = ensure_dir(resolve_path(args.results_dir, SCRIPT_DIR, must_exist=False))
    ensure_dir(resolve_path(args.models_dir, SCRIPT_DIR, must_exist=False))
    index_df, report, paths = load_index(dataset_dir)
    device = choose_device(args.device)
    split_ids = parse_splits(args.splits)
    print("Dataset valid rows: {0}".format(len(index_df)))
    print("Checkpoint: {0}".format(checkpoint))
    print("Device: {0}".format(device))
    print("Splits: {0}".format(",".join(str(x) for x in split_ids)))
    rows = []
    started = time.time()
    for split_id in split_ids:
        rows.append(run_split(args, split_id, index_df, dataset_dir, checkpoint, device))
    df = pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
    comparison_csv = results_dir / "Comparison_per_split.csv"
    df.to_csv(comparison_csv, index=False)
    aggregate = aggregate_comparisons(rows)
    aggregate["checkpoint_original"] = str(checkpoint)
    aggregate["dataset_dir"] = str(dataset_dir)
    aggregate["elapsed_seconds"] = float(time.time() - started)
    save_json(aggregate, results_dir / "Comparison_aggregate.json")
    write_markdown_report(results_dir / "Comparison_report.md", aggregate, rows, args, checkpoint)
    print("Comparison completed")
    print("  per split: {0}".format(comparison_csv))
    print("  aggregate: {0}".format(results_dir / "Comparison_aggregate.json"))
    print("  report: {0}".format(results_dir / "Comparison_report.md"))
    return aggregate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate original pretrained CAPLA and fine-tune it on URV v3b official splits.")
    parser.add_argument("--mode", default="all", choices=["debug", "all", "report"])
    parser.add_argument("--dataset-dir", default="../data/urv_dataset_v3b_prepared")
    parser.add_argument("--checkpoint-original", default="../models/pretrained/best_model.pt")
    parser.add_argument("--results-dir", default="../outputs/pretrained_vs_finetuned")
    parser.add_argument("--models-dir", default="../models/finetuned")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--splits", default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=DEFAULT_MIN_EPOCHS_BEFORE_STOPPING)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-pkt-len", type=int, default=DEFAULT_MAX_PKT_LEN)
    parser.add_argument("--max-smi-len", type=int, default=DEFAULT_MAX_SMI_LEN)
    return parser.parse_args(argv)


def validate_args(args):
    if args.epochs < 1:
        raise RuntimeError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    if args.lr <= 0:
        raise RuntimeError("--lr must be > 0")
    if args.weight_decay < 0:
        raise RuntimeError("--weight-decay must be >= 0")
    if args.early_stopping_rounds < 1:
        raise RuntimeError("--early-stopping-rounds must be >= 1")
    if args.min_epochs_before_stopping < 1:
        raise RuntimeError("--min-epochs-before-stopping must be >= 1")
    if args.num_workers < 0:
        raise RuntimeError("--num-workers must be >= 0")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    started = time.time()
    try:
        if args.mode == "debug":
            run_debug(args)
        elif args.mode == "report":
            run_report_only(args)
        else:
            run_all(args)
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        raise
    print("Elapsed seconds: {0:.1f}".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
