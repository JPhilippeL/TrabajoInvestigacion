#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run DEAttentionDTA on the official URV 5-split protocol used in this TFM.

Python 3.6 compatible.

This script is intentionally standalone because the existing TFM_Implementation
helper scripts use Python 3.9+ annotations and random k-fold logic. Here we use
only the official files inside:

    data/urv_dataset_v3b/Info.csv
    data/urv_dataset_v3b/Splits/*.txt

Default outputs:

    data/urv_dataset_v3b_prepared/
    models/from_scratch/
    outputs/from_scratch/

Examples
--------
Prepare and validate the CSVs only:

    python DEAttentionDTA/core/Run_URV_5Splits.py --mode prepare

Run a fast dry forward pass:

    python DEAttentionDTA/core/Run_URV_5Splits.py --mode debug --device cpu --batch-size 2

Train and test the 5 official splits using already prepared Version2-derived positions:

    python DEAttentionDTA/core/Run_URV_5Splits.py --mode all --skip-prepare --device cuda --epochs 50 --batch-size 16

Train only one split for testing the pipeline:

    python DEAttentionDTA/core/Run_URV_5Splits.py --mode all --splits 1 --device cpu --epochs 1 --batch-size 4
"""

from __future__ import print_function

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


SMI_CHAR = {
    "<MASK>": 0, "C": 1, ")": 2, "(": 3, "c": 4, "O": 5, "]": 6, "[": 7,
    "@": 8, "1": 9, "=": 10, "H": 11, "N": 12, "2": 13, "n": 14,
    "3": 15, "o": 16, "+": 17, "-": 18, "S": 19, "F": 20, "p": 21,
    "l": 22, "/": 23, "4": 24, "#": 25, "B": 26, "\\": 27, "5": 28,
    "r": 29, "s": 30, "6": 31, "I": 32, "7": 33, "%": 34, "8": 35,
    "e": 36, "P": 37, "9": 38, "R": 39, "u": 40, "0": 41, "i": 42,
    ".": 43, "A": 44, "t": 45, "h": 46, "V": 47, "g": 48, "b": 49,
    "Z": 50, "T": 51, "M": 52,
}

PROTEIN_CHAR = {
    "<MASK>": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
    "H": 7, "K": 8, "I": 9, "L": 10, "M": 11, "N": 12, "P": 13,
    "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "Y": 19, "W": 20,
}

# Fallback Mpro pocket residues used when the URV dataset only provides Info.csv and no
# interaction JSON/PDB residue maps. These are 1-based positions.
M_PRO_BINDING_SITE_RESIDUES = [
    24, 25, 26, 27, 41, 44, 45, 46, 49, 54, 140, 141, 142, 143, 144,
    145, 163, 164, 165, 166, 167, 168, 172, 187, 188, 189, 190, 191,
]

DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_SMI_LEN = 256
SCRIPT_NAME = "Run_URV_5Splits"
MODEL_NAME = "DEAttentionDTA"


def repo_root_from_script():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def join_root(repo_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def read_split_file(path):
    with open(path, "r") as handle:
        value = ast.literal_eval(handle.read())
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("Expected 5 split lists in {0}".format(path))
    out = []
    for split in value:
        out.append([str(x).strip().upper() for x in split])
    return out


def load_official_splits(urv_dir):
    split_dir = os.path.join(urv_dir, "Splits")
    return {
        "train": read_split_file(os.path.join(split_dir, "train_index_folder.txt")),
        "valid": read_split_file(os.path.join(split_dir, "valid_index_folder.txt")),
        "test": read_split_file(os.path.join(split_dir, "test_index_folder.txt")),
    }


def unknown_chars(text, vocab):
    return sorted(set(str(text)) - set(vocab.keys()))


def pocket_positions_for_sequence(sequence):
    seq_len = len(sequence)
    return [pos for pos in M_PRO_BINDING_SITE_RESIDUES if 1 <= pos <= seq_len]


def pocket_string(sequence, positions):
    chars = []
    for pos in positions:
        if 1 <= pos <= len(sequence):
            chars.append(sequence[pos - 1])
    return "".join(chars)


def select_consensus_sequence(info_df):
    valid = info_df["Proteine Sequence"].astype(str).str.strip().str.upper()
    valid = valid[valid.str.len() > 0]
    if len(valid) == 0:
        return ""
    return valid.value_counts().index[0]


def prepare_urv_v3b(args, repo_root):
    urv_dir = join_root(repo_root, args.urv_dir)
    prepared_dir = ensure_dir(join_root(repo_root, args.prepared_dir))
    split_out_dir = ensure_dir(os.path.join(prepared_dir, "splits"))
    reports_dir = ensure_dir(os.path.join(prepared_dir, "reports"))

    info_path = os.path.join(urv_dir, "Info.csv")
    if not os.path.isfile(info_path):
        raise IOError("Info.csv not found: {0}".format(info_path))

    splits = load_official_splits(urv_dir)
    info_df = pd.read_csv(info_path, sep=";", dtype=str, keep_default_na=False)
    required = ["PDB_ID", "SMILES", "Proteine Sequence", "pIC50"]
    missing = [col for col in required if col not in info_df.columns]
    if missing:
        raise ValueError("Info.csv is missing columns: {0}".format(missing))

    info_df["PDBname"] = info_df["PDB_ID"].astype(str).str.strip().str.upper()
    info_df["Smile"] = info_df["SMILES"].astype(str).str.strip()
    info_df["Sequence"] = info_df["Proteine Sequence"].astype(str).str.strip().str.upper()
    info_df["affinity"] = pd.to_numeric(info_df["pIC50"], errors="coerce")

    consensus_sequence = select_consensus_sequence(info_df)
    rows = []
    dropped = []
    imputed = []

    for _, row in info_df.iterrows():
        pdb_id = row["PDBname"]
        smile = str(row["Smile"]).strip()
        sequence = str(row["Sequence"]).strip().upper()
        affinity = row["affinity"]
        reasons = []

        if not pdb_id:
            reasons.append("missing_pdb_id")
        if not smile:
            reasons.append("missing_smiles")
        if pd.isnull(affinity):
            reasons.append("missing_or_invalid_pIC50")
        if unknown_chars(smile, SMI_CHAR):
            reasons.append("unknown_smiles_tokens:" + "".join(unknown_chars(smile, SMI_CHAR)))
        if sequence and unknown_chars(sequence, PROTEIN_CHAR):
            reasons.append("unknown_sequence_tokens:" + "".join(unknown_chars(sequence, PROTEIN_CHAR)))

        if not sequence:
            if args.zero_seq_policy == "impute-consensus" and consensus_sequence:
                sequence = consensus_sequence
                imputed.append(pdb_id)
            else:
                reasons.append("empty_sequence")

        positions = pocket_positions_for_sequence(sequence) if sequence else []
        if not positions:
            reasons.append("empty_position_list")

        if reasons:
            dropped.append({
                "PDBname": pdb_id,
                "DropReasons": ";".join(reasons),
                "SMILES": smile,
                "pIC50": row.get("pIC50", ""),
            })
            continue

        rows.append({
            "PDBname": pdb_id,
            "Smile": smile,
            "Sequence": sequence,
            "Pocket": pocket_string(sequence, positions),
            "Position": str(list(positions)),
            "affinity": float(affinity),
        })

    prepared_df = pd.DataFrame(rows).drop_duplicates(subset=["PDBname"], keep="first")
    prepared_df = prepared_df.sort_values("PDBname").reset_index(drop=True)
    affinity_df = prepared_df[["PDBname", "affinity"]].copy()
    seq_df = prepared_df[["PDBname", "Smile", "Sequence", "Pocket", "Position"]].copy()

    affinity_all_path = os.path.join(prepared_dir, "affinity_all.csv")
    seq_all_path = os.path.join(prepared_dir, "seq_data_all.csv")
    dropped_path = os.path.join(reports_dir, "dropped_rows.csv")
    affinity_df.to_csv(affinity_all_path, index=False)
    seq_df.to_csv(seq_all_path, index=False)
    pd.DataFrame(dropped).to_csv(dropped_path, index=False)

    available_ids = set(prepared_df["PDBname"].tolist())
    original_ids = set(info_df["PDBname"].tolist())

    manifest_rows = []
    split_reports = []
    for split_idx in range(5):
        split_name = "split_{0:02d}".format(split_idx + 1)
        split_dir = ensure_dir(os.path.join(split_out_dir, split_name))
        split_record = {"split": split_idx + 1}

        for role in ["train", "valid", "test"]:
            ids = splits[role][split_idx]
            ids_set = set(ids)
            missing_original = sorted(ids_set - original_ids)
            dropped_in_role = sorted(ids_set - available_ids)
            role_df = prepared_df[prepared_df["PDBname"].isin(ids_set)].copy()
            role_df = role_df.sort_values("PDBname").reset_index(drop=True)
            seq_role_path = os.path.join(split_dir, "seq_{0}.csv".format(role))
            aff_role_path = os.path.join(split_dir, "affinity_{0}.csv".format(role))
            role_df[["PDBname", "Smile", "Sequence", "Pocket", "Position"]].to_csv(seq_role_path, index=False)
            role_df[["PDBname", "affinity"]].to_csv(aff_role_path, index=False)

            manifest_rows.append({
                "split": split_idx + 1,
                "role": role,
                "official_ids": len(ids),
                "exported_rows": len(role_df),
                "missing_from_info": len(missing_original),
                "dropped_after_preparation": len(dropped_in_role),
                "seq_csv": seq_role_path,
                "affinity_csv": aff_role_path,
            })
            split_record[role + "_official"] = len(ids)
            split_record[role + "_exported"] = len(role_df)
            split_record[role + "_dropped_ids"] = dropped_in_role

        train_set = set(splits["train"][split_idx])
        valid_set = set(splits["valid"][split_idx])
        test_set = set(splits["test"][split_idx])
        split_record["overlap_train_valid"] = len(train_set & valid_set)
        split_record["overlap_train_test"] = len(train_set & test_set)
        split_record["overlap_valid_test"] = len(valid_set & test_set)
        split_reports.append(split_record)

    manifest_path = os.path.join(prepared_dir, "split_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    report = {
        "script": SCRIPT_NAME,
        "mode": "prepare",
        "info_csv": info_path,
        "urv_dir": urv_dir,
        "prepared_dir": prepared_dir,
        "zero_seq_policy": args.zero_seq_policy,
        "input_rows": int(len(info_df)),
        "prepared_rows": int(len(prepared_df)),
        "dropped_rows": int(len(dropped)),
        "imputed_empty_sequence_ids": imputed,
        "affinity_all_csv": affinity_all_path,
        "seq_all_csv": seq_all_path,
        "manifest_csv": manifest_path,
        "dropped_rows_csv": dropped_path,
        "split_reports": split_reports,
        "position_source": "fallback_Mpro_binding_site_residues",
        "position_residues_1_based": M_PRO_BINDING_SITE_RESIDUES,
    }
    report_path = os.path.join(reports_dir, "prepare_report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("Prepared URV dataset")
    print("  rows in Info.csv:       {0}".format(len(info_df)))
    print("  prepared rows:          {0}".format(len(prepared_df)))
    print("  dropped rows:           {0}".format(len(dropped)))
    if imputed:
        print("  imputed empty sequence: {0}".format(", ".join(imputed)))
    print("  prepared_dir:           {0}".format(prepared_dir))
    print("  manifest:               {0}".format(manifest_path))
    return report


def parse_position(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def position_seq(seq, positions):
    res = ["<MASK>"] * len(seq)
    for pos in positions:
        if 1 <= pos <= len(seq):
            res[pos - 1] = seq[pos - 1]
        else:
            raise ValueError("Position {0} out of range for sequence length {1}".format(pos, len(seq)))
    return res


def encode_smiles(smile, max_len):
    arr = np.zeros(max_len, dtype=np.int64)
    for i, token in enumerate(str(smile)[:max_len]):
        arr[i] = SMI_CHAR[token]
    return arr


def encode_sequence(seq_or_tokens, max_len):
    arr = np.zeros(max_len, dtype=np.int64)
    if isinstance(seq_or_tokens, str):
        tokens = list(seq_or_tokens[:max_len])
    else:
        tokens = list(seq_or_tokens)[:max_len]
    for i, token in enumerate(tokens):
        arr[i] = PROTEIN_CHAR[token]
    return arr


class URVDEAttentionDataset(Dataset):
    def __init__(self, seq_csv, affinity_csv, max_seq_len, max_smi_len):
        seq_df = pd.read_csv(seq_csv)
        aff_df = pd.read_csv(affinity_csv)
        seq_df["PDBname"] = seq_df["PDBname"].astype(str).str.strip().str.upper()
        aff_df["PDBname"] = aff_df["PDBname"].astype(str).str.strip().str.upper()
        aff_df["affinity"] = pd.to_numeric(aff_df["affinity"], errors="coerce")
        merged = seq_df.merge(aff_df[["PDBname", "affinity"]], on="PDBname", how="inner")
        merged = merged.sort_values("PDBname").reset_index(drop=True)
        if len(merged) == 0:
            raise ValueError("Dataset has zero rows after merging {0} and {1}".format(seq_csv, affinity_csv))
        self.df = merged
        self.max_seq_len = int(max_seq_len)
        self.max_smi_len = int(max_smi_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pdb_id = str(row["PDBname"])
        smile = str(row["Smile"])
        sequence = str(row["Sequence"])
        positions = parse_position(row["Position"])
        pocket = position_seq(sequence, positions)
        smi_encode = torch.tensor(encode_smiles(smile, self.max_smi_len)).long()
        seq_encode = torch.tensor(encode_sequence(sequence, self.max_seq_len)).long()
        pocket_encode = torch.tensor(encode_sequence(pocket, self.max_seq_len)).long()
        affinity = torch.tensor(float(row["affinity"]), dtype=torch.float32)
        return pdb_id, smi_encode, seq_encode, pocket_encode, affinity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_class(repo_root):
    bestmodel_path = os.path.join(repo_root, "original", "src", "bestmodel.py")
    if not os.path.isfile(bestmodel_path):
        raise IOError("Cannot find upstream model file: {0}".format(bestmodel_path))
    spec = importlib.util.spec_from_file_location("deattentiondta_bestmodel", bestmodel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyModule"):
        raise RuntimeError("original/src/bestmodel.py does not expose MyModule")
    return module.MyModule, bestmodel_path


def to_device(batch, device):
    pdb_id, smi, seq, pocket, affinity = batch
    return (
        pdb_id,
        smi.to(device),
        seq.to(device),
        pocket.to(device),
        affinity.to(device),
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2)) if len(diff) else float("nan")
    rmse = math.sqrt(mse) if not math.isnan(mse) else float("nan")
    mae = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
    if len(y_true) >= 2 and float(np.std(y_true)) > 0.0 and float(np.std(y_pred)) > 0.0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "Pearson": pearson, "N": int(len(y_true))}


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true = []
    y_pred = []
    ids = []
    with torch.no_grad():
        for batch in dataloader:
            pdb_id, smi, seq, pocket, affinity = to_device(batch, device)
            output = model(seq, smi, pocket).reshape(-1)
            target = affinity.reshape(-1)
            loss = criterion(output, target)
            n = int(target.numel())
            total_loss += float(loss.item()) * n
            total_n += n
            ids.extend(list(pdb_id))
            y_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
            y_pred.extend(output.detach().cpu().numpy().reshape(-1).tolist())
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / float(total_n) if total_n else float("nan")
    return metrics, ids, y_true, y_pred


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, split_id):
    model.train()
    total_loss = 0.0
    total_n = 0
    iterator = tqdm(dataloader, desc="split {0:02d} train epoch {1:03d}".format(split_id, epoch))
    for batch in iterator:
        _, smi, seq, pocket, affinity = to_device(batch, device)
        optimizer.zero_grad()
        output = model(seq, smi, pocket).reshape(-1)
        target = affinity.reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        n = int(target.numel())
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / float(total_n) if total_n else float("nan")


def save_predictions(path, ids, y_true, y_pred, split_id, role):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "role", "PDBname", "y_true", "y_pred"])
        for pdb_id, true_value, pred_value in zip(ids, y_true, y_pred):
            writer.writerow([split_id, role, pdb_id, true_value, pred_value])


def save_metrics(path, metrics, split_id, role):
    row = {"split": split_id, "role": role}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(path, index=False)


def save_scatter(path, y_true, y_pred, title):
    if plt is None:
        return False
    if len(y_true) == 0:
        return False
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, alpha=0.75)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Real pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def split_paths(prepared_dir, split_id):
    split_dir = os.path.join(prepared_dir, "splits", "split_{0:02d}".format(split_id))
    return {
        "train_seq": os.path.join(split_dir, "seq_train.csv"),
        "train_aff": os.path.join(split_dir, "affinity_train.csv"),
        "valid_seq": os.path.join(split_dir, "seq_valid.csv"),
        "valid_aff": os.path.join(split_dir, "affinity_valid.csv"),
        "test_seq": os.path.join(split_dir, "seq_test.csv"),
        "test_aff": os.path.join(split_dir, "affinity_test.csv"),
    }


def ensure_prepared_exists(args, repo_root):
    prepared_dir = join_root(repo_root, args.prepared_dir)
    required = []
    for split_id in parse_splits_arg('all'):
        paths = split_paths(prepared_dir, split_id)
        required.extend([paths['train_seq'], paths['train_aff'], paths['valid_seq'], paths['valid_aff'], paths['test_seq'], paths['test_aff']])
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise IOError('Prepared files are missing. Run Prepare_URV_Positions_From_V2_Dataset.py first or omit --skip-prepare. First missing: {0}'.format(missing[0]))
    return prepared_dir


def maybe_prepare(args, repo_root):
    if getattr(args, 'skip_prepare', False):
        return ensure_prepared_exists(args, repo_root)
    prepare_urv_v3b(args, repo_root)
    return join_root(repo_root, args.prepared_dir)


def run_debug(args, repo_root):
    prepared_dir = maybe_prepare(args, repo_root)
    paths = split_paths(prepared_dir, 1)
    device = choose_device(args.device)
    model_cls, bestmodel_path = load_model_class(repo_root)
    dataset = URVDEAttentionDataset(paths["train_seq"], paths["train_aff"], args.max_seq_len, args.max_smi_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model_cls().to(device)
    model.eval()
    first_batch = next(iter(loader))
    _, smi, seq, pocket, affinity = to_device(first_batch, device)
    with torch.no_grad():
        output = model(seq, smi, pocket).reshape(-1)
    report = {
        "script": SCRIPT_NAME,
        "mode": "debug",
        "device": str(device),
        "bestmodel_path": bestmodel_path,
        "batch_size": int(args.batch_size),
        "output_shape": list(output.detach().cpu().size()),
        "target_shape": list(affinity.detach().cpu().size()),
        "output_has_nan": bool(torch.isnan(output.detach().cpu()).any().item()),
        "output_has_inf": bool(torch.isinf(output.detach().cpu()).any().item()),
        "train_split_1_rows": int(len(dataset)),
    }
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    path = os.path.join(results_dir, "debug_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("Debug forward pass OK")
    print("  device:       {0}".format(device))
    print("  output_shape: {0}".format(report["output_shape"]))
    print("  report:       {0}".format(path))
    return report


def train_and_test_split(args, repo_root, split_id):
    prepared_dir = join_root(repo_root, args.prepared_dir)
    models_dir = ensure_dir(join_root(repo_root, args.models_dir))
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    split_result_dir = ensure_dir(os.path.join(results_dir, "split_{0:02d}".format(split_id)))
    split_model_dir = ensure_dir(os.path.join(models_dir, "split_{0:02d}".format(split_id)))
    paths = split_paths(prepared_dir, split_id)

    for key in paths:
        if not os.path.isfile(paths[key]):
            raise IOError("Missing prepared split file {0}: {1}".format(key, paths[key]))

    set_seed(args.seed + split_id)
    device = choose_device(args.device)
    model_cls, bestmodel_path = load_model_class(repo_root)

    train_dataset = URVDEAttentionDataset(paths["train_seq"], paths["train_aff"], args.max_seq_len, args.max_smi_len)
    valid_dataset = URVDEAttentionDataset(paths["valid_seq"], paths["valid_aff"], args.max_seq_len, args.max_smi_len)
    test_dataset = URVDEAttentionDataset(paths["test_seq"], paths["test_aff"], args.max_seq_len, args.max_smi_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = model_cls().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    rounds_without_improvement = 0
    history_rows = []

    print("Training split {0}/5 | train={1} valid={2} test={3} | device={4}".format(
        split_id, len(train_dataset), len(valid_dataset), len(test_dataset), device
    ))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, split_id)
        val_metrics, _, _, _ = evaluate(model, valid_loader, criterion, device)
        history_rows.append({
            "split": split_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_Pearson": val_metrics["Pearson"],
        })
        print("split {0:02d} epoch {1:03d} train_loss={2:.6f} val_loss={3:.6f} val_RMSE={4:.6f}".format(
            split_id, epoch, train_loss, val_metrics["loss"], val_metrics["RMSE"]
        ))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if args.early_stopping_rounds > 0 and rounds_without_improvement >= args.early_stopping_rounds:
                print("Early stopping split {0} at epoch {1}".format(split_id, epoch))
                break

    if best_state is None:
        raise RuntimeError("No best state was captured for split {0}".format(split_id))

    model.load_state_dict(best_state)
    val_metrics, val_ids, val_true, val_pred = evaluate(model, valid_loader, criterion, device)
    test_metrics, test_ids, test_true, test_pred = evaluate(model, test_loader, criterion, device)

    history_path = os.path.join(split_result_dir, "train_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    save_predictions(os.path.join(split_result_dir, "Predictions_validation.csv"), val_ids, val_true, val_pred, split_id, "validation")
    save_predictions(os.path.join(split_result_dir, "Predictions_test.csv"), test_ids, test_true, test_pred, split_id, "test")
    save_metrics(os.path.join(split_result_dir, "Metrics_validation.csv"), val_metrics, split_id, "validation")
    save_metrics(os.path.join(split_result_dir, "Metrics_test.csv"), test_metrics, split_id, "test")
    save_scatter(os.path.join(split_result_dir, "Scatter_validation.png"), val_true, val_pred,
                 "DEAttentionDTA URV split {0} validation".format(split_id))
    save_scatter(os.path.join(split_result_dir, "Scatter_test.png"), test_true, test_pred,
                 "DEAttentionDTA URV split {0} test".format(split_id))

    model_path = os.path.join(split_model_dir, "DEAttentionDTA_URV_v3b_split{0:02d}.pt".format(split_id))
    bundle = {
        "format_version": "urv_5split_v1",
        "model_name": MODEL_NAME,
        "script": SCRIPT_NAME,
        "split_id": int(split_id),
        "best_epoch": int(best_epoch),
        "state_dict": best_state,
        "bestmodel_path": bestmodel_path,
        "input_paths": paths,
        "max_seq_len": int(args.max_seq_len),
        "max_smi_len": int(args.max_smi_len),
        "vocab": {"SMI_CHAR": SMI_CHAR, "PROTEIN_CHAR": PROTEIN_CHAR},
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed + split_id),
        },
        "metrics": {"validation": val_metrics, "test": test_metrics},
    }
    torch.save(bundle, model_path)

    summary = {
        "split": int(split_id),
        "model_pt": model_path,
        "results_dir": split_result_dir,
        "best_epoch": int(best_epoch),
        "train_rows": int(len(train_dataset)),
        "validation_rows": int(len(valid_dataset)),
        "test_rows": int(len(test_dataset)),
        "val_RMSE": val_metrics["RMSE"],
        "val_MAE": val_metrics["MAE"],
        "val_Pearson": val_metrics["Pearson"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_Pearson": test_metrics["Pearson"],
    }
    with open(os.path.join(split_result_dir, "split_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("Finished split {0}: test_RMSE={1:.6f} model={2}".format(split_id, test_metrics["RMSE"], model_path))
    return summary


def parse_splits_arg(value):
    if value.strip().lower() in ("all", "*"):
        return [1, 2, 3, 4, 5]
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        split_id = int(part)
        if split_id < 1 or split_id > 5:
            raise ValueError("Split id must be between 1 and 5: {0}".format(split_id))
        out.append(split_id)
    return sorted(set(out))


def run_all(args, repo_root):
    maybe_prepare(args, repo_root)
    summaries = []
    for split_id in parse_splits_arg(args.splits):
        summaries.append(train_and_test_split(args, repo_root, split_id))

    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    summary_csv = os.path.join(results_dir, "Summary_5splits.csv")
    summary_df = pd.DataFrame(summaries).sort_values("split")
    summary_df.to_csv(summary_csv, index=False)

    if len(summary_df) > 0:
        aggregate = {
            "splits_completed": int(len(summary_df)),
            "test_RMSE_mean": float(summary_df["test_RMSE"].mean()),
            "test_RMSE_std": float(summary_df["test_RMSE"].std(ddof=1)) if len(summary_df) > 1 else float("nan"),
            "test_MAE_mean": float(summary_df["test_MAE"].mean()),
            "test_MAE_std": float(summary_df["test_MAE"].std(ddof=1)) if len(summary_df) > 1 else float("nan"),
            "test_Pearson_mean": float(summary_df["test_Pearson"].mean()),
            "test_Pearson_std": float(summary_df["test_Pearson"].std(ddof=1)) if len(summary_df) > 1 else float("nan"),
        }
    else:
        aggregate = {"splits_completed": 0}

    aggregate_path = os.path.join(results_dir, "Aggregate_metrics.json")
    with open(aggregate_path, "w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)

    print("All requested splits completed")
    print("  summary_csv: {0}".format(summary_csv))
    print("  aggregate:   {0}".format(aggregate_path))
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description="DEAttentionDTA official URV 5-split runner, Python 3.6 compatible.")
    parser.add_argument("--mode", choices=["prepare", "debug", "all"], default="all")
    parser.add_argument("--urv-dir", default="data/urv_dataset_v3b", help="Path to URV dataset directory containing Info.csv and Splits/. Used only when --mode prepare or when --skip-prepare is not set.")
    parser.add_argument("--prepared-dir", default="data/urv_dataset_v3b_prepared", help="Path to prepared DEAttentionDTA CSVs generated by Prepare_URV_Positions_From_V2_Dataset.py.")
    parser.add_argument("--models-dir", default="models/from_scratch", help="Output directory for trained .pt models, one subfolder per split.")
    parser.add_argument("--results-dir", default="outputs/from_scratch", help="Output directory for metrics, predictions, scatter plots and training histories.")
    parser.add_argument("--zero-seq-policy", choices=["drop", "impute-consensus"], default="drop")
    parser.add_argument("--skip-prepare", action="store_true", help="Do not regenerate prepared CSVs. Use this after running Prepare_URV_Positions_From_V2_Dataset.py so Version2-derived Position/Pocket are not overwritten.")
    parser.add_argument("--splits", default="all", help="Comma-separated split ids, e.g. 1,2,3,4,5 or all")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4, help="Learning rate used by Adam. Alias: --learning-rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=990721)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-smi-len", type=int, default=DEFAULT_MAX_SMI_LEN)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = repo_root_from_script()
    started = time.time()
    try:
        if args.mode == "prepare":
            prepare_urv_v3b(args, repo_root)
        elif args.mode == "debug":
            run_debug(args, repo_root)
        else:
            run_all(args, repo_root)
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    print("Elapsed seconds: {0:.1f}".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
