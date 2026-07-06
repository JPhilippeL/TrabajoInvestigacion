"""Dataset utilities for CAPLA.

The original CAPLA repository couples dataset resolution to fixed phase names such
as ``training`` or ``Test2016_290``. This module replaces that with explicit paths
for affinity, SMILES, global features, and pocket features.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import ensure_dir

CHAR_SMI_SET: Dict[str, int] = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
    "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
    "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
    "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
    "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
    "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
    "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
    "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
}
CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40
KNOWN_AFFINITY_COLUMNS = ["-logKd/Ki", "-logkd/ki", "affinity", "pic50", "target"]


class DatasetPaths(object):
    """Explicit CAPLA dataset paths."""

    def __init__(self, affinity_csv, smi_csv, global_dir, pocket_dir):
        self.affinity_csv = Path(affinity_csv)
        self.smi_csv = Path(smi_csv)
        self.global_dir = Path(global_dir)
        self.pocket_dir = Path(pocket_dir)


class ValidationReport(object):
    """Structured validation output for CAPLA inputs."""

    def __init__(
        self,
        affinity_csv,
        smi_csv,
        global_dir,
        pocket_dir,
        affinity_column,
        affinity_rows,
        smi_rows,
        duplicated_affinity_ids,
        duplicated_smi_ids,
        affinity_null_rows,
        smi_null_rows,
        missing_global_ids,
        missing_pocket_ids,
        orphan_global_ids,
        orphan_pocket_ids,
        common_ids_count,
        valid_ids_count,
        unknown_smiles,
        invalid_global_files,
        invalid_pocket_files,
    ):
        self.affinity_csv = affinity_csv
        self.smi_csv = smi_csv
        self.global_dir = global_dir
        self.pocket_dir = pocket_dir
        self.affinity_column = affinity_column
        self.affinity_rows = affinity_rows
        self.smi_rows = smi_rows
        self.duplicated_affinity_ids = duplicated_affinity_ids
        self.duplicated_smi_ids = duplicated_smi_ids
        self.affinity_null_rows = affinity_null_rows
        self.smi_null_rows = smi_null_rows
        self.missing_global_ids = missing_global_ids
        self.missing_pocket_ids = missing_pocket_ids
        self.orphan_global_ids = orphan_global_ids
        self.orphan_pocket_ids = orphan_pocket_ids
        self.common_ids_count = common_ids_count
        self.valid_ids_count = valid_ids_count
        self.unknown_smiles = unknown_smiles
        self.invalid_global_files = invalid_global_files
        self.invalid_pocket_files = invalid_pocket_files

    def to_dict(self):
        return self.__dict__.copy()


class CAPLADataError(ValueError):
    """Raised when the external dataset does not match the expected CAPLA format."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_column(df: pd.DataFrame, expected: str) -> Optional[str]:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    return lookup.get(expected.strip().lower())


def _select_affinity_column(df: pd.DataFrame) -> str:
    non_pdb_cols = [col for col in df.columns if str(col).strip().lower() != "pdbid"]
    if len(non_pdb_cols) == 1:
        return non_pdb_cols[0]
    lowered = {str(col).strip().lower(): col for col in non_pdb_cols}
    for candidate in KNOWN_AFFINITY_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    numeric_candidates = [col for col in non_pdb_cols if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise CAPLADataError(
        "Could not infer the affinity column. Expected exactly one non-'pdbid' column or a known affinity name."
    )


def encode_smiles(smiles: str, max_smi_len: int) -> np.ndarray:
    """Encode a SMILES string into the 0-based CAPLA vocabulary indices."""
    encoded = np.zeros(max_smi_len, dtype=np.int64)
    for idx, ch in enumerate(smiles[:max_smi_len]):
        if ch not in CHAR_SMI_SET:
            raise CAPLADataError(f"Unsupported SMILES character {ch!r} in string {smiles!r}")
        encoded[idx] = CHAR_SMI_SET[ch] - 1
    return encoded


def find_unknown_smiles_characters(smiles: str) -> List[str]:
    """Return the sorted list of SMILES characters not present in the original vocabulary."""
    return sorted({ch for ch in smiles if ch not in CHAR_SMI_SET})


def _read_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    for col in [col for col in df.columns if str(col).lower().startswith("unnamed:")]:
        df = df.drop(columns=[col])
    if df.shape[1] != PT_FEATURE_SIZE:
        raise CAPLADataError(
            f"Feature file {path} has {df.shape[1]} usable columns, expected {PT_FEATURE_SIZE}."
        )
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise CAPLADataError(f"Feature file {path} contains non-numeric columns: {non_numeric}")
    return df


def _load_affinity_df(affinity_csv: Path) -> Tuple[pd.DataFrame, str]:
    df = _normalize_columns(pd.read_csv(affinity_csv))
    pdbid_col = _find_column(df, "pdbid")
    if pdbid_col is None:
        raise CAPLADataError(f"Missing required column 'pdbid' in {affinity_csv}")
    affinity_col = _select_affinity_column(df)
    out = df[[pdbid_col, affinity_col]].rename(columns={pdbid_col: "pdbid", affinity_col: "affinity"})
    out["pdbid"] = out["pdbid"].astype(str).str.strip()
    return out, affinity_col


def _load_smiles_df(smi_csv: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(smi_csv))
    pdbid_col = _find_column(df, "pdbid")
    smiles_col = _find_column(df, "smiles")
    if pdbid_col is None or smiles_col is None:
        raise CAPLADataError(f"Expected columns 'pdbid' and 'smiles' in {smi_csv}")
    out = df[[pdbid_col, smiles_col]].rename(columns={pdbid_col: "pdbid", smiles_col: "smiles"})
    out["pdbid"] = out["pdbid"].astype(str).str.strip()
    out["smiles"] = out["smiles"].astype(str)
    return out


def build_dataset_index(
    affinity_csv: Path,
    smi_csv: Path,
    global_dir: Path,
    pocket_dir: Path,
    strict: bool = True,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Build a normalized dataset index and a detailed validation report.

    The returned dataframe contains one row per valid `pdbid` and the resolved
    file paths needed by the dataset loader.
    """
    affinity_df, affinity_col = _load_affinity_df(affinity_csv)
    smi_df = _load_smiles_df(smi_csv)

    duplicated_affinity_ids = sorted(affinity_df.loc[affinity_df["pdbid"].duplicated(), "pdbid"].unique().tolist())
    duplicated_smi_ids = sorted(smi_df.loc[smi_df["pdbid"].duplicated(), "pdbid"].unique().tolist())
    affinity_null_rows = int(affinity_df[["pdbid", "affinity"]].isnull().any(axis=1).sum())
    smi_null_rows = int(smi_df[["pdbid", "smiles"]].isnull().any(axis=1).sum())

    affinity_df = affinity_df.drop_duplicates(subset=["pdbid"], keep="first")
    smi_df = smi_df.drop_duplicates(subset=["pdbid"], keep="first")
    affinity_ids = set(affinity_df["pdbid"])
    smi_ids = set(smi_df["pdbid"])

    global_files = {path.stem: path for path in sorted(Path(global_dir).glob("*.csv"))}
    pocket_files = {path.stem: path for path in sorted(Path(pocket_dir).glob("*.csv"))}
    global_ids = set(global_files)
    pocket_ids = set(pocket_files)

    common_ids = affinity_ids & smi_ids
    missing_global_ids = sorted(common_ids - global_ids)
    missing_pocket_ids = sorted(common_ids - pocket_ids)
    orphan_global_ids = sorted(global_ids - common_ids)
    orphan_pocket_ids = sorted(pocket_ids - common_ids)

    unknown_smiles: Dict[str, List[str]] = {}
    for row in smi_df.itertuples(index=False):
        unknown = find_unknown_smiles_characters(row.smiles)
        if unknown:
            unknown_smiles[str(row.pdbid)] = unknown

    valid_ids = sorted(common_ids - set(missing_global_ids) - set(missing_pocket_ids))

    invalid_global_files: Dict[str, str] = {}
    invalid_pocket_files: Dict[str, str] = {}
    for pdbid in valid_ids:
        try:
            _read_feature_table(global_files[pdbid])
        except Exception as exc:  # noqa: BLE001 - we want the message in the report
            invalid_global_files[pdbid] = str(exc)
        try:
            _read_feature_table(pocket_files[pdbid])
        except Exception as exc:  # noqa: BLE001 - we want the message in the report
            invalid_pocket_files[pdbid] = str(exc)

    invalid_ids = set(invalid_global_files) | set(invalid_pocket_files)
    valid_ids = [pdbid for pdbid in valid_ids if pdbid not in invalid_ids]

    affinity_series = affinity_df.set_index("pdbid")["affinity"]
    smi_series = smi_df.set_index("pdbid")["smiles"]
    index_df = pd.DataFrame(
        {
            "pdbid": valid_ids,
            "affinity": [float(affinity_series[pdbid]) for pdbid in valid_ids],
            "smiles": [str(smi_series[pdbid]) for pdbid in valid_ids],
            "global_path": [str(global_files[pdbid]) for pdbid in valid_ids],
            "pocket_path": [str(pocket_files[pdbid]) for pdbid in valid_ids],
        }
    )

    report = ValidationReport(
        affinity_csv=str(Path(affinity_csv).resolve()),
        smi_csv=str(Path(smi_csv).resolve()),
        global_dir=str(Path(global_dir).resolve()),
        pocket_dir=str(Path(pocket_dir).resolve()),
        affinity_column=affinity_col,
        affinity_rows=len(affinity_df),
        smi_rows=len(smi_df),
        duplicated_affinity_ids=duplicated_affinity_ids,
        duplicated_smi_ids=duplicated_smi_ids,
        affinity_null_rows=affinity_null_rows,
        smi_null_rows=smi_null_rows,
        missing_global_ids=missing_global_ids,
        missing_pocket_ids=missing_pocket_ids,
        orphan_global_ids=orphan_global_ids,
        orphan_pocket_ids=orphan_pocket_ids,
        common_ids_count=len(common_ids),
        valid_ids_count=len(valid_ids),
        unknown_smiles=unknown_smiles,
        invalid_global_files=invalid_global_files,
        invalid_pocket_files=invalid_pocket_files,
    )

    if strict:
        messages: List[str] = []
        if duplicated_affinity_ids:
            messages.append(f"Duplicate affinity ids: {duplicated_affinity_ids[:10]}")
        if duplicated_smi_ids:
            messages.append(f"Duplicate SMILES ids: {duplicated_smi_ids[:10]}")
        if affinity_null_rows or smi_null_rows:
            messages.append(f"Null rows detected (affinity={affinity_null_rows}, smiles={smi_null_rows})")
        if missing_global_ids:
            messages.append(f"Missing global feature files for {len(missing_global_ids)} ids")
        if missing_pocket_ids:
            messages.append(f"Missing pocket feature files for {len(missing_pocket_ids)} ids")
        if unknown_smiles:
            messages.append(f"Unsupported SMILES characters found for {len(unknown_smiles)} ids")
        if invalid_global_files:
            messages.append(f"Invalid global files for {len(invalid_global_files)} ids")
        if invalid_pocket_files:
            messages.append(f"Invalid pocket files for {len(invalid_pocket_files)} ids")
        if messages:
            raise CAPLADataError("Dataset validation failed: " + "; ".join(messages))

    return index_df, report


def validate_capla_inputs(
    affinity_csv: Path,
    smi_csv: Path,
    global_dir: Path,
    pocket_dir: Path,
    strict: bool = False,
) -> ValidationReport:
    """Validate dataset inputs and return a structured report."""
    _, report = build_dataset_index(affinity_csv, smi_csv, global_dir, pocket_dir, strict=strict)
    return report


class CAPLADataset(Dataset):
    """PyTorch dataset for explicit CAPLA data paths."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        max_seq_len: int,
        max_pkt_len: int,
        max_smi_len: int,
    ) -> None:
        if index_df.empty:
            raise CAPLADataError("The dataset index is empty after validation.")
        self.index_df = index_df.reset_index(drop=True).copy()
        self.max_seq_len = int(max_seq_len)
        self.max_pkt_len = int(max_pkt_len)
        self.max_smi_len = int(max_smi_len)

    def __len__(self) -> int:
        return len(self.index_df)

    def _load_feature_tensor(self, path: str, max_len: int) -> np.ndarray:
        df = _read_feature_table(Path(path))
        values = df.values[:max_len]
        out = np.zeros((max_len, PT_FEATURE_SIZE), dtype=np.float32)
        out[: len(values)] = values.astype(np.float32)
        return out

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.index_df.iloc[idx]
        pdbid = str(row["pdbid"])
        seq_tensor = torch.from_numpy(self._load_feature_tensor(str(row["global_path"]), self.max_seq_len))
        pkt_tensor = torch.from_numpy(self._load_feature_tensor(str(row["pocket_path"]), self.max_pkt_len))
        smi_tensor = torch.from_numpy(encode_smiles(str(row["smiles"]), self.max_smi_len))
        affinity = torch.tensor(float(row["affinity"]), dtype=torch.float32)
        return pdbid, seq_tensor, pkt_tensor, smi_tensor, affinity


def create_debug_subset(
    affinity_csv: Path,
    smi_csv: Path,
    global_dir: Path,
    pocket_dir: Path,
    output_dir: Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> DatasetPaths:
    """Create a reproducible on-disk subset for quick debugging.

    The subset includes filtered copies of the affinity and SMILES files plus the
    corresponding `global/` and `pocket/` feature files.
    """
    index_df, _ = build_dataset_index(affinity_csv, smi_csv, global_dir, pocket_dir, strict=True)
    n_samples = min(max_samples, len(index_df))
    subset_df = index_df.sample(n=n_samples, random_state=seed).sort_values("pdbid").reset_index(drop=True)

    output_dir = ensure_dir(output_dir)
    global_out = ensure_dir(output_dir / "global")
    pocket_out = ensure_dir(output_dir / "pocket")

    affinity_df, affinity_col = _load_affinity_df(Path(affinity_csv))
    smi_df = _load_smiles_df(Path(smi_csv))
    selected_ids = set(subset_df["pdbid"])

    affinity_subset = affinity_df[affinity_df["pdbid"].isin(selected_ids)].rename(columns={"affinity": affinity_col})
    smi_subset = smi_df[smi_df["pdbid"].isin(selected_ids)]

    affinity_path = output_dir / "affinity_data_debug.csv"
    smi_path = output_dir / "dataset_debug_smi.csv"
    affinity_subset.to_csv(affinity_path, index=False)
    smi_subset.to_csv(smi_path, index=False)

    for row in subset_df.itertuples(index=False):
        shutil.copy2(row.global_path, global_out / Path(row.global_path).name)
        shutil.copy2(row.pocket_path, pocket_out / Path(row.pocket_path).name)

    return DatasetPaths(
        affinity_csv=affinity_path,
        smi_csv=smi_path,
        global_dir=global_out,
        pocket_dir=pocket_out,
    )
