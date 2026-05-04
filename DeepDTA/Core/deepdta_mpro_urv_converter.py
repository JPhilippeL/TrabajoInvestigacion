"""
@file deepdta_mpro_urv_converter.py
@author Mohamed EL BOUKHIARI
@brief Converter from MPro-URV_Version2 format to DeepDTA-compatible format.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def canonicalize_smiles_for_deepdta(smiles: str) -> str:
    """
    Convert an input SMILES into a DeepDTA-compatible canonical SMILES.

    DeepDTA uses a fixed molecular alphabet. MPro-URV contains stereochemical
    symbols such as '@', '/', and '\\', which are not supported by the original
    DeepDTA one-hot encoder. Disabling isomeric SMILES removes these symbols
    without changing the model input dimensionality.
    """
    smiles = smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def read_pic50(pic50_path: Path) -> Dict[str, float]:
    targets: Dict[str, float] = {}

    if not pic50_path.exists():
        raise FileNotFoundError(f"pIC50 file not found: {pic50_path}")

    with pic50_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            pdb_id = parts[0].strip()
            value = float(parts[1])
            targets[pdb_id] = value

    return targets


def read_smiles(smiles_dir: Path) -> Dict[str, str]:
    smiles_by_id: Dict[str, str] = {}

    if not smiles_dir.exists():
        raise FileNotFoundError(f"SMILES directory not found: {smiles_dir}")

    for smi_path in sorted(smiles_dir.glob("*.smi")):
        pdb_id = smi_path.stem.split("_")[0]

        with smi_path.open("r", encoding="utf-8") as file:
            first_line = file.readline().strip()

        if not first_line:
            continue

        raw_smiles = first_line.split()[0]
        smiles = canonicalize_smiles_for_deepdta(raw_smiles)
        smiles_by_id[pdb_id] = smiles

    if not smiles_by_id:
        raise RuntimeError(f"No .smi files found in {smiles_dir}")

    return smiles_by_id


def extract_sequence_from_pdb(pdb_path: Path) -> str:
    residues: List[Tuple[str, str, str]] = []
    seen = set()

    if not pdb_path.exists():
        raise FileNotFoundError(f"Protein PDB file not found: {pdb_path}")

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("ATOM"):
                continue

            residue_name = line[17:20].strip()
            chain_id = line[21].strip()
            residue_number = line[22:26].strip()

            key = (chain_id, residue_number, residue_name)
            if key in seen:
                continue

            seen.add(key)

            if residue_name in AA3_TO_AA1:
                residues.append(key)

    sequence = "".join(AA3_TO_AA1[residue_name] for _, _, residue_name in residues)

    if not sequence:
        raise RuntimeError(f"No protein sequence could be extracted from {pdb_path}")

    return sequence


def read_split_file(split_path: Path) -> List[List[str]]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    text = split_path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval(text)

    if not isinstance(data, list):
        raise ValueError(f"Invalid split format in {split_path}")

    return data


def convert_splits_to_indices(
    split_ids: List[List[str]],
    id_to_index: Dict[str, int],
    split_name: str,
) -> List[List[int]]:
    converted: List[List[int]] = []
    skipped = 0

    for fold_idx, fold_ids in enumerate(split_ids):
        fold_indices: List[int] = []

        for pdb_id in fold_ids:
            if pdb_id not in id_to_index:
                skipped += 1
                continue

            fold_indices.append(id_to_index[pdb_id])

        if not fold_indices:
            raise ValueError(
                f"{split_name} fold {fold_idx} is empty after DeepDTA filtering."
            )

        converted.append(fold_indices)

    if skipped:
        print(
            f"[WARNING] {split_name}: skipped {skipped} IDs because they were removed "
            "during DeepDTA filtering."
        )

    return converted


def save_json_txt(data: object, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)


def save_pickle(data: object, output_path: Path) -> None:
    with output_path.open("wb") as file:
        pickle.dump(data, file)


def find_default_protein_pdb(pdb_dir: Path) -> Path:
    if not pdb_dir.exists():
        raise FileNotFoundError(f"Protein PDB directory not found: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))

    if not pdb_files:
        raise FileNotFoundError(f"No PDB file found in {pdb_dir}")

    return pdb_files[0]


def convert_mpro_urv_to_deepdta(
    source_root: str,
    output_root: str,
    protein_pdb_path: str | None = None,
) -> None:
    source_dir = Path(source_root).resolve()
    output_dir = Path(output_root).resolve()

    pic50_path = source_dir / "pIC50.txt"
    smiles_dir = source_dir / "Ligand" / "Ligand_SMI"
    split_dir = source_dir / "Splits"
    pdb_dir = source_dir / "Protein" / "Protein_PDB"

    if protein_pdb_path is None:
        protein_path = find_default_protein_pdb(pdb_dir)
    else:
        protein_path = Path(protein_pdb_path).resolve()

    targets = read_pic50(pic50_path)
    smiles_by_id = read_smiles(smiles_dir)
    protein_sequence = extract_sequence_from_pdb(protein_path)

    common_ids_raw = sorted(set(targets) & set(smiles_by_id))

    missing_targets = sorted(set(smiles_by_id) - set(targets))
    missing_smiles = sorted(set(targets) - set(smiles_by_id))

    if missing_targets:
        print(
            f"[WARNING] {len(missing_targets)} ligands have SMILES but no pIC50 target."
        )

    if missing_smiles:
        print(f"[WARNING] {len(missing_smiles)} targets have no SMILES file.")

    if not common_ids_raw:
        raise RuntimeError("No common IDs found between pIC50.txt and Ligand_SMI.")

    canonical_smiles_by_id: Dict[str, str] = {}
    skipped_too_long: Dict[str, int] = {}

    for pdb_id in common_ids_raw:
        canonical_smiles = smiles_by_id[pdb_id]

        if len(canonical_smiles) > 50:
            skipped_too_long[pdb_id] = len(canonical_smiles)
            continue

        canonical_smiles_by_id[pdb_id] = canonical_smiles

    common_ids = sorted(canonical_smiles_by_id)

    if not common_ids:
        raise RuntimeError("No ligands left after DeepDTA SMILES length filtering.")

    if skipped_too_long:
        print(
            f"[WARNING] {len(skipped_too_long)} ligands skipped because "
            "canonical SMILES length > 50."
        )

    if len(protein_sequence) > 600:
        raise ValueError(
            f"Protein sequence length is {len(protein_sequence)}, "
            "but DeepDTA expects max protein length 600."
        )

    ligands = {pdb_id: canonical_smiles_by_id[pdb_id] for pdb_id in common_ids}
    proteins = {"Mpro": protein_sequence}

    y_values = np.array([[targets[pdb_id]] for pdb_id in common_ids], dtype=np.float32)

    id_to_index = {pdb_id: index for index, pdb_id in enumerate(common_ids)}

    train_ids = read_split_file(split_dir / "train_index_folder.txt")
    valid_ids = read_split_file(split_dir / "valid_index_folder.txt")
    test_ids = read_split_file(split_dir / "test_index_folder.txt")

    train_indices = convert_splits_to_indices(train_ids, id_to_index, "train")
    valid_indices = convert_splits_to_indices(valid_ids, id_to_index, "valid")
    test_indices = convert_splits_to_indices(test_ids, id_to_index, "test")

    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    save_json_txt(ligands, output_dir / "ligands_can.txt")
    save_json_txt(proteins, output_dir / "proteins.txt")
    save_pickle(y_values, output_dir / "Y")

    save_json_txt(train_indices, folds_dir / "train_fold_setting1.txt")
    save_json_txt(valid_indices, folds_dir / "valid_fold_setting1.txt")
    save_json_txt(test_indices, folds_dir / "test_fold_setting1.txt")

    metadata = {
        "dataset": "mpro_urv",
        "source_root": str(source_dir),
        "protein_pdb": str(protein_path),
        "num_ligands": len(common_ids),
        "num_proteins": 1,
        "y_shape": list(y_values.shape),
        "target_min": float(np.min(y_values)),
        "target_max": float(np.max(y_values)),
        "target_mean": float(np.mean(y_values)),
        "target_std": float(np.std(y_values)),
        "num_train_folds": len(train_indices),
        "num_valid_folds": len(valid_indices),
        "num_test_folds": len(test_indices),
        "skipped_smiles_longer_than_50": len(skipped_too_long),
    }

    save_json_txt(metadata, output_dir / "metadata.json")

    print("[OK] MPro-URV converted to DeepDTA format")
    print(f"[OK] Source directory: {source_dir}")
    print(f"[OK] Output directory: {output_dir}")
    print(f"[OK] Protein PDB: {protein_path}")
    print(f"[OK] Ligands kept: {len(common_ids)}")
    print(f"[OK] Ligands skipped length > 50: {len(skipped_too_long)}")
    print(f"[OK] Protein sequence length: {len(protein_sequence)}")
    print(f"[OK] Y shape: {y_values.shape}")
    print(
        f"[OK] Target range: "
        f"{metadata['target_min']:.4f} -> {metadata['target_max']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MPro-URV_Version2 into DeepDTA-compatible format."
    )

    parser.add_argument(
        "--source-root",
        required=True,
        help=(
            "Path to the real MPro-URV_Version2 dataset root containing "
            "pIC50.txt, Ligand/, Protein/, Splits/."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="DeepDTA/data/mpro_urv",
        help="Output directory for the converted DeepDTA dataset.",
    )

    parser.add_argument(
        "--protein-pdb-path",
        default=None,
        help=(
            "Optional explicit protein PDB file. If omitted, the first .pdb "
            "in Protein/Protein_PDB is used."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    convert_mpro_urv_to_deepdta(
        source_root=args.source_root,
        output_root=args.output_root,
        protein_pdb_path=args.protein_pdb_path,
    )
