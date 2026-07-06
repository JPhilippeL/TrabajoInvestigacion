#!/usr/bin/env python3
"""Generate DCML descriptor matrices for the MPro-URV staging tree.

Input expected from prepare_mpro_urv_for_dcml.py:
  <work-root>/pdbbind_like/URV/refined/<PDB_ID>/<PDB_ID>_pocket.pdb
  <work-root>/pdbbind_like/URV/refined/<PDB_ID>/<PDB_ID>_ligand.mol2
  <work-root>/dcml_input/MProURV_all_sample_ids.csv

Output:
  <work-root>/dcml_input/MProURV_all_feature.npy     shape (n, 118360) by default
  <work-root>/dcml_input/MProURV_all_feature.zip     optional, one .npy inside

The first 63,360 columns reproduce distance DCML features from distance2007.py.
The next 55,000 columns reproduce charge DCML features from charge2007.py when
PQR files are supplied. If --charge-mode zeros is used, the charge block is filled
with zeros so the current trained 118,360-column TFM model can be executed as an
immediate baseline/smoke test. Use --charge-mode pqr for the final scientific run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

AA_LIST = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "HSE", "HSD", "SEC",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "PYL",
}
DIST_PROTEIN_ATOMS = ["C", "N", "O", "S"]
DIST_LIGAND_ATOMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
CHARGE_PROTEIN_ATOMS = ["C", "N", "O", "S", "H"]
CHARGE_LIGAND_ATOMS = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]
MOMENT_POWERS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
DISTANCE_N_FEATURES = 63360
CHARGE_N_FEATURES = 55000
FULL_N_FEATURES = DISTANCE_N_FEATURES + CHARGE_N_FEATURES


@dataclass(frozen=True)
class Atom:
    element: str
    xyz: tuple[float, float, float]
    charge: float = 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_id(value: str) -> str:
    return value.strip().upper()


def get_atom_family(element: str, families: Sequence[str]) -> str | None:
    e = element.strip()
    if not e:
        return None
    low = e.lower()
    if low == "cl":
        return "Cl" if "Cl" in families else None
    if low == "br":
        return "Br" if "Br" in families else None
    first = e[0].upper()
    return first if first in families else None


def infer_element_from_atom_name(atom_name: str) -> str:
    token = atom_name.strip()
    if not token:
        return ""
    letters = "".join(ch for ch in token if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].lower() in {"cl", "br"}:
        return letters[:2].capitalize()
    return letters[0].upper()


def parse_float_field(text: str, fallback: float | None = None) -> float:
    try:
        return float(text)
    except Exception:
        if fallback is None:
            raise
        return fallback


def read_protein_pdb(path: Path, families: Sequence[str]) -> dict[str, list[Atom]]:
    atoms = {family: [] for family in families}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA_LIST:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                element = infer_element_from_atom_name(atom_name)
            family = get_atom_family(element, families)
            if family is None:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                parts = line.split()
                if len(parts) < 9:
                    continue
                x, y, z = map(float, parts[6:9])
            atoms[family].append(Atom(family, (x, y, z), 0.0))
    return atoms


def read_protein_pqr(path: Path, families: Sequence[str]) -> dict[str, list[Atom]]:
    atoms = {family: [] for family in families}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA_LIST:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                element = infer_element_from_atom_name(atom_name)
            family = get_atom_family(element, families)
            if family is None:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                parts = line.split()
                if len(parts) < 9:
                    continue
                x, y, z = map(float, parts[-5:-2])
            charge = 0.0
            fixed_charge = line[54:62].strip() if len(line) >= 62 else ""
            if fixed_charge:
                try:
                    charge = float(fixed_charge)
                except ValueError:
                    pass
            if charge == 0.0:
                parts = line.split()
                # PQR usually ends with: x y z charge radius
                for idx in (-2, -1):
                    if len(parts) >= abs(idx):
                        try:
                            candidate = float(parts[idx])
                            if -5.0 <= candidate <= 5.0:
                                charge = candidate
                                break
                        except ValueError:
                            pass
            atoms[family].append(Atom(family, (x, y, z), charge))
    return atoms


def read_ligand_mol2(path: Path, families: Sequence[str]) -> dict[str, list[Atom]]:
    atoms = {family: [] for family in families}
    in_atom = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            atom_name = parts[1]
            try:
                x, y, z = map(float, parts[2:5])
            except ValueError:
                continue
            atom_type = parts[5]
            element = atom_type.split(".", 1)[0]
            if element.lower() in {"du", "any"} or not element:
                element = infer_element_from_atom_name(atom_name)
            if element.upper() == "CL":
                element = "Cl"
            elif element.upper() == "BR":
                element = "Br"
            else:
                element = element.capitalize() if len(element) > 1 else element.upper()
            family = get_atom_family(element, families)
            if family is None:
                continue
            charge = 0.0
            # MOL2 atom charge is commonly the last field; keep 0 if absent/non-numeric.
            try:
                charge = float(parts[-1])
            except ValueError:
                pass
            atoms[family].append(Atom(family, (x, y, z), charge))
    return atoms


def read_ligand_sdf(path: Path, families: Sequence[str]) -> dict[str, list[Atom]]:
    atoms = {family: [] for family in families}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        return atoms
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
    except ValueError:
        parts = counts.split()
        if not parts:
            return atoms
        n_atoms = int(parts[0])
    for line in lines[4:4 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = map(float, parts[0:3])
        except ValueError:
            continue
        element = parts[3]
        family = get_atom_family(element, families)
        if family is None:
            continue
        atoms[family].append(Atom(family, (x, y, z), 0.0))
    return atoms


def atoms_to_array(atoms: Sequence[Atom], with_charge: bool) -> np.ndarray:
    if not atoms:
        return np.zeros((0, 4 if with_charge else 3), dtype=np.float64)
    if with_charge:
        return np.array([[a.xyz[0], a.xyz[1], a.xyz[2], a.charge] for a in atoms], dtype=np.float64)
    return np.array([[a.xyz[0], a.xyz[1], a.xyz[2]] for a in atoms], dtype=np.float64)


def distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    diff = a[:, None, :3] - b[None, :, :3]
    return np.sqrt(np.sum(diff * diff, axis=2))


def charge_filtration_values(p_atoms: np.ndarray, l_atoms: np.ndarray, geom_dist: np.ndarray) -> np.ndarray:
    if p_atoms.size == 0 or l_atoms.size == 0:
        return np.zeros((p_atoms.shape[0], l_atoms.shape[0]), dtype=np.float64)
    qprod = p_atoms[:, None, 3] * l_atoms[None, :, 3]
    safe_dist = np.where(geom_dist <= 1e-12, 1e-12, geom_dist)
    temp1 = 100.0 * qprod / safe_dist
    # Numerically stable sigmoid equivalent to original 1/(1+exp(-temp1)).
    temp1 = np.clip(temp1, -700, 700)
    return 1.0 / (1.0 + np.exp(-temp1))


def build_dowker_simplices(n_p: int, n_l: int, p_keep: Sequence[int], l_keep: Sequence[int], sorted_edges: Sequence[tuple[int, int, float]]):
    # Protein-side Dowker component.
    simplices_p: list[list[float | int]] = []
    for idx, p in enumerate(p_keep):
        simplices_p.append([idx, 0.0, 0, p])
    l_to_rel = {l: idx for idx, l in enumerate(l_keep)}
    p_to_rel = {p: idx for idx, p in enumerate(p_keep)}
    l_neigh: list[list[int]] = [[] for _ in l_keep]
    edge_seen_p: set[tuple[int, int]] = set()
    count_p = len(simplices_p)
    for p, l, filtration in sorted_edges:
        l_index = l_to_rel[l]
        neigh = l_neigh[l_index]
        if not neigh:
            neigh.append(p)
            continue
        for one in neigh:
            a, b = (one, p) if one <= p else (p, one)
            key = (a, b)
            if key not in edge_seen_p:
                edge_seen_p.add(key)
                simplices_p.append([count_p, float(filtration), 1, a, b])
                count_p += 1
        neigh.append(p)

    # Ligand-side Dowker component.
    simplices_l: list[list[float | int]] = []
    for idx, l in enumerate(l_keep):
        simplices_l.append([idx, 0.0, 0, l])
    p_neigh: list[list[int]] = [[] for _ in p_keep]
    edge_seen_l: set[tuple[int, int]] = set()
    count_l = len(simplices_l)
    for p, l, filtration in sorted_edges:
        p_index = p_to_rel[p]
        neigh = p_neigh[p_index]
        if not neigh:
            neigh.append(l)
            continue
        for one in neigh:
            a, b = (one, l) if one <= l else (l, one)
            key = (a, b)
            if key not in edge_seen_l:
                edge_seen_l.add(key)
                simplices_l.append([count_l, float(filtration), 1, a, b])
                count_l += 1
        neigh.append(l)
    return simplices_p, simplices_l


def spectral_moments(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0] * len(MOMENT_POWERS)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.abs(arr) > 1e-9]
    if arr.size == 0:
        return [0.0] * len(MOMENT_POWERS)
    out: list[float] = []
    for power in MOMENT_POWERS:
        if power == 0:
            out.append(float(arr.size))
        else:
            out.append(float(np.sum(np.power(arr, power))))
    return out


def eigenvalues_at_thresholds(simplices: Sequence[Sequence[float | int]], thresholds: Sequence[float]) -> list[list[float]]:
    if not simplices:
        return [[] for _ in thresholds]
    # Vertices are always first and filtration 0. Edges are already sorted by filtration.
    vertices = [int(s[3]) for s in simplices if int(s[2]) == 0]
    edge_simplices = [s for s in simplices if int(s[2]) == 1]
    result: list[list[float]] = []
    edge_pos = 0
    active_edges: list[tuple[int, int]] = []
    vertex_index = {v: i for i, v in enumerate(vertices)}
    n_vertices = len(vertices)
    for threshold in thresholds:
        while edge_pos < len(edge_simplices) and float(edge_simplices[edge_pos][1]) <= threshold:
            s = edge_simplices[edge_pos]
            active_edges.append((int(s[3]), int(s[4])))
            edge_pos += 1
        if n_vertices == 0 or not active_edges:
            result.append([0.0] * n_vertices)
            continue
        boundary = np.zeros((n_vertices, len(active_edges)), dtype=np.float64)
        valid_col = 0
        for c, (a, b) in enumerate(active_edges):
            ia = vertex_index.get(a)
            ib = vertex_index.get(b)
            if ia is None or ib is None:
                continue
            boundary[ia, valid_col] = -1.0
            boundary[ib, valid_col] = 1.0
            valid_col += 1
        if valid_col == 0:
            result.append([0.0] * n_vertices)
            continue
        if valid_col != boundary.shape[1]:
            boundary = boundary[:, :valid_col]
        laplacian = boundary @ boundary.T
        result.append(np.linalg.eigvalsh(laplacian).tolist())
    return result


def dowker_feature_block(
    protein_by_family: dict[str, list[Atom]],
    ligand_by_family: dict[str, list[Atom]],
    protein_families: Sequence[str],
    ligand_families: Sequence[str],
    *,
    use_charge_filtration: bool,
    cutoff: float,
    filtration: float,
    distance_thresholds: Sequence[float],
    charge_thresholds: Sequence[float],
) -> np.ndarray:
    features: list[float] = []
    thresholds = charge_thresholds if use_charge_filtration else distance_thresholds
    for p_family in protein_families:
        for l_family in ligand_families:
            p_arr = atoms_to_array(protein_by_family.get(p_family, []), with_charge=use_charge_filtration)
            l_arr = atoms_to_array(ligand_by_family.get(l_family, []), with_charge=use_charge_filtration)
            n_p, n_l = p_arr.shape[0], l_arr.shape[0]
            if n_p == 0 or n_l == 0:
                features.extend([0.0] * (len(thresholds) * len(MOMENT_POWERS) * 2))
                continue
            geom = distance_matrix(p_arr, l_arr)
            p_keep = np.where(np.any(geom <= cutoff, axis=1))[0].astype(int).tolist()
            l_keep = list(range(n_l))
            if not p_keep or not l_keep:
                features.extend([0.0] * (len(thresholds) * len(MOMENT_POWERS) * 2))
                continue
            if use_charge_filtration:
                filt = charge_filtration_values(p_arr, l_arr, geom)
                mask = geom <= filtration
            else:
                filt = geom
                mask = geom <= filtration
            sorted_edges: list[tuple[int, int, float]] = []
            for p in p_keep:
                for l in l_keep:
                    if mask[p, l]:
                        sorted_edges.append((p, l, float(filt[p, l])))
            sorted_edges.sort(key=lambda x: x[2])
            simplices_p, simplices_l = build_dowker_simplices(n_p, n_l, p_keep, l_keep, sorted_edges)
            eig_p = eigenvalues_at_thresholds(simplices_p, thresholds)
            eig_l = eigenvalues_at_thresholds(simplices_l, thresholds)
            for vals in eig_p:
                features.extend(spectral_moments(vals))
            for vals in eig_l:
                features.extend(spectral_moments(vals))
    return np.asarray(features, dtype=np.float64)


def find_pqr_for_id(pdb_id: str, pqr_root: Path | None, refined_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if pqr_root is not None:
        candidates.extend([
            pqr_root / f"{pdb_id}_pocket.pqr",
            pqr_root / f"{pdb_id}.pqr",
            pqr_root / pdb_id / f"{pdb_id}_pocket.pqr",
            pqr_root / pdb_id / f"{pdb_id}.pqr",
            pqr_root / pdb_id / f"{pdb_id}_protein.pqr",
        ])
    candidates.extend([
        refined_dir / pdb_id / f"{pdb_id}_pocket.pqr",
        refined_dir / pdb_id / f"{pdb_id}.pqr",
        refined_dir / pdb_id / f"{pdb_id}_protein.pqr",
    ])
    for path in candidates:
        if path.is_file():
            return path
    if pqr_root is not None and pqr_root.exists():
        hits = sorted(p for p in pqr_root.rglob("*.pqr") if pdb_id.lower() in p.name.lower())
        if hits:
            return hits[0]
    return None


def compute_one(args_tuple):
    (
        row_index,
        pdb_id,
        refined_dir_str,
        pqr_root_str,
        charge_mode,
        output_mode,
        dtype_name,
    ) = args_tuple
    refined_dir = Path(refined_dir_str)
    pqr_root = Path(pqr_root_str) if pqr_root_str else None
    complex_dir = refined_dir / pdb_id
    protein_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"
    ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    if not protein_pdb.is_file():
        raise FileNotFoundError(f"Missing protein PDB for {pdb_id}: {protein_pdb}")
    if ligand_mol2.is_file():
        ligand_distance = read_ligand_mol2(ligand_mol2, DIST_LIGAND_ATOMS)
    elif ligand_sdf.is_file():
        ligand_distance = read_ligand_sdf(ligand_sdf, DIST_LIGAND_ATOMS)
    else:
        raise FileNotFoundError(f"Missing ligand MOL2/SDF for {pdb_id}: {complex_dir}")

    distance_thresholds = [2.0 + 0.1 * i for i in range(80)]  # original feature loop ii=1..80 -> 2.0..9.9
    charge_thresholds = [0.02 * i for i in range(50)]          # original feature loop ii=1..50 -> 0.00..0.98

    protein_distance = read_protein_pdb(protein_pdb, DIST_PROTEIN_ATOMS)
    dist_feat = dowker_feature_block(
        protein_distance,
        ligand_distance,
        DIST_PROTEIN_ATOMS,
        DIST_LIGAND_ATOMS,
        use_charge_filtration=False,
        cutoff=10.0,
        filtration=10.0,
        distance_thresholds=distance_thresholds,
        charge_thresholds=charge_thresholds,
    )
    if dist_feat.shape[0] != DISTANCE_N_FEATURES:
        raise RuntimeError(f"Distance feature length mismatch for {pdb_id}: {dist_feat.shape[0]} != {DISTANCE_N_FEATURES}")
    if output_mode == "distance-only":
        feat = dist_feat
        metadata = {"pdb_id": pdb_id, "charge_source": None, "n_features": int(feat.shape[0])}
        return row_index, feat.astype(dtype_name, copy=False), metadata

    if charge_mode == "zeros":
        charge_feat = np.zeros(CHARGE_N_FEATURES, dtype=np.float64)
        charge_source = "zeros"
    else:
        pqr_path = find_pqr_for_id(pdb_id, pqr_root, refined_dir)
        if pqr_path is None:
            if charge_mode == "auto":
                charge_feat = np.zeros(CHARGE_N_FEATURES, dtype=np.float64)
                charge_source = "zeros_no_pqr_found"
            else:
                raise FileNotFoundError(f"Missing PQR for {pdb_id}. Pass --pqr-root or use --charge-mode zeros/auto.")
        else:
            ligand_charge = read_ligand_mol2(ligand_mol2, CHARGE_LIGAND_ATOMS) if ligand_mol2.is_file() else read_ligand_sdf(ligand_sdf, CHARGE_LIGAND_ATOMS)
            protein_charge = read_protein_pqr(pqr_path, CHARGE_PROTEIN_ATOMS)
            charge_feat = dowker_feature_block(
                protein_charge,
                ligand_charge,
                CHARGE_PROTEIN_ATOMS,
                CHARGE_LIGAND_ATOMS,
                use_charge_filtration=True,
                cutoff=10.0,
                filtration=10.0,
                distance_thresholds=distance_thresholds,
                charge_thresholds=charge_thresholds,
            )
            charge_source = str(pqr_path)
    if charge_feat.shape[0] != CHARGE_N_FEATURES:
        raise RuntimeError(f"Charge feature length mismatch for {pdb_id}: {charge_feat.shape[0]} != {CHARGE_N_FEATURES}")
    feat = np.concatenate([dist_feat, charge_feat], axis=0)
    if feat.shape[0] != FULL_N_FEATURES:
        raise RuntimeError(f"Full feature length mismatch for {pdb_id}: {feat.shape[0]} != {FULL_N_FEATURES}")
    metadata = {"pdb_id": pdb_id, "charge_source": charge_source, "n_features": int(feat.shape[0])}
    return row_index, feat.astype(dtype_name, copy=False), metadata


def read_ids_from_sample_csv(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "PDB_ID" not in (reader.fieldnames or []):
            raise ValueError(f"Sample CSV must contain PDB_ID: {path}")
        for row in reader:
            pdb_id = norm_id(row.get("PDB_ID", ""))
            if pdb_id:
                ids.append(pdb_id)
    if not ids:
        raise ValueError(f"No PDB_ID rows found in {path}")
    return ids


def write_zip(zip_path: Path, npy_path: Path, internal_name: str = "urv_feature.npy") -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, arcname=internal_name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate DCML features for MPro-URV prepared staging tree.")
    p.add_argument("--work-root", type=Path, required=True, help="Folder created by prepare_mpro_urv_for_dcml.py, e.g. .../data_urv")
    p.add_argument("--pdbbind-root", type=Path, default=None, help="Default: <work-root>/pdbbind_like/URV")
    p.add_argument("--sample-csv", type=Path, default=None, help="Default: <work-root>/dcml_input/MProURV_all_sample_ids.csv")
    p.add_argument("--out-npy", type=Path, default=None, help="Default: <work-root>/dcml_input/MProURV_all_feature.npy")
    p.add_argument("--out-zip", type=Path, default=None, help="Optional feature.zip path. Default with --write-zip: <work-root>/dcml_input/MProURV_all_feature.zip")
    p.add_argument("--write-zip", action="store_true", help="Also write a DCML-compatible ZIP containing one .npy")
    p.add_argument("--ids", default=None, help="Comma-separated PDB IDs to process, e.g. 7B2J,7GBZ. Overrides --limit for those IDs.")
    p.add_argument("--limit", type=int, default=None, help="Process first N rows only; useful for smoke tests.")
    p.add_argument("--output-mode", choices=["full", "distance-only"], default="full", help="full = 118360 cols; distance-only = 63360 cols")
    p.add_argument("--charge-mode", choices=["zeros", "pqr", "auto"], default="zeros", help="zeros baseline, pqr required, or auto use PQR when found else zeros")
    p.add_argument("--pqr-root", type=Path, default=None, help="Folder containing PQR files for --charge-mode pqr/auto")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes. Start with 1 or 2 under WSL.")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_root = args.work_root.resolve()
    pdbbind_root = (args.pdbbind_root or (work_root / "pdbbind_like" / "URV")).resolve()
    refined_dir = pdbbind_root / "refined"
    sample_csv = (args.sample_csv or (work_root / "dcml_input" / "MProURV_all_sample_ids.csv")).resolve()
    out_npy = (args.out_npy or (work_root / "dcml_input" / "MProURV_all_feature.npy")).resolve()
    out_zip = (args.out_zip or (work_root / "dcml_input" / "MProURV_all_feature.zip")).resolve()

    if not refined_dir.is_dir():
        raise FileNotFoundError(f"refined folder not found: {refined_dir}")
    if not sample_csv.is_file():
        raise FileNotFoundError(f"sample CSV not found: {sample_csv}")
    if out_npy.exists() and not args.force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {out_npy}")
    if args.write_zip and out_zip.exists() and not args.force:
        raise FileExistsError(f"ZIP output exists. Use --force to overwrite: {out_zip}")

    ids_all = read_ids_from_sample_csv(sample_csv)
    if args.ids:
        requested = [norm_id(x) for x in args.ids.split(",") if x.strip()]
        missing = [x for x in requested if x not in set(ids_all)]
        if missing:
            raise ValueError(f"Requested IDs not found in sample CSV: {missing}")
        ids = requested
    else:
        ids = ids_all[: args.limit] if args.limit is not None else ids_all
    if not ids:
        raise ValueError("No IDs selected")

    n_features = FULL_N_FEATURES if args.output_mode == "full" else DISTANCE_N_FEATURES
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(ids), n_features), dtype=np.dtype(args.dtype))

    print(json.dumps({
        "started_at_utc": now_iso(),
        "work_root": str(work_root),
        "refined_dir": str(refined_dir),
        "sample_csv": str(sample_csv),
        "n_samples": len(ids),
        "n_features": n_features,
        "output_mode": args.output_mode,
        "charge_mode": args.charge_mode,
        "pqr_root": str(args.pqr_root.resolve()) if args.pqr_root else None,
        "out_npy": str(out_npy),
        "out_zip": str(out_zip) if args.write_zip else None,
        "workers": args.workers,
    }, indent=2))

    tasks = [
        (i, pdb_id, str(refined_dir), str(args.pqr_root.resolve()) if args.pqr_root else "", args.charge_mode, args.output_mode, args.dtype)
        for i, pdb_id in enumerate(ids)
    ]
    metadata: list[dict] = [None] * len(ids)  # type: ignore[list-item]
    t0 = time.time()
    done = 0
    if args.workers <= 1:
        for task in tasks:
            row, feat, meta = compute_one(task)
            matrix[row, :] = feat
            metadata[row] = meta
            done += 1
            elapsed = time.time() - t0
            print(f"[{done}/{len(ids)}] {meta['pdb_id']} ok | elapsed={elapsed:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(compute_one, task) for task in tasks]
            for fut in as_completed(futures):
                row, feat, meta = fut.result()
                matrix[row, :] = feat
                metadata[row] = meta
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{len(ids)}] {meta['pdb_id']} ok | elapsed={elapsed:.1f}s", flush=True)

    if not np.isfinite(matrix).all():
        raise ValueError("Generated matrix contains NaN or infinite values")
    zero_rows = np.where(np.all(matrix == 0, axis=1))[0]
    if len(zero_rows) > 0:
        raise ValueError(f"Generated matrix has fully zero rows at indices: {zero_rows[:20].tolist()}")

    np.save(out_npy, matrix, allow_pickle=False)
    if args.write_zip:
        write_zip(out_zip, out_npy)

    report = {
        "finished_at_utc": now_iso(),
        "elapsed_seconds": round(time.time() - t0, 3),
        "n_samples": len(ids),
        "n_features": int(matrix.shape[1]),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "output_mode": args.output_mode,
        "charge_mode": args.charge_mode,
        "out_npy": str(out_npy),
        "out_zip": str(out_zip) if args.write_zip else None,
        "metadata": metadata,
    }
    report_path = out_npy.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
