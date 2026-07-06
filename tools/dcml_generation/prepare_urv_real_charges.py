#!/usr/bin/env python3
"""Prepare real-charge inputs for the MPro-URV DCML final run.

Version 2 adds robust recovery for PDB2PQR failures:
  1. Try PDB2PQR on original PDB.
  2. Try PDB2PQR on cleaned protein-only PDB variants.
  3. Optional fallback: Open Babel PDB->PQR with partial charges.

Outputs:
  <work-root>/pdbbind_like/URV/pqr/<PDB_ID>_pocket.pqr
  regenerated ligand MOL2 files with partial charges in:
  <work-root>/pdbbind_like/URV/refined/<PDB_ID>/<PDB_ID>_ligand.mol2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
RESNAME_MAP = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS",
}
ELEMENT_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_id(value: str) -> str:
    return value.strip().upper()


def executable_exists(name: str) -> bool:
    return shutil.which(name) is not None


def read_ids(sample_csv: Path) -> list[str]:
    if not sample_csv.is_file():
        raise FileNotFoundError(f"sample CSV not found: {sample_csv}")
    ids: list[str] = []
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "PDB_ID" not in (reader.fieldnames or []):
            raise ValueError(f"PDB_ID column not found in {sample_csv}")
        for row in reader:
            pdb_id = norm_id(row.get("PDB_ID", ""))
            if pdb_id:
                ids.append(pdb_id)
    if not ids:
        raise ValueError(f"No PDB IDs found in {sample_csv}")
    return ids


def run_cmd(cmd: list[str], timeout: int | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_pqr_charge_stats(path: Path) -> dict:
    n_atoms = 0
    charges: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            n_atoms += 1
            q = None
            fixed = line[54:62].strip() if len(line) >= 62 else ""
            if fixed:
                try:
                    q = float(fixed)
                except ValueError:
                    q = None
            if q is None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        q = float(parts[-2])  # PQR: ... x y z charge radius
                    except ValueError:
                        q = None
            if q is not None:
                charges.append(q)
    if not charges:
        return {"n_atoms": n_atoms, "n_charge_values": 0, "nonzero_charges": 0, "sum_abs_charge": 0.0}
    nonzero = sum(1 for q in charges if abs(q) > 1e-8)
    return {
        "n_atoms": n_atoms,
        "n_charge_values": len(charges),
        "nonzero_charges": nonzero,
        "sum_abs_charge": float(sum(abs(q) for q in charges)),
        "min_charge": float(min(charges)),
        "max_charge": float(max(charges)),
    }


def parse_mol2_charge_stats(path: Path) -> dict:
    n_atoms = 0
    charges: list[float] = []
    in_atom = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom or not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            n_atoms += 1
            try:
                charges.append(float(parts[-1]))
            except ValueError:
                pass
    if not charges:
        return {"n_atoms": n_atoms, "n_charge_values": 0, "nonzero_charges": 0, "sum_abs_charge": 0.0}
    nonzero = sum(1 for q in charges if abs(q) > 1e-8)
    return {
        "n_atoms": n_atoms,
        "n_charge_values": len(charges),
        "nonzero_charges": nonzero,
        "sum_abs_charge": float(sum(abs(q) for q in charges)),
        "min_charge": float(min(charges)),
        "max_charge": float(max(charges)),
    }


def pdb2pqr_candidates(ff: str, ph: float | None, keep_chain: bool) -> list[list[str]]:
    base_opts = [f"--ff={ff}"]
    if keep_chain:
        base_opts.append("--keep-chain")
    variants: list[list[str]] = []
    if ph is not None:
        variants.append(base_opts + [f"--with-ph={ph}", "{inp}", "{out}"])
        variants.append(base_opts + ["--titration-state-method=propka", f"--with-ph={ph}", "{inp}", "{out}"])
    variants.append(base_opts + ["{inp}", "{out}"])
    variants.append([f"--ff={ff}", "{inp}", "{out}"])
    return variants


def run_pdb2pqr(
    input_pdb: Path,
    output_pqr: Path,
    ff: str,
    ph: float | None,
    keep_chain: bool,
    timeout: int,
    ff_fallbacks: list[str] | None = None,
) -> tuple[bool, str, str]:
    executables = [x for x in ["pdb2pqr30", "pdb2pqr"] if executable_exists(x)]
    attempts: list[str] = []
    errors: list[str] = []
    if not executables:
        return False, "", "PDB2PQR executable not found. Install with: conda install -c conda-forge pdb2pqr"

    output_pqr.parent.mkdir(parents=True, exist_ok=True)
    ffs = [ff]
    if ff_fallbacks:
        for fallback in ff_fallbacks:
            if fallback and fallback not in ffs:
                ffs.append(fallback)
    for current_ff in ffs:
        for exe in executables:
            for opts in pdb2pqr_candidates(ff=current_ff, ph=ph, keep_chain=keep_chain):
                cmd = [exe] + [str(input_pdb) if x == "{inp}" else str(output_pqr) if x == "{out}" else x for x in opts]
                attempts.append(" ".join(cmd))
                if output_pqr.exists():
                    output_pqr.unlink()
                code, out, err = run_cmd(cmd, timeout=timeout)
                if code == 0 and output_pqr.is_file() and output_pqr.stat().st_size > 0:
                    stats = parse_pqr_charge_stats(output_pqr)
                    if stats.get("nonzero_charges", 0) > 0:
                        return True, "\n".join(attempts[-1:]) + "\n" + out, err
                    errors.append(f"PDB2PQR output had no nonzero charges: {' '.join(cmd)}")
                else:
                    if err.strip():
                        errors.append(err.strip()[-1200:])
                    elif out.strip():
                        errors.append(out.strip()[-1200:])
    return False, "\n".join(attempts), "All PDB2PQR attempts failed. Last errors:\n" + "\n---\n".join(errors[-5:])


def safe_element_from_line(line: str) -> str:
    if len(line) >= 78 and line[76:78].strip():
        return line[76:78].strip().upper()
    atom = line[12:16].strip() if len(line) >= 16 else ""
    atom = "".join(ch for ch in atom if ch.isalpha()).upper()
    if len(atom) >= 2 and atom[:2] in {"CL", "BR", "SE"}:
        return atom[:2]
    return atom[:1] or "C"


def clean_pdb(
    input_pdb: Path,
    output_pdb: Path,
    *,
    remove_h: bool,
    require_backbone: bool,
    renumber_residues: bool,
    chain_filter: str | None = None,
) -> dict:
    raw_records: list[str] = []
    residue_atoms: dict[tuple[str, str, str], set[str]] = {}
    with input_pdb.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if len(line) < 54:
                continue
            chain = line[21].strip() or "A"
            if chain_filter and chain != chain_filter:
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A", "1"):
                continue
            resname_raw = line[17:20].strip().upper()
            resname = RESNAME_MAP.get(resname_raw, resname_raw)
            if resname not in STANDARD_AA:
                continue
            element = safe_element_from_line(line)
            atom = line[12:16].strip().upper()
            if remove_h and (element == "H" or atom.startswith("H")):
                continue
            # Normalize MSE selenium atom to MET sulfur for force-field compatibility.
            fixed = line.rstrip("\n")
            if resname_raw == "MSE":
                fixed = "ATOM  " + fixed[6:]
                fixed = fixed[:17] + "MET" + fixed[20:]
                if atom == "SE":
                    fixed = fixed[:12] + " SD " + fixed[16:]
                    if len(fixed) >= 78:
                        fixed = fixed[:76] + " S" + fixed[78:]
            else:
                fixed = "ATOM  " + fixed[6:]
                fixed = fixed[:17] + f"{resname:>3}" + fixed[20:]
            # Clear alternate location field.
            fixed = fixed[:16] + " " + fixed[17:]
            if len(fixed) < 80:
                fixed = fixed.ljust(80)
            key = (chain, fixed[22:26].strip(), fixed[26].strip())
            residue_atoms.setdefault(key, set()).add(fixed[12:16].strip().upper())
            raw_records.append(fixed)

    allowed_residues = set(residue_atoms)
    if require_backbone:
        allowed_residues = {k for k, atoms in residue_atoms.items() if {"N", "CA", "C", "O"}.issubset(atoms)}

    renum: dict[tuple[str, str, str], int] = {}
    if renumber_residues:
        counters: dict[str, int] = {}
        for fixed in raw_records:
            key = ((fixed[21].strip() or "A"), fixed[22:26].strip(), fixed[26].strip())
            if key not in allowed_residues:
                continue
            chain = key[0]
            if key not in renum:
                counters[chain] = counters.get(chain, 0) + 1
                renum[key] = counters[chain]

    kept: list[str] = []
    atom_serial = 1
    for fixed in raw_records:
        key = ((fixed[21].strip() or "A"), fixed[22:26].strip(), fixed[26].strip())
        if key not in allowed_residues:
            continue
        line = fixed
        if renumber_residues:
            line = line[:22] + f"{renum[key]:4d}" + " " + line[27:]
        line = line[:6] + f"{atom_serial:5d}" + line[11:]
        kept.append(line[:80] + "\n")
        atom_serial += 1

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        handle.writelines(kept)
        handle.write("TER\nEND\n")
    return {"input": str(input_pdb), "output": str(output_pdb), "atoms": len(kept), "residues": len(allowed_residues)}


def extract_chains(input_pdb: Path) -> list[str]:
    chains: set[str] = set()
    with input_pdb.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 22:
                chains.add(line[21].strip() or "A")
    return sorted(chains)


def run_obabel_ligand(input_sdf: Path, output_mol2: Path, method: str, timeout: int) -> tuple[bool, str, str]:
    obabel = shutil.which("obabel")
    if not obabel:
        return False, "", "Open Babel executable 'obabel' not found. Install with: conda install -c conda-forge openbabel"
    output_mol2.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_mol2.with_suffix(".charged.tmp.mol2")
    if tmp.exists():
        tmp.unlink()
    cmd = [obabel, "-isdf", str(input_sdf), "-omol2", "-O", str(tmp), "--partialcharge", method]
    code, out, err = run_cmd(cmd, timeout=timeout)
    if code != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        return False, out, err
    if output_mol2.exists():
        backup = output_mol2.with_suffix(".pre_real_charge.mol2")
        if not backup.exists():
            shutil.copy2(output_mol2, backup)
    shutil.move(str(tmp), str(output_mol2))
    return True, out, err


def run_obabel_protein_to_pqr(input_pdb: Path, output_pqr: Path, method: str, timeout: int) -> tuple[bool, str, str]:
    obabel = shutil.which("obabel")
    if not obabel:
        return False, "", "Open Babel executable 'obabel' not found. Install with: conda install -c conda-forge openbabel"
    output_pqr.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[str] = []
    errors: list[str] = []
    variants = [
        [obabel, "-ipdb", str(input_pdb), "-opqr", "-O", str(output_pqr), "--partialcharge", method],
        [obabel, "-ipdb", str(input_pdb), "-opqr", "-O", str(output_pqr), "-h", "--partialcharge", method],
        [obabel, "-ipdb", str(input_pdb), "-opqr", "-O", str(output_pqr)],
        [obabel, "-ipdb", str(input_pdb), "-opqr", "-O", str(output_pqr), "-h"],
    ]
    for cmd in variants:
        attempts.append(" ".join(cmd))
        if output_pqr.exists():
            output_pqr.unlink()
        code, out, err = run_cmd(cmd, timeout=timeout)
        if code == 0 and output_pqr.is_file() and output_pqr.stat().st_size > 0:
            stats = parse_pqr_charge_stats(output_pqr)
            if stats.get("nonzero_charges", 0) > 0:
                return True, "\n".join(attempts[-1:]) + "\n" + out, err
            errors.append("Open Babel PQR produced no nonzero charges")
        else:
            errors.append((err or out).strip()[-1200:])
    return False, "\n".join(attempts), "Open Babel protein PQR fallback failed:\n" + "\n---\n".join(errors[-5:])


@dataclass
class ItemResult:
    pdb_id: str
    protein_pdb: str
    pqr_path: str
    pqr_ok: bool
    pqr_method: str
    pqr_charge_stats: dict
    ligand_sdf: str
    ligand_mol2: str
    ligand_ok: bool
    ligand_charge_stats: dict
    error: str = ""


def write_log(log_dir: Path, pdb_id: str, name: str, content: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{pdb_id}_{name}.log").write_text(content, encoding="utf-8", errors="replace")


def prepare_protein_pqr(
    pdb_id: str,
    protein_pdb: Path,
    pqr_path: Path,
    repair_dir: Path,
    ff: str,
    ph: float | None,
    keep_chain: bool,
    timeout: int,
    protein_fallback: str,
    protein_charge_method: str,
) -> tuple[bool, str, str]:
    log_dir = repair_dir / pdb_id
    # 1) Original input.
    ok, out, err = run_pdb2pqr(protein_pdb, pqr_path, ff=ff, ph=ph, keep_chain=keep_chain, timeout=timeout)
    write_log(log_dir, pdb_id, "01_original_pdb2pqr_stdout", out)
    write_log(log_dir, pdb_id, "01_original_pdb2pqr_stderr", err)
    if ok:
        return True, "pdb2pqr_original", ""

    if protein_fallback in {"none"}:
        return False, "", f"PDB2PQR failed on original input and fallback is disabled: {err}\n{out}"

    # 2) Cleaned PDB variants for PDB2PQR.
    cleaned_variants: list[tuple[str, Path]] = []
    variants = [
        ("clean_std", dict(remove_h=False, require_backbone=False, renumber_residues=False, chain_filter=None)),
        ("clean_no_h", dict(remove_h=True, require_backbone=False, renumber_residues=False, chain_filter=None)),
        ("clean_no_h_renum", dict(remove_h=True, require_backbone=False, renumber_residues=True, chain_filter=None)),
        ("clean_backbone_no_h_renum", dict(remove_h=True, require_backbone=True, renumber_residues=True, chain_filter=None)),
    ]
    for chain in extract_chains(protein_pdb):
        variants.append((f"clean_chain_{chain}_no_h_renum", dict(remove_h=True, require_backbone=False, renumber_residues=True, chain_filter=chain)))

    for name, kwargs in variants:
        cleaned = log_dir / f"{pdb_id}_{name}.pdb"
        stats = clean_pdb(protein_pdb, cleaned, **kwargs)
        write_log(log_dir, pdb_id, f"{name}_stats", json.dumps(stats, indent=2))
        if stats["atoms"] <= 0:
            continue
        cleaned_variants.append((name, cleaned))
        ok, out, err = run_pdb2pqr(
            cleaned,
            pqr_path,
            ff=ff,
            ph=ph,
            keep_chain=keep_chain,
            timeout=timeout,
            ff_fallbacks=["PARSE", "CHARMM"],
        )
        write_log(log_dir, pdb_id, f"02_{name}_pdb2pqr_stdout", out)
        write_log(log_dir, pdb_id, f"02_{name}_pdb2pqr_stderr", err)
        if ok:
            return True, f"pdb2pqr_{name}", ""

    # 3) Open Babel protein PQR fallback. This keeps aligned coordinates and uses computed partial charges.
    if protein_fallback in {"auto", "obabel"}:
        obabel_inputs = cleaned_variants + [("original", protein_pdb)]
        # Put the most conservative cleaned variant first.
        order = ["clean_no_h_renum", "clean_no_h", "clean_std", "clean_backbone_no_h_renum", "original"]
        obabel_inputs = sorted(obabel_inputs, key=lambda x: order.index(x[0]) if x[0] in order else 99)
        for name, inp in obabel_inputs:
            ok, out, err = run_obabel_protein_to_pqr(inp, pqr_path, method=protein_charge_method, timeout=timeout)
            write_log(log_dir, pdb_id, f"03_{name}_obabel_pqr_stdout", out)
            write_log(log_dir, pdb_id, f"03_{name}_obabel_pqr_stderr", err)
            if ok:
                return True, f"obabel_pqr_{name}_{protein_charge_method}", ""

    return False, "", f"All protein charge generation attempts failed for {pdb_id}. See logs in {log_dir}"


def prepare_one(
    pdb_id: str,
    refined_dir: Path,
    pqr_root: Path,
    repair_dir: Path,
    ff: str,
    ph: float | None,
    keep_chain: bool,
    ligand_charge_method: str,
    protein_fallback: str,
    protein_charge_method: str,
    force_pqr: bool,
    force_ligand: bool,
    timeout: int,
) -> ItemResult:
    complex_dir = refined_dir / pdb_id
    protein_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    ligand_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"
    pqr_path = pqr_root / f"{pdb_id}_pocket.pqr"

    result = ItemResult(
        pdb_id=pdb_id,
        protein_pdb=str(protein_pdb),
        pqr_path=str(pqr_path),
        pqr_ok=False,
        pqr_method="",
        pqr_charge_stats={},
        ligand_sdf=str(ligand_sdf),
        ligand_mol2=str(ligand_mol2),
        ligand_ok=False,
        ligand_charge_stats={},
    )
    try:
        if not protein_pdb.is_file():
            raise FileNotFoundError(f"Missing protein PDB: {protein_pdb}")
        if not ligand_sdf.is_file():
            raise FileNotFoundError(f"Missing ligand SDF: {ligand_sdf}")

        if pqr_path.exists() and not force_pqr:
            stats = parse_pqr_charge_stats(pqr_path)
            if stats.get("nonzero_charges", 0) > 0:
                result.pqr_ok = True
                result.pqr_method = "existing"
                result.pqr_charge_stats = stats
            else:
                force_pqr = True

        if force_pqr or not result.pqr_ok:
            ok, method, err = prepare_protein_pqr(
                pdb_id=pdb_id,
                protein_pdb=protein_pdb,
                pqr_path=pqr_path,
                repair_dir=repair_dir,
                ff=ff,
                ph=ph,
                keep_chain=keep_chain,
                timeout=timeout,
                protein_fallback=protein_fallback,
                protein_charge_method=protein_charge_method,
            )
            result.pqr_ok = ok
            result.pqr_method = method
            if not ok:
                raise RuntimeError(err)
            result.pqr_charge_stats = parse_pqr_charge_stats(pqr_path)

        if result.pqr_charge_stats.get("nonzero_charges", 0) <= 0:
            raise RuntimeError(f"PQR has no nonzero charges for {pdb_id}: {pqr_path}")

        needs_ligand = force_ligand or (not ligand_mol2.is_file())
        if not needs_ligand:
            stats = parse_mol2_charge_stats(ligand_mol2)
            needs_ligand = stats.get("nonzero_charges", 0) <= 0
        if needs_ligand:
            ok, out, err = run_obabel_ligand(ligand_sdf, ligand_mol2, method=ligand_charge_method, timeout=timeout)
            result.ligand_ok = ok
            if not ok:
                raise RuntimeError(f"Open Babel ligand charge failed for {pdb_id}: {err}\n{out}")
        else:
            result.ligand_ok = True
        result.ligand_charge_stats = parse_mol2_charge_stats(ligand_mol2)
        if result.ligand_charge_stats.get("nonzero_charges", 0) <= 0:
            raise RuntimeError(f"Ligand MOL2 has no nonzero charges for {pdb_id}: {ligand_mol2}")
    except Exception as exc:
        result.error = str(exc)
    return result


def write_csv(path: Path, rows: list[ItemResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pdb_id", "pqr_ok", "pqr_method", "pqr_atoms", "pqr_nonzero_charges", "pqr_sum_abs_charge",
        "ligand_ok", "ligand_atoms", "ligand_nonzero_charges", "ligand_sum_abs_charge", "error",
        "pqr_path", "ligand_mol2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "pdb_id": r.pdb_id,
                "pqr_ok": r.pqr_ok,
                "pqr_method": r.pqr_method,
                "pqr_atoms": r.pqr_charge_stats.get("n_atoms", 0),
                "pqr_nonzero_charges": r.pqr_charge_stats.get("nonzero_charges", 0),
                "pqr_sum_abs_charge": r.pqr_charge_stats.get("sum_abs_charge", 0.0),
                "ligand_ok": r.ligand_ok,
                "ligand_atoms": r.ligand_charge_stats.get("n_atoms", 0),
                "ligand_nonzero_charges": r.ligand_charge_stats.get("nonzero_charges", 0),
                "ligand_sum_abs_charge": r.ligand_charge_stats.get("sum_abs_charge", 0.0),
                "error": r.error,
                "pqr_path": r.pqr_path,
                "ligand_mol2": r.ligand_mol2,
            })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate real charge input files for MPro-URV DCML.")
    p.add_argument("--work-root", type=Path, required=True, help="Existing data_urv folder created by prepare_mpro_urv_for_dcml.py")
    p.add_argument("--sample-csv", type=Path, default=None, help="Default: <work-root>/dcml_input/MProURV_all_sample_ids.csv")
    p.add_argument("--ids", default=None, help="Comma-separated PDB IDs for smoke tests, e.g. 7B2J,7GBZ")
    p.add_argument("--pqr-root", type=Path, default=None, help="Default: <work-root>/pdbbind_like/URV/pqr")
    p.add_argument("--repair-dir", type=Path, default=None, help="Default: <work-root>/reports/pdb2pqr_repair")
    p.add_argument("--ff", default="AMBER", help="PDB2PQR force field, e.g. AMBER, PARSE, CHARMM")
    p.add_argument("--ph", type=float, default=7.0, help="pH passed to PDB2PQR when supported. Use --ph -1 to disable.")
    p.add_argument("--no-keep-chain", action="store_true", help="Do not pass --keep-chain to PDB2PQR")
    p.add_argument("--ligand-charge-method", default="gasteiger", help="Open Babel partial charge method for ligands")
    p.add_argument("--protein-fallback", choices=["none", "clean", "obabel", "auto"], default="auto", help="Recovery strategy after original PDB2PQR failure. auto = clean PDB2PQR then Open Babel PQR fallback")
    p.add_argument("--protein-charge-method", default="gasteiger", help="Open Babel partial charge method for protein PQR fallback")
    p.add_argument("--force-pqr", action="store_true", help="Regenerate existing PQR files")
    p.add_argument("--force-ligand", action="store_true", help="Regenerate existing ligand MOL2 files")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers. Use 1 first; 2-4 after smoke test.")
    p.add_argument("--timeout", type=int, default=240, help="Timeout seconds per protein/ligand command")
    p.add_argument("--allow-failures", action="store_true", help="Write reports even if some IDs fail; default exits non-zero on any failure")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_root = args.work_root.resolve()
    refined_dir = work_root / "pdbbind_like" / "URV" / "refined"
    pqr_root = (args.pqr_root or (work_root / "pdbbind_like" / "URV" / "pqr")).resolve()
    repair_dir = (args.repair_dir or (work_root / "reports" / "pdb2pqr_repair")).resolve()
    sample_csv = (args.sample_csv or (work_root / "dcml_input" / "MProURV_all_sample_ids.csv")).resolve()
    report_dir = work_root / "reports" / "real_charge_preparation"
    report_json = report_dir / "real_charge_preparation_report.json"
    report_csv = report_dir / "real_charge_preparation_report.csv"

    if not refined_dir.is_dir():
        raise FileNotFoundError(f"refined folder not found: {refined_dir}")
    ids_all = read_ids(sample_csv)
    if args.ids:
        requested = [norm_id(x) for x in args.ids.split(",") if x.strip()]
        missing = [x for x in requested if x not in set(ids_all)]
        if missing:
            raise ValueError(f"Requested IDs not present in sample CSV: {missing}")
        ids = requested
    else:
        ids = ids_all
    if not ids:
        raise ValueError("No IDs selected")

    if not executable_exists("obabel"):
        raise RuntimeError("Missing Open Babel. Install: conda install -c conda-forge openbabel")
    if not (executable_exists("pdb2pqr") or executable_exists("pdb2pqr30")):
        raise RuntimeError("Missing PDB2PQR. Install: conda install -c conda-forge pdb2pqr")

    ph = None if args.ph < 0 else args.ph
    print(json.dumps({
        "script_version": "prepare_urv_real_charges_v2",
        "started_at_utc": now_iso(),
        "work_root": str(work_root),
        "refined_dir": str(refined_dir),
        "pqr_root": str(pqr_root),
        "repair_dir": str(repair_dir),
        "n_ids": len(ids),
        "ff": args.ff,
        "ph": ph,
        "ligand_charge_method": args.ligand_charge_method,
        "protein_fallback": args.protein_fallback,
        "protein_charge_method": args.protein_charge_method,
        "workers": args.workers,
        "force_pqr": args.force_pqr,
        "force_ligand": args.force_ligand,
    }, indent=2))

    t0 = time.time()
    rows: list[ItemResult] = []
    if args.workers <= 1:
        for i, pdb_id in enumerate(ids, 1):
            r = prepare_one(
                pdb_id, refined_dir, pqr_root, repair_dir, args.ff, ph, not args.no_keep_chain,
                args.ligand_charge_method, args.protein_fallback, args.protein_charge_method,
                args.force_pqr, args.force_ligand, args.timeout,
            )
            rows.append(r)
            status = "ok" if not r.error else "FAIL"
            method = f" ({r.pqr_method})" if r.pqr_method else ""
            print(f"[{i}/{len(ids)}] {pdb_id} {status}{method}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    prepare_one, pdb_id, refined_dir, pqr_root, repair_dir, args.ff, ph, not args.no_keep_chain,
                    args.ligand_charge_method, args.protein_fallback, args.protein_charge_method,
                    args.force_pqr, args.force_ligand, args.timeout,
                ): pdb_id for pdb_id in ids
            }
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                done += 1
                status = "ok" if not r.error else "FAIL"
                method = f" ({r.pqr_method})" if r.pqr_method else ""
                print(f"[{done}/{len(ids)}] {r.pdb_id} {status}{method}", flush=True)
    rows.sort(key=lambda r: ids.index(r.pdb_id))

    failures = [r for r in rows if r.error]
    pqr_ok = sum(1 for r in rows if r.pqr_ok and not r.error)
    lig_ok = sum(1 for r in rows if r.ligand_ok and not r.error)
    method_counts: dict[str, int] = {}
    for r in rows:
        method_counts[r.pqr_method or "failed"] = method_counts.get(r.pqr_method or "failed", 0) + 1
    report = {
        "script_version": "prepare_urv_real_charges_v2",
        "finished_at_utc": now_iso(),
        "elapsed_seconds": round(time.time() - t0, 3),
        "work_root": str(work_root),
        "refined_dir": str(refined_dir),
        "pqr_root": str(pqr_root),
        "repair_dir": str(repair_dir),
        "n_ids": len(ids),
        "n_success": len(rows) - len(failures),
        "n_failures": len(failures),
        "pqr_ok": pqr_ok,
        "ligand_ok": lig_ok,
        "pqr_method_counts": method_counts,
        "ff": args.ff,
        "ph": ph,
        "ligand_charge_method": args.ligand_charge_method,
        "protein_fallback": args.protein_fallback,
        "protein_charge_method": args.protein_charge_method,
        "csv_report": str(report_csv),
        "items": [asdict(r) for r in rows],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report_csv, rows)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2, ensure_ascii=False))

    if failures and not args.allow_failures:
        print("Failures:", file=sys.stderr)
        for r in failures[:20]:
            print(f"  {r.pdb_id}: {r.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
