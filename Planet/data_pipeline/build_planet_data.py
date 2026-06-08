import csv
import json
import os
import pickle
import random
import shutil

import pandas as pd

try:
    from .chemutils import ComplexPocket
except ImportError:
    from chemutils import ComplexPocket


def load_pic50(pic50_path):
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )

    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.upper()
    df["pIC50"] = df["pIC50"].astype(float)

    return dict(zip(df["pdb_id"], df["pIC50"]))


def ensure_directory(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    ensure_directory(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def get_input_files(pdb_dir, sdf_dir, pdb_id):
    pdb_id = str(pdb_id).strip().upper()
    pdb_id_lower = pdb_id.lower()

    protein_candidates = [
        os.path.join(pdb_dir, f"{pdb_id}.pdb"),
        os.path.join(pdb_dir, f"{pdb_id}_protein.pdb"),
        os.path.join(pdb_dir, f"{pdb_id_lower}.pdb"),
        os.path.join(pdb_dir, f"{pdb_id_lower}_protein.pdb"),
    ]

    ligand_candidates = [
        os.path.join(sdf_dir, f"{pdb_id}.sdf"),
        os.path.join(sdf_dir, f"{pdb_id}_ligand.sdf"),
        os.path.join(sdf_dir, f"{pdb_id_lower}.sdf"),
        os.path.join(sdf_dir, f"{pdb_id_lower}_ligand.sdf"),
    ]

    protein_path = None
    ligand_path = None

    for path in protein_candidates:
        if os.path.isfile(path):
            protein_path = path
            break

    for path in ligand_candidates:
        if os.path.isfile(path):
            ligand_path = path
            break

    if protein_path is None:
        raise FileNotFoundError(f"PDB file not found for {pdb_id}. Tried: {protein_candidates}")

    if ligand_path is None:
        raise FileNotFoundError(f"SDF file not found for {pdb_id}. Tried: {ligand_candidates}")

    return protein_path, ligand_path


def copy_file(src, dst, overwrite=False):
    ensure_directory(os.path.dirname(dst))

    if os.path.exists(dst):
        if not overwrite:
            return dst
        os.remove(dst)

    shutil.copy2(src, dst)
    return dst


def prepare_raw_and_labels(pdb_dir, sdf_dir, pic50_path, output_dir, overwrite=False):
    pic50 = load_pic50(pic50_path)

    raw_dir = os.path.join(output_dir, "raw")
    metadata_dir = os.path.join(output_dir, "metadata")

    ensure_directory(raw_dir)
    ensure_directory(metadata_dir)

    prepared = []
    ok_ids = []

    for pdb_id, value in pic50.items():
        problems = []
        protein_src = None
        ligand_src = None

        try:
            protein_src, ligand_src = get_input_files(pdb_dir, sdf_dir, pdb_id)
        except Exception as exc:
            problems.append(str(exc))

        complex_dir = os.path.join(raw_dir, pdb_id)

        protein_dst = os.path.join(complex_dir, f"{pdb_id}.pdb")
        ligand_dst = os.path.join(complex_dir, f"{pdb_id}.sdf")

        if len(problems) == 0:
            ensure_directory(complex_dir)
            copy_file(protein_src, protein_dst, overwrite=overwrite)
            copy_file(ligand_src, ligand_dst, overwrite=overwrite)
            ok_ids.append(pdb_id)

        prepared.append(
            {
                "complex_id": pdb_id,
                "pIC50": value,
                "source_protein_pdb": "" if protein_src is None else protein_src,
                "source_ligand_sdf": "" if ligand_src is None else ligand_src,
                "raw_protein_pdb": protein_dst if os.path.exists(protein_dst) else "",
                "raw_ligand_sdf": ligand_dst if os.path.exists(ligand_dst) else "",
                "ok": len(problems) == 0,
                "problems": "; ".join(problems),
            }
        )

    labels_csv = os.path.join(metadata_dir, "labels.csv")
    prepared_csv = os.path.join(metadata_dir, "prepared.csv")

    with open(labels_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["complex_id", "PDB_code", "affinity", "pK", "pIC50"],
        )
        writer.writeheader()

        for pdb_id in ok_ids:
            writer.writerow(
                {
                    "complex_id": pdb_id,
                    "PDB_code": pdb_id,
                    "affinity": pic50[pdb_id],
                    "pK": pic50[pdb_id],
                    "pIC50": pic50[pdb_id],
                }
            )

    with open(prepared_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "complex_id",
            "pIC50",
            "source_protein_pdb",
            "source_ligand_sdf",
            "raw_protein_pdb",
            "raw_ligand_sdf",
            "ok",
            "problems",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in prepared:
            writer.writerow(row)

    return ok_ids, pic50, raw_dir, metadata_dir, labels_csv, prepared_csv


def split_ids(ids, train_ratio=0.8, valid_ratio=0.1, seed=42):
    ids = list(ids)

    random.seed(seed)
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_ids = ids[:n_train]
    valid_ids = ids[n_train : n_train + n_valid]
    core_ids = ids[n_train + n_valid :]

    if len(core_ids) == 0 and len(valid_ids) > 1:
        core_ids.append(valid_ids.pop())

    if len(valid_ids) == 0 and len(train_ids) > 1:
        valid_ids.append(train_ids.pop())

    return train_ids, valid_ids, core_ids


def write_split_csv(path, ids, pic50, split_type="REFINED"):
    ensure_directory(os.path.dirname(path))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["", "PDB_code", "pK", "type"],
        )
        writer.writeheader()

        for index, pdb_id in enumerate(ids):
            writer.writerow(
                {
                    "": index,
                    "PDB_code": pdb_id,
                    "pK": pic50[pdb_id],
                    "type": split_type,
                }
            )

    return path


def build_metadata_file(
    metadata_dir,
    ids,
    pic50,
    train_ratio=0.8,
    valid_ratio=0.1,
    seed=42,
):
    train_ids, valid_ids, core_ids = split_ids(
        ids,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_csv = write_split_csv(
        os.path.join(metadata_dir, "train.csv"),
        train_ids,
        pic50,
    )

    valid_csv = write_split_csv(
        os.path.join(metadata_dir, "valid.csv"),
        valid_ids,
        pic50,
    )

    core_csv = write_split_csv(
        os.path.join(metadata_dir, "core.csv"),
        core_ids,
        pic50,
    )

    return {
        "train": train_ids,
        "valid": valid_ids,
        "core": core_ids,
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "core_csv": core_csv,
    }


def build_one_pocket(raw_dir, pocket_root, pdb_id, affinity, overwrite=False):
    protein_path = os.path.join(raw_dir, pdb_id, f"{pdb_id}.pdb")
    ligand_sdf_path = os.path.join(raw_dir, pdb_id, f"{pdb_id}.sdf")

    if not os.path.exists(protein_path):
        raise FileNotFoundError(f"{protein_path} does not exist")

    if not os.path.exists(ligand_sdf_path):
        raise FileNotFoundError(f"{ligand_sdf_path} does not exist")

    pocket_dir = os.path.join(pocket_root, pdb_id)
    ensure_directory(pocket_dir)

    pocket_path = os.path.join(pocket_dir, f"{pdb_id}_pocket.pkl")

    if os.path.exists(pocket_path) and not overwrite:
        return pocket_path

    pocket = ComplexPocket(
        protein_path,
        ligand_sdf_path,
        float(affinity),
        None,
    )

    if getattr(pocket, "res_count", 0) <= 0:
        raise ValueError(f"Pocket construction produced zero residues for {pdb_id}")

    with open(pocket_path, "wb") as f:
        pickle.dump(pocket, f, pickle.HIGHEST_PROTOCOL)

    return pocket_path


def read_split_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    required_columns = ["PDB_code", "pK", "type"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"{column} not in {csv_path}")

    return df


def build_split_pocket(
    raw_dir,
    pocket_root,
    csv_path,
    output_pickle,
    overwrite_pockets=False,
):
    df = read_split_csv(csv_path)

    records = []
    skipped = []

    for _, row in df.iterrows():
        pdb_id = str(row["PDB_code"]).strip().upper()
        affinity = float(row["pK"])

        try:
            pocket_path = build_one_pocket(
                raw_dir,
                pocket_root,
                pdb_id,
                affinity,
                overwrite=overwrite_pockets,
            )

            records.append([pocket_path, affinity])

        except Exception as exc:
            skipped.append(
                {
                    "complex_id": pdb_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"Skipping {pdb_id}: {exc}")

    ensure_directory(os.path.dirname(output_pickle))

    with open(output_pickle, "wb") as f:
        pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

    return records, skipped


def build_all_pickle(raw_dir, metadata_dir, output_dir, overwrite_pockets=False):
    pkl_dir = os.path.join(metadata_dir, "pkl")
    pocket_root = os.path.join(pkl_dir, "pockets")

    ensure_directory(pkl_dir)
    ensure_directory(pocket_root)

    split_files = {
        "train": os.path.join(metadata_dir, "train.csv"),
        "valid": os.path.join(metadata_dir, "valid.csv"),
        "core": os.path.join(metadata_dir, "core.csv"),
    }

    summary = {
        "pkl_dir": pkl_dir,
        "pocket_root": pocket_root,
        "splits": {},
        "skipped": {},
    }

    for split_name, csv_path in split_files.items():
        output_pickle = os.path.join(pkl_dir, f"{split_name}.pkl")

        records, skipped = build_split_pocket(
            raw_dir,
            pocket_root,
            csv_path,
            output_pickle,
            overwrite_pockets=overwrite_pockets,
        )

        summary["splits"][split_name] = {
            "csv_path": csv_path,
            "pickle_path": output_pickle,
            "count_built": len(records),
        }

        summary["skipped"][split_name] = skipped

        print(f"{split_name} built {len(records)} records -> {output_pickle}")

    save_json(summary, os.path.join(pkl_dir, "build_pkl_summary.json"))

    return summary


def build_data(
    output_dir,
    pdb_dir,
    sdf_dir,
    pic50_path,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    overwrite=False,
    overwrite_pockets=False,
    log_callback=None,
):
    output_dir = os.path.abspath(output_dir)
    pdb_dir = os.path.abspath(pdb_dir)
    sdf_dir = os.path.abspath(sdf_dir)
    pic50_path = os.path.abspath(pic50_path)

    ensure_directory(output_dir)

    ok_ids, pic50, raw_dir, metadata_dir, labels_csv, prepared_csv = prepare_raw_and_labels(
        pdb_dir,
        sdf_dir,
        pic50_path,
        output_dir,
        overwrite=overwrite,
    )

    split_info = build_metadata_file(
        metadata_dir,
        ok_ids,
        pic50,
        train_ratio=train_ratio,
        valid_ratio=val_ratio,
        seed=seed,
    )

    pkl_summary = build_all_pickle(
        raw_dir,
        metadata_dir,
        output_dir,
        overwrite_pockets=overwrite_pockets,
    )

    manifest = {
        "pdb_dir": pdb_dir,
        "sdf_dir": sdf_dir,
        "pic50_path": pic50_path,
        "output_dir": output_dir,
        "raw_dir": raw_dir,
        "metadata_dir": metadata_dir,
        "labels_csv": labels_csv,
        "prepared_csv": prepared_csv,
        "n_labels": len(pic50),
        "n_prepared_ok": len(ok_ids),
        "split_counts": {
            "train": len(split_info["train"]),
            "valid": len(split_info["valid"]),
            "core": len(split_info["core"]),
        },
        "pkl_summary": pkl_summary,
    }

    manifest_path = os.path.join(metadata_dir, "build_data_manifest.json")
    save_json(manifest, manifest_path)
    if log_callback:
        log_callback.info(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest
