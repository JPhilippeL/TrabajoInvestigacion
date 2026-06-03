import argparse
import csv
import json
import os
import pickle
import random
import shutil
import sys

import pandas as pd


def load_pic50(pic50_path):
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )
    return dict(zip(df["pdb_id"], df["pIC50"]))


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    ensure_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def get_input_files(pdb_dir, sdf_dir, pdb_id):
    pdb_id = str(pdb_id).strip().upper()

    protein_path = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
    sdf_path = os.path.join(sdf_dir, f"{pdb_id}_ligand.sdf")

    if not os.path.isfile(protein_path):
        raise FileNotFoundError(f"{pdb_id}.pdb")
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"{pdb_id}.sdf")
    return protein_path, sdf_path


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
        protein_src, ligand_src = get_input_files(pdb_dir, sdf_dir, pdb_id)
        problems = []
        if protein_src is None:
            problems.append(f"Missing Proteine {pdb_id} ")
        if ligand_src is None:
            problems.append(f"Missing Ligand {pdb_id} ")
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
            fieldnames=["complex_id", "PDB_code", "affinity", "pk", "pIC50"],
        )
        writer.writeheader()
        for pdb_id in ok_ids:
            writer.writerow({
                "complex_id": pdb_id,
                "PDB_code": pdb_id,
                "affinity": pic50[pdb_id],
                "pk": pic50[pdb_id],
                "pIC50": pic50[pdb_id],
            })

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
        writer = csv.DictWriter(f, fieldnames)
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
    valid_ids = ids[n_train:n_train + n_valid]
    core_ids = ids[n_train + n_valid:]

    if len(train_ids) == 0 and len(valid_ids) > 1:
        core_ids.append(valid_ids.pop())
    if len(valid_ids) == 0 and len(train_ids) > 1:
        core_ids.append(train_ids.pop())

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
            writer.writerow({
                "": index,
                "PDB_code": pdb_id,
                "pK": pic50[pdb_id],
                "type": split_type
            })

    return path


def build_metadata_file(metadata_dir, ids, pic50, train_ratio=0.8, valid_ratio=0.1, seed=42):
    train_ids, valid_ids, core_ids = split_ids(ids, train_ratio, valid_ratio, seed=seed)

    train_csv = write_split_csv(os.path.join(metadata_dir, "train.csv"), train_ids, pic50)
    valid_csv = write_split_csv(os.path.join(metadata_dir, "valid.csv"), valid_ids, pic50)
    core_csv = write_split_csv(os.path.join(metadata_dir, "core.csv"), core_ids, pic50)

    return {
        "train": train_ids,
        "valid": valid_ids,
        "core": core_ids,
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "core_csv": core_csv,
    }


def add_planet_to_root(planet_root):
    planet_root = os.path.abspath(planet_root)
    if planet_root not in sys.path:
        sys.path.insert(0, planet_root)

    return planet_root


def build_one_pocket(planet_root, raw_dir, pocket_root, pdb_id, affinity, overwrite=False):
    add_planet_to_root(planet_root)
    from chemutils import ComplexPocket
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

    pocket = ComplexPocket(protein_path, ligand_sdf_path, float(affinity), None, )

    if getattr(pocket, "res_count", 0) <= 0:
        raise ValueError(f"Pocket construction produce zero residues {pdb_id}")

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


def build_split_pocket(planet_root, raw_dir, pocket_root, csv_path, output_pickle, overwrite_pockets=False):
    df = read_split_csv(csv_path)
    records = []
    skipped = []

    for _, row in df.iterrows():
        pdb_id = str(row["PDB_code"]).strip().upper()
        affinity = float(row["pK"])

        try:
            pocket_path = build_one_pocket(
                planet_root,
                raw_dir,
                pocket_root,
                pdb_id,
                affinity,
                overwrite=overwrite_pockets,
            )
            records.append([pocket_path, affinity])
        except Exception as e:
            skipped.append({
                "complex_id": pdb_id,
                "error": f"{type(e).__name__}: {pdb_id}",

            })
            print(f"Skipping {pdb_id} {e}")
    ensure_directory(os.path.dirname(output_pickle))

    with open(output_pickle, "wb") as f:
        pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

    return records, skipped


def build_all_pickle(planet_root, raw_dir, metadata_dir, output, overwrite_pockets=False):
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
        output_pickles = os.path.join(pkl_dir, f"{split_name}.pkl")
        records, skipped = build_split_pocket(
            planet_root,
            raw_dir,
            pocket_root,
            csv_path,
            output_pickles,
            overwrite_pockets=overwrite_pockets
        )
        summary["splits"][split_name] = {
            "csv_path": csv_path,
            "pickle_path": output_pickles,
            "count_built": len(records),
        }
        summary["skipped"][split_name] = skipped

        print(f"{split_name} built {len(records)} records {output_pickles}")

    save_json(summary, os.path.join(pkl_dir, "build_pkl_summary.json"))
    return summary


def build_data(args):
    output_dir = os.path.abspath(args.output_dir)
    pdb_dir = os.path.abspath(args.pdb_dir)
    sdf_dir = os.path.abspath(args.sdf_dir)
    pic50_path = os.path.abspath(args.pic50)
    planet_root = os.path.abspath(args.planet_root)

    ensure_directory(output_dir)

    ok_ids, pic50, raw_dir, metadata_dir, labels_csv, prepared_csv = prepare_raw_and_labels(
        pdb_dir,
        sdf_dir,
        pic50_path,
        output_dir,
        overwrite=args.overwrite,
    )
    split_info = build_metadata_file(metadata_dir, ok_ids, pic50, train_ratio=args.train_ratio,
                                     valid_ratio=args.val_ratio, seed=args.seed)

    pkl_summary = build_all_pickle(planet_root, raw_dir, metadata_dir, output_dir, args.overwrite_pockets)

    manifest = {
        "pdb_dir": pdb_dir,
        "sdf_dir": sdf_dir,
        "pic50_path": pic50_path,
        "planet_root": planet_root,
        "output_dir": output_dir,
        "raw_dir": raw_dir,
        "metadata_dir": metadata_dir,
        "labels_csv": labels_csv,
        "prepared_csv": prepared_csv,
        "n_labels": len(pic50),
        "n_prepared_ok": len(ok_ids),
        "split_counts":
            {
                "train": len(split_info["train"]),
                "valid": len(split_info["valid"]),
                "core": len(split_info["core"]),

            },
        "pkl_summary": pkl_summary
    }

    manifest_path = os.path.join(metadata_dir, "build_data_manifest.json")
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--sdf-dir", required=True)
    parser.add_argument("--pic50", required=True)
    parser.add_argument("--planet-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-pockets", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    build_data(args)


if __name__ == "__main__":
    main()
