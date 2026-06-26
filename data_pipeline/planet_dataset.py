import os
import pickle
import random
import sys
from torch.utils.data import DataLoader, Dataset

from pathlib import Path


class PocketDatasetRecord:
    def __init__(self, complex_id, pocket_path, affinity=0.0):
        self.complex_id = complex_id
        self.pocket_path = pocket_path
        self.affinity = float(affinity)


def add_planet_to_path(planet_root):
    if planet_root is None:
        return None

    planet_root = Path(planet_root).expanduser().resolve()

    if not planet_root.is_dir():
        raise ValueError(f"planet_root is not a directory: {planet_root}")

    planet_root_str = str(planet_root)

    if planet_root_str not in sys.path:
        sys.path.insert(0, planet_root_str)

    return planet_root


def normalize_record(record):
    if isinstance(record, PocketDatasetRecord):
        return record

    if isinstance(record, (list, tuple)) and len(record) >= 2:
        pocket_path = record[0]
        affinity = record[1]

        file_name = os.path.basename(str(pocket_path))
        complex_id = file_name.replace("_pocket.pkl", "")

        return PocketDatasetRecord(
            complex_id=complex_id,
            pocket_path=pocket_path,
            affinity=affinity,
        )

    if isinstance(record, dict):
        pocket_path = record.get("pocket_path") or record.get("path")
        affinity = (
                record.get("affinity")
                or record.get("pK")
                or record.get("pk")
                or record.get("pIC50")
                or 0.0
        )

        complex_id = record.get("complex_id")

        if complex_id is None and pocket_path is not None:
            file_name = os.path.basename(str(pocket_path))
            complex_id = file_name.replace("_pocket.pkl", "")

        return PocketDatasetRecord(
            complex_id=complex_id,
            pocket_path=pocket_path,
            affinity=affinity,
        )

    raise ValueError(f"Unsupported record format: {record}")


def load_planet_pickle_records(dataset_pkl):
    if not os.path.isfile(dataset_pkl):
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl}")

    with open(dataset_pkl, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {dataset_pkl}, got {type(payload)}")

    records = []

    for item in payload:
        record = normalize_record(item)

        if not os.path.isfile(record.pocket_path):
            raise FileNotFoundError(
                f"Pocket file not found for {record.complex_id}: {record.pocket_path}",
            )

        records.append(record)

    return records


def split_records_into_batches(records, batch_size, shuffle=True, seed=42):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    records = list(records)

    if shuffle:
        random.seed(seed)
        random.shuffle(records)

    batches = []

    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        batches.append(batch)

    return batches


def tensorize_records(records, planet_root=None, decoy_flag=True):
    if planet_root is not None:
        add_planet_to_path(planet_root)

    from data_pipeline.chemutils import tensorize_all

    pocket_batch = []

    for record in records:
        with open(record.pocket_path, "rb") as f:
            pocket = pickle.load(f)

        pocket_batch.append(pocket)

    (
        res_feature_batch,
        mol_feature_batch,
        mol_interactions,
        pro_lig_interactions,
        pks,
        pk_flags,
        complex_labels,
    ) = tensorize_all(pocket_batch, decoy_flag=decoy_flag)

    targets = (
        mol_interactions,
        pro_lig_interactions,
        pks,
        pk_flags,
        complex_labels,
    )

    return res_feature_batch, mol_feature_batch, targets


class PlanetPocketDataset(Dataset):
    def __init__(
            self,
            dataset_pkl,
            batch_size,
            planet_root=None,
            shuffle=True,
            seed=42,
            decoy_flag=True,
    ):
        self.dataset_pkl = dataset_pkl
        self.planet_root = planet_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.decoy_flag = decoy_flag

        self.records = load_planet_pickle_records(dataset_pkl)

        self.batches = split_records_into_batches(
            records=self.records,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_records = self.batches[index]

        return tensorize_records(
            records=batch_records,
            planet_root=self.planet_root,
            decoy_flag=self.decoy_flag,
        )

    def get_record_count(self):
        return len(self.records)

    def get_batch_count(self):
        return len(self.batches)

    def get_batch_records(self, index):
        return self.batches[index]


def make_planet_dataloader(dataset, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=lambda batch: batch[0],
    )


def create_datasets_from_output(
        output_dir,
        planet_root,
        batch_size,
        seed=42,
        decoy_flag=True,
):
    pkl_dir = os.path.join(output_dir, "metadata", "pkl")

    train_pkl = os.path.join(pkl_dir, "train.pkl")
    valid_pkl = os.path.join(pkl_dir, "valid.pkl")
    core_pkl = os.path.join(pkl_dir, "core.pkl")

    train_dataset = PlanetPocketDataset(
        dataset_pkl=train_pkl,
        planet_root=planet_root,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        decoy_flag=decoy_flag,
    )

    valid_dataset = PlanetPocketDataset(
        dataset_pkl=valid_pkl,
        planet_root=planet_root,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        decoy_flag=decoy_flag,
    )

    core_dataset = PlanetPocketDataset(
        dataset_pkl=core_pkl,
        planet_root=planet_root,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        decoy_flag=decoy_flag,
    )

    return train_dataset, valid_dataset, core_dataset


def create_dataloaders_from_output(
        output_dir,
        planet_root,
        batch_size,
        seed=42,
        decoy_flag=True,
        num_workers=0,
):
    train_dataset, valid_dataset, core_dataset = create_datasets_from_output(
        output_dir=output_dir,
        planet_root=planet_root,
        batch_size=batch_size,
        seed=seed,
        decoy_flag=decoy_flag,
    )

    train_loader = make_planet_dataloader(
        train_dataset,
        num_workers=num_workers,
    )

    valid_loader = make_planet_dataloader(
        valid_dataset,
        num_workers=num_workers,
    )

    core_loader = make_planet_dataloader(
        core_dataset,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, core_loader
