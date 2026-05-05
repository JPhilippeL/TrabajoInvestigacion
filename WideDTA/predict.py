"""
@file predict.py
@author Mohamed EL BOUKHIARI
@brief Command-line entry point for WideDTA prediction/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from WideDTA.Core.widedta_trainer import (
    RMSLoss,
    build_dataloaders,
    evaluate,
    get_dataset_paths,
    load_checkpoint,
    prepare_batch,
    resolve_device,
)
from WideDTA.data import WideDTADataset


def predict_to_csv(
    checkpoint_path: str,
    dataset_name: str,
    output_csv: str,
    device: str = "auto",
    batch_size: int = 1,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
) -> dict:
    torch_device = resolve_device(device)
    ligand_path, protein_path, motif_path, affinity_path = get_dataset_paths(dataset_name)
    dataset = WideDTADataset(ligand_path, protein_path, motif_path, affinity_path)

    _train_loader, _val_loader, test_loader = build_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    model = load_checkpoint(checkpoint_path, dataset, torch_device)
    model.eval()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            protein, ligand, motif, target = prepare_batch(batch, torch_device)
            output = model(protein, ligand, motif)

            predictions = output.detach().cpu().numpy().reshape(-1).tolist()
            targets = target.detach().cpu().numpy().reshape(-1).tolist()

            for pred, true in zip(predictions, targets):
                rows.append({"prediction": pred, "target": true})

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["prediction", "target"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = evaluate(model, test_loader, RMSLoss(), torch_device)

    return {
        "status": "success",
        "dataset": dataset_name,
        "checkpoint_path": checkpoint_path,
        "output_csv": output_csv,
        "num_predictions": len(rows),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained WideDTA checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="mpro_urv", choices=["davis", "kiba", "mpro_urv"])
    parser.add_argument("--output-csv", default="WideDTA/results/widedta_predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    results = predict_to_csv(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        output_csv=args.output_csv,
        device=args.device,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    print(json.dumps(results, indent=4, default=str))


if __name__ == "__main__":
    main()
