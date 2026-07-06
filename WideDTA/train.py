"""
@file train.py
@author Mohamed EL BOUKHIARI
@brief Command-line entry point for WideDTA training.
"""

from __future__ import annotations

import argparse
import json

from WideDTA.Core.widedta_trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WideDTA.")
    parser.add_argument("--dataset", default="mpro_urv", choices=["davis", "kiba", "mpro_urv"])
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--use-dataset-folds", action="store_true")
    parser.add_argument("--no-dataset-folds", action="store_true")

    args = parser.parse_args()

    use_dataset_folds = args.use_dataset_folds and not args.no_dataset_folds

    results = train(
        dataset_name=args.dataset,
        output_base=args.output_base,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        val_split=args.val_split,
        test_split=args.test_split,
        max_train_batches=None if args.max_train_batches == 0 else args.max_train_batches,
        fold_index=args.fold_index,
        use_dataset_folds=use_dataset_folds,
    )

    print(json.dumps(results, indent=4, default=str))


if __name__ == "__main__":
    main()
