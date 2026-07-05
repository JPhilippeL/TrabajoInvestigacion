"""
@file predict.py
@brief Command-line entry point for DeepDTA prediction/evaluation.
"""

from __future__ import annotations

import argparse
import json

from DeepDTA.Core.deepdta_evaluator import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepDTA checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="mpro_urv", choices=["davis", "kiba", "mpro_urv"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--use-dataset-folds", action="store_true", default=True)
    parser.add_argument("--no-dataset-folds", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        fold_index=args.fold_index,
        use_dataset_folds=args.use_dataset_folds and not args.no_dataset_folds,
        split=args.split,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(json.dumps(results, indent=4, default=str))


if __name__ == "__main__":
    main()
