import argparse
import gc
import os
import sys
import time
import torch

sys.path.append(os.getcwd())

from GNNs.hyperparameter_search import run_hyperparameter_search


def parse_list(value, cast_type=str):
    if value is None or str(value).strip() == "":
        return []

    items = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            items.append(cast_type(item))

    return items


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_sdf_dir", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument("--eval_sdf_dir", default="")
    parser.add_argument("--eval_targets_file", default="")

    parser.add_argument("--model_names", default="GIN,GINE,GAT,EGAT,GraphTransformer")

    parser.add_argument("--lr_values", default="0.001,0.0005,0.0001")
    parser.add_argument("--batch_size_values", default="16,32")
    parser.add_argument("--hidden_dim_values", default="64,128")
    parser.add_argument("--num_layers_values", default="2,3")
    parser.add_argument("--atom_emb_dim_values", default="0.4")
    parser.add_argument("--hibrid_emb_dim_values", default="0.5")
    parser.add_argument("--bond_emb_dim_values", default="1")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--valid_split", type=float, default=0.2)

    parser.add_argument("--objective_metric", default="RMSE")
    parser.add_argument("--objective_mode", default="min")

    parser.add_argument("--resume", default="true")
    parser.add_argument("--rerun_failed", default="false")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    try:
        start = time.time()

        print("STARTED|Hyperparameter search started", flush=True)

        model_names = parse_list(args.model_names, str)

        if not model_names:
            raise ValueError("No model selected for hyperparameter search.")

        search_space = {
            "lr": parse_list(args.lr_values, float),
            "batch_size": parse_list(args.batch_size_values, int),
            "hidden_dim": parse_list(args.hidden_dim_values, int),
            "num_layers": parse_list(args.num_layers_values, int),
            "atom_emb_dim": parse_list(args.atom_emb_dim_values, float),
            "hibrid_emb_dim": parse_list(args.hibrid_emb_dim_values, float),
            "bond_emb_dim": parse_list(args.bond_emb_dim_values, float),
        }

        for key, values in search_space.items():
            if not values:
                raise ValueError(f"Empty search space for parameter: {key}")

        results = run_hyperparameter_search(
            train_sdf_dir=args.train_sdf_dir,
            target_file=args.target_file,
            output_root=args.output_root,
            model_names=model_names,
            search_space=search_space,
            eval_sdf_dir=args.eval_sdf_dir or None,
            eval_targets_file=args.eval_targets_file or None,
            epochs=args.epochs,
            patience=args.patience,
            valid_split=args.valid_split,
            objective_metric=args.objective_metric,
            objective_mode=args.objective_mode,
            resume=parse_bool(args.resume),
            rerun_failed=parse_bool(args.rerun_failed),
            seed=args.seed,
        )

        elapsed = time.time() - start

        status = results.get("status", "")
        message = results.get("message", "")
        trials_csv = results.get("trials_csv", "")
        failed_trials_csv = results.get("failed_trials_csv", "")
        best_config_json = results.get("best_config_json", "")
        best_config_yaml = results.get("best_config_yaml", "")

        print(
            f"FINISHED|{status}|{message}|{trials_csv}|"
            f"{failed_trials_csv}|{best_config_json}|{best_config_yaml}|{elapsed:.2f}",
            flush=True,
        )

    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)

    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
