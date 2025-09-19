# trainer_worker.py
import argparse
import time
import torch
import gc
import sys
from ML.model_trainer import train_and_save_model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir")
    parser.add_argument("--target_file")
    parser.add_argument("--modelo_nombre")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()

    try:
        start = time.time()
        path = train_and_save_model(
            sdf_dir=args.sdf_dir,
            target_file=args.target_file,
            modelo_nombre=args.modelo_nombre,
            epochs=args.epochs,
            save_path=args.save_path,
            batch_size=args.batch_size,
            lr=args.lr,
            valid_split=args.valid_split,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            patience=args.patience
        )
        elapsed = time.time() - start

        print(f"FINISHED|{path}|{elapsed:.2f}", flush=True)
    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
