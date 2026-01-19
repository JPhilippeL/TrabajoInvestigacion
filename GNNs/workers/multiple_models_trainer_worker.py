# multiple_models_trainer_worker.py
import argparse
import time
import torch
import gc
from ML.model_trainer import train_multiple_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir")
    parser.add_argument("--target_file")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--atom_emb_dim", type=float)
    parser.add_argument("--hibrid_emb_dim", type=float)
    parser.add_argument("--bond_emb_dim", type=float)
    args = parser.parse_args()

    try:
        start = time.time()
        paths = train_multiple_models(
            sdf_dir=args.sdf_dir,
            target_file=args.target_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            valid_split=args.valid_split,
            hidden_dim=args.hidden_dim,
            patience=args.patience,
            atom_emb_dim=args.atom_emb_dim,
            hibrid_emb_dim=args.hibrid_emb_dim,
            bond_emb_dim=args.bond_emb_dim
        )
        elapsed = time.time() - start

        print(f"FINISHED||{elapsed:.2f}", flush=True)
    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()