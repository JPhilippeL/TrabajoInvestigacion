# trainer_worker.py
import argparse
import time
import torch
import gc
from GNNs.model_trainer import train_and_save_model_from_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file")
    parser.add_argument("--model_type")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--valid_split", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--atom_emb_dim", type=float)
    parser.add_argument("--hibrid_emb_dim", type=float)
    parser.add_argument("--bond_emb_dim", type=float)
    args = parser.parse_args()

    try:
        start = time.time()
        path = train_and_save_model_from_pt(
            pt_file = args.pt_file,
            model_type=args.model_type,
            epochs=args.epochs,
            model_name=args.model_name,
            batch_size=args.batch_size,
            lr=args.lr,
            valid_split=args.valid_split,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            patience=args.patience,
            atom_emb_dim=args.atom_emb_dim,
            hibrid_emb_dim=args.hibrid_emb_dim,
            bond_emb_dim=args.bond_emb_dim
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
