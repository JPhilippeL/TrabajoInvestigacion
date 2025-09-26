# transfer_trainer_worker.py

import argparse
import time
import torch
import gc
from ML.transfer_trainer_flexible import transfer_train_flexible

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--transfer_mode", default="fine_tuning", choices=["fine_tuning", "feature_extraction"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()

    try:
        start_time = time.time()
        path = transfer_train_flexible(
            pretrained_model_path=args.pretrained_model_path,
            sdf_dir=args.sdf_dir,
            target_file=args.target_file,
            transfer_mode=args.transfer_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            valid_split=args.valid_split,
            save_path=args.save_path
        )
        elapsed = time.time() - start_time
        print(f"FINISHED|{path}|{elapsed:.2f}", flush=True)

    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)

    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
