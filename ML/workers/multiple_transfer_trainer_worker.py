# multiple_models_trainer_worker.py
import argparse
import time
import torch
import gc
from ML.transfer_trainer import transfer_train_multiple_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_directory_path")
    parser.add_argument("--sdf_dir")
    parser.add_argument("--target_file")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--transfer_mode", type = int, default = 0)  # 0: both, 1: feature extraction, 2: fine-tuning
    args = parser.parse_args()

    try:
        start = time.time()
        paths = transfer_train_multiple_models(
            pretrained_model_directory_path=args.pretrained_model_directory_path,
            sdf_dir=args.sdf_dir,
            target_file=args.target_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            valid_split=args.valid_split,
            patience=args.patience
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