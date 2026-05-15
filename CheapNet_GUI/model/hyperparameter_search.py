import os
import shutil
from time import time

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from torch_geometric.loader import DataLoader

from CheapNet_GUI.model.CheapNet_model import CheapNet
from CheapNet_GUI.model.utils import (
    load_split_txt,
    seed_everything,
    URVGraphDataset,
    val,
)


def train_cheapnet(config, train_file, test_file, val_file, log_callback, graph_directory):
    # Quantiles of train set
    q_lig = [0, 20, 28, 37, 177]
    q_pro = [0, 130, 156, 186, 500]
    q_i_lig = 2
    q_i_pro = 2
    num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trial_dir = tune.get_context().get_trial_dir()
    os.makedirs(trial_dir, exist_ok=True)

    train_splits = load_split_txt(train_file)
    val_splits = load_split_txt(val_file)
    test_splits = load_split_txt(test_file)

    debut_train = time()

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []

    for split_id in range(len(train_splits)):
        if log_callback:
            log_callback.info("\n==============================")
            log_callback.info(f"        SPLIT {split_id:02d}")
            log_callback.info("==============================")

        seed_everything(split_id)

        train_ids = train_splits[split_id]
        val_ids = val_splits[split_id]
        test_ids = test_splits[split_id]

        train_set = URVGraphDataset(graph_directory, train_ids)
        val_set = URVGraphDataset(graph_directory, val_ids)
        test_set = URVGraphDataset(graph_directory, test_ids)

        train_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        all_train_y = []

        for data in train_loader:
            all_train_y.append(data.y)

        all_train_y = torch.cat(all_train_y)

        y_mean = all_train_y.mean().to(device)
        y_std = all_train_y.std().to(device)

        if log_callback:
            log_callback.info(f"Target mean: {y_mean.item():.4f}")
            log_callback.info(f"Target std: {y_std.item():.4f}")
            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")

        model = CheapNet(
            config["NODE_DIM"],
            config["hidden_dim"],
            config["drop_out"],
            num_clusters,
        ).to(device)

        if log_callback:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log_callback.info(f"Trainable params: {n_params}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0
        patience = config["PATIENCE"]

        split_save_dir = os.path.join(trial_dir, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)

        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        # ---------------- Training loop ----------------
        for epoch in range(config["EPOCHS"]):
            model.train()

            epoch_loss = 0.0
            n_train = 0

            for data in train_loader:
                data = data.to(device)

                pred = model(data).view(-1)

                y = (data.y.view(-1) - y_mean) / y_std

                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_batch_size = data.y.view(-1).size(0)
                epoch_loss += loss.item() * current_batch_size
                n_train += current_batch_size

            train_rmse = np.sqrt(epoch_loss / n_train)

            test_rmse, test_pr = val(model, test_loader, device, y_mean, y_std)
            val_rmse, _ = val(model, val_loader, device, y_mean, y_std)

            if log_callback:
                log_callback.info(
                    f"Epoch {epoch:03d} | "
                    f"Train RMSE: {train_rmse:.4f} | "
                    f"Test RMSE: {test_rmse:.4f} | "
                    f"Val RMSE: {val_rmse:.4f} | "
                    f"Pearson: {test_pr:.4f}"
                )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "split_id": split_id,
                        "best_epoch": epoch,
                        "best_val_rmse": best_val_rmse,
                        "y_mean": y_mean,
                        "y_std": y_std,
                    },
                    best_model_path,
                )

                if log_callback:
                    log_callback.info(">>> Nuevo mejor modelo guardado")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activado")
                break

        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        y_mean = checkpoint["y_mean"].to(device)
        y_std = checkpoint["y_std"].to(device)

        test_rmse, test_pr = val(model, test_loader, device, y_mean, y_std)

        split_best_val_rmses.append(best_val_rmse)
        split_test_rmses.append(test_rmse)
        split_test_pearsons.append(test_pr)

        if log_callback:
            log_callback.info(
                f"Best RMSE split {split_id:02d}: {best_val_rmse:.4f}"
            )
            log_callback.info(
                f"Test RMSE split {split_id:02d}: {test_rmse:.4f}"
            )
            log_callback.info(
                f"Test Pearson split {split_id:02d}: {test_pr:.4f}"
            )

    end_train = time()
    train_time = end_train - debut_train

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    mean_test_rmse = float(np.mean(split_test_rmses))
    mean_test_pearson = float(np.mean(split_test_pearsons))

    std_val_rmse = float(np.std(split_best_val_rmses))
    std_test_rmse = float(np.std(split_test_rmses))
    std_test_pearson = float(np.std(split_test_pearsons))

    hyperparameter_path = os.path.join(trial_dir, "cheapnet_hyperparameter.txt")

    with open(hyperparameter_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")
        f.write(f"mean_val_rmse: {mean_val_rmse}\n")
        f.write(f"std_val_rmse: {std_val_rmse}\n")
        f.write(f"mean_test_rmse: {mean_test_rmse}\n")
        f.write(f"std_test_rmse: {std_test_rmse}\n")
        f.write(f"mean_test_pearson: {mean_test_pearson}\n")
        f.write(f"std_test_pearson: {std_test_pearson}\n")
        f.write(f"training_time: {train_time}\n")

    if log_callback:
        log_callback.info(f"\nIt took {train_time:.2f} seconds to train all splits.")
        log_callback.info("\nTraining completed for CheapNet.")
        log_callback.info(f"Mean Val RMSE: {mean_val_rmse:.4f}")
        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f}")
        log_callback.info(f"Mean Test Pearson: {mean_test_pearson:.4f}")
        log_callback.info("Hyperparameters saved in cheapnet_hyperparameter.txt")

    tune.report(
        {
            "mean_val_rmse": mean_val_rmse,
            "std_val_rmse": std_val_rmse,
            "mean_test_rmse": mean_test_rmse,
            "std_test_rmse": std_test_rmse,
            "mean_test_pearson": mean_test_pearson,
            "std_test_pearson": std_test_pearson,
        }
    )


def run_hyperparameter_search(
        save_directory,
        train_file,
        test_file,
        val_file,
        search_space,
        graph_directory,
        cpu_used_per_trials,
        gpu_used_per_trials,
        number_of_trials,
        log_callback,
):
    ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=20,
    )

    trainable = tune.with_parameters(
        train_cheapnet,
        train_file=train_file,
        test_file=test_file,
        val_file=val_file,
        log_callback=log_callback,
        graph_directory=graph_directory,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": cpu_used_per_trials,
                "gpu": gpu_used_per_trials,
            },
        ),
        tune_config=tune.TuneConfig(
            metric="mean_val_rmse",
            mode="min",
            num_samples=number_of_trials,
        ),
        param_space=search_space,
        run_config=ray.tune.RunConfig(
            name="CheapNet_hyperparameter_tuning",
            storage_path=save_directory,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(
        metric="mean_val_rmse",
        mode="min",
    )

    best_trial_dir = best_result.path

    final_best_dir = os.path.join(save_directory, "best_trial")

    if os.path.exists(final_best_dir):
        shutil.rmtree(final_best_dir)

    shutil.copytree(best_trial_dir, final_best_dir)

    if log_callback:
        log_callback.info(f"Best trial copied to: {final_best_dir}")
        log_callback.info(f"Best config: {best_result.config}")
        log_callback.info(
            f"Best mean val RMSE: {best_result.metrics['mean_val_rmse']}"
        )

    experiment_dir = os.path.dirname(best_trial_dir)

    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)

        if item_path == final_best_dir:
            continue

        if item_path == best_trial_dir:
            continue

        if os.path.isdir(item_path):
            shutil.rmtree(item_path, ignore_errors=True)

    return best_result
