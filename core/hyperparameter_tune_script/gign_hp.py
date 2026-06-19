import os
from time import time

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from data_pipeline.common import load_split_txt, save_all_trials_results_csv, seed_everything
from data_pipeline.URVGraphDataset import URVGraphDataset
from models.gign.GIGN import GIGN


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)

        pred_list.append(pred.cpu().numpy())
        label_list.append(data.y.cpu().numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


def train_gign(config, train_file, test_file, val_file, log_callback, graph_directory):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trial_dir = tune.get_context().get_trial_dir()
    os.makedirs(trial_dir, exist_ok=True)

    training_start_time = time()

    train_splits = load_split_txt(train_file)
    val_splits = load_split_txt(val_file)
    test_splits = load_split_txt(test_file)

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []
    split_best_train_rmses = []

    for split_id in range(len(train_splits)):
        if log_callback:
            log_callback.info(f"SPLIT {split_id}")

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
            drop_last=False,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        model = GIGN(
            config["NODE_DIM"],
            config["hidden_dim"],
            config["drop_out"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0
        patience = config["PATIENCE"]
        best_train_rmse = None

        split_save_dir = os.path.join(trial_dir, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)

        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        for epoch in range(config["EPOCHS"]):
            model.train()

            epoch_loss = 0.0
            n_train = 0

            for data in train_loader:
                data = data.to(device)

                pred = model(data).view(-1)
                target = data.y.view(-1)

                loss = criterion(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = target.size(0)
                epoch_loss += loss.item() * batch_size
                n_train += batch_size

            train_rmse = np.sqrt(epoch_loss / n_train)
            val_rmse, _ = val(model, val_loader, device)

            if log_callback:
                log_callback.info(
                    f"Split {split_id:02d} | "
                    f"Epoch {epoch:03d} | "
                    f"Train RMSE: {train_rmse:.4f} | "
                    f"Val RMSE: {val_rmse:.4f}"
                )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                patience_counter = 0
                best_train_rmse = float(train_rmse)
                torch.save(
                    {
                        "split_id": split_id,
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val_rmse,
                        "best_train_rmse": best_train_rmse,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    best_model_path,
                )

                if log_callback:
                    log_callback.info(f">>> Best model saved for split {split_id:02d}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activated")
                break

        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_rmse, test_pr = val(model, test_loader, device)

        split_best_val_rmses.append(best_val_rmse)
        split_test_rmses.append(test_rmse)
        split_test_pearsons.append(test_pr)
        split_best_train_rmses.append(float(checkpoint["best_train_rmse"]))

        if log_callback:
            log_callback.info(
                f"Split {split_id:02d} | "
                f"Best Val RMSE: {best_val_rmse:.4f} | "
                f"Test RMSE: {test_rmse:.4f} | "
                f"Test Pearson: {test_pr:.4f}"
            )

    mean_train_rmse = float(np.mean(split_best_train_rmses))
    std_train_rmse = float(np.std(split_best_train_rmses))

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    std_val_rmse = float(np.std(split_best_val_rmses))

    mean_test_rmse = float(np.mean(split_test_rmses))
    std_test_rmse = float(np.std(split_test_rmses))

    mean_test_pearson = float(np.mean(split_test_pearsons))
    std_test_pearson = float(np.std(split_test_pearsons))
    hyperparameter_path = os.path.join(trial_dir, "hyperparameters.txt")

    with open(hyperparameter_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")
        f.write(f"mean_train_rmse: {mean_train_rmse}\n")
        f.write(f"std_train_rmse: {std_train_rmse}\n")
        f.write(f"mean_val_rmse: {mean_val_rmse}\n")
        f.write(f"std_val_rmse: {std_val_rmse}\n")
        f.write(f"mean_test_rmse: {mean_test_rmse}\n")
        f.write(f"std_test_rmse: {std_test_rmse}\n")
        f.write(f"mean_test_pearson: {mean_test_pearson}\n")
        f.write(f"std_test_pearson: {std_test_pearson}\n")
    end_training = time()

    if log_callback:
        log_callback.info("Training completed for all splits.")
        log_callback.info(f"Total training time: {end_training - training_start_time:.2f} seconds")
        log_callback.info(f"Mean Val RMSE: {mean_val_rmse:.4f}")
        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f}")
        log_callback.info(f"Mean Test Pearson: {mean_test_pearson:.4f}")

    tune.report(
        {
            "mean_train_rmse": mean_train_rmse,
            "std_train_rmse": std_train_rmse,
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
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=20)

    trainable = tune.with_parameters(
        train_gign,
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
            name="GIGN_hyperparameter_tuning",
            storage_path=save_directory,
        ),
    )

    results = tuner.fit()
    csv_path = save_all_trials_results_csv(
        results=results,
        save_directory=save_directory,
    )

    best_result = results.get_best_result(
        metric="mean_val_rmse",
        mode="min",
    )

    best_trial_dir = best_result.path
    experiment_dir = os.path.dirname(best_trial_dir)

    if log_callback:
        log_callback.info(f"Best trial kept at: {best_trial_dir}")
        log_callback.info(f"Best config: {best_result.config}")
        log_callback.info(f"Best mean val RMSE: {best_result.metrics['mean_val_rmse']}")
        log_callback.info(f"Best mean test RMSE: {best_result.metrics['mean_test_rmse']}")
        log_callback.info(f"Best mean test Pearson: {best_result.metrics['mean_test_pearson']}")
        log_callback(f"Best mean train RMSE: {best_result.metrics['mean_train_rmse']}")
        log_callback(f"Best std train RMSE: {best_result.metrics['std_train_rmse']}")

        log_callback(f"Best std val RMSE: {best_result.metrics['std_val_rmse']}")

        log_callback(f"Best std test RMSE: {best_result.metrics['std_test_rmse']}")

        log_callback(f"Best std test Pearson: {best_result.metrics['std_test_pearson']}")

        log_callback(f"All trials CSV saved at: {csv_path}")

    return best_result
