import os
from time import time

import numpy as np
import ray
import torch
from models.graphdta.registry import get_graph_dta_model
from ray import tune
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from data_pipeline.common import (
    load_split_txt,
    save_all_trials_results_csv,
    seed_everything,
)
from data_pipeline.URVGraphDataset import URVGraphDataset


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).view(-1)
            target = data.y.view(-1).float()

        pred_list.append(pred.cpu().numpy())
        label_list.append(target.cpu().numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


def train_dta(
    config,
    model_name,
    output_dir,
    train_split_file,
    val_split_file,
    test_split_file,
    graph_dir,
    log_callback=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trial_dir = tune.get_context().get_trial_dir()
    os.makedirs(output_dir, exist_ok=True)
    train_split = load_split_txt(train_split_file)
    val_split = load_split_txt(val_split_file)
    test_split = load_split_txt(test_split_file)
    split_best_train_rmses = []
    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []
    patience = 15
    trial_config = {
        "model_name": model_name,
        "n_filters": config["n_filters"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "patience": patience,
        "epochs": config["epochs"],
    }
    for split_id in range(len(train_split)):
        if log_callback:
            log_callback.info(f"SPLIT: {split_id}")

        seed_everything(split_id)
        patience_counter = 0

        train_ids = train_split[split_id]
        test_ids = test_split[split_id]
        val_ids = val_split[split_id]

        train_set = URVGraphDataset(graph_dir, train_ids)
        sample = train_set[0]

        if log_callback:
            log_callback.info(f"Graph directory used: {graph_dir}")
            log_callback.info(f"First graph x shape: {sample.x.shape}")
            log_callback.info(f"First graph keys: {sample.keys()}")
        test_set = URVGraphDataset(graph_dir, test_ids)
        val_set = URVGraphDataset(graph_dir, val_ids)

        train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], shuffle=True, drop_last=False
        )
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

        model_class = get_graph_dta_model(model_name)

        model = model_class(
            n_filters=config["n_filters"],
            dropout=config["dropout"],
        ).to(device)
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        best_val_rmse = float("inf")
        best_epoch = -1
        best_train_rmse = None

        split_save_dir = os.path.join(trial_dir, f"split_{split_id:02}")
        os.makedirs(split_save_dir, exist_ok=True)
        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        for epoch in range(config["epochs"]):
            model.train()
            epoch_loss = 0.0
            n_train = 0
            for data in train_loader:
                data = data.to(device)
                pred = model(data)
                target = data.y.view(-1, 1).float()
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_current = target.size(0)
                epoch_loss += loss.item() * batch_size_current
                n_train += batch_size_current

            train_rmse = np.sqrt(epoch_loss / n_train)
            val_rmse, val_pr = val(model, val_loader, device)
            if log_callback:
                log_callback.info(
                    f"Epoch {epoch:03d} | "
                    f"Train RMSE: {train_rmse:.4f} | "
                    f"Val RMSE: {val_rmse:.4f} | "
                    f"Val Pearson: {val_pr:.4f}"
                )
            if val_rmse < best_val_rmse:
                best_val_rmse = float(val_rmse)
                best_epoch = epoch
                best_train_rmse = float(train_rmse)
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_train_rmse": best_train_rmse,
                        "best_val_rmse": best_val_rmse,
                        "config": trial_config,
                    },
                    best_model_path,
                )

                patience_counter = 0
                if log_callback:
                    log_callback.info(">>> Better model stocked")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early Stopped")
                break
        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        checkpoint_config = checkpoint["config"]

        best_model_class = get_graph_dta_model(checkpoint["model_name"])

        best_model = best_model_class(
            n_filters=checkpoint_config["n_filters"],
            dropout=checkpoint_config["dropout"],
        ).to(device)

        best_model.load_state_dict(checkpoint["model_state_dict"])

        test_rmse, test_pr = val(best_model, test_loader, device)
        if log_callback:
            log_callback.info(
                f"[SPLIT {split_id}] "
                f"Best epoch: {best_epoch} | "
                f"Best Train RMSE: {checkpoint['best_train_rmse']:.4f} | "
                f"Best Val RMSE: {checkpoint['best_val_rmse']:.4f} | "
                f"Test RMSE: {test_rmse:.4f} | "
                f"Test Pearson: {test_pr:.4f}"
            )
        split_best_train_rmses.append(float(checkpoint["best_train_rmse"]))
        split_best_val_rmses.append(float(checkpoint["best_val_rmse"]))
        split_test_rmses.append(float(test_rmse))
        split_test_pearsons.append(float(test_pr))
    mean_train_rmse = float(np.mean(split_best_train_rmses))
    std_train_rmse = float(np.std(split_best_train_rmses))

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    std_val_rmse = float(np.std(split_best_val_rmses))

    mean_test_rmse = float(np.mean(split_test_rmses))
    std_test_rmse = float(np.std(split_test_rmses))

    mean_test_pearson = float(np.mean(split_test_pearsons))
    std_test_pearson = float(np.std(split_test_pearsons))

    if log_callback:
        log_callback.info("Training finished for all splits")
        log_callback.info(
            f"Mean Best Val RMSE: {mean_val_rmse:.4f} std {np.std(split_best_val_rmses):.4f}"
        )
        log_callback.info(
            f"Mean Test RMSE: {mean_test_rmse:.4f} std  {np.std(split_test_rmses):.4f}"
        )
        log_callback.info(
            f"Mean Test Pearson: {mean_test_pearson:.4f} std {np.std(split_test_pearsons):.4f}"
        )

    hyperparameter_path = os.path.join(trial_dir, f"hyperparameters_{model_name}.txt")

    with open(hyperparameter_path, "w") as f:
        for key, value in trial_config.items():
            f.write(f"{key}: {value}\n")

        f.write(f"mean_train_rmse: {mean_train_rmse}\n")
        f.write(f"std_train_rmse: {std_train_rmse}\n")
        f.write(f"mean_val_rmse: {mean_val_rmse}\n")
        f.write(f"std_val_rmse: {std_val_rmse}\n")
        f.write(f"mean_test_rmse: {mean_test_rmse}\n")
        f.write(f"std_test_rmse: {std_test_rmse}\n")
        f.write(f"mean_test_pearson: {mean_test_pearson}\n")
        f.write(f"std_test_pearson: {std_test_pearson}\n")

    if log_callback:
        log_callback.info("Training completed for all splits.")

        log_callback.info(f"Mean Train RMSE: {mean_train_rmse:.4f}")
        log_callback.info(f"Std Train RMSE: {std_train_rmse:.4f}")

        log_callback.info(f"Mean Val RMSE: {mean_val_rmse:.4f}")
        log_callback.info(f"Std Val RMSE: {std_val_rmse:.4f}")

        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f}")
        log_callback.info(f"Std Test RMSE: {std_test_rmse:.4f}")

        log_callback.info(f"Mean Test Pearson: {mean_test_pearson:.4f}")
        log_callback.info(f"Std Test Pearson: {std_test_pearson:.4f}")
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


def launch_hyperparametre_search(
    config,
    cpu_per_trial,
    gpu_per_trial,
    num_samples,
    output_dir,
    model_name,
    train_split_file,
    val_split_file,
    test_split_file,
    graph_dir,
    log_callback=None,
):
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    os.makedirs(output_dir, exist_ok=True)

    trainable = tune.with_parameters(
        train_dta,
        model_name=model_name,
        output_dir=output_dir,
        train_split_file=train_split_file,
        val_split_file=val_split_file,
        test_split_file=test_split_file,
        graph_dir=graph_dir,
        log_callback=log_callback,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": cpu_per_trial,
                "gpu": gpu_per_trial,
            },
        ),
        tune_config=tune.TuneConfig(
            metric="mean_val_rmse",
            mode="min",
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=ray.tune.RunConfig(
            name=f"GRAPHDTA_{model_name}",
            storage_path=output_dir,
        ),
    )
    results = tuner.fit()
    csv_path = save_all_trials_results_csv(
        results=results,
        save_directory=output_dir,
    )

    best_result = results.get_best_result(
        metric="mean_val_rmse",
        mode="min",
    )
    best_trial_dir = best_result.path
    if log_callback:
        log_callback.info(f"Best trial kept at: {best_trial_dir}")
        log_callback.info(f"Best config: {best_result.config}")
        log_callback.info(f"Best mean train RMSE: {best_result.metrics['mean_train_rmse']}")
        log_callback.info(f"Best std train RMSE: {best_result.metrics['std_train_rmse']}")
        log_callback.info(f"Best mean val RMSE: {best_result.metrics['mean_val_rmse']}")
        log_callback.info(f"Best std val RMSE: {best_result.metrics['std_val_rmse']}")
        log_callback.info(f"Best mean test RMSE: {best_result.metrics['mean_test_rmse']}")
        log_callback.info(f"Best std test RMSE: {best_result.metrics['std_test_rmse']}")
        log_callback.info(f"Best mean test Pearson: {best_result.metrics['mean_test_pearson']}")
        log_callback.info(f"Best std test Pearson: {best_result.metrics['std_test_pearson']}")
        log_callback.info(f"All trials CSV saved at: {csv_path}")
    return best_result
