import os
import shutil

import numpy as np
import ray
import torch
from ray import tune
from torch_geometric.loader import DataLoader

from GraphDTA.model.utils import val, initialize_model, seed_everything, load_split_txt, URVGraphDataset


def train_dta(config, model_name, output_dir, train_split_file, val_split_file, test_split_file, graph_dir,
              log_callback=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trial_dir = tune.get_context().get_trial_dir()
    os.makedirs(output_dir, exist_ok=True)
    train_split = load_split_txt(train_split_file)
    val_split = load_split_txt(val_split_file)
    test_split = load_split_txt(test_split_file)

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []

    patience = 15
    for split_id in range(len(train_split)):
        if log_callback:
            log_callback.info(f"SPLIT: {split_id}")

        seed_everything(split_id)
        patience_counter = 0

        train_ids = train_split[split_id]
        test_ids = test_split[split_id]
        val_ids = val_split[split_id]

        train_set = URVGraphDataset(graph_dir, train_ids)
        test_set = URVGraphDataset(graph_dir, test_ids)
        val_set = URVGraphDataset(graph_dir, val_ids)

        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

        model = initialize_model(model_name=model_name, n_filters=config["n_filters"], drop_out=config["dropout"]).to(
            device)
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        best_rmse = float("inf")
        best_epoch = -1
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
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = epoch

                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_rmse,
                        "config": {
                            "n_filters": config["n_filters"],
                            "dropout": config["dropout"],
                            "batch_size": config["batch_size"],
                            "lr": config["lr"],
                            "weight_decay": config["weight_decay"],
                            "patience": patience,
                            "epochs": 50,
                        },
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
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

        checkpoint_config = checkpoint["config"]

        best_model = initialize_model(
            model_name=checkpoint["model_name"],
            n_filters=checkpoint_config["n_filters"],
            drop_out=checkpoint_config["dropout"],
        ).to(device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
        test_rmse, test_pr = val(best_model, test_loader, device)
        if log_callback:
            log_callback.info(
                f"[SPLIT {split_id}] "
                f"Best epoch: {best_epoch} | "
                f"Best Val RMSE: {best_rmse:.4f} | "
                f"Test RMSE: {test_rmse:.4f} | "
                f"Test Pearson: {test_pr:.4f}"
            )

        split_best_val_rmses.append(best_rmse)
        split_test_rmses.append(test_rmse)
        split_test_pearsons.append(test_pr)

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    mean_test_rmse = float(np.mean(split_test_rmses))
    mean_test_pearson = float(np.mean(split_test_pearsons))

    if log_callback:
        log_callback.info("Training finished for all splits")
        log_callback.info(
            f"Mean Best Val RMSE: {mean_val_rmse:.4f} std {np.std(split_best_val_rmses):.4f}")
        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f} std  {np.std(split_test_rmses):.4f}")
        log_callback.info(
            f"Mean Test Pearson: {mean_test_pearson:.4f} std {np.std(split_test_pearsons):.4f}")

    hyperparameter_path = os.path.join(trial_dir, f"hyperparameters_{model_name}.txt")
    with open(hyperparameter_path, "w") as f:
        for key, value in checkpoint_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"Mean Best Val RMSE: {mean_val_rmse:.4f} std {np.std(split_best_val_rmses):.4f}")
        f.write(f"Mean Test RMSE: {mean_test_rmse:.4f} std  {np.std(split_test_rmses):.4f}")
        f.write(f"Mean Test Pearson: {mean_test_pearson:.4f} std {np.std(split_test_pearsons):.4f}")

    tune.report(
        {
            "mean_val_rmse": mean_val_rmse,
            "mean_test_rmse": mean_test_rmse,
            "mean_test_pearson": mean_test_pearson,
        }
    )


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    output_dir = "/home/administrateur/Bureau/TrabajoInvestigacion/GraphDTA/results_hp"
    os.makedirs(output_dir, exist_ok=True)
    train_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/train_index_folder.txt"
    val_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/valid_index_folder.txt"
    test_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/test_index_folder.txt"
    graph_dir = "/home/administrateur/Bureau/TrabajoInvestigacion/GraphDTA/graph"
    model_name = ["GINConvNet", "GAT", "GCN", "GAT_GCN"]
    config = {
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "dropout": tune.choice([0.0, 0.05, 0.1, 0.2]),
        "n_filters": tune.choice([16, 32, 64, 128]),
        "epochs": 50,
    }
    for name in model_name:
        trainable = tune.with_parameters(
            train_dta,
            model_name=name,
            output_dir=output_dir,
            train_split_file=train_split_file,
            val_split_file=val_split_file,
            test_split_file=test_split_file,
            graph_dir=graph_dir,
            log_callback=None

        )

        tuner = tune.Tuner(
            tune.with_resources(
                trainable,
                resources={
                    "cpu": 5,
                    "gpu": 1,
                },
            ),
            tune_config=tune.TuneConfig(
                metric="mean_val_rmse",
                mode="min",
                num_samples=30,
            ),
            param_space=config,
            run_config=ray.tune.RunConfig(
                name=f"GRAPHDTA_{name}",
                storage_path=output_dir,
            ),
        )
        results = tuner.fit()

        best_result = results.get_best_result(
            metric="mean_val_rmse",
            mode="min",
        )
        best_trial_dir = best_result.path
        expirement_dir = os.path.join(best_trial_dir)
        print(f"Best trial kept at: {best_trial_dir}")
        print(f"Best config: {best_result.config}")
        print(f"Best mean val RMSE: {best_result.metrics['mean_val_rmse']}")
        print(f"Best mean test RMSE: {best_result.metrics['mean_test_rmse']}")
        print(f"Best mean test Pearson: {best_result.metrics['mean_test_pearson']}")

        for item in os.listdir(expirement_dir):
            item_path = os.path.join(expirement_dir, item)

            if os.path.abspath(item_path) == os.path.abspath(best_trial_dir):
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
