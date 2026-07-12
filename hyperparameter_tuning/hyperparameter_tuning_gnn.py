import datetime
import logging
import os
import shutil
import tempfile
from time import time

import ray
import torch
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GNNs.data_processing import prepare_sdf_training_data
from GNNs.model_trainer import (
    calc_dim,
    create_model,
)
from ui.utils.constants import (
    N_BOND_TYPES,
    OTHER_EDGE_FEATURES,
    OTHER_NODE_FEATURES,
    hybridization_types,
    periodic_elements,
)

"""
This script performs hyperparameter tuning on the GNNs models using Ray Tune.
We define a training function that trains the model and reports the training and validation losses to Ray Tune.
The hyperparameters scale are in the search_space_dictionary dictionary.

By the way, I only used the train function already written in the model_trainer.py in GNNs. I added ray tune configurations
and removed some unused features (Thanks for that clean code and easy to understand).

All my work and experiments are based on the PyTorch tutorial for hyperparameter tuning with Ray Tune, which can be found at the following link:
@reference: https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
@reference: https://docs.ray.io/en/latest/tune/key-concepts.html
"""


def train(
        search_space_dictionary,
        sdf_directory,
        target_file,
        gnn_model_name,
        valid_split,
        gnn_model_pseudo,
):
    best_model_tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"{gnn_model_pseudo}_{os.getpid()}_best_model_tmp.pt",
    )

    calc_atom_emb_dim = calc_dim(len(periodic_elements) * search_space_dictionary["atom_emb_dim"])

    calc_hybrid_emb_dim = calc_dim(
        len(hybridization_types) * search_space_dictionary["hybrid_emb_dim"]
    )

    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * search_space_dictionary["bond_emb_dim"])

    input_dim = calc_atom_emb_dim + calc_hybrid_emb_dim + OTHER_NODE_FEATURES

    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    train_loader, val_loader, device, _ = prepare_sdf_training_data(
        sdf_directory,
        target_file,
        batch_size=search_space_dictionary["batch_size"],
        valid_split=valid_split,
    )

    model = create_model(
        gnn_model_name,
        input_dim,
        calc_atom_emb_dim,
        calc_hybrid_emb_dim,
        calc_bond_emb_dim,
        hidden_dim=search_space_dictionary["hidden_dim"],
        num_layers=search_space_dictionary["num_layers"],
        edge_dim=edge_dim,
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=search_space_dictionary["lr"])

    patience_scheduler = (
        max(10, search_space_dictionary["patience"] // 4)
        if search_space_dictionary["patience"] > 0
        else 15
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience_scheduler)

    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = search_space_dictionary["EPOCHS"]
    avg_train_loss_saved = None

    for epoch in range(1, search_space_dictionary["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        avg_val_loss = None
        current_lr = optimizer.param_groups[0]["lr"]

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs

            avg_val_loss = val_loss / len(val_loader.dataset)

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch
                avg_train_loss_saved = avg_train_loss
                torch.save(model.state_dict(), best_model_tmp_path)
                # Move the best model checkpoint to a temporary directory to ensure it is saved even if the trial is stopped
                with tempfile.TemporaryDirectory() as tmp_dir:
                    checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
                    shutil.copy(best_model_tmp_path, checkpoint_path)
            else:
                if search_space_dictionary["patience"] > 0:
                    patience_counter += 1
            #
            # Report the training and validation losses to Ray Tune
            tune.report(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "lr": current_lr,
                }
            )

            if 0 < search_space_dictionary["patience"] <= patience_counter:
                logging.info(f"Early stopping on epoch {epoch}")
                break
        else:
            # Report only the training loss to Ray Tune if no validation set is used
            tune.report(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                }
            )

        if avg_val_loss is not None:
            logging.info(
                f"Epoch {epoch:03d} | LR: {current_lr:.6f} | "
                f"Train MSE: {avg_train_loss:.4f} | \
                Validation MSE: {avg_val_loss:.4f}"
            )
        else:
            logging.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f}")

    if os.path.exists(best_model_tmp_path):
        model.load_state_dict(torch.load(best_model_tmp_path, map_location=device))
        os.remove(best_model_tmp_path)
        logging.info(
            f"Best model saved at epoch {best_epoch} | "
            f"Train MSE: {avg_train_loss_saved:.4f} | \
            Validation MSE: {best_val_loss:.4f}"
        )


def write_results_to_file(model_name, best_results, hyperparemeters, output_file):
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Validation Loss (MSE): {best_results['best_val_loss']:.4f}\n")
        f.write("Best Hyperparameters:\n")
        for param, value in hyperparemeters.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")


if __name__ == "__main__":
    debut_tuning = time()
    models = ["GINE", "GraphTransformer"]
    for model in models:
        logging.info(f"Starting hyperparameter tuning for model: {model}")
        ray.shutdown()
        # we can limit the number cpus in order to avoid overloading the cpus by num_cpus passed to ray.init
        ray.init(ignore_reinit_error=True, include_dashboard=False)

        sdf_dir = "/home/andromeda/Documentos/mohamedA/DeepGNN/MPro-URV_Version2/Ligand/Ligand_SDF"
        target_file = "/home/andromeda/Documentos/mohamedA/DeepGNN/MPro-URV_Version2/pIC50.txt"

        model_name = "prueba_gnn_" + model + "_hyperparameter_tuning"

        valid_split = 0.2

        if not os.path.exists(sdf_dir):
            raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Target file not found: {target_file}")

        # Define the hyperparameter search space for Ray Tune
        """ Note that some parameters are not changeable via the gui so we have to change them from the file GNNs/model_trainer.py.
            such as drop_out.
        """
        # Please check the range of dialog_inputs for each hyperparameter (specially the embedding dimensions) and the patience value
        config = {
            "batch_size": tune.choice([4, 8, 16]),
            "atom_emb_dim": tune.choice([0.2, 0.3, 0.4]),
            "hybrid_emb_dim": tune.choice([0.3, 0.4, 0.5]),
            "bond_emb_dim": tune.choice([0.5, 1.0]),
            "hidden_dim": tune.choice([32, 64, 128, 256]),
            "num_layers": tune.choice([1, 2, 3]),
            "fc_hidden_dim": tune.choice([32, 64, 128, 256]),
            "drop_out": tune.choice([0.0, 0.05, 0.1]),
            "lr": tune.loguniform(1e-4, 1e-3),
            "patience": 15,
            "EPOCHS": 50,
        }

        trainable = tune.with_parameters(
            train,
            sdf_directory=sdf_dir,
            target_file=target_file,
            gnn_model_name=model,
            valid_split=valid_split,
            gnn_model_pseudo=model_name,
        )
        cpu_per_trials = 6
        gpu_per_trials = 0
        # number of combinations of hyperparameters to try.
        num_trials = 20

        # Tuner object to run and report the results
        tuner = tune.Tuner(
            tune.with_resources(
                trainable,
                resources={"cpu": cpu_per_trials, "gpu": gpu_per_trials},
            ),
            tune_config=tune.TuneConfig(
                metric="best_val_loss",
                mode="min",
                num_samples=num_trials,
            ),
            param_space=config,
            run_config=RunConfig(
                name="hyperparameter_tuning_" + model,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=False,
                ),
                storage_path="/tmp/ray_results",
            ),
        )

        results = tuner.fit()

        best_result = results.get_best_result(metric="best_val_loss", mode="min")
        write_results_to_file(
            model,
            best_result.metrics,
            best_result.config,
            "hyperparameter_tuning_results_" + model + ".txt",
        )
    end_tuning = time()
    elapsed_time = end_tuning - debut_tuning
    with open("hyperparameter_tuning_results.txt", "a") as f:
        f.write(f"Tuning time: {elapsed_time}\n")
        f.write(f"---\tend\t---{datetime.now()}\n")

    logging.info(f"it took {elapsed_time} to tune hyperparameter for all GNNs models")
