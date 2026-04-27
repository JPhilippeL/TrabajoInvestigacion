import logging
import os
import shutil

from GNNs.data_processing import prepare_sdf_training_data
from GNNs.model_trainer import GINNet, GINENet, GATNet, EGATNet, GraphTransformerNet, create_model, calc_dim, \
    get_unique_name
import torch
import tempfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ray
from ray import tune
from ray.train import Checkpoint
from ray.air import RunConfig, CheckpointConfig

from ui.utils.constants import periodic_elements, hybridization_types, N_BOND_TYPES, OTHER_NODE_FEATURES, \
    OTHER_EDGE_FEATURES, MODELOS_DIR


def train(config, sdf_dir, target_file, model_type, valid_split, model_name):
    best_model_tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"{model_name}_{os.getpid()}_best_model_tmp.pt"
    )

    calc_atom_emb_dim = calc_dim(len(periodic_elements) * config["atom_emb_dim"])
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * config["hibrid_emb_dim"])
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * config["bond_emb_dim"])
    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    train_loader, val_loader, device, targetname = prepare_sdf_training_data(
        sdf_dir,
        target_file,
        batch_size=config["batch_size"],
        valid_split=valid_split
    )

    model = create_model(
        model_type,
        input_dim,
        calc_atom_emb_dim,
        calc_hibrid_emb_dim,
        calc_bond_emb_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        edge_dim=edge_dim,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    patience_scheduler = max(10, config["patience"] // 4) if config["patience"] > 0 else 15
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience_scheduler)

    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = config["EPOCHS"]
    avg_train_loss_saved = None

    for epoch in range(1, config["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        avg_val_loss = None
        current_lr = optimizer.param_groups[0]["lr"]
        checkpoint = None

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

                with tempfile.TemporaryDirectory() as tmp_dir:
                    checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
                    shutil.copy(best_model_tmp_path, checkpoint_path)
                    checkpoint = Checkpoint.from_directory(tmp_dir)
            else:
                if config["patience"] > 0:
                    patience_counter += 1

            tune.report(
                {"epoch": epoch,
                 "train_loss": avg_train_loss,
                 "val_loss": avg_val_loss,
                 "best_val_loss": best_val_loss,
                 "lr": current_lr, }
            )

            if 0 < config["patience"] <= patience_counter:
                logging.info(f"Early stopping on epoch {epoch}")
                break
        else:
            train.report(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                }
            )

        if avg_val_loss is not None:
            logging.info(
                f"Epoch {epoch:03d} | LR: {current_lr:.6f} | "
                f"Train MSE: {avg_train_loss:.4f} | Validation MSE: {avg_val_loss:.4f}"
            )
        else:
            logging.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f}")

    if os.path.exists(best_model_tmp_path):
        model.load_state_dict(torch.load(best_model_tmp_path, map_location=device))
        os.remove(best_model_tmp_path)
        logging.info(
            f"Best model saved at epoch {best_epoch} | "
            f"Train MSE: {avg_train_loss_saved:.4f} | Validation MSE: {best_val_loss:.4f}"
        )


if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_cpus=20, ignore_reinit_error=True, include_dashboard=False)

    sdf_dir = "/home/andromeda/Documentos/mohamedA/DeepGNN/MPro-URV_Version2/Ligand/Ligand_SDF"
    target_file = "/home/andromeda/Documentos/mohamedA/DeepGNN/MPro-URV_Version2/pIC50.txt"
    model_type = "GAT"
    model_name = "GAThp"
    valid_split = 0.2

    config = {
        "batch_size": tune.choice([4, 8, 16]),
        "atom_emb_dim": tune.choice([0.2, 0.3, 0.4]),
        "hibrid_emb_dim": tune.choice([0.3, 0.4, 0.5]),
        "bond_emb_dim": tune.choice([0.5, 1.0]),
        "hidden_dim": tune.choice([32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3]),
        "fc_hidden_dim": tune.choice([32, 64, 128, 256]),
        "drop_out": tune.choice([0.0, 0.05, 0.1]),
        "lr": tune.loguniform(1e-4, 1e-3),
        "patience": 15,
        "EPOCHS": 50,
    }

    final_model_name = get_unique_name(model_name, MODELOS_DIR, extension=".pt")

    trainable = tune.with_parameters(
        train,
        sdf_dir=sdf_dir,
        target_file=target_file,
        model_type=model_type,
        valid_split=valid_split,
        model_name=final_model_name,
    )

    cpu_per_trials = 6
    gpu_per_trials = 1
    num_trials = 60

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
            name="hyperparameter_tuning_GNN",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=False,
            ),
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="best_val_loss", mode="min")
    logging.info(f"best_result: {best_result}")
