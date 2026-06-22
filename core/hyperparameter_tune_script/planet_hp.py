from pathlib import Path
from time import time

import numpy as np
import ray
import torch
from ray import tune

from data_pipeline.common import save_all_trials_results_csv, seed_everything
from data_pipeline.planet_dataset import PlanetPocketDataset, make_planet_dataloader
from models.planet.architecture.planet import PLANET


def log_message(log_callback, message):
    if log_callback is not None:
        if hasattr(log_callback, "info"):
            log_callback.info(message)
        else:
            log_callback(message)
    else:
        print(message)


def safe_pearson(pred, label):
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)

    if len(pred) < 2:
        return 0.0

    if np.std(pred) == 0 or np.std(label) == 0:
        return 0.0

    value = float(np.corrcoef(pred, label)[0, 1])

    if np.isnan(value):
        return 0.0

    return value


def move_batch_to_device(batch, device):
    res_batch, mol_batch, targets = batch

    fresidues, res_map, res_scope, alpha_coordinates = res_batch
    fatoms, fbonds, agraph, bgraph, lig_scope = mol_batch

    moved_res_batch = (
        fresidues.to(device),
        res_map.to(device) if isinstance(res_map, torch.Tensor) else res_map,
        res_scope,
        alpha_coordinates.to(device),
    )

    moved_mol_batch = (
        fatoms.to(device),
        fbonds.to(device),
        agraph.to(device),
        bgraph.to(device),
        lig_scope,
    )

    moved_targets = []

    for target in targets:
        if isinstance(target, torch.Tensor):
            moved_targets.append(target.to(device))
        else:
            moved_targets.append(target)

    return moved_res_batch, moved_mol_batch, tuple(moved_targets)


def extract_affinity(predictions, targets):
    predicted_affinities = predictions[2].detach().cpu().view(-1).numpy()
    pks = targets[2].detach().cpu().view(-1).numpy()
    pk_flags = targets[3].detach().cpu().view(-1).numpy()

    mask = pk_flags > 0

    return predicted_affinities[mask], pks[mask]


def evaluate_planet(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []

    skipped_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            try:
                res_batch, mol_batch, targets = move_batch_to_device(batch, device)

                predictions = model(res_batch, mol_batch)

                pred, label = extract_affinity(predictions, targets)

                if len(label) > 0:
                    pred_list.append(pred)
                    label_list.append(label)

            except Exception as exc:
                skipped_batches += 1
                print(f"Skipping eval batch: {type(exc).__name__}: {exc}")

    if not pred_list:
        model.train()
        return float("inf"), 0.0, skipped_batches

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
    pearson = safe_pearson(pred, label)

    model.train()

    return rmse, pearson, skipped_batches


def create_planet_loaders(data_output_path, batch_size, seed, num_workers):
    pkl_dir = Path(data_output_path) / "metadata" / "pkl"

    train_pkl = pkl_dir / "train.pkl"
    valid_pkl = pkl_dir / "valid.pkl"
    core_pkl = pkl_dir / "core.pkl"

    for pkl_path in [train_pkl, valid_pkl, core_pkl]:
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing PLANET pkl file: {pkl_path}")

    train_dataset = PlanetPocketDataset(
        dataset_pkl=train_pkl,
        planet_root=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        decoy_flag=True,
    )

    valid_dataset = PlanetPocketDataset(
        dataset_pkl=valid_pkl,
        planet_root=None,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        decoy_flag=False,
    )

    core_dataset = PlanetPocketDataset(
        dataset_pkl=core_pkl,
        planet_root=None,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        decoy_flag=False,
    )

    train_loader = make_planet_dataloader(
        train_dataset,
        num_workers=num_workers,
    )

    valid_loader = make_planet_dataloader(
        valid_dataset,
        num_workers=num_workers,
    )

    core_loader = make_planet_dataloader(
        core_dataset,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, core_loader


def build_planet_model(config, device):
    model = PLANET(
        feature_dims=config["feature_dims"],
        nheads=config["nheads"],
        key_dims=config["key_dims"],
        value_dims=config["value_dims"],
        pro_update_inters=config["pro_update_inters"],
        lig_update_iters=config["lig_update_iters"],
        pro_lig_update_iters=config["pro_lig_update_iters"],
        device=device,
    )

    return model.to(device)


def compute_total_loss(lig_loss, pro_lig_loss, affinity_loss, global_step, beta_start_step):
    beta = 0.0 if global_step <= beta_start_step else 1.0
    total_loss = lig_loss + pro_lig_loss + beta * affinity_loss

    return total_loss, beta


def train_planet(config, data_output_path, log_callback=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trial_dir = Path(tune.get_context().get_trial_dir())
    trial_dir.mkdir(parents=True, exist_ok=True)

    training_start_time = time()

    seed_everything(config["seed"])

    train_loader, valid_loader, core_loader = create_planet_loaders(
        data_output_path=data_output_path,
        batch_size=config["batch_size"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    model = build_planet_model(config, device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_message(log_callback, f"Trainable params: {trainable_params}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_train_rmse = float("inf")
    best_val_pearson = 0.0
    patience_counter = 0
    global_step = 0

    best_model_path = trial_dir / "best_model.pt"

    for epoch in range(config["epochs"]):
        model.train()

        epoch_affinity_preds = []
        epoch_affinity_labels = []

        skipped_train_batches = 0
        last_beta = 0.0

        for batch in train_loader:
            try:
                res_batch, mol_batch, targets = move_batch_to_device(batch, device)

                predictions = model(res_batch, mol_batch)

                lig_loss, pro_lig_loss, affinity_loss = model.compute_loss(
                    predictions,
                    targets,
                    res_batch,
                    mol_batch,
                )

                total_loss, beta = compute_total_loss(
                    lig_loss=lig_loss,
                    pro_lig_loss=pro_lig_loss,
                    affinity_loss=affinity_loss,
                    global_step=global_step,
                    beta_start_step=config["beta_start_step"],
                )

                last_beta = beta

                optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    config["clip_norm"],
                )

                optimizer.step()

                pred, label = extract_affinity(predictions, targets)

                if len(label) > 0:
                    epoch_affinity_preds.append(pred)
                    epoch_affinity_labels.append(label)

                global_step += 1

            except Exception as exc:
                skipped_train_batches += 1
                log_message(
                    log_callback,
                    f"Skipping train batch: {type(exc).__name__}: {exc}",
                )

        if epoch_affinity_preds:
            train_pred = np.concatenate(epoch_affinity_preds)
            train_label = np.concatenate(epoch_affinity_labels)

            train_rmse = float(np.sqrt(np.mean((train_pred - train_label) ** 2)))
        else:
            train_rmse = float("inf")

        val_rmse, val_pearson, skipped_val_batches = evaluate_planet(
            model=model,
            dataloader=valid_loader,
            device=device,
        )

        log_message(
            log_callback,
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val Pearson: {val_pearson:.4f} | "
            f"beta: {last_beta:.1f} | "
            f"Skipped train: {skipped_train_batches} | "
            f"Skipped val: {skipped_val_batches}",
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_pearson = val_pearson
            best_train_rmse = train_rmse
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": "PLANET",
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                    "best_val_pearson": best_val_pearson,
                    "best_train_rmse": best_train_rmse,
                    "config": {
                        "feature_dims": config["feature_dims"],
                        "nheads": config["nheads"],
                        "key_dims": config["key_dims"],
                        "value_dims": config["value_dims"],
                        "pro_update_inters": config["pro_update_inters"],
                        "lig_update_iters": config["lig_update_iters"],
                        "pro_lig_update_iters": config["pro_lig_update_iters"],
                        "batch_size": config["batch_size"],
                        "lr": config["lr"],
                        "weight_decay": config["weight_decay"],
                        "patience": config["patience"],
                        "epochs": config["epochs"],
                        "clip_norm": config["clip_norm"],
                        "beta_start_step": config["beta_start_step"],
                    },
                },
                best_model_path,
            )

            log_message(log_callback, ">>> Better model stocked")

        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            log_message(log_callback, ">>> Early Stopped")
            break

    if not best_model_path.exists():
        raise RuntimeError(
            "No valid best model was saved. "
            "Check validation loader, pk_flags, skipped batches, or tensorization errors."
        )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    best_model = build_planet_model(checkpoint["config"], device)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    train_rmse, train_pearson, skipped_train_eval = evaluate_planet(
        model=best_model,
        dataloader=train_loader,
        device=device,
    )

    val_rmse, val_pearson, skipped_val_eval = evaluate_planet(
        model=best_model,
        dataloader=valid_loader,
        device=device,
    )

    test_rmse, test_pearson, skipped_test_eval = evaluate_planet(
        model=best_model,
        dataloader=core_loader,
        device=device,
    )

    end_training = time()
    training_time = float(end_training - training_start_time)

    hyperparameter_path = trial_dir / "hyperparameters.txt"

    with hyperparameter_path.open("w", encoding="utf-8") as f:
        f.write("=== Hyperparameters ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Results ===\n")
        f.write(f"train_rmse: {train_rmse}\n")
        f.write(f"train_pearson: {train_pearson}\n")
        f.write(f"val_rmse: {val_rmse}\n")
        f.write(f"val_pearson: {val_pearson}\n")
        f.write(f"test_rmse: {test_rmse}\n")
        f.write(f"test_pearson: {test_pearson}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_rmse: {best_val_rmse}\n")
        f.write(f"training_time: {training_time}\n")
        f.write(f"best_model_path: {best_model_path}\n")
        f.write(f"skipped_train_eval: {skipped_train_eval}\n")
        f.write(f"skipped_val_eval: {skipped_val_eval}\n")
        f.write(f"skipped_test_eval: {skipped_test_eval}\n")

    log_message(log_callback, "PLANET trial completed")
    log_message(log_callback, f"Train RMSE: {train_rmse:.4f}")
    log_message(log_callback, f"Val RMSE: {val_rmse:.4f}")
    log_message(log_callback, f"Test/Core RMSE: {test_rmse:.4f}")
    log_message(log_callback, f"Test/Core Pearson: {test_pearson:.4f}")
    log_message(log_callback, f"Best epoch: {best_epoch}")
    log_message(log_callback, f"Trial time: {training_time:.2f} seconds")

    tune.report(
        {
            "train_rmse": float(train_rmse),
            "train_pearson": float(train_pearson),
            "val_rmse": float(val_rmse),
            "val_pearson": float(val_pearson),
            "test_rmse": float(test_rmse),
            "test_pearson": float(test_pearson),
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val_rmse),
            "training_time": float(training_time),
            "skipped_train_eval": int(skipped_train_eval),
            "skipped_val_eval": int(skipped_val_eval),
            "skipped_test_eval": int(skipped_test_eval),
        }
    )


def run_hyperparameter_search(
    save_directory,
    data_output_path,
    search_space,
    cpu_used_per_trials,
    gpu_used_per_trials,
    number_of_trials,
    log_callback=None,
):
    ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=20,
    )

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    trainable = tune.with_parameters(
        train_planet,
        data_output_path=data_output_path,
        log_callback=log_callback,
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
            metric="val_rmse",
            mode="min",
            num_samples=number_of_trials,
        ),
        param_space=search_space,
        run_config=ray.tune.RunConfig(
            name="PLANET_hyperparameter_tuning",
            storage_path=str(save_directory),
        ),
    )

    results = tuner.fit()

    csv_path = save_all_trials_results_csv(
        results=results,
        save_directory=save_directory,
    )

    best_result = results.get_best_result(
        metric="val_rmse",
        mode="min",
    )

    best_trial_dir = best_result.path

    if log_callback:
        log_message(log_callback, f"Best trial kept at: {best_trial_dir}")
        log_message(log_callback, f"Best config: {best_result.config}")
        log_message(log_callback, f"Best val RMSE: {best_result.metrics['val_rmse']}")
        log_message(log_callback, f"Best val Pearson: {best_result.metrics['val_pearson']}")
        log_message(log_callback, f"Best test RMSE: {best_result.metrics['test_rmse']}")
        log_message(log_callback, f"Best test Pearson: {best_result.metrics['test_pearson']}")
        log_message(log_callback, f"Best train RMSE: {best_result.metrics['train_rmse']}")
        log_message(log_callback, f"Best train Pearson: {best_result.metrics['train_pearson']}")
        log_message(log_callback, f"Best epoch: {best_result.metrics['best_epoch']}")
        log_message(log_callback, f"All trials CSV saved at: {csv_path}")

    return best_result
