from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from URVDEEPTAF.Core.urvdtaf_model import (
    GNN_MODELS,
    MODEL_DICT,
    test,
)
from URVDEEPTAF.Core.urvdtaf_trainer import (
    get_data_loaders,
    process_batch_inputs,
    setup_environment,
)
from URVDEEPTAF.hyperparameter_search.utils import (
    save_all_trials_results_csv,
    save_split_results_csv,
)


def discover_split_paths(mother_dir):

    mother_dir = Path(mother_dir)

    if not mother_dir.exists():
        raise FileNotFoundError(f"Mother directory not found: {mother_dir}")

    split_paths = []

    for split_dir in sorted(mother_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        training_dir = split_dir / "training"
        validation_dir = split_dir / "validation"

        if training_dir.exists() and validation_dir.exists():
            split_paths.append(split_dir)
        else:
            print(f"Ignored {split_dir.name}: missing training/validation folders")

    if not split_paths:
        raise RuntimeError(f"No valid splits found in {mother_dir}")

    return split_paths


def train_one_split_for_tuning(
    config,
    split_path,
    split_id,
    model_name,
    save_best_epoch,
    device,
    seed,
    num_workers,
    output_base,
    log_callback=None,
):

    device, seed, dirs = setup_environment(
        model_name=model_name,
        device=device,
        seed=seed + split_id,
        output_base=output_base,
    )

    use_gnn = GNN_MODELS.get(model_name, False)
    writer = SummaryWriter(dirs["run"])

    model = MODEL_DICT[model_name]().to(device)

    loaders = get_data_loaders(
        str(split_path),
        config["max_seq_len"],
        config["max_pkt_len"],
        config["max_smi_len"],
        config["batch_size"],
        num_workers,
        device,
        use_gnn,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(loaders["training"]),
    )

    loss_function = nn.MSELoss(reduction="sum")

    best_val_rmse = float("inf")
    best_train_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    patience = config.get("patience", 10)

    start_time = datetime.now()

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        tbar = tqdm(
            loaders["training"],
            desc=f"{model_name} | Split {split_id:02d} | Epoch {epoch}",
        )

        for *x, y in tbar:
            processed_x = process_batch_inputs(x, device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = model(*processed_x)

            loss = loss_function(
                y_hat.view(-1),
                y.view(-1),
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            tbar.set_description(
                f"{model_name} | Split {split_id:02d} | "
                f"Epoch {epoch} | Loss: {loss.item() / y.size(0):.4f}"
            )

        # 4. EVALUATION
        model.eval()

        train_metrics, _, _ = test(
            model,
            loaders["training"],
            loss_function,
            device,
            False,
        )

        val_metrics, _, _ = test(
            model,
            loaders["validation"],
            loss_function,
            device,
            False,
        )

        train_rmse = float(train_metrics["RMSE"])
        val_rmse = float(val_metrics["RMSE"])

        # 5. LOGGING
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f"train/{metric_name}", metric_value, epoch)

        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

        if log_callback:
            log_callback(
                f"{model_name} | Split {split_id:02d} | "
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f}"
            )

        # 6. CHECKPOINTING
        if epoch >= save_best_epoch and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_train_rmse = train_rmse
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": model_name,
                    "split_id": split_id,
                    "split_path": str(split_path),
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                    "best_train_rmse": best_train_rmse,
                    "model_state_dict": model.state_dict(),
                    "config": dict(config),
                },
                dirs["run"] / "best_model.pt",
            )

            if log_callback:
                log_callback(
                    f">>> Best model saved | {model_name} | Split {split_id:02d} | Epoch {epoch}"
                )

        else:
            patience_counter += 1

        if patience_counter >= patience:
            if log_callback:
                log_callback(f">>> Early stopping | {model_name} | Split {split_id:02d}")
            break

    writer.close()
    end_time = datetime.now()

    best_model_path = dirs["run"] / "best_model.pt"

    if not best_model_path.exists():
        raise RuntimeError(
            f"No best_model.pt saved for split {split_id}. "
            f"Check save_best_epoch={save_best_epoch} and epochs={config['epochs']}."
        )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_rmse = float("nan")
    test_pearson = float("nan")

    if "test" in loaders:
        test_metrics, _, _ = test(
            model,
            loaders["test"],
            loss_function,
            device,
            False,
        )

        if "RMSE" in test_metrics:
            test_rmse = float(test_metrics["RMSE"])

        if "CORR" in test_metrics:
            test_pearson = float(test_metrics["CORR"])
        elif "Pearson" in test_metrics:
            test_pearson = float(test_metrics["Pearson"])
        elif "pearson" in test_metrics:
            test_pearson = float(test_metrics["pearson"])

    result = {
        "split_id": split_id,
        "split_path": str(split_path),
        "run_dir": str(dirs["run"]),
        "best_epoch": best_epoch,
        "best_train_rmse": best_train_rmse,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_rmse,
        "test_pearson": test_pearson,
        "best_model_path": str(best_model_path),
    }

    if log_callback:
        log_callback(
            f"{model_name} | Split {split_id:02d} finished | "
            f"Best Val RMSE: {best_val_rmse:.4f} | "
            f"Duration: {end_time - start_time}"
        )

    return result


def train_model_on_all_splits(
    config,
    mother_dir,
    model_name,
    save_best_epoch,
    device,
    seed,
    num_workers,
    log_callback=None,
):

    trial_dir = tune.get_context().get_trial_dir()

    split_paths = discover_split_paths(mother_dir)

    split_results = []

    for split_id, split_path in enumerate(split_paths):
        split_output_base = Path(trial_dir) / f"split_{split_id:02d}"
        split_output_base.mkdir(parents=True, exist_ok=True)

        result = train_one_split_for_tuning(
            config=config,
            split_path=split_path,
            split_id=split_id,
            model_name=model_name,
            save_best_epoch=save_best_epoch,
            device=device,
            seed=seed,
            num_workers=num_workers,
            output_base=str(split_output_base),
            log_callback=log_callback,
        )

        split_results.append(result)

    save_split_results_csv(
        split_results=split_results,
        trial_dir=trial_dir,
    )

    train_rmses = [
        result["best_train_rmse"]
        for result in split_results
        if not np.isnan(result["best_train_rmse"])
    ]

    val_rmses = [
        result["best_val_rmse"] for result in split_results if not np.isnan(result["best_val_rmse"])
    ]

    test_rmses = [
        result["test_rmse"] for result in split_results if not np.isnan(result["test_rmse"])
    ]

    test_pearsons = [
        result["test_pearson"] for result in split_results if not np.isnan(result["test_pearson"])
    ]

    mean_train_rmse = float(np.mean(train_rmses))
    std_train_rmse = float(np.std(train_rmses))

    mean_val_rmse = float(np.mean(val_rmses))
    std_val_rmse = float(np.std(val_rmses))

    mean_test_rmse = float(np.mean(test_rmses)) if test_rmses else float("nan")
    std_test_rmse = float(np.std(test_rmses)) if test_rmses else float("nan")

    mean_test_pearson = float(np.mean(test_pearsons)) if test_pearsons else float("nan")

    std_test_pearson = float(np.std(test_pearsons)) if test_pearsons else float("nan")

    trial_summary_path = Path(trial_dir) / "trial_summary.txt"

    with open(trial_summary_path, "w", encoding="utf-8") as file:
        file.write("=== TRIAL SUMMARY ===\n")
        file.write(f"Model: {model_name}\n")
        file.write(f"Trial dir: {trial_dir}\n\n")

        file.write("=== HYPERPARAMETERS ===\n")
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

        file.write("\n=== RESULTS OVER SPLITS ===\n")
        file.write(f"mean_train_rmse: {mean_train_rmse}\n")
        file.write(f"std_train_rmse: {std_train_rmse}\n")
        file.write(f"mean_val_rmse: {mean_val_rmse}\n")
        file.write(f"std_val_rmse: {std_val_rmse}\n")
        file.write(f"mean_test_rmse: {mean_test_rmse}\n")
        file.write(f"std_test_rmse: {std_test_rmse}\n")
        file.write(f"mean_test_pearson: {mean_test_pearson}\n")
        file.write(f"std_test_pearson: {std_test_pearson}\n")

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


def default_search_space():
    return {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([4, 8, 16, 32, 64]),
        "epochs": 50,
        "patience": 15,
        "max_seq_len": 1000,
        "max_pkt_len": 63,
        "max_smi_len": 150,
    }


def run_hyperparameter_search(
    model_name,
    mother_dir,
    output_path,
    search_space=None,
    cpu_used_per_trial=4,
    gpu_used_per_trial=1,
    number_of_trials=50,
    seed=42,
    num_workers=0,
    save_best_epoch=1,
    log_callback=None,
):

    if model_name not in MODEL_DICT:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available models: {list(MODEL_DICT.keys())}"
        )

    if search_space is None:
        search_space = default_search_space()

    model_output_dir = Path(output_path) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if log_callback:
        log_callback(f"Starting hyperparameter search for model: {model_name}")
        log_callback(f"Device: {device}")
        log_callback(f"Output directory: {model_output_dir}")

    trainable = tune.with_parameters(
        train_model_on_all_splits,
        mother_dir=mother_dir,
        model_name=model_name,
        save_best_epoch=save_best_epoch,
        device=device,
        seed=seed,
        num_workers=num_workers,
        log_callback=log_callback,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": cpu_used_per_trial,
                "gpu": gpu_used_per_trial,
            },
        ),
        tune_config=tune.TuneConfig(
            metric="mean_val_rmse",
            mode="min",
            num_samples=number_of_trials,
        ),
        param_space=search_space,
        run_config=ray.tune.RunConfig(
            name=f"{model_name}_hyperparameter_tuning",
            storage_path=str(model_output_dir),
        ),
    )

    results = tuner.fit()

    csv_path = save_all_trials_results_csv(
        results=results,
        save_directory=model_output_dir,
    )

    best_result = results.get_best_result(
        metric="mean_val_rmse",
        mode="min",
    )

    if log_callback:
        log_callback(f"Best trial path: {best_result.path}")
        log_callback(f"Best config: {best_result.config}")
        log_callback(f"Best mean val RMSE: {best_result.metrics['mean_val_rmse']}")
        log_callback(f"Best std val RMSE: {best_result.metrics['std_val_rmse']}")
        log_callback(f"Best mean test RMSE: {best_result.metrics['mean_test_rmse']}")
        log_callback(f"Best std test RMSE: {best_result.metrics['std_test_rmse']}")
        log_callback(f"All trials CSV saved at: {csv_path}")

    return best_result


if __name__ == "__main__":
    data = None
    output = None
    search_space = default_search_space()
    cpu_used_per_trial = 4
    gpu_used_per_trial = 1
    number_of_trials = 50
    seed = 42
    num_workers = 0
    save_best_epoch = 1

    for model in MODEL_DICT:
        run_hyperparameter_search(
            model_name=model,
            mother_dir=data,
            output_path=output,
            search_space=search_space,
            cpu_used_per_trial=cpu_used_per_trial,
            gpu_used_per_trial=gpu_used_per_trial,
            seed=seed,
            num_workers=num_workers,
            save_best_epoch=save_best_epoch,
            log_callback=print,
        )
