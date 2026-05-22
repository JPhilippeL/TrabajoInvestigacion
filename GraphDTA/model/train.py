import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GraphDTA.model.utils import val, initialize_model, seed_everything, load_split_txt, URVGraphDataset


def train(model_name, output_dir, train_split_file, val_split_file, test_split_file, graph_dir, batch_size, lr,
          n_filters, dropout, weight_decay, log_callback=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = initialize_model(model_name=model_name, n_filters=n_filters, drop_out=dropout).to(device)
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_rmse = float("inf")
        best_epoch = -1
        split_save_dir = os.path.join(output_dir, f"split_{split_id:02}")
        os.makedirs(split_save_dir, exist_ok=True)
        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        for epoch in range(50):
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

                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                if log_callback:
                    log_callback.info(">>> Better model stocked")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early Stopped")
                break
        best_model = initialize_model(model_name=model_name, n_filters=n_filters, drop_out=dropout)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.to(device)

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
    if log_callback:
        log_callback.info("Training finished for all splits")
        log_callback.info(
            f"Mean Best Val RMSE: {np.mean(split_best_val_rmses):.4f} std {np.std(split_best_val_rmses):.4f}")
        log_callback.info(f"Mean Test RMSE: {np.mean(split_test_rmses):.4f} std  {np.std(split_test_rmses):.4f}")
        log_callback.info(
            f"Mean Test Pearson: {np.mean(split_test_pearsons):.4f} std {np.std(split_test_pearsons):.4f}")
