from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from .dataset import MyDataset, collate_gnn
from .model import (
    MODEL_DICT,
    GNN_MODELS,
    test
)

"""Train a model and save the best checkpoint.
    
    Args:
        data_path: Path to data directory
        model_name: Name of model to train (must be in MODEL_DICT)
        batch_size: Batch size for training
        epochs: Number of epochs to train
        save_best_epoch: Epoch from which to start saving best model
        lr: Learning rate for optimizer
        device: Device to use for training (cuda/cpu)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        generate_plots: Whether to generate and save plots
        output_base: Base directory for outputs
        max_seq_len: Maximum sequence length
        max_pkt_len: Maximum pocket length
        max_smi_len: Maximum SMILES length
    
    Returns:
        str: Path to the directory where model and results are saved
    """
def train(
    data_path: str, model_name: str, batch_size: int, epochs: int,
    save_best_epoch: int, lr: float, device: str, seed: int,
    num_workers: int, generate_plots: bool, output_base: str = "runs",
    max_seq_len: int = 1000, max_pkt_len: int = 63, max_smi_len: int = 150
) -> str:
    """Función principal de entrenamiento re-estructurada."""
    
    # 1. SETUP
    device, seed, dirs = setup_environment(model_name, device, seed, output_base)
    use_gnn = GNN_MODELS.get(model_name, False)
    csv_paths, csv_headers = initialize_csv_loggers(dirs['results'])
    writer = SummaryWriter(dirs['run'])
    
    # 2. MODEL & DATA
    model = MODEL_DICT[model_name]().to(device)
    loaders = get_data_loaders(data_path, max_seq_len, max_pkt_len, max_smi_len, 
                               batch_size, num_workers, device, use_gnn)
                               
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                                              steps_per_epoch=len(loaders['training']))
    loss_function = nn.MSELoss(reduction='sum')

    # Historial para gráficos
    train_history = {m: [] for m in ['loss', 'RMSE', 'MAE', 'c_index', 'CORR', 'R2', 'MSE', 'SD']}
    val_history   = {m: [] for m in ['loss', 'RMSE', 'MAE', 'c_index', 'CORR', 'R2', 'MSE', 'SD']}
    epochs_list, errors_warnings = [], []
    best_val_loss, best_epoch = float('inf'), -1
    start_time = datetime.now()

    # 3. TRAINING LOOP
    for epoch in range(1, epochs + 1):
        epoch_start = datetime.now()
        model.train()
        
        tbar = tqdm(loaders['training'], desc=f"Epoch {epoch}")
        for *x, y in tbar:
            processed_x = process_batch_inputs(x, device)
            y = y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(*processed_x)
            loss = loss_function(y_hat.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            tbar.set_description(f"Epoch {epoch}, Loss: {loss.item()/y.size(0):.4f}")

        # Evaluación (usando tu función test externa)
        model.eval()
        train_metrics, _, _ = test(model, loaders['training'], loss_function, device, False)
        val_metrics, _, _ = test(model, loaders['validation'], loss_function, device, False)

        # 4. LOGGING & CHECKPOINTING
        # (Aquí se podría hacer otra sub-función para guardar en CSV/Tensorboard, 
        # pero mantenemos la lógica simplificada)
        for m_name, m_val in train_metrics.items(): writer.add_scalar(f'train/{m_name}', m_val, epoch)
        for m_name, m_val in val_metrics.items(): writer.add_scalar(f'val/{m_name}', m_val, epoch)

        if generate_plots:
            epochs_list.append(epoch)
            for m in train_history:
                if m in train_metrics:
                    train_history[m].append(train_metrics[m])
                    val_history[m].append(val_metrics[m])

        if epoch >= save_best_epoch and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), dirs['run'] / 'best_model.pt')
            print(f'✅ Best model saved at epoch {epoch}')

    # 5. FINALIZACIÓN Y REPORTES
    end_time = datetime.now()
    writer.close()
    model.load_state_dict(torch.load(dirs['run'] / 'best_model.pt'))

    if generate_plots:
        generate_training_plots(epochs_list, train_history, val_history, dirs['plots'], model_name)
        
    params_dict = {'batch_size': batch_size, 'epochs': epochs, 'lr': lr} # Resumen
    save_final_reports(dirs['run'], dirs['results'], model_name, best_epoch, 
                       end_time - start_time, start_time, end_time, device, seed, 
                       params_dict, model, loaders, loss_function, errors_warnings)

    return str(dirs['run'])

def setup_environment(model_name: str, device: str, seed: int, output_base: str):
    """Configura el dispositivo, semilla aleatoria y crea la estructura de directorios."""
    actual_device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    actual_seed = seed if seed is not None else np.random.randint(10000, 100000)
    
    # Reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    
    # Estructura de directorios
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{model_name}_{timestamp}_{actual_seed}"
    run_dir = Path(output_base) / run_name
    
    dirs = {
        'run': run_dir,
        'training': run_dir / "training_files",
        'plots': run_dir / "training_files" / "training_plots",
        'results': run_dir / "training_files" / "more_results"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    return actual_device, actual_seed, dirs

def initialize_csv_loggers(results_dir: Path):
    """Crea los archivos CSV y escribe las cabeceras."""
    csv_paths = {
        'train': results_dir / "training_results_by_epoch.csv",
        'val': results_dir / "validation_results_by_epoch.csv",
        'all': results_dir / "results_by_epoch.csv"
    }
    
    headers = ["epoch", "model", "phase", "timestamp", "learning_rate", "loss", "batch_size", 
               "MSE", "RMSE", "MAE", "c_index", "SD", "CORR", "R2", "elapsed_time"]
               
    for path in csv_paths.values():
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
            
    return csv_paths, headers

def get_data_loaders(data_path, max_seq_len, max_pkt_len, max_smi_len, 
                     batch_size, num_workers, device, use_gnn):
    """Inicializa y devuelve los DataLoaders de entrenamiento y validación."""
    collate_fn = collate_gnn if use_gnn else None
    
    loaders = {}
    for phase in ['training', 'validation']:
        dataset = MyDataset(
            data_path, phase, max_seq_len, max_pkt_len, max_smi_len,
            use_gnn=use_gnn, pkt_window=None, pkt_stride=None
        )
        loaders[phase] = DataLoader(
            dataset, batch_size=batch_size, pin_memory=(device.type == 'cuda'),
            num_workers=num_workers, shuffle=(phase == 'training'), collate_fn=collate_fn
        )
    return loaders

def process_batch_inputs(x, device):
    """Mueve los datos complejos (tuplas, grafos GNN o tensores) al dispositivo correspondiente."""
    processed_x = []
    for i in range(len(x)):
        if isinstance(x[i], (tuple, list)):
            if len(x[i]) == 4: # GNN graph data
                graph_data = tuple(t.to(device) if torch.is_tensor(t) else t for t in x[i])
                processed_x.append(graph_data)
            else:
                processed_data = [item.to(device) if torch.is_tensor(item) else item for item in x[i]]
                processed_x.append(tuple(processed_data) if isinstance(x[i], tuple) else processed_data)
        elif torch.is_tensor(x[i]):
            processed_x.append(x[i].to(device))
        else:
            processed_x.append(x[i])
    return processed_x

def generate_training_plots(epochs_list, train_history, val_history, plots_dir, model_name):
    """Genera y guarda gráficos de las métricas usando matplotlib/seaborn."""
    sns.set_theme(style="whitegrid")
    all_metrics = ['loss', 'MSE', 'RMSE', 'MAE', 'c_index', 'CORR', 'R2', 'SD']
    
    for metric in all_metrics:
        if metric not in train_history: continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_list, train_history[metric], marker='o', linestyle='-', label='Training')
        ax.plot(epochs_list, val_history[metric], marker='s', linestyle='--', label='Validation')
        ax.set_title(f"{model_name} — {metric} per Epoch", fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_ylim(0, None)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.savefig(plots_dir / f"{metric}_vs_epoch.png", dpi=300)
        plt.close(fig)

def save_final_reports(run_dir, more_results_dir, model_name, best_epoch, 
                       total_duration, start, end, device, seed, params_dict,
                       model, data_loaders, loss_function, errors_warnings):
    """Genera el reporte final en archivos TXT."""
    with open(run_dir / 'results.txt', 'w') as f:
        f.write("=== TRAINING REPORT ===\n")
        f.write(f"Model: {model_name}\nBest Epoch: {best_epoch}\nDuration: {total_duration}\n\n")
        
        for phase in ['training', 'validation']:
            performance, _, _ = test(model, data_loaders[phase], loss_function, device, False)
            f.write(f"{phase.upper()} METRICS\n" + "-"*18 + "\n")
            for key, value in performance.items():
                f.write(f"{key:12}: {value:.6f}\n")
            f.write("\n")