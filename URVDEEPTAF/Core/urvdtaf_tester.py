from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from .urvdtaf_dataset import MyDataset, collate_gnn
from .urvdtaf_model import (
    MODEL_DICT,
    GNN_MODELS,
    test
)

def test_model(
    model_path: str, data_path: str, batch_size: int, device: str,
    generate_plots: bool, predictions: bool = False, output_base: str = "visuals/saved",
    max_seq_len: int = 1000, max_pkt_len: int = 63, max_smi_len: int = 150
) -> dict:
    """Test a trained model and generate reports."""
    
    model_path, data_path = Path(model_path), Path(data_path)
    device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # 1. Configuración e Identificación
    model_name, use_gnn = get_model_info_from_path(model_path)
    run_dir, test_files_dir, test_plots_dir = setup_test_directories(model_path, generate_plots)
    
    print(f"Testing {model_name} on {device}...")

    # 2. Carga de Modelo y Datos
    model, test_loader, test_dataset = load_test_environment(
        model_name, model_path, data_path, max_seq_len, max_pkt_len, 
        max_smi_len, batch_size, device, use_gnn
    )

    # 3. Ejecución del Test
    start_time = datetime.now()
    loss_function = nn.MSELoss(reduction='sum')
    metrics, preds, labels = test(model, test_loader, loss_function, device, True)
    duration = datetime.now() - start_time

    # 4. Reportes y Exportaciones
    save_test_report(run_dir, model_name, model_path, data_path, batch_size, device, duration, metrics)
    
    if predictions:
        save_predictions_csv(test_files_dir, test_dataset, labels, preds)
        
    if generate_plots:
        generate_test_plots(test_plots_dir, labels, preds, model_name)

    return metrics

def get_model_info_from_path(model_path: Path) -> tuple:
    """Extrae el nombre exacto del modelo a partir del nombre del directorio."""
    run_name = model_path.parent.name
    exact_model_name = next(
        (key for key in sorted(MODEL_DICT.keys(), key=len, reverse=True) if key in run_name), 
        None
    )
    if exact_model_name is None:
        raise ValueError(f"Could not determine model type from run name: {run_name}")
        
    return exact_model_name, GNN_MODELS.get(exact_model_name, False)

def setup_test_directories(model_path: Path, generate_plots: bool) -> tuple:
    """Crea y devuelve las rutas para los archivos y gráficos de prueba."""
    run_dir = model_path.parent
    test_files_dir = run_dir / "test_files"
    test_plots_dir = test_files_dir / "test_plots"
    
    test_files_dir.mkdir(parents=True, exist_ok=True)
    if generate_plots:
        test_plots_dir.mkdir(parents=True, exist_ok=True)
        
    return run_dir, test_files_dir, test_plots_dir

def load_test_environment(model_name: str, model_path: Path, data_path: Path, 
                          max_seq_len, max_pkt_len, max_smi_len, batch_size, device, use_gnn):
    """Carga el modelo pre-entrenado y prepara el DataLoader para testing."""
    # Inicializar y cargar modelo
    model = MODEL_DICT[model_name]()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Preparar datos
    test_dataset = MyDataset(
        data_path, 'test', max_seq_len, max_pkt_len, max_smi_len,
        use_gnn=use_gnn, pkt_window=None, pkt_stride=None
    )
    
    collate_fn = collate_gnn if use_gnn else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=(device.type == 'cuda'),
        shuffle=False, collate_fn=collate_fn
    )
    
    return model, test_loader, test_dataset

def save_test_report(run_dir: Path, model_name: str, model_path: Path, data_path: Path, 
                     batch_size: int, device: torch.device, duration, metrics: dict):
    """Genera un reporte de texto con las métricas finales."""
    report_path = run_dir / 'test_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== TEST REPORT ===\n\n")
        f.write(f"Model: {model_name}\nPath: {model_path}\nDuration: {duration}\n\n")
        f.write("METRICS\n" + "-"*18 + "\n")
        for key, value in metrics.items():
            display_key = 'MSE' if key == 'loss' else key
            f.write(f"{display_key:12}: {value:.6f}\n")
    print(f"✅ Test report saved to: {report_path}")

def save_predictions_csv(test_files_dir: Path, dataset, labels, preds):
    """Genera un archivo CSV comparando las predicciones con los valores reales."""
    y_true = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    y_pred = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    
    # Obtener PDB IDs
    if hasattr(dataset, 'seq_path'):
        pdbids = [Path(path).stem for path in dataset.seq_path]
    else:
        pdbids = [f"sample_{i}" for i in range(len(dataset))]

    df = pd.DataFrame({
        "pdbid": pdbids[:len(y_true)],
        "real": y_true,
        "predicted": y_pred,
        "set": "test"
    })
    
    csv_path = test_files_dir / "test_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Predictions saved to: {csv_path}")

def generate_test_plots(test_plots_dir: Path, labels, preds, model_name: str):
    """Genera gráficos de diagnóstico: Dispersión, Residuales y Distribución de Errores."""
    sns.set_theme(style="whitegrid")
    y_true = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    y_pred = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    errors = y_pred - y_true

    plots = [
        ('pred_vs_actual.png', 'Predicted vs Actual', lambda: [
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k'),
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ]),
        ('residuals_plot.png', 'Residuals', lambda: [
            plt.scatter(y_pred, errors, alpha=0.6, edgecolor='k'),
            plt.axhline(0, color='red', linestyle='--')
        ]),
        ('error_histogram.png', 'Error Distribution', lambda: [
            plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ])
    ]

    for filename, title, plot_fn in plots:
        plt.figure(figsize=(7, 6))
        plot_fn()
        plt.title(f'{model_name}: {title}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(test_plots_dir / filename, dpi=300)
        plt.close()
    
    print(f"✅ Saved test plots to: {test_plots_dir}")