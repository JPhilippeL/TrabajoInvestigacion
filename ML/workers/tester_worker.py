import argparse
import time
import torch
import os
import gc
import csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from ML.model_tester import cargar_modelo, read_targets, load_data_from_sdf, test_model_on_directory
from ui.utils import RESULTADOS_DIR
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--targets_file", required=True)
    args = parser.parse_args()

    try:
        start = time.time()
        models = [f for f in os.listdir(args.models_dir) if f.endswith((".pt", ".pth"))]
        total = len(models)

        # CSV de resumen
        resumen_file_name = f"resumen_metrics_{os.path.basename(args.models_dir)}.csv"
        resumen_path = os.path.join(RESULTADOS_DIR, resumen_file_name)
        os.makedirs(RESULTADOS_DIR, exist_ok=True)

        # Cargar dataset
        target_dict = read_targets(args.targets_file)
        data_list = load_data_from_sdf(args.sdf_dir, target_dict)

        resultados = []

        for i, fname in enumerate(models, start=1):
            model_path = os.path.join(args.models_dir, fname)
            try:
                # Cargar modelo
                model, device, target_name = cargar_modelo(model_path)

                # Predecir
                y_true, y_pred = [], []
                for data in data_list:
                    data = data.to(device)
                    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(data.x, data.edge_index, data.edge_attr, batch)
                        pred = out.squeeze().item()
                    y_pred.append(pred)
                    y_true.append(data.y.item())

                # Métricas
                rmse = sqrt(mean_squared_error(y_true, y_pred))
                pearson_r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (float("nan"), float("nan"))

                # Guardar plots y predicciones
                test_model_on_directory(model_path, args.sdf_dir, args.targets_file)

                resultados.append((fname, f"{rmse:.4f}", f"{pearson_r:.4f}"))
                #print(f"RESULT|{fname}|{rmse:.4f}|{pearson_r:.4f}", flush=True)

            except Exception as e:
                resultados.append((fname, f"ERROR ({str(e)})", "ERROR"))
                print(f"ERROR|{fname}: {str(e)}", flush=True)

        # Ordenar alfabéticamente por nombre de modelo
        resultados.sort(key=lambda x: x[0].lower())

        # Guardar CSV
        with open(resumen_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Modelo", "RMSE", "Pearson"])
            for row in resultados:
                writer.writerow(row)

        elapsed = time.time() - start
        print(f"FINISHED|{resumen_path}|{elapsed:.2f}", flush=True)

    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)

    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
