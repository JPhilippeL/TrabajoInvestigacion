import argparse
import time
import torch
import gc
import sys
import os

# Aseguramos que Python encuentre el módulo GNNs si ejecutas desde la raíz
sys.path.append(os.getcwd())

from GNNs.model_tester import test_all_models_in_directory

def main():
    parser = argparse.ArgumentParser()
    # Recibimos los argumentos necesarios para la función
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--targets_file", required=True)
    args = parser.parse_args()

    try:
        start = time.time()
        
        # Llamamos a la función importada.
        # Es CRÍTICO que test_all_models_in_directory tenga 'return resumen_path' al final.
        csv_path = test_all_models_in_directory(
            models_dir=args.models_dir,
            sdf_dir=args.sdf_dir,
            targets_file=args.targets_file
        )
        
        elapsed = time.time() - start

        # Imprimimos en el formato que espera tu UI
        # Si csv_path es None (porque olvidaste el return), esto fallará visualmente,
        # así que asegúrate del Paso 1.
        print(f"FINISHED|{csv_path}|{elapsed:.2f}", flush=True)

    except Exception as e:
        # Captura cualquier error dentro de la función y lo envía a la UI
        print(f"ERROR|{str(e)}", flush=True)
        
    finally:
        # Limpieza de memoria GPU/RAM
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()