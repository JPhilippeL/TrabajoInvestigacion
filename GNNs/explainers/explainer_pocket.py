import pandas as pd
import os
import sys

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_padre = os.path.abspath(os.path.join(dir_actual, "../.."))
sys.path.insert(0, dir_padre)

from GNNs.explainers.model_Graph_explainer import obtener_graph_explainer
from GNNs.data_processing import prepare_pt_training_data
from ui.utils.constants import RESULTADOS_DIR

def calcular_y_guardar_importancias_beta(
    data_list, 
    checkpoint_path, 
    csv_resumen_path=RESULTADOS_DIR, 
    csv_matriz_path=RESULTADOS_DIR
):
    """
    Calcula la importancia 'beta' de los nodos en un dataset usando un explainer,
    genera estadísticas (media, desviación, conteo) y guarda los resultados en CSV.

    Parámetros:
    - data_list: Lista de objetos Data de PyTorch Geometric (tu dataset).
    - checkpoint_path: Ruta al archivo del modelo (.ckpt) necesario para el explainer.
    - csv_resumen_path: Ruta donde se guardará el CSV con el resumen estadístico.
    - csv_matriz_path: (Opcional) Ruta para guardar la matriz completa de todos los grafos.
    
    Retorna:
    - resumen_nodos (DataFrame): Tabla con las estadísticas ordenadas.
    - df_crudo (DataFrame): Matriz completa original.
    """
    print(f"Iniciando análisis sobre {len(data_list)} grafos...")
    filas_tabla = []

    # 1. Bucle por cada grafo
    for idx, data in enumerate(data_list):
        
        # Llamada a tu explainer
        resultados = obtener_graph_explainer(
            checkpoint_path=checkpoint_path, 
            data_indices=data,
            batch_mode=True
        )
        nombre = data.name
        print(f"Molecula {nombre} explicada")
        
        # 2. Extraer beta y node_ids
        beta = resultados['beta'] 
        original_ids = data.node_ids 
        
        # 3. Preparar la fila para la tabla
        fila_actual = {
            'id_grafo': idx, 
            'mol_name': resultados.get('mol_name', f'Grafo_{idx}')
        }
        
        # Unir pesos con IDs
        for importancia, n_id in zip(beta, original_ids):
            if n_id is not None:
                # Limpiar el node_id (Ej: "ALA_163_B_N" -> "ALA_163_N")
                partes = n_id.split('_')
                if len(partes) >= 4:
                    # Unimos el residuo (0), el número (1) y el átomo (último)
                    n_id_limpio = f"{partes[0]}_{partes[1]}_{partes[-1]}"
                else:
                    # Por precaución, si algún nodo no tiene ese formato exacto, lo dejamos igual
                    n_id_limpio = n_id
                
                fila_actual[n_id_limpio] = importancia.item()
                
        filas_tabla.append(fila_actual)
        print (f"Molecula {nombre} añadida a la lista")

    print("Procesamiento finalizado. Construyendo DataFrames...")

    # 4. Crear la matriz completa de datos
    df_crudo = pd.DataFrame(filas_tabla)
    df_crudo.set_index('id_grafo', inplace=True)

    # Separamos columnas de texto para el cálculo matemático
    if 'mol_name' in df_crudo.columns:
        df_nodos = df_crudo.drop(columns=['mol_name'])
    else:
        df_nodos = df_crudo

    # 5. Calcular estadísticas (Media, Desviación Estándar y Conteo)
    resumen_nodos = pd.DataFrame({
        'media_importancia': df_nodos.mean(),
        'desviacion_estandar': df_nodos.std(),
        'veces_aparecido': df_nodos.count()
    })

    # Ordenar por la media (descendente)
    resumen_nodos = resumen_nodos.sort_values(by='media_importancia', ascending=False)

    # 6. Guardar en CSV
    save_path_resumen = os.path.join(csv_resumen_path, "importancias_beta_resumen.csv")
    resumen_nodos.to_csv(save_path_resumen, index_label='node_id')
    print(f"✅ Resumen estadístico guardado en: {csv_resumen_path}")

    # if csv_matriz_path is not None:
    #     df_crudo.to_csv(csv_matriz_path)
    #     print(f"✅ Matriz completa guardada en: {csv_matriz_path}")

    return resumen_nodos, df_crudo

# ==========================================
# EJEMPLO DE USO:
# ==========================================
# (Asegúrate de tener data_list cargado)

RUTA_MODELO = "Modelos/NuevaPruebaNuevasFeatures/split_0/GraphTransformer_4capas_pIC50.pt"
DATA_PATH = "/home/philippe/Documents/Databases/3_A_node_id/pocket_BD.pt"

train_loader, val_loader, device, targetname = prepare_pt_training_data(
    pt_file_path=DATA_PATH, 
    batch_size=1,     # <-- Un grafo por lote
    valid_split=0.0   # <-- 0% validación, 100% de los datos van a train_loader
)

# 2. Le pasas directamente el dataset completo a tu función
ruta_resumen = f"{RESULTADOS_DIR}/importancias_beta_{targetname}.csv"

df_resumen, df_crudo = calcular_y_guardar_importancias_beta(
    data_list=train_loader.dataset,  # <-- Usamos .dataset para evitar el empaquetado del DataLoader
    checkpoint_path=RUTA_MODELO, 
)
