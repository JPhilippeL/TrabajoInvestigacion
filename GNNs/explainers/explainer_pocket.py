import pandas as pd
import os
import sys
import torch
from torch_geometric.data import Batch

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_padre = os.path.abspath(os.path.join(dir_actual, "../.."))
sys.path.insert(0, dir_padre)

from GNNs.explainers.model_Graph_explainer import obtener_graph_explainer
from GNNs.data_processing import prepare_pt_training_data
from GNNs.model_tester import cargar_modelo, predecir_molecula
from ui.utils.constants import RESULTADOS_DIR, BOND_CLASS_NAMES

MAX_ERROR = 1

# Función de ayuda para limpiar los IDs
def limpiar_id(n_id, idx_tensor):
    if n_id is None:
        return f"LIGAND_{idx_tensor}" # Si es None, es del ligando
    partes = n_id.split('_')
    if len(partes) >= 4:
        return f"{partes[0]}_{partes[1]}_{partes[-1]}"
    return n_id

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

    # --- NUEVO: 1. Cargar el modelo antes del bucle para hacer el filtrado ---
    # Asumo que tienes tu función cargar_modelo disponible aquí
    model, device, target_name = cargar_modelo(checkpoint_path)
    model.eval() # Asegurarse de que está en modo inferencia
    
    filas_tabla = []
    mol_omitidas = 0 # Contador para las estadísticas finales

    # 2. Bucle por cada grafo
    for idx, data in enumerate(data_list):
        nombre = data.name
        
        # --- NUEVO: 2. Control de Error ---
        
        # 1. Empaquetamos el grafo único en un Batch y LO ENVIAMOS AL DEVICE
        data_batch = Batch.from_data_list([data]).to(device)
        
        # 2. Hacemos la predicción sobre ese Batch
        prediccion = predecir_molecula(model, data_batch, device)
        
        # Extraer los valores numéricos limpios para la matemática
        pred_val = prediccion.item() if isinstance(prediccion, torch.Tensor) else prediccion
        
        # Asegurarnos de que real_val también sea un float estándar
        if isinstance(data.y, torch.Tensor):
            real_val = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        else:
            real_val = float(data.y)
        
        # Calcular el error absoluto
        error = abs(pred_val - real_val)
        
        # Evaluar la condición
        if error >= MAX_ERROR:
            print(f"⏩ Molécula {nombre} omitida (Error: {error:.3f} >= {MAX_ERROR})")
            mol_omitidas += 1
            continue # Salta el resto del código y va a la siguiente molécula
            
        print(f"✅ Molécula {nombre} aceptada (Error: {error:.3f} < {MAX_ERROR}). Explicando...")
        # ----------------------------------
        
        # Llamada a tu explainer
        resultados = obtener_graph_explainer(
            checkpoint_path=checkpoint_path, 
            data_indices=data,
            batch_mode=True
        )
        
        # Extraer beta y node_ids
        beta = resultados['beta'] 
        original_ids = data.node_ids 
        
        # Preparar la fila para la tabla
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
                    n_id_limpio = f"{partes[0]}_{partes[1]}_{partes[-1]}"
                else:
                    n_id_limpio = n_id
                
                fila_actual[n_id_limpio] = importancia.item()
                
        filas_tabla.append(fila_actual)
        print (f"Molécula {nombre} añadida a la lista")

    print(f"\nProcesamiento finalizado. {len(filas_tabla)} procesadas, {mol_omitidas} omitidas por alto error.")
    print("Construyendo DataFrames...")

    # Crear la matriz completa de datos
    df_crudo = pd.DataFrame(filas_tabla)
    
    # Manejo de seguridad por si NINGUNA molécula superó el filtro
    if df_crudo.empty:
        print("⚠️ Ninguna molécula cumplió con el criterio de MAX_ERROR. No se guardaron archivos.")
        return None, None

    df_crudo.set_index('id_grafo', inplace=True)

    # Separamos columnas de texto para el cálculo matemático
    if 'mol_name' in df_crudo.columns:
        df_nodos = df_crudo.drop(columns=['mol_name'])
    else:
        df_nodos = df_crudo

    # Calcular estadísticas (Media, Desviación Estándar y Conteo)
    resumen_nodos = pd.DataFrame({
        'media_importancia': df_nodos.mean(),
        'desviacion_estandar': df_nodos.std(),
        'veces_aparecido': df_nodos.count()
    })

    # Ordenar por la media (descendente)
    resumen_nodos = resumen_nodos.sort_values(by='media_importancia', ascending=False)

    # Guardar en CSV
    save_path_resumen = os.path.join(csv_resumen_path, "importancias_beta_resumen.csv")
    resumen_nodos.to_csv(save_path_resumen, index_label='node_id')
    print(f"✅ Resumen estadístico guardado en: {save_path_resumen}")

    return resumen_nodos, df_crudo

def explicar_y_guardar_molecula_individual(
    data_list, 
    checkpoint_path, 
    target_mol_name,
    csv_dir=RESULTADOS_DIR,
):
    """
    Busca una molécula, comprueba su error, calcula la importancia 
    'beta' (nodos) y 'delta' (enlaces), filtra enlaces no covalentes y guarda los CSVs.
    """
    print(f"Buscando la molécula '{target_mol_name}' en el dataset...")
    
    # 1. Buscar la molécula
    target_data = None
    for data in data_list:
        if data.name == target_mol_name:
            target_data = data
            break
            
    if target_data is None:
        print(f"❌ Error: No se encontró la molécula '{target_mol_name}'.")
        return None, None
        
    print(f"✅ Molécula encontrada. Cargando modelo...")

    # 2. Cargar el modelo
    model, device, _ = cargar_modelo(checkpoint_path)
    model.eval()
    
    # 3. Control de Error
    data_batch = Batch.from_data_list([target_data]).to(device)
    prediccion = predecir_molecula(model, data_batch, device)
    
    pred_val = prediccion.item() if isinstance(prediccion, torch.Tensor) else prediccion
    if isinstance(target_data.y, torch.Tensor):
        real_val = target_data.y.item() if target_data.y.numel() == 1 else target_data.y[0].item()
    else:
        real_val = float(target_data.y)
        
    error = abs(pred_val - real_val)
    
    if error >= MAX_ERROR:
        print(f"⏩ Operación cancelada. El error ({error:.3f}) >= MAX_ERROR ({MAX_ERROR}).")
        return None, None
        
    print(f"✅ Error aceptable ({error:.3f} < {MAX_ERROR}). Explicando...")

    # 4. Obtener explicación
    resultados = obtener_graph_explainer(
        checkpoint_path=checkpoint_path, 
        data_indices=target_data,
        batch_mode=True
    )
    
    # ==========================================================
    # PROCESAMIENTO DE NODOS (BETA)
    # ==========================================================
    beta = resultados['beta'] 
    # Protección por si el data no tiene node_ids definido
    original_ids = getattr(target_data, 'node_ids', [None] * len(beta))
    nodos_data = []

    # CAMBIO: Sin condicional 'if n_id is not None'. Agregamos todos.
    for i, (importancia, n_id) in enumerate(zip(beta, original_ids)):
        n_id_limpio = limpiar_id(n_id, i)
        nodos_data.append({
            'node_id': n_id_limpio,
            'importancia': float(importancia.item())
        })

    df_nodos = pd.DataFrame(nodos_data)
    df_nodos = df_nodos.sort_values(by='importancia', ascending=False)
    
    save_path_nodos = os.path.join(csv_dir, f"importancias_nodos_{target_mol_name}.csv")
    df_nodos.to_csv(save_path_nodos, index=False)
    print(f"✅ CSV Nodos guardado (Total: {len(df_nodos)}): {save_path_nodos}")

    # ==========================================================
    # PROCESAMIENTO DE ENLACES (DELTA)
    # ==========================================================
    delta = resultados.get('delta')
    df_edges = None
    
    if delta is not None and target_data.edge_index is not None:
        edge_index = target_data.edge_index
        edge_attr = target_data.edge_attr
        
        # CAMBIO: Usaremos un diccionario para agrupar la ida y la vuelta
        diccionario_enlaces = {}
        
        for i in range(edge_index.shape[1]):
            nodo_u = edge_index[0, i].item()
            nodo_v = edge_index[1, i].item()
            
            id_u = original_ids[nodo_u]
            id_v = original_ids[nodo_v]
            bond_idx = int(edge_attr[i, 0].item())
            
            u_limpio = limpiar_id(id_u, nodo_u)
            v_limpio = limpiar_id(id_v, nodo_v)
            
            # Ordenar alfabéticamente para crear una clave única
            par_nodos = tuple(sorted([u_limpio, v_limpio]))
            
            # Si es la primera vez que vemos este enlace, lo creamos
            if par_nodos not in diccionario_enlaces:
                if bond_idx < len(BOND_CLASS_NAMES):
                    nombre_enlace = BOND_CLASS_NAMES[bond_idx]
                else:
                    nombre_enlace = "UNKNOWN"
                    
                diccionario_enlaces[par_nodos] = {
                    'interaccion': f"{par_nodos[0]} <-> {par_nodos[1]}",
                    'nodo_1': par_nodos[0],
                    'nodo_2': par_nodos[1],
                    'tipo_enlace': nombre_enlace,
                    'importancias_temporales': [] # Aquí guardaremos los valores
                }
            
            # Guardamos la importancia de esta dirección (ida o vuelta)
            diccionario_enlaces[par_nodos]['importancias_temporales'].append(float(delta[i].item()))
            
        # Ahora construimos la lista final resolviendo los duplicados
        edges_data = []
        for par, datos in diccionario_enlaces.items():
            # Nos quedamos con el valor máximo entre la ida y la vuelta para no perder el 1.0
            importancia_final = max(datos['importancias_temporales'])
            
            edges_data.append({
                'interaccion': datos['interaccion'],
                'nodo_1': datos['nodo_1'],
                'nodo_2': datos['nodo_2'],
                'tipo_enlace': datos['tipo_enlace'],
                'importancia': importancia_final
            })
                        
        if edges_data:
            df_edges = pd.DataFrame(edges_data)
            df_edges = df_edges.sort_values(by='importancia', ascending=False)
            
            save_path_edges = os.path.join(csv_dir, f"importancias_enlaces_{target_mol_name}.csv")
            df_edges.to_csv(save_path_edges, index=False)
            print(f"✅ CSV Enlaces guardado (Total: {len(df_edges)}): {save_path_edges}")
        else:
            print("⚠️ No se encontraron enlaces para guardar.")

    return df_nodos, df_edges

# ==========================================
# EJEMPLO DE USO:
# ==========================================
# (Asegúrate de tener data_list cargado)

RUTA_MODELO = "Modelos/Explainer_BindingAffinity/GT_4_Split3.pt"
DATA_PATH = "/home/philippe/Documents/Databases/3_A_node_id/pocket_BD.pt"
MOL_OBJETIVO = "7EN8"
OBJETIVOS = ["7GBZ", "7GHB", "5RGX", "7GCK","7GH7", "7UR9", "9GIL", "8DZ0", "9KSK", "9VS1"]

train_loader, val_loader, device, targetname = prepare_pt_training_data(
    pt_file_path=DATA_PATH, 
    batch_size=1,     # <-- Un grafo por lote
    valid_split=0.0   # <-- 0% validación, 100% de los datos van a train_loader
)

# df_resumen, df_crudo = calcular_y_guardar_importancias_beta(
#     data_list=train_loader.dataset,  # <-- Usamos .dataset para evitar el empaquetado del DataLoader
#     checkpoint_path=RUTA_MODELO, 
# )

# Ejecutar la función para una sola molécula
for molecula in OBJETIVOS:
    df_resultado = explicar_y_guardar_molecula_individual(
        data_list=train_loader.dataset,
        checkpoint_path=RUTA_MODELO,
        target_mol_name=molecula
    )
