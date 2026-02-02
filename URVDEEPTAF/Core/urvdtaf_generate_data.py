import os
import glob
import re
import shutil
import pandas as pd
import numpy as np
import time
import platform
import datetime
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
import ast  # Importante: para convertir el string del txt a lista de listas

# Esto añade la carpeta 'TrabajoInvestigacion' al path de búsqueda de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Ahora ya puedes importar
from URVDEEPTAF.utils.constants import PROCESSED_DATA_DIR, c1, c2, structure_types, amino_acids
import logging

logger = logging.getLogger(__name__)

# Funciones de utilidad pura (sin estado, pueden ir fuera de la clase)
def clean_pdbid(pid):
    """Limpia agresivamente el ID."""
    pid = str(pid).strip() # Quita espacios delante y detras
    # Quita comillas simples o dobles si se colaron en el string
    pid = pid.replace("'", "").replace('"', "")
    # Quita saltos de linea
    pid = pid.replace("\n", "").replace("\r", "")
    return pid.replace("E+", "E").replace("e+", "E")

def f1(aa, key):
    if aa == 'X': return 1 / len(c1)
    return 1 if aa in c1[key] else 0

def f2(aa, key):
    if aa == 'X': return 1 / len(c2)
    return 1 if aa in c2[key] else 0

# ==============================================================================
# CLASE PRINCIPAL DEL GENERADOR
# ==============================================================================
class URVDataGenerator:
    def __init__(self):
        # El estado ahora vive DENTRO de la instancia, no en el espacio global
        self.processing_details = {}
        self.start_time = None

    def generate(self, dssp_dir, ligand_dir, pocket_file, output_dir=None, 
                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                 random_seed=42, cleanup_processed=True,
                 custom_split_ids=None): # <--- Nuevo parámetro: una tupla (train, val, test)
        
        self.start_time = time.time()
        
        # Si NO hay splits externos, validamos que los ratios sumen 1
        if custom_split_ids is None:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError(f"Ratios must sum to 1.0, got: {train_ratio + val_ratio + test_ratio}")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            error_msg = f"Ratios must sum to 1.0, got: {train_ratio + val_ratio + test_ratio}"
            logger.exception(error_msg)
            raise ValueError(error_msg)
        
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), PROCESSED_DATA_DIR)
        
        # Create temporary processing directory
        proc_dir = os.path.join(output_dir, "processed")
        proc_global_dir = os.path.join(proc_dir, "global")
        proc_pocket_dir = os.path.join(proc_dir, "pocket")
        ligand_output_csv = os.path.join(proc_dir, "ligands.csv")
        
        os.makedirs(proc_global_dir, exist_ok=True)
        os.makedirs(proc_pocket_dir, exist_ok=True)
        os.makedirs(os.path.dirname(ligand_output_csv), exist_ok=True)
        
        print(f"Processing files for dataset...")
        print(f"Output directory: {output_dir}")
        
        # Inicialización del atributo de instancia (Adiós Global)
        self.processing_details = {
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "dssp_dir": os.path.abspath(dssp_dir),
                "ligand_dir": os.path.abspath(ligand_dir),
                "pocket_file": os.path.abspath(pocket_file),
                "output_dir": os.path.abspath(output_dir),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "random_seed": random_seed,
                "cleanup_processed": cleanup_processed
            },
            "system_info": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "node": platform.node()
            },
            "file_counts": {
                "input": {"dssp_files": 0, "ligand_files": 0, "pocket_entries": 0},
                "output": {}
            }
        }

        # -------------------------
        # Flujo de Ejecución (Pipeline)
        # -------------------------
        print("\nStep 1/5: Processing protein files...")
        self._process_protein_files(dssp_dir, proc_global_dir)
        
        print("\nStep 2/5: Processing pocket data...")
        self._process_pocket_file(pocket_file, dssp_dir, proc_pocket_dir)
        
        print("\nStep 3/5: Processing ligand files...")
        self._process_ligand_files(ligand_dir, ligand_output_csv)
        
        # -------------------------
        # Paso 4 modificado: Splitting
        # -------------------------
        print("\nStep 4/5: Splitting data...")
        
        if custom_split_ids is not None:
            # Usamos las listas pasadas externamente
            train_ids, val_ids, test_ids = custom_split_ids
            # Limpiamos IDs por seguridad
            train_ids = [clean_pdbid(i) for i in train_ids]
            val_ids   = [clean_pdbid(i) for i in val_ids]
            test_ids  = [clean_pdbid(i) for i in test_ids]
            
            self.processing_details["split_info"] = {
                "method": "external_custom_lists",
                "total_pdbids": len(train_ids) + len(val_ids) + len(test_ids),
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "test_count": len(test_ids)
            }
        else:
            # Comportamiento original por defecto
            train_ids, val_ids, test_ids = self._split_pdbids(
                ligand_output_csv, train_ratio, val_ratio, test_ratio, random_seed
            )
        
        # Estructura de salida
        dirs = {
            'train_global': os.path.join(output_dir, "training", "global"),
            'train_pocket': os.path.join(output_dir, "training", "pocket"),
            'val_global': os.path.join(output_dir, "validation", "global"),
            'val_pocket': os.path.join(output_dir, "validation", "pocket"),
            'test_global': os.path.join(output_dir, "test", "global"),
            'test_pocket': os.path.join(output_dir, "test", "pocket")
        }
        for d in dirs.values(): os.makedirs(d, exist_ok=True)
        
        # Copia y Combinación
        print("Copying feature files...")
        tg_c, _ = self._copy_files(proc_global_dir, train_ids, dirs['train_global'])
        vg_c, _ = self._copy_files(proc_global_dir, val_ids, dirs['val_global'])
        teg_c, _ = self._copy_files(proc_global_dir, test_ids, dirs['test_global'])
        
        tp_c, _ = self._copy_files(proc_pocket_dir, train_ids, dirs['train_pocket'])
        vp_c, _ = self._copy_files(proc_pocket_dir, val_ids, dirs['val_pocket'])
        tep_c, _ = self._copy_files(proc_pocket_dir, test_ids, dirs['test_pocket'])
        
        tl, vl, tel = self._split_ligand_csv(ligand_output_csv, train_ids, val_ids, test_ids, output_dir)
        
        print("Creating combined sequence files...")
        t_seq_c, _ = self._combine_seq(dssp_dir, train_ids, os.path.join(output_dir, "training_seq_.csv"))
        v_seq_c, _ = self._combine_seq(dssp_dir, val_ids, os.path.join(output_dir, "validation_seq_.csv"))
        te_seq_c, _ = self._combine_seq(dssp_dir, test_ids, os.path.join(output_dir, "test_seq_.csv"), include_index=False)
        
        t_pkt_c, _ = self._combine_pocket_seq(pocket_file, train_ids, os.path.join(output_dir, "training_pocket_.csv"))
        v_pkt_c, _ = self._combine_pocket_seq(pocket_file, val_ids, os.path.join(output_dir, "validation_pocket_.csv"))
        te_pkt_c, _ = self._combine_pocket_seq(pocket_file, test_ids, os.path.join(output_dir, "test_pocket_.csv"), include_index=False)
        
        # Actualización de contadores finales
        self.processing_details["file_counts"]["output"] = {
            "global_features": tg_c + vg_c + teg_c,
            "pocket_features": tp_c + vp_c + tep_c,
            "ligand_entries": tl + vl + tel,
            "global_sequences": t_seq_c + v_seq_c + te_seq_c,
            "pocket_sequences": t_pkt_c + v_pkt_c + te_pkt_c
        }
        
        # Finalización
        print("\nStep 5/5: Finalizing...")
        details_path = os.path.join(output_dir, "details.txt")
        self._write_details_file(details_path)
        
        if cleanup_processed and os.path.exists(proc_dir):
            shutil.rmtree(proc_dir)
            print("Cleaned up temporary processed files.")
            
        return {
            "total_pdbids": self.processing_details['split_info']['total_pdbids'],
            "train_count": self.processing_details['split_info']['train_count'],
            "val_count": self.processing_details['split_info']['val_count'],
            "test_count": self.processing_details['split_info']['test_count'],
            "output_dir": output_dir,
            "details_file": details_path,
        }

    # ==============================================================================
    # MÉTODOS INTERNOS (Uso de 'self' para modificar el estado)
    # ==============================================================================
    
    def _process_protein_files(self, dssp_dir, output_global_dir):
        global_seq_dir = os.path.join(os.path.dirname(output_global_dir), "global_seq")
        os.makedirs(global_seq_dir, exist_ok=True)
        
        dssp_files = glob.glob(os.path.join(dssp_dir, "*_protein.dssp"))
        self.processing_details["file_counts"]["input"]["dssp_files"] = len(dssp_files) # <--- Uso de self
        
        success, failed = 0, 0
        
        for dssp_file in tqdm(dssp_files, desc="Processing DSSP files"):
            try:
                protein_id = clean_pdbid(os.path.basename(dssp_file).split('_')[0])
                df = self._parse_dssp_file(dssp_file)
                if df.empty:
                    failed += 1
                    continue
                
                seq = "".join(df["aa"].tolist())
                pd.DataFrame({"id": [protein_id], "seq": [seq]}).to_csv(os.path.join(global_seq_dir, protein_id + "_seq.csv"), index=True)
                
                df = self._compute_features(df)
                idx_func = self._idx_df_l_init(df)
                df['idx'] = df.apply(idx_func, axis=1)
                df_out = df.drop(['residue', 'chain', 'aa', 'structure'], axis=1)
                df_out.index.name = None
                df_out.to_csv(os.path.join(output_global_dir, protein_id + ".csv"), index=True, float_format='%.6f')
                success += 1
            except Exception as e:
                logger.exception(f"Error processing {dssp_file}: {e}")
                failed += 1
                
        self.processing_details["protein_processing"] = {"success_count": success, "failed_count": failed}

    def _process_pocket_file(self, pocket_csv, dssp_dir, output_pocket_dir):
        pocket_seq_dir = os.path.join(os.path.dirname(output_pocket_dir), "pocket_seq")
        os.makedirs(pocket_seq_dir, exist_ok=True)
        
        try:
            df_pocket = pd.read_csv(pocket_csv, sep=None, engine='python', dtype={'PDB': str})
            df_pocket.columns = df_pocket.columns.str.strip()
            df_pocket.rename(columns=lambda x: x.upper() if x.upper() == 'PDB' else x, inplace=True)
            
            self.processing_details["file_counts"]["input"]["pocket_entries"] = len(df_pocket)
            
            df_pocket[["PDB", "seq"]].rename(columns={"PDB": "id"}).to_csv(os.path.join(pocket_seq_dir, "all_pocket_seq.csv"), index=True)
            
            success, skipped = 0, 0
            for _, row in df_pocket.iterrows():
                try:
                    protein_id = clean_pdbid(str(row['PDB']))
                    seq = str(row['seq']).strip()
                    positions = str(row.get('Positions', '')).strip().split() if not pd.isna(row.get('Positions')) else []
                    
                    if not positions or len(seq) != len(positions):
                        skipped += 1
                        continue
                        
                    dssp_path = os.path.join(dssp_dir, f"{protein_id}_protein.dssp")
                    df_protein = self._parse_dssp_file(dssp_path)
                    
                    df_protein['residue'] = df_protein['residue'].astype(str)
                    structure_map = df_protein.set_index(['chain', 'residue'])['structure'].to_dict()
                    
                    records = [{
                        'residue': pos, 'chain': 'A', 'aa': aa,
                        'structure': structure_map.get(('A', str(pos)), 'C')
                    } for pos, aa in zip(positions, seq)]
                    
                    df_pocket_residues = self._compute_features(pd.DataFrame(records))
                    df_pocket_residues['idx'] = df_pocket_residues['chain'] + df_pocket_residues['residue'].astype(str)
                    
                    cols_to_drop = [c for c in ['residue', 'chain', 'aa', 'structure'] if c in df_pocket_residues.columns]
                    df_out = df_pocket_residues.drop(cols_to_drop, axis=1)
                    
                    if 'idx' in df_out.columns:
                        df_out = df_out[['idx'] + [c for c in df_out.columns if c != 'idx']]
                        
                    df_out.to_csv(os.path.join(output_pocket_dir, f"{protein_id}.csv"), index=True)
                    success += 1
                except Exception as e:
                    logger.exception(f"Error on pocket {row.get('PDB')}: {e}")
                    skipped += 1
                    
            self.processing_details["pocket_processing"] = {"success_count": success, "skipped_count": skipped}
        except Exception as e:
            logger.critical(f"Critical error on pocket file: {e}")

    def _process_ligand_files(self, ligand_dir, output_csv):
        ligand_files = glob.glob(os.path.join(ligand_dir, "*.smi"))
        self.processing_details["file_counts"]["input"]["ligand_files"] = len(ligand_files)
        
        records, success, failed = [], 0, 0
        
        for ligand_file in tqdm(ligand_files, desc="Processing ligand files"):
            try:
                pdbid = clean_pdbid(os.path.basename(ligand_file).split('_')[0])
                with open(ligand_file, 'r') as f:
                    parts = re.split(r'\s+', f.readline().strip())
                    if parts:
                        records.append({'pdbid': pdbid, 'smiles': parts[0]})
                        success += 1
                    else:
                        failed += 1
            except Exception:
                failed += 1
                
        df_ligand = pd.DataFrame(records) if records else pd.DataFrame(columns=['pdbid', 'smiles'])
        df_ligand['pdbid'] = df_ligand['pdbid'].astype(str)
        df_ligand.to_csv(output_csv, index=False, float_format='%.6f')
        self.processing_details["ligand_processing"] = {"success_count": success, "failed_count": failed, "total_ligands": len(records)}

    def _split_pdbids(self, ligand_csv, train_ratio, val_ratio, test_ratio, random_seed):
        df = pd.read_csv(ligand_csv, dtype={'pdbid': str})
        pdbids = [clean_pdbid(x) for x in df['pdbid'].unique().tolist()]
        train_ids, temp_ids = train_test_split(pdbids, train_size=train_ratio, random_state=random_seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=test_ratio/(val_ratio+test_ratio), random_state=random_seed)
        
        self.processing_details["split_info"] = {
            "total_pdbids": len(pdbids),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids)
        }
        return train_ids, val_ids, test_ids

    def _copy_files(self, source_dir, split_ids, target_dir):
        copied, missing = 0, 0
        for pdbid in split_ids:
            src = os.path.join(source_dir, pdbid + ".csv")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_dir, pdbid + ".csv"))
                copied += 1
            else:
                missing += 1
        return copied, missing

    def _split_ligand_csv(self, ligand_csv, train_ids, val_ids, test_ids, output_dir):
        df = pd.read_csv(ligand_csv, dtype={'pdbid': str})
        for ids, name in zip([train_ids, val_ids, test_ids], ["training", "validation", "test"]):
            df[df['pdbid'].isin(ids)].to_csv(os.path.join(output_dir, f"{name}_smi.csv"), index=False, float_format='%.6f')
        return len(df[df['pdbid'].isin(train_ids)]), len(df[df['pdbid'].isin(val_ids)]), len(df[df['pdbid'].isin(test_ids)])

    def _combine_seq(self, dssp_dir, split_ids, output_filename, include_index=True):
        records = []
        for pdbid in split_ids:
            files = glob.glob(os.path.join(dssp_dir, f"{pdbid}_protein.dssp"))
            if files:
                df = self._parse_dssp_file(files[0])
                records.append({"id": pdbid, "seq": "".join(df["aa"].tolist())})
        if records:
            pd.DataFrame(records).to_csv(output_filename, index=include_index, float_format='%.6f')
            return len(records), len(split_ids) - len(records)
        return 0, len(split_ids)

    def _combine_pocket_seq(self, pocket_csv, split_ids, output_filename, include_index=True):
        df = pd.read_csv(pocket_csv, dtype={'PDB': str})
        df["PDB"] = df["PDB"].apply(clean_pdbid)
        df_subset = df[df["PDB"].isin(split_ids)].rename(columns={"PDB": "id"})[["id", "seq"]]
        df_subset.to_csv(output_filename, index=include_index, float_format='%.6f')
        return len(df_subset), len(split_ids) - len(df_subset)

    # --- PURE DATA PROCESSING HELPERS ---
    def _parse_dssp_file(self, dssp_path):
        try:
            with open(dssp_path, 'r') as f:
                lines = f.readlines()
            records = []
            for line in lines[28:]:
                if len(line) < 17 or line[13] == '!': continue
                records.append({
                    'residue': line[5:11].strip(),
                    'chain': line[11:12].strip() or 'A',
                    'aa': (line[12:14].strip() or 'X')[0],
                    'structure': line[16:17].strip() or 'C'
                })
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame(columns=['residue', 'chain', 'aa', 'structure'])

    def _compute_features(self, df):
        df['aa'] = df['aa'].apply(lambda x: x if re.match('[A-Z]', x) else 'C')
        df['structure'] = df['structure'].apply(lambda x: x if re.match('[A-Z]', x) else 'C')
        for k in c1: df[k] = df['aa'].apply(lambda x: f1(x, k))
        for k in c2: df[f'c2_{k}'] = df['aa'].apply(lambda x: f2(x, k))
        for si in structure_types: df[f's2_{si}'] = df['structure'].apply(lambda x: 1 if x == si else 0)
        for ai in amino_acids: df[f'a_{ai}'] = df['aa'].apply(lambda x: 1 if x == ai else 0)
        return df

    def _idx_df_l_init(self, df):
        chains = df['chain'].unique()
        if len(chains) == 1: return lambda row: chains[0] + row['residue']
        i = -1
        def idx_df_l(row):
            nonlocal i
            while True:
                i += 1
                row_g = df.iloc[i]
                if row_g['residue'] == row['residue'] and row['aa'] == row_g['aa'] and (not row['chain'] or row['chain'] == row_g['chain']):
                    break
            return row_g['chain'] + row_g['residue']
        return idx_df_l

    def _write_details_file(self, output_path):
        with open(output_path, 'w') as f:
            # (El código de escritura se mantiene igual, usando self.processing_details y self.start_time)
            f.write("Database Generation Details\n")
            f.write(f"Run started: {self.processing_details['start_time']}\n")
            f.write(f"Total runtime: {time.time() - self.start_time:.2f} seconds\n")
            # ... (sección abreviada por legibilidad, copiarías tu lógica original aquí)

    # Para hacer los splits ya dados
    def load_external_splits(self, train_path, val_path, test_path, split_index=0):
        """
        Carga los IDs de 3 archivos TXT. 
        split_index: entero de 0 a 4 para seleccionar cuál de las 5 listas usar.
        """
        def read_txt(path):
            with open(path, 'r') as f:
                # ast.literal_eval convierte el texto "[[...], [...]]" en lista de listas
                data = ast.literal_eval(f.read().strip())
                return data[split_index]

        try:
            train_ids = [str(pid).strip() for pid in read_txt(train_path)]
            val_ids   = [str(pid).strip() for pid in read_txt(val_path)]
            test_ids  = [str(pid).strip() for pid in read_txt(test_path)]
            
            return train_ids, val_ids, test_ids
        except IndexError:
            raise IndexError(f"El split_index {split_index} no existe. Los archivos deben tener 5 listas.")
        except Exception as e:
            raise Exception(f"Error leyendo los archivos de splits: {e}")


# ==============================================================================
# WRAPPER COMPATIBLE CON EL RESTO DE TU CÓDIGO (menu_URVDEEPTAF.py)
# ==============================================================================
def DB_Generation(**kwargs):
    """
    Función envoltorio para mantener la compatibilidad con tu código de la interfaz gráfica.
    Simplemente instancia la clase y llama al método generate.
    """
    generator = URVDataGenerator()
    return generator.generate(**kwargs)

# ==============================================================================
# EJECUCIÓN AUTOMÁTICA DE LOS 5 SPLITS
# ==============================================================================
if __name__ == "__main__":
    # 1. Configura tus rutas aquí
    CONFIG = {
        "dssp_dir": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/Protein/Protein_DSSP",
        "ligand_dir": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/Ligand/Ligand_SMI",
        "pocket_file": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/Binding/Binding_CSV/Protein_Residuoes_5.csv",
        "base_output": "",
        "txt_train": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/train_index_folder.txt",
        "txt_val": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/valid_index_folder.txt",
        "txt_test": "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/test_index_folder.txt"
    }

    generator = URVDataGenerator()

    for i in range(5):
        print(f"\n{'='*50}")
        print(f"INICIANDO PROCESAMIENTO SPLIT {i}")
        print(f"{'='*50}")
        
        try:
            # Obtener IDs del split actual
            ids = generator.load_external_splits(
                CONFIG["txt_train"], 
                CONFIG["txt_val"], 
                CONFIG["txt_test"], 
                split_index=i
            )
            
            # Definir carpeta de salida única para este split
            output_split = os.path.join(CONFIG["base_output"], f"split_{i}")
            
            # Ejecutar
            generator.generate(
                dssp_dir=CONFIG["dssp_dir"],
                ligand_dir=CONFIG["ligand_dir"],
                pocket_file=CONFIG["pocket_file"],
                output_dir=output_split,
                custom_split_ids=ids,
                cleanup_processed=True
            )
            
        except Exception as e:
            print(f"Error procesando el split {i}: {e}")
            continue

    print("\nPROCESO COMPLETO: Los 5 datasets han sido generados.")