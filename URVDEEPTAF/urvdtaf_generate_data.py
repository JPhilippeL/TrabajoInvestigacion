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
from sklearn.model_selection import train_test_split

from .utils.constants import PROCESSED_DATA_DIR, c1, c2, structure_types, amino_acids
import logging

logger = logging.getLogger(__name__)
COLUMNA_ID_LIGAND = "id"
processing_details = []

def DB_Generation(
    # Required parameters
    dssp_dir,                 # Directory containing protein DSSP files
    ligand_dir,               # Directory containing ligand SMI files
    pocket_file,              # Path to pocket CSV file
    
    # Optional parameters with defaults
    output_dir=None,          # Output directory (defaults to PROCESSED_DATA_DIR in current directory)
    train_ratio=0.70,         # Train set ratio
    val_ratio=0.15,           # Validation set ratio
    test_ratio=0.15,          # Test set ratio
    random_seed=42,           # Random seed for reproducibility
    cleanup_processed=True    # Whether to clean up temporary processed files
):
    # Start timing
    start_time = time.time()
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        error_msg = f"Ratios must sum to 1.0, got: {train_ratio + val_ratio + test_ratio}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), PROCESSED_DATA_DIR)
    # else:
    #    output_dir = os.path.join(output_dir, PROCESSED_DATA_DIR)
    
    # Create temporary processing directory
    proc_dir = os.path.join(output_dir, "processed")
    proc_global_dir = os.path.join(proc_dir, "global")
    proc_pocket_dir = os.path.join(proc_dir, "pocket")
    ligand_output_csv = os.path.join(proc_dir, "ligands.csv")
    
    # Create directories
    os.makedirs(proc_global_dir, exist_ok=True)
    os.makedirs(proc_pocket_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ligand_output_csv), exist_ok=True)
    
    print(f"Processing files for dataset...")
    print(f"Output directory: {output_dir}")
    
    # Track status for details.txt
    processing_details = {
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
            "input": {
                "dssp_files": 0,
                "ligand_files": 0,
                "pocket_entries": 0
            },
            "output": {}
        }
    }

    # -------------------------
    # Main Execution Block
    # -------------------------
    
    print(f"Starting database generation...")
    
    # Step 1: Process protein files to generate global features and sequences
    print("\nStep 1/5: Processing protein files...")
    process_protein_files(dssp_dir, proc_global_dir)
    
    # Step 2: Process pocket data
    print("\nStep 2/5: Processing pocket data...")
    process_pocket_file(pocket_file, dssp_dir, proc_pocket_dir)
    
    # Step 3: Process ligand files
    print("\nStep 3/5: Processing ligand files...")
    process_ligand_files(ligand_dir, ligand_output_csv)
    
    # Step 4: Split data into train/validation/test sets
    print("\nStep 4/5: Splitting data...")
    train_ids, val_ids, test_ids = split_pdbids(
        ligand_output_csv, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Create output subdirectories
    train_global_dir = os.path.join(output_dir, "training", "global")
    train_pocket_dir = os.path.join(output_dir, "training", "pocket")
    val_global_dir = os.path.join(output_dir, "validation", "global")
    val_pocket_dir = os.path.join(output_dir, "validation", "pocket")
    test_global_dir = os.path.join(output_dir, "test", "global")
    test_pocket_dir = os.path.join(output_dir, "test", "pocket")
    
    os.makedirs(train_global_dir, exist_ok=True)
    os.makedirs(train_pocket_dir, exist_ok=True)
    os.makedirs(val_global_dir, exist_ok=True)
    os.makedirs(val_pocket_dir, exist_ok=True)
    os.makedirs(test_global_dir, exist_ok=True)
    os.makedirs(test_pocket_dir, exist_ok=True)
    
    # Copy files to their respective directories
    print("Copying global feature files...")
    train_global_copied, train_global_missing = copy_files_for_split(proc_global_dir, train_ids, train_global_dir)
    val_global_copied, val_global_missing = copy_files_for_split(proc_global_dir, val_ids, val_global_dir)
    test_global_copied, test_global_missing = copy_files_for_split(proc_global_dir, test_ids, test_global_dir)
    
    print("Copying pocket feature files...")
    train_pocket_copied, train_pocket_missing = copy_files_for_split(proc_pocket_dir, train_ids, train_pocket_dir)
    val_pocket_copied, val_pocket_missing = copy_files_for_split(proc_pocket_dir, val_ids, val_pocket_dir)
    test_pocket_copied, test_pocket_missing = copy_files_for_split(proc_pocket_dir, test_ids, test_pocket_dir)
    
    # Split and save ligand CSV files
    train_ligands, val_ligands, test_ligands = split_ligand_csv(ligand_output_csv, train_ids, val_ids, test_ids)
    
    # Create combined sequence files
    print("Creating combined sequence files...")
    train_global_seq_path = os.path.join(output_dir, "training_seq_.csv")
    val_global_seq_path = os.path.join(output_dir, "validation_seq_.csv")
    test_global_seq_path = os.path.join(output_dir, "test_seq_.csv")
    
    train_pocket_seq_path = os.path.join(output_dir, "training_pocket_.csv")
    val_pocket_seq_path = os.path.join(output_dir, "validation_pocket_.csv")
    test_pocket_seq_path = os.path.join(output_dir, "test_pocket_.csv")
    
    train_global_seq_count, _ = combine_global_seq_for_split(dssp_dir, train_ids, train_global_seq_path)
    val_global_seq_count, _ = combine_global_seq_for_split(dssp_dir, val_ids, val_global_seq_path)
    test_global_seq_count, _ = combine_global_seq_for_split(dssp_dir, test_ids, test_global_seq_path, include_index=False)
    
    train_pocket_seq_count, _ = combine_pocket_seq_for_split(pocket_file, train_ids, train_pocket_seq_path)
    val_pocket_seq_count, _ = combine_pocket_seq_for_split(pocket_file, val_ids, val_pocket_seq_path)
    test_pocket_seq_count, _ = combine_pocket_seq_for_split(pocket_file, test_ids, test_pocket_seq_path, include_index=False)
    
    # Update output file counts
    processing_details["file_counts"]["output"] = {
        "global_features": train_global_copied + val_global_copied + test_global_copied,
        "pocket_features": train_pocket_copied + val_pocket_copied + test_pocket_copied,
        "ligand_entries": train_ligands + val_ligands + test_ligands,
        "global_sequences": train_global_seq_count + val_global_seq_count + test_global_seq_count,
        "pocket_sequences": train_pocket_seq_count + val_pocket_seq_count + test_pocket_seq_count
    }
    
    # Step 5: Clean up and finalize
    print("\nStep 5/5: Finalizing...")
    # Write detailed processing information
    details_path = os.path.join(output_dir, "details.txt")
    write_details_file(details_path, processing_details, start_time)
    print(f"Processing details written to {details_path}")
    
    # Clean up temporary files if requested
    if cleanup_processed:
        cleanup_processed_dir(proc_dir)
    
    # Output success summary
    print("\nDatabase generation completed!")
    print(f"Data directory: {output_dir}")
    print(f"Total PDBIDs: {processing_details['split_info']['total_pdbids']}")
    print(f"  Training: {processing_details['split_info']['train_count']}")
    print(f"  Validation: {processing_details['split_info']['val_count']}")
    print(f"  Test: {processing_details['split_info']['test_count']}")
    
    # Return a dictionary with the summary of results
    return {
        "total_pdbids": processing_details['split_info']['total_pdbids'],
        "train_count": processing_details['split_info']['train_count'],
        "val_count": processing_details['split_info']['val_count'],
        "test_count": processing_details['split_info']['test_count'],
        "output_dir": output_dir,
        "details_file": details_path,
    }

def process_protein_files(dssp_dir, output_global_dir):
        """
        Process each DSSP file in dssp_dir:
          - Compute 40-dimensional features and advanced idx mapping.
          - Save feature file (CSV with index=True) in output_global_dir.
          - Also extract and save global sequence (id and concatenated seq) in a sibling folder "global_seq".
          (For proteins, the 'idx' column remains as the last column.)
        """
        if not os.path.exists(output_global_dir):
            os.makedirs(output_global_dir)
        global_seq_dir = os.path.join(os.path.dirname(output_global_dir), "global_seq")
        if not os.path.exists(global_seq_dir):
            os.makedirs(global_seq_dir)
        dssp_files = glob.glob(os.path.join(dssp_dir, "*_protein.dssp"))
        
        # Update file count information
        processing_details["file_counts"]["input"]["dssp_files"] = len(dssp_files)
        
        success_count = 0
        failed_count = 0
        
        for dssp_file in tqdm(dssp_files, desc="Processing DSSP files"):
            try:
                protein_id = clean_pdbid(os.path.basename(dssp_file).split('_')[0])
                df = parse_dssp_file(dssp_file)
                
                if df.empty:
                    error_msg = f"Error: Empty DataFrame for {protein_id} (DSSP parse failed)"
                    logger.error(error_msg)
                    failed_count += 1
                    continue
                
                # Extract and save global sequence:
                seq = "".join(df["aa"].tolist())
                seq_out_path = os.path.join(global_seq_dir, protein_id + "_seq.csv")
                pd.DataFrame({"id": [protein_id], "seq": [seq]}).to_csv(seq_out_path, index=True, float_format='%.6f')
                
                # Compute features and save feature file:
                df = compute_features(df, use_structure=True)
                idx_func = idx_df_l_init(df)
                df['idx'] = df.apply(idx_func, axis=1)
                df_out = df.drop(['residue', 'chain', 'aa', 'structure'], axis=1)
                # For protein files, leave the order as is (idx as the last column)
                df_out.index.name = None
                feature_out_path = os.path.join(output_global_dir, protein_id + ".csv")
                df_out.to_csv(feature_out_path, index=True, float_format='%.6f')
                success_count += 1
            except Exception as e:
                error_msg = f"Error processing DSSP file {dssp_file}: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                failed_count += 1
                
        # Update processing details
        processing_details["protein_processing"] = {
            "success_count": success_count,
            "failed_count": failed_count
        }

def process_pocket_file(pocket_csv, dssp_dir, output_pocket_dir):
        """
        Process pocket CSV using protein DSSP files to get true secondary structures.
        """
        if not os.path.exists(output_pocket_dir):
            os.makedirs(output_pocket_dir)
        pocket_seq_dir = os.path.join(os.path.dirname(output_pocket_dir), "pocket_seq")
        if not os.path.exists(pocket_seq_dir):
            os.makedirs(pocket_seq_dir)
        
        try:
            # --- CORRECCIÓN 1: Lectura Robusta ---
            # Usamos sep=None y engine='python' para que detecte automáticamente 
            # si es coma, punto y coma o tabulador.
            df_pocket = pd.read_csv(pocket_csv, sep=None, engine='python', dtype={'PDB': str})
            
            # --- CORRECCIÓN 2: Limpieza de Columnas ---
            # Eliminamos espacios en blanco al principio/final de los nombres de columnas
            df_pocket.columns = df_pocket.columns.str.strip()
            
            # Opcional: Si tus columnas a veces vienen en minúscula ('pdb', 'seq'), esto lo estandariza
            df_pocket.rename(columns=lambda x: x.upper() if x.upper() == 'PDB' else x, inplace=True)
            # Nota: 'seq' lo mantenemos como venga o lo forzamos si es necesario.
            # Si tu CSV trae 'Seq' o 'SEQUENCE', puedes añadir una línea similar para renombrarlo a 'seq'.

            # Verificación antes de continuar
            required_cols = ['PDB', 'seq']
            if not all(col in df_pocket.columns for col in required_cols):
                missing = [c for c in required_cols if c not in df_pocket.columns]
                raise ValueError(f"El CSV no tiene las columnas esperadas {required_cols}. Columnas encontradas: {list(df_pocket.columns)}")

            processing_details["file_counts"]["input"]["pocket_entries"] = len(df_pocket)
            
            all_seq_out = os.path.join(pocket_seq_dir, "all_pocket_seq.csv")
            
            # Aquí es donde fallaba si las columnas no se detectaban bien
            df_pocket[["PDB", "seq"]].rename(columns={"PDB": "id"}).to_csv(all_seq_out, index=True)
            
            success_count = 0
            skipped_count = 0
            
            for _, row in df_pocket.iterrows():
                try:
                    # Aseguramos que accedemos a 'PDB' correctamente
                    protein_id = clean_pdbid(str(row['PDB']))
                    seq = str(row['seq']).strip()
                    
                    # Manejo seguro de 'Positions' por si no existe en el CSV
                    raw_positions = row.get('Positions', '')
                    # Si es NaN (vacío en pandas), convertir a string vacío
                    if pd.isna(raw_positions):
                        raw_positions = ''
                    positions = str(raw_positions).strip().split()
                    
                    # Validate
                    if not positions or len(seq) != len(positions):
                        # Mensaje de error más descriptivo
                        error_msg = f"Skipping {protein_id}: Invalid Positions (SeqLen: {len(seq)} vs PosLen: {len(positions)})"
                        logger.error(error_msg)
                        print(error_msg)
                        skipped_count += 1
                        continue
                    
                    # Get corresponding protein's DSSP data
                    dssp_path = os.path.join(dssp_dir, f"{protein_id}_protein.dssp")
                    if not os.path.exists(dssp_path):
                        error_msg = f"Skipping {protein_id}: DSSP file not found at {dssp_path}"
                        logger.error(error_msg)
                        print(error_msg)
                        skipped_count += 1
                        continue
                    
                    # Parse protein's DSSP to map residue positions to structures
                    df_protein = parse_dssp_file(dssp_path)
                    chain = 'A'  # Assuming pocket residues belong to chain A
                    
                    # --- CORRECCIÓN 3: Seguridad en tipos de datos ---
                    # Asegurarse de que el mapeo usa strings para evitar mismatch de tipos (int vs str)
                    df_protein['residue'] = df_protein['residue'].astype(str)
                    structure_map = df_protein.set_index(['chain', 'residue'])['structure'].to_dict()
                    
                    # Build pocket records with real structures
                    records = []
                    for pos, aa in zip(positions, seq):
                        # Get true structure from protein's DSSP (chain A)
                        # Convertimos pos a string para asegurar match con el diccionario
                        structure = structure_map.get((chain, str(pos)), 'C')  
                        records.append({
                            'residue': pos,
                            'chain': chain,
                            'aa': aa,
                            'structure': structure
                        })
                    
                    # Compute features WITH structural data (use_structure=True)
                    df_pocket_residues = pd.DataFrame(records)
                    df_pocket_residues = compute_features(df_pocket_residues, use_structure=True)
                    df_pocket_residues['idx'] = df_pocket_residues['chain'] + df_pocket_residues['residue'].astype(str)
                    
                    # Save features (40 columns)
                    # Verificamos qué columnas existen antes de hacer drop
                    cols_to_drop = ['residue', 'chain', 'aa', 'structure']
                    existing_cols_to_drop = [c for c in cols_to_drop if c in df_pocket_residues.columns]
                    
                    df_out = df_pocket_residues.drop(existing_cols_to_drop, axis=1)
                    
                    # Reordenar columnas poniendo 'idx' primero
                    if 'idx' in df_out.columns:
                        cols = ['idx'] + [c for c in df_out.columns if c != 'idx']
                        df_out = df_out[cols]
                    
                    df_out.to_csv(os.path.join(output_pocket_dir, f"{protein_id}.csv"), index=True)
                    success_count += 1
                except Exception as e:
                    # Capturar error específico por fila para no detener todo el proceso
                    error_msg = f"Error processing pocket for {row.get('PDB', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    print(error_msg)
                    skipped_count += 1
            
            # Update processing details
            processing_details["pocket_processing"] = {
                "success_count": success_count,
                "skipped_count": skipped_count
            }
        except Exception as e:
            error_msg = f"CRITICAL Error processing pocket file {pocket_csv}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            processing_details["pocket_processing"] = {
                "success_count": 0,
                "skipped_count": 0,
                "error": str(e)
            }

def process_ligand_files(ligand_dir, output_csv):
        """
        Process ligand .smi files and combine them into one CSV.
        Export with index=False.
        """
        # CORRECCIÓN 1: Añadido el '*' antes de .smi
        pattern = os.path.join(ligand_dir, "*.smi")
        ligand_files = glob.glob(pattern)
        
        processing_details["file_counts"]["input"]["ligand_files"] = len(ligand_files)
        print(f"Encontrados {len(ligand_files)} archivos de ligando en: {ligand_dir}")
        
        records = []
        success_count = 0
        failed_count = 0
        
        # Si no hay archivos, avisar para no procesar nada
        if not ligand_files:
            print("ADVERTENCIA: No se encontraron archivos .smi. Verifica la ruta.")
        
        for ligand_file in tqdm(ligand_files, desc="Processing ligand files"):
            try:
                pdbid = clean_pdbid(os.path.basename(ligand_file).split('_')[0])
                with open(ligand_file, 'r') as f:
                    line = f.readline().strip()
                    # Usar regex para separar por espacios o tabs
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 1: # A veces el SMILES es lo único en la línea
                        smiles = parts[0]
                        records.append({'pdbid': pdbid, 'smiles': smiles})
                        success_count += 1
                    else:
                        error_msg = f"Warning: File {ligand_file} is empty or invalid format"
                        logger.error(error_msg)
                        print(error_msg)
                        failed_count += 1
            except Exception as e:
                error_msg = f"Error processing ligand file {ligand_file}: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                failed_count += 1
                
        # CORRECCIÓN 2: Crear DataFrame asegurando las columnas, aunque esté vacío
        if records:
            df_ligand = pd.DataFrame(records)
        else:
            # Crea las columnas vacías para evitar el KeyError 'pdbid'
            df_ligand = pd.DataFrame(columns=['pdbid', 'smiles'])

        # Conversión segura a string
        df_ligand['pdbid'] = df_ligand['pdbid'].astype(str)
        
        df_ligand.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"Saved ligand CSV: {output_csv}")
        
        # Update processing details
        processing_details["ligand_processing"] = {
            "success_count": success_count,
            "failed_count": failed_count,
            "total_ligands": len(records)
        }

# -------------------------
# Splitting Functions: Split pdbids and Copy Files
# -------------------------
def split_pdbids(ligand_csv, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits pdbids (from ligand CSV) into training, validation, and test lists.
    """
    df = pd.read_csv(ligand_csv, dtype={'pdbid': str})
    pdbids = [clean_pdbid(x) for x in df['pdbid'].unique().tolist()]
    train_ids, temp_ids = train_test_split(pdbids, train_size=train_ratio, random_state=random_seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_ratio/(val_ratio+test_ratio), random_state=random_seed)
    
    # Update processing details
    processing_details["split_info"] = {
        "total_pdbids": len(pdbids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids)
    }
    
    return train_ids, val_ids, test_ids

def copy_files_for_split(source_dir, split_ids, target_dir):
        """
        Copies individual CSV files (named as <pdbid>.csv) from source_dir to target_dir for each pdbid in split_ids.
        """
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        copied_count = 0
        missing_count = 0
        for pdbid in split_ids:
            src_file = os.path.join(source_dir, pdbid + ".csv")
            if os.path.exists(src_file):
                dst_file = os.path.join(target_dir, pdbid + ".csv")
                shutil.copy(src_file, dst_file)
                copied_count += 1
            else:
                error_msg = f"Warning: {src_file} not found."
                logger.error(error_msg)
                print(error_msg)
                missing_count += 1
        return copied_count, missing_count

def split_ligand_csv(ligand_csv, train_ids, val_ids, test_ids, output_dir):
        """
        Splits the ligand CSV into three files based on split pdbids.
        No index column in any of the files.
        """
        df = pd.read_csv(ligand_csv, dtype={'pdbid': str})
        df_train = df[df['pdbid'].isin(train_ids)]
        df_val = df[df['pdbid'].isin(val_ids)]
        df_test = df[df['pdbid'].isin(test_ids)]
        
        train_path = os.path.join(output_dir, "training_smi.csv")
        val_path = os.path.join(output_dir, "validation_smi.csv")
        test_path = os.path.join(output_dir, "test_smi.csv")
        
        df_train.to_csv(train_path, index=False, float_format='%.6f')
        df_val.to_csv(val_path, index=False, float_format='%.6f')
        df_test.to_csv(test_path, index=False, float_format='%.6f')
        
        print(f"Saved ligand splits: {train_path}, {val_path}, {test_path}")
        return len(df_train), len(df_val), len(df_test)

# -------------------------
# Combined Sequence Files for Global and Pocket Data
# -------------------------
def combine_global_seq_for_split(dssp_dir, split_ids, output_filename, include_index=True):
    """
    For each pdbid in split_ids, read its DSSP file from dssp_dir, extract the global sequence,
    and combine into a CSV with columns: id, seq.
    Option to include/exclude index.
    """
    records = []
    missing_count = 0
    for pdbid in split_ids:
        pattern = os.path.join(dssp_dir, f"{pdbid}_protein.dssp")
        files = glob.glob(pattern)
        if files:
            dssp_file = files[0]
            df = parse_dssp_file(dssp_file)
            seq = "".join(df["aa"].tolist())
            records.append({"id": pdbid, "seq": seq})
        else:
            error_msg = f"Warning: DSSP file for {pdbid} not found."
            logger.error(error_msg)
            print(error_msg)
            missing_count += 1
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_filename, index=include_index, float_format='%.6f')
        print(f"Saved combined global seq file: {output_filename}")
        return len(df), missing_count
    else:
        error_msg = f"No global seq records combined for {output_filename}"
        logger.error(error_msg)
        print(error_msg)
        return 0, missing_count

def combine_pocket_seq_for_split(pocket_csv, split_ids, output_filename, include_index=True):
    """
    Filter the original pocket CSV (columns: PDB, Ligand, seq) for split_ids and output a CSV with id and seq.
    Option to include/exclude index.
    """
    df = pd.read_csv(pocket_csv, dtype={'PDB': str})
    df["PDB"] = df["PDB"].apply(clean_pdbid)
    df_subset = df[df["PDB"].isin(split_ids)].rename(columns={"PDB": "id"})[["id", "seq"]]
    missing_count = len(split_ids) - len(df_subset)
    if missing_count > 0:
        error_msg = f"Warning: {missing_count} pocket sequences not found for split IDs."
        logger.error(error_msg)
        print(error_msg)
    df_subset.to_csv(output_filename, index=include_index, float_format='%.6f')
    print(f"Saved combined pocket seq file: {output_filename}")
    return len(df_subset), missing_count

# -------------------------
# Create detailed summary
# -------------------------
def write_details_file(output_path, details, start_time):
    """
    Write processing details to a text file
    """
    with open(output_path, 'w') as f:
        f.write("Database Generation Details\n")
        f.write("=====================================\n\n")
        
        # Time and system information
        f.write(f"Run started: {details['start_time']}\n")
        f.write(f"Run completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {time.time() - start_time:.2f} seconds\n\n")
        
        f.write("System Information:\n")
        f.write(f"  Python version: {details['system_info']['python_version']}\n")
        f.write(f"  Platform: {details['system_info']['platform']}\n")
        f.write(f"  Node: {details['system_info']['node']}\n\n")
        
        # Input parameters
        f.write("Input Parameters:\n")
        for param, value in details['parameters'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # Input files
        f.write("Input Files:\n")
        f.write(f"  DSSP files: {details['file_counts']['input']['dssp_files']}\n")
        f.write(f"  Ligand files: {details['file_counts']['input']['ligand_files']}\n")
        f.write(f"  Pocket entries: {details['file_counts']['input']['pocket_entries']}\n\n")
        
        # Processing statistics
        f.write("Processing Statistics:\n")
        if 'protein_processing' in details:
            f.write("  Protein Processing:\n")
            f.write(f"    Success: {details['protein_processing']['success_count']}\n")
            f.write(f"    Failed: {details['protein_processing']['failed_count']}\n")
        
        if 'pocket_processing' in details:
            f.write("  Pocket Processing:\n")
            f.write(f"    Success: {details['pocket_processing']['success_count']}\n")
            f.write(f"    Skipped: {details['pocket_processing']['skipped_count']}\n")
        
        if 'ligand_processing' in details:
            f.write("  Ligand Processing:\n")
            f.write(f"    Success: {details['ligand_processing']['success_count']}\n")
            f.write(f"    Failed: {details['ligand_processing']['failed_count']}\n")
            f.write(f"    Total ligands: {details['ligand_processing']['total_ligands']}\n")
        f.write("\n")
        
        # Data split information
        if 'split_info' in details:
            f.write("Data Split Information:\n")
            f.write(f"  Total PDBIDs: {details['split_info']['total_pdbids']}\n")
            f.write(f"  Training set: {details['split_info']['train_count']} PDBIDs\n")
            f.write(f"  Validation set: {details['split_info']['val_count']} PDBIDs\n")
            f.write(f"  Test set: {details['split_info']['test_count']} PDBIDs\n\n")
        
        # Output files
        if 'output_files' in details['file_counts']:
            f.write("Output Files:\n")
            for key, count in details['file_counts']['output'].items():
                f.write(f"  {key}: {count}\n")
            f.write("\n")

def parse_dssp_file(dssp_path):
        """
        Parse a DSSP file and return a DataFrame with columns:
          residue, chain, aa, structure.
        Assumes data lines start at line 29 (after a 28-line header).
        Residue number: columns 6-11, chain: column 12, aa: columns 13-15, structure: column 17.
        """
        try:
            with open(dssp_path, 'r') as f:
                lines = f.readlines()
            records = []
            for line in lines[28:]:
                if len(line) < 17:
                    continue
                if line[13] == '!':
                    continue
                residue = line[5:11].strip()   # global residue number as string
                chain = line[11:12].strip() or 'A'
                aa = line[12:14].strip() or 'X'
                structure = line[16:17].strip() or 'C'
                records.append({
                    'residue': residue,
                    'chain': chain,
                    'aa': aa[0],
                    'structure': structure
                })
            return pd.DataFrame(records)
        except Exception as e:
            error_msg = f"Error parsing DSSP file {dssp_path}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            return pd.DataFrame(columns=['residue', 'chain', 'aa', 'structure'])

def compute_features(df, use_structure=True):
        """
        Compute 40 features for each residue:
          - 4 features from c1
          - 7 features from c2
          - 8 one-hot features for secondary structure (s2_*)
          - 21 one-hot features for amino acids (a_*)
        Cleaning: if 'aa' or 'structure' do not match [A-Z], they are set to 'C'.
        """
        df['aa'] = df['aa'].apply(lambda x: x if re.match('[A-Z]', x) else 'C')
        df['structure'] = df['structure'].apply(lambda x: x if re.match('[A-Z]', x) else 'C')
        
        for key in c1:
            new_feature = df['aa'].apply(lambda x: f1(x, key))
            df = df.assign(**{key: new_feature})
        for key in c2:
            new_feature = df['aa'].apply(lambda x: f2(x, key))
            df = df.assign(**{f'c2_{key}': new_feature})
        for si in structure_types:
            new_feature = df['structure'].apply(lambda x: 1 if x == si else 0)
            df = df.assign(**{f's2_{si}': new_feature})
        for ai in amino_acids:
            new_feature = df['aa'].apply(lambda x: 1 if x == ai else 0)
            df = df.assign(**{f'a_{ai}': new_feature})
        return df

# -------------------------
# Advanced idx mapping (mimicking the notebook)
# -------------------------
def idx_df_l_init(df):
    chains = df['chain'].unique()
    if len(chains) == 1:
        return lambda row: chains[0] + row['residue']
    i = -1
    def idx_df_l(row):
        nonlocal i
        while True:
            i += 1
            row_g = df.iloc[i]
            if row_g['residue'] == row['residue']:
                if row['aa'] != row_g['aa']:
                    continue
                if row['chain'] and row['chain'] != row_g['chain']:
                    continue
                break
        return row_g['chain'] + row_g['residue']
    return idx_df_l

# -------------------------
# Helper function to clean up processed directory
# -------------------------
def cleanup_processed_dir(directory):
    """
    Remove the entire directory tree if it exists
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleaned up {directory}")
        return True
    return False

# -------------------------
# Helper function to clean pdbid strings
# -------------------------
def clean_pdbid(pid):
    """Convert pid to string and remove any '+' from exponential notation."""
    pid = str(pid)
    return pid.replace("E+", "E").replace("e+", "E")

def f1(aa, key):
    if aa == 'X':
        return 1 / len(c1)
    return 1 if aa in c1[key] else 0

def f2(aa, key):
    if aa == 'X':
        return 1 / len(c2)
    return 1 if aa in c2[key] else 0