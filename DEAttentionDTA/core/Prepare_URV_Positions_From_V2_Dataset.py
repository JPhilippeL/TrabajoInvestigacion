#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare URV for DEAttentionDTA by deriving Position/Pocket from the older
MPro-URV Version2 structural files.

Python 3.6 compatible.

Inputs expected:
  - URV official dataset, version v3b in this TFM:
      data/urv_dataset_v3b/Info.csv
      data/urv_dataset_v3b/Splits/*.txt

  - MPro-URV Version2 full dataset somewhere on disk, containing one or more of:
      Interaction/<PDB_ID>_ligand.json
      Complex/ALIGNED/<PDB_ID>.cif
      Protein/Protein_PDB/<PDB_ID>_protein.pdb

Outputs:
  data/urv_dataset_v3b_prepared/
      seq_data_all.csv
      affinity_all.csv
      splits/split_01/seq_train.csv
      splits/split_01/affinity_train.csv
      ...
      reports/position_report.json
      reports/position_report.csv
      reports/dropped_rows.csv

Recommended usage from the DEAttentionDTA project root:

  python DEAttentionDTA/core/Prepare_URV_Positions_From_V2_Dataset.py \
    --urv-dir data/urv_dataset_v3b \
    --urv-v2-dir "/mnt/c/Users/pcmsi/Desktop/URV/2025-2026 MASTER THESIS (17685301) VIRTUAL/MPro-URV_Version2"

The script first tries Interaction JSON. If that fails or gives no positions,
it falls back to a distance-based pocket from Complex/ALIGNED/*.cif.
"""

from __future__ import print_function

import argparse
import ast
import csv
import json
import math
import os
import re
import shlex
import sys
import traceback

import pandas as pd

AA_CODES = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # frequent protonation/alternate histidine names
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'MSE': 'M'
}

PROTEIN_ALPHABET = set(list('ACDEFGHIKLMNPQRSTVWY'))


def repo_root_from_script():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def join_root(repo_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def clean_id(value):
    text = str(value).strip().upper()
    text = text.replace('\xa0', '').replace(' ', '')
    # Fix Excel-like corruption observed in old Info.csv, e.g. 7,00 GEK -> 7GEK
    text = text.replace(',00', '').replace(',', '')
    text = re.sub('[^A-Z0-9]', '', text)
    return text


def read_split_file(path):
    with open(path, 'r') as handle:
        value = ast.literal_eval(handle.read())
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError('Expected 5 split lists in {0}'.format(path))
    out = []
    for split in value:
        out.append([clean_id(x) for x in split])
    return out


def load_splits(urv_v3b_dir):
    split_dir = os.path.join(urv_v3b_dir, 'Splits')
    return {
        'train': read_split_file(os.path.join(split_dir, 'train_index_folder.txt')),
        'valid': read_split_file(os.path.join(split_dir, 'valid_index_folder.txt')),
        'test': read_split_file(os.path.join(split_dir, 'test_index_folder.txt'))
    }


def pocket_string(sequence, positions):
    chars = []
    for pos in positions:
        if 1 <= int(pos) <= len(sequence):
            chars.append(sequence[int(pos) - 1])
    return ''.join(chars)


def parse_chain_list(value):
    text = str(value).strip()
    if not text:
        return []
    parts = re.split('[,;/ ]+', text)
    return [p.strip() for p in parts if p.strip()]


def is_hydrogen(atom):
    symbol = str(atom.get('element', '')).strip().upper()
    atom_name = str(atom.get('atom_name', '')).strip().upper()
    return symbol == 'H' or atom_name.startswith('H')


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_int_resseq(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ['', '.', '?']:
        return None
    match = re.search(r'-?\d+', text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def residue_key(chain, resseq, icode=None):
    chain_text = str(chain).strip() if chain is not None else ''
    resseq_text = str(resseq).strip() if resseq is not None else ''
    icode_text = str(icode).strip() if icode not in [None, '', '.', '?'] else ''
    return (chain_text, resseq_text, icode_text)


def parse_pdb_residue_order(pdb_path, allowed_chains):
    """Return mapping (chain, resseq, icode) -> 1-based sequence position."""
    mapping = {}
    order = []
    if not pdb_path or not os.path.isfile(pdb_path):
        return mapping, ''
    with open(pdb_path, 'r') as handle:
        for line in handle:
            if not line.startswith('ATOM'):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA_CODES:
                continue
            chain = line[21].strip()
            if allowed_chains and chain not in allowed_chains:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = residue_key(chain, resseq, icode)
            if key not in mapping:
                mapping[key] = len(order) + 1
                order.append((key, resname))
    seq = ''.join([AA_CODES.get(resname, 'X') for _, resname in order])
    return mapping, seq


def tokenize_cif_line(line):
    try:
        return shlex.split(line, comments=False, posix=True)
    except Exception:
        return line.strip().split()


def parse_mmcif_atoms(cif_path):
    """Minimal mmCIF atom_site parser. No BioPython dependency."""
    atoms = []
    if not cif_path or not os.path.isfile(cif_path):
        return atoms
    with open(cif_path, 'r') as handle:
        lines = handle.readlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line != 'loop_':
            i += 1
            continue
        i += 1
        headers = []
        while i < n and lines[i].strip().startswith('_'):
            headers.append(lines[i].strip())
            i += 1
        if not headers or not headers[0].startswith('_atom_site.'):
            continue
        keys = []
        for h in headers:
            if '.' in h:
                keys.append(h.split('.', 1)[1])
            else:
                keys.append(h)
        while i < n:
            raw = lines[i]
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                i += 1
                if stripped.startswith('#'):
                    break
                continue
            if stripped == 'loop_' or stripped.startswith('_') or stripped.startswith('data_'):
                break
            parts = tokenize_cif_line(raw)
            if len(parts) >= len(keys):
                rec = dict(zip(keys, parts[:len(keys)]))
                x = parse_float(rec.get('Cartn_x'))
                y = parse_float(rec.get('Cartn_y'))
                z = parse_float(rec.get('Cartn_z'))
                if x is not None and y is not None and z is not None:
                    atom = {
                        'record': rec.get('group_PDB', ''),
                        'atom_name': rec.get('auth_atom_id', rec.get('label_atom_id', '')),
                        'element': rec.get('type_symbol', ''),
                        'comp_id': rec.get('auth_comp_id', rec.get('label_comp_id', '')),
                        'chain': rec.get('auth_asym_id', rec.get('label_asym_id', '')),
                        'resseq': rec.get('auth_seq_id', rec.get('label_seq_id', '')),
                        'icode': rec.get('pdbx_PDB_ins_code', ''),
                        'x': x,
                        'y': y,
                        'z': z
                    }
                    atoms.append(atom)
            i += 1
    return atoms


def build_residue_order_from_atoms(atoms, sequence, allowed_chains):
    mapping = {}
    order = []
    for atom in atoms:
        if str(atom.get('record', '')).upper() != 'ATOM':
            continue
        resname = str(atom.get('comp_id', '')).upper()
        if resname not in AA_CODES:
            continue
        chain = str(atom.get('chain', '')).strip()
        if allowed_chains and chain not in allowed_chains:
            continue
        key = residue_key(chain, atom.get('resseq'), atom.get('icode'))
        if key not in mapping:
            mapping[key] = len(order) + 1
            order.append((key, resname))
    # If numbering is canonical 1..len(sequence), prefer direct resseq mapping.
    # Keep the order mapping as fallback.
    direct = {}
    for key, _ in order:
        resnum = parse_int_resseq(key[1])
        if resnum is not None and 1 <= resnum <= len(sequence):
            direct[key] = resnum
    if len(direct) >= max(1, int(0.8 * len(mapping))):
        mapping.update(direct)
    return mapping


def find_existing_file(candidates):
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def collect_numbers_near_residue_words(text):
    # Very conservative string fallback: collect numbers in strings containing residue terms.
    t = str(text)
    tl = t.lower()
    if not any(w in tl for w in ['residue', 'resnr', 'resnum', 'residue_number', 'amino', 'protein']):
        return []
    nums = []
    for m in re.finditer(r'(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])', t):
        try:
            nums.append(int(m.group(0)))
        except Exception:
            pass
    return nums


def traverse_json_for_residues(obj, found):
    """Collect possible residue numbers from heterogeneous interaction JSON files.

    The exact Version2 JSON schema is unknown here, so this function accepts
    common field names used by PLIP-like/protein-ligand interaction outputs.
    """
    residue_keys = set([
        'resnr', 'resnum', 'residue_number', 'residue_num', 'residueid',
        'residue_id', 'resseq', 'res_seq', 'resid', 'res_id', 'position'
    ])
    chain_keys = set(['chain', 'chain_id', 'chainid', 'protchain', 'protein_chain'])
    if isinstance(obj, dict):
        lower = {}
        for k, v in obj.items():
            lower[str(k).lower()] = v
        chain = None
        for ck in chain_keys:
            if ck in lower:
                chain = str(lower[ck]).strip()
                break
        for rk in residue_keys:
            if rk in lower:
                resnum = parse_int_resseq(lower[rk])
                if resnum is not None:
                    found.append((chain, resnum, 'json_field:' + rk))
        # PLIP frequently uses fields like restype/resnr/reschain together.
        for k, v in obj.items():
            kl = str(k).lower()
            if 'res' in kl and not isinstance(v, (dict, list)):
                nums = collect_numbers_near_residue_words(str(k) + ' ' + str(v))
                for num in nums:
                    found.append((chain, num, 'json_string:' + str(k)))
            traverse_json_for_residues(v, found)
    elif isinstance(obj, list):
        for item in obj:
            traverse_json_for_residues(item, found)
    elif isinstance(obj, str):
        for num in collect_numbers_near_residue_words(obj):
            found.append((None, num, 'json_string'))


def positions_from_interaction_json(json_path, sequence, allowed_chains, residue_map):
    if not json_path or not os.path.isfile(json_path):
        return [], 'missing_interaction_json'
    try:
        with open(json_path, 'r') as handle:
            data = json.load(handle)
    except Exception as exc:
        return [], 'json_read_error:{0}'.format(exc)

    raw = []
    traverse_json_for_residues(data, raw)
    positions = set()
    for chain, resnum, _source in raw:
        if resnum is None:
            continue
        mapped = None
        if residue_map:
            candidate_keys = []
            if chain:
                candidate_keys.append(residue_key(chain, str(resnum), ''))
            for key in residue_map.keys():
                if parse_int_resseq(key[1]) == int(resnum):
                    if not chain or str(key[0]) == str(chain):
                        candidate_keys.append(key)
            for key in candidate_keys:
                if key in residue_map:
                    mapped = residue_map[key]
                    break
        if mapped is None and 1 <= int(resnum) <= len(sequence):
            mapped = int(resnum)
        if mapped is not None and 1 <= int(mapped) <= len(sequence):
            positions.add(int(mapped))
    if not positions:
        return [], 'no_positions_in_interaction_json'
    return sorted(positions), 'interaction_json'


def squared_distance(a, b):
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    dz = a['z'] - b['z']
    return dx * dx + dy * dy + dz * dz


def positions_from_distance_cif(cif_path, sequence, ligand_code, allowed_chains, cutoff):
    atoms = parse_mmcif_atoms(cif_path)
    if not atoms:
        return [], 'missing_or_empty_cif'
    residue_map = build_residue_order_from_atoms(atoms, sequence, allowed_chains)
    ligand_code = str(ligand_code).strip().upper()
    ligand_atoms = []
    protein_atoms = []
    for atom in atoms:
        if is_hydrogen(atom):
            continue
        record = str(atom.get('record', '')).upper()
        comp_id = str(atom.get('comp_id', '')).upper()
        chain = str(atom.get('chain', '')).strip()
        if record == 'ATOM' and comp_id in AA_CODES:
            if not allowed_chains or chain in allowed_chains:
                protein_atoms.append(atom)
        elif record == 'HETATM' and comp_id == ligand_code:
            ligand_atoms.append(atom)
    if not ligand_atoms:
        return [], 'no_ligand_atoms_for_code:{0}'.format(ligand_code)
    if not protein_atoms:
        return [], 'no_protein_atoms'
    cutoff2 = float(cutoff) * float(cutoff)
    pos = set()
    for patom in protein_atoms:
        near = False
        for latom in ligand_atoms:
            if squared_distance(patom, latom) <= cutoff2:
                near = True
                break
        if not near:
            continue
        key = residue_key(patom.get('chain'), patom.get('resseq'), patom.get('icode'))
        mapped = residue_map.get(key)
        if mapped is None:
            resnum = parse_int_resseq(patom.get('resseq'))
            if resnum is not None and 1 <= resnum <= len(sequence):
                mapped = resnum
        if mapped is not None and 1 <= int(mapped) <= len(sequence):
            pos.add(int(mapped))
    if not pos:
        return [], 'no_residues_within_{0}_angstrom'.format(cutoff)
    return sorted(pos), 'distance_cif_cutoff_{0}'.format(cutoff)


def derive_positions_for_row(row, urv_v2_dir, cutoff):
    pdb_id = row['PDBname']
    sequence = row['Sequence']
    ligand_code = row.get('Ligand', '')
    allowed_chains = parse_chain_list(row.get('Chain ID', ''))

    json_path = find_existing_file([
        os.path.join(urv_v2_dir, 'Interaction', pdb_id + '_ligand.json'),
        os.path.join(urv_v2_dir, 'Interaction', pdb_id.lower() + '_ligand.json')
    ])
    pdb_path = find_existing_file([
        os.path.join(urv_v2_dir, 'Protein', 'Protein_PDB', pdb_id + '_protein.pdb'),
        os.path.join(urv_v2_dir, 'Protein', 'Protein_PDB', pdb_id.lower() + '_protein.pdb')
    ])
    cif_path = find_existing_file([
        os.path.join(urv_v2_dir, 'Complex', 'ALIGNED', pdb_id + '.cif'),
        os.path.join(urv_v2_dir, 'Complex', 'ALIGNED', pdb_id.lower() + '.cif')
    ])

    residue_map, pdb_seq = parse_pdb_residue_order(pdb_path, allowed_chains)
    positions, source = positions_from_interaction_json(json_path, sequence, allowed_chains, residue_map)
    if positions:
        return positions, source, json_path, pdb_path, cif_path

    positions, source2 = positions_from_distance_cif(cif_path, sequence, ligand_code, allowed_chains, cutoff)
    if positions:
        return positions, source2, json_path, pdb_path, cif_path

    combined_source = source + ';' + source2
    return [], combined_source, json_path, pdb_path, cif_path


def validate_row(row):
    reasons = []
    if not row.get('PDBname'):
        reasons.append('missing_pdb_id')
    if not row.get('Smile'):
        reasons.append('missing_smiles')
    if not row.get('Sequence'):
        reasons.append('empty_sequence')
    else:
        bad = sorted(set(row.get('Sequence')) - PROTEIN_ALPHABET)
        if bad:
            reasons.append('unknown_sequence_tokens:' + ''.join(bad))
    try:
        float(row.get('affinity'))
    except Exception:
        reasons.append('missing_or_invalid_pIC50')
    return reasons


def prepare(args):
    repo_root = repo_root_from_script()
    urv_v3b_dir = join_root(repo_root, args.urv_dir)
    urv_v2_dir = join_root(repo_root, args.urv_v2_dir)
    out_dir = ensure_dir(join_root(repo_root, args.out_dir))
    reports_dir = ensure_dir(os.path.join(out_dir, 'reports'))
    split_root = ensure_dir(os.path.join(out_dir, 'splits'))

    info_path = os.path.join(urv_v3b_dir, 'Info.csv')
    if not os.path.isfile(info_path):
        raise IOError('Cannot find URV Info.csv: {0}'.format(info_path))
    if not os.path.isdir(urv_v2_dir):
        raise IOError('Cannot find Version2 directory: {0}'.format(urv_v2_dir))

    info_df = pd.read_csv(info_path, sep=';', dtype=str, keep_default_na=False)
    required_cols = ['PDB_ID', 'SMILES', 'Proteine Sequence', 'pIC50', 'Ligand', 'Chain ID']
    missing = [c for c in required_cols if c not in info_df.columns]
    if missing:
        raise ValueError('Info.csv missing required columns: {0}'.format(missing))

    info_df['PDBname'] = info_df['PDB_ID'].apply(clean_id)
    info_df['Smile'] = info_df['SMILES'].astype(str).str.strip()
    info_df['Sequence'] = info_df['Proteine Sequence'].astype(str).str.strip().str.upper()
    info_df['affinity'] = pd.to_numeric(info_df['pIC50'], errors='coerce')

    rows = []
    dropped = []
    position_reports = []

    for _, raw_row in info_df.iterrows():
        row = raw_row.to_dict()
        base = {
            'PDBname': row.get('PDBname', ''),
            'Smile': row.get('Smile', ''),
            'Sequence': row.get('Sequence', ''),
            'affinity': row.get('affinity')
        }
        reasons = validate_row(base)
        positions = []
        source = ''
        json_path = ''
        pdb_path = ''
        cif_path = ''

        if not reasons:
            positions, source, json_path, pdb_path, cif_path = derive_positions_for_row(row, urv_v2_dir, args.distance_cutoff)
            if not positions:
                reasons.append('no_position:' + source)

        report_row = {
            'PDBname': base['PDBname'],
            'position_source': source,
            'n_positions': len(positions),
            'positions': str(positions),
            'interaction_json': json_path or '',
            'protein_pdb': pdb_path or '',
            'complex_cif': cif_path or '',
            'drop_reasons': ';'.join(reasons)
        }
        position_reports.append(report_row)

        if reasons:
            dropped.append({
                'PDBname': base['PDBname'],
                'DropReasons': ';'.join(reasons),
                'SMILES': base['Smile'],
                'pIC50': row.get('pIC50', '')
            })
            continue

        rows.append({
            'PDBname': base['PDBname'],
            'Smile': base['Smile'],
            'Sequence': base['Sequence'],
            'Pocket': pocket_string(base['Sequence'], positions),
            'Position': str(list(positions)),
            'affinity': float(base['affinity'])
        })

    prepared_df = pd.DataFrame(rows).drop_duplicates(subset=['PDBname'], keep='first')
    prepared_df = prepared_df.sort_values('PDBname').reset_index(drop=True)

    seq_all = prepared_df[['PDBname', 'Smile', 'Sequence', 'Pocket', 'Position']]
    aff_all = prepared_df[['PDBname', 'affinity']]
    seq_all_path = os.path.join(out_dir, 'seq_data_all.csv')
    aff_all_path = os.path.join(out_dir, 'affinity_all.csv')
    seq_all.to_csv(seq_all_path, index=False)
    aff_all.to_csv(aff_all_path, index=False)

    pd.DataFrame(position_reports).to_csv(os.path.join(reports_dir, 'position_report.csv'), index=False)
    pd.DataFrame(dropped).to_csv(os.path.join(reports_dir, 'dropped_rows.csv'), index=False)

    splits = load_splits(urv_v3b_dir)
    available_ids = set(prepared_df['PDBname'].tolist())
    original_ids = set(info_df['PDBname'].tolist())
    manifest_rows = []
    split_reports = []
    for split_idx in range(5):
        split_id = split_idx + 1
        split_name = 'split_{0:02d}'.format(split_id)
        split_dir = ensure_dir(os.path.join(split_root, split_name))
        split_record = {'split': split_id}
        for role in ['train', 'valid', 'test']:
            official_ids = splits[role][split_idx]
            official_set = set(official_ids)
            role_df = prepared_df[prepared_df['PDBname'].isin(official_set)].copy()
            role_df = role_df.sort_values('PDBname').reset_index(drop=True)
            seq_path = os.path.join(split_dir, 'seq_{0}.csv'.format(role))
            aff_path = os.path.join(split_dir, 'affinity_{0}.csv'.format(role))
            role_df[['PDBname', 'Smile', 'Sequence', 'Pocket', 'Position']].to_csv(seq_path, index=False)
            role_df[['PDBname', 'affinity']].to_csv(aff_path, index=False)
            dropped_ids = sorted(official_set - available_ids)
            missing_info = sorted(official_set - original_ids)
            manifest_rows.append({
                'split': split_id,
                'role': role,
                'official_ids': len(official_ids),
                'exported_rows': len(role_df),
                'missing_from_info': len(missing_info),
                'dropped_after_preparation': len(dropped_ids),
                'dropped_ids': ','.join(dropped_ids),
                'seq_csv': seq_path,
                'affinity_csv': aff_path
            })
            split_record[role + '_official'] = len(official_ids)
            split_record[role + '_exported'] = len(role_df)
            split_record[role + '_dropped_ids'] = dropped_ids
        split_reports.append(split_record)

    manifest_path = os.path.join(out_dir, 'split_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    n_with_position = len(prepared_df)
    n_total = len(info_df)
    summary = {
        'position_source': 'Interaction JSON from MPro-URV Version2; fallback distance from Complex/ALIGNED CIF',
        'urv_info_csv': info_path,
        'urv_v2_dir': urv_v2_dir,
        'distance_cutoff_angstrom': args.distance_cutoff,
        'n_samples_total': int(n_total),
        'n_samples_with_position': int(n_with_position),
        'n_samples_without_position': int(n_total - n_with_position),
        'dropped_rows_csv': os.path.join(reports_dir, 'dropped_rows.csv'),
        'position_report_csv': os.path.join(reports_dir, 'position_report.csv'),
        'seq_data_all_csv': seq_all_path,
        'affinity_all_csv': aff_all_path,
        'split_manifest_csv': manifest_path,
        'split_reports': split_reports
    }
    with open(os.path.join(reports_dir, 'position_report.json'), 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print('Prepared URV dataset with Version2-derived Position/Pocket')
    print('  total samples:           {0}'.format(n_total))
    print('  samples with position:   {0}'.format(n_with_position))
    print('  samples without position:{0}'.format(n_total - n_with_position))
    print('  output:                  {0}'.format(out_dir))
    print('  report:                  {0}'.format(os.path.join(reports_dir, 'position_report.json')))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description='Prepare URV DEAttentionDTA files using Position/Pocket derived from MPro-URV Version2. Python 3.6 compatible.')
    parser.add_argument('--urv-dir', '--urv-v3b-dir', dest='urv_dir', default='data/urv_dataset_v3b', help='Path to the official URV dataset directory containing Info.csv and Splits/. The old alias --urv-v3b-dir is also accepted.')
    parser.add_argument('--urv-v2-dir', required=True, help='Path to full MPro-URV Version2 directory containing Interaction, Complex, Protein.')
    parser.add_argument('--out-dir', default='data/urv_dataset_v3b_prepared')
    parser.add_argument('--distance-cutoff', type=float, default=4.5, help='Distance cutoff in Angstroms for CIF fallback pocket extraction.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except Exception as exc:
        print('ERROR: {0}'.format(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
