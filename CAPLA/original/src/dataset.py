from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv

import math
from typing import List

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X

def rotation(x,y,r):
    if r == 0:
        return (x,y)
    else:
        tempx = x * math.cos(r) + y * math.sin(r)
        tempy = y * math.cos(r) - x * math.sin(r)
        return (tempx,tempy)

def td_rotation(x,y,z,x_t,y_t,z_t,r1,r2,r3):
    x -= x_t
    y -= y_t
    z -= z_t
    x,y = rotation(x,y,r1)
    y,z = rotation(y,z,r2)
    x,z = rotation(x,z,r3)
    x += x_t
    y += y_t
    z += z_t
    return (x,y,z)

class MyDataset(Dataset):
    def __init__(self, data_path, td_data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None):
        data_path = Path(data_path)

        td_data_path = Path(td_data_path)

        global_path = td_data_path / 'protein_3D' / phase
        self.global_path = sorted(list(global_path.glob('*')))
        self.max_global_len = max_seq_len
        pocket_path = td_data_path / 'pocket_3D' / phase
        self.max_pkt_len = max_pkt_len
        self.pocket_path = sorted(list(pocket_path.glob('*')))

        ligand_path = td_data_path / 'ligand_3D' / phase
        self.max_ligand_len = max_smi_len
        self.ligand_path = sorted(list(ligand_path.glob('*')))

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len

        seq_path = data_path / phase / 'global'
        self.seq_path = sorted(list(seq_path.glob('*')))
        self.max_seq_len = max_seq_len

        pkt_path = data_path / phase / 'pocket'
        self.pkt_path = sorted(list(pkt_path.glob('*')))
        self.max_pkt_len = max_pkt_len
        self.pkt_window = pkt_window
        self.pkt_stride = pkt_stride
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        assert len(self.seq_path) == len(self.pkt_path)
        assert len(self.seq_path) == len(self.smi)

        self.length = len(self.smi)
        with open("pdbid.csv","w+") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(self.global_path)

    def __getitem__(self, idx):
        seq = self.seq_path[idx]#path 对象 有 name 属性
        pkt = self.pkt_path[idx]
        assert seq.name == pkt.name

#        td_glo = self.global_path[idx]#3D数据
#        td_pkt = self.pocket_path[idx]
#        td_lig = self.ligand_path[idx]

#        assert td_glo.name == td_pkt.name
#        assert td_pkt.name == td_lig.name
#
#        _tdglo_tensor = F.pdist(torch.tensor(pd.read_csv(td_glo, index_col=0).drop(['res_name'], axis=1).values[:self.max_global_len]))
#        _tdglo_tensor = _tdglo_tensor.view(-1,1)
#        torch.tensor(
#        print(_tdglo_tensor)
#        gloDisLen = int(self.max_global_len * (self.max_global_len - 1) / 2)
#        tdglo_tensor = np.zeros((gloDisLen, 1))
#        tdglo_tensor[:len(_tdglo_tensor)] = _tdglo_tensor#全局三维坐标
#        print(tdglo_tensor)

#        _tdpkt_tensor = F.pdist(torch.tensor(pd.read_csv(td_pkt, index_col=0).drop(['res_name'], axis=1).values[:self.max_pkt_len]))
#        _tdpkt_tensor = _tdpkt_tensor.view(-1,1)
#        pktDisLen = int(self.max_pkt_len * (self.max_pkt_len - 1) / 2)
#        tdpkt_tensor = np.zeros((pktDisLen, 1))
#        tdpkt_tensor[:len(_tdpkt_tensor)] = _tdpkt_tensor#口袋的三维坐标
#
#        _tdlig_tensor = F.pdist(torch.tensor(pd.read_csv(td_lig, index_col=0).drop(['res_name'], axis=1).values[:self.max_ligand_len]))
#        _tdlig_tensor = _tdlig_tensor.view(-1,1)
#        ligDisLen = int(self.max_ligand_len * (self.max_ligand_len - 1) / 2)
#        tdlig_tensor = np.zeros((ligDisLen, 1))
#        tdlig_tensor[:len(_tdlig_tensor)] = _tdlig_tensor#配体的三维坐标

#        tdglo_tensordis = np.zeros((self.max_global_len, self.max_global_len))
#        tdpkt_tensordis = np.zeros((self.max_pkt_len, self.max_pkt_len))
#        tdlig_tensordis = np.zeros((self.max_ligand_len, self.max_ligand_len))

#        for i in range(len(_tdglo_tensor)):
#            for j in range(i):
#                tdglo_tensordis[i][j] = (math.sqrt(((tdglo_tensor[i][0]-tdglo_tensor[j][0])**2)+((tdglo_tensor[i][1]-tdglo_tensor[j][1])**2)+((tdglo_tensor[i][2]-tdglo_tensor[j][2])**2)))
#        
#        for i in range(len(_tdpkt_tensor)):
#            for j in range(i):
#                tdpkt_tensordis[i][j] = (math.sqrt(((tdpkt_tensor[i][0]-tdpkt_tensor[j][0])**2)+((tdpkt_tensor[i][1]-tdpkt_tensor[j][1])**2)+((tdpkt_tensor[i][2]-tdpkt_tensor[j][2])**2))) 
#
#        for i in range(len(_tdlig_tensor)):
#            for j in range(i):
#                tdlig_tensordis[i][j] = (math.sqrt(((tdlig_tensor[i][0]-tdlig_tensor[j][0])**2)+((tdlig_tensor[i][1]-tdlig_tensor[j][1])**2)+((tdlig_tensor[i][2]-tdlig_tensor[j][2])**2)))  
#        
#        baisx = np.random.random()
#        baisy = np.random.random()
#        baisz = np.random.random()
#        degree = np.random.random()

#        for i in range(len(_tdglo_tensor)):
#            tdglo_tensor[i][0],tdglo_tensor[i][1],tdglo_tensor[i][2] = (0.0,0.0,0.0)
#
#        for i in range(len(_tdpkt_tensor)):
#            tdpkt_tensor[i][0],tdpkt_tensor[i][1],tdpkt_tensor[i][2] = (0.0,0.0,0.0)
#
#        for i in range(len(_tdlig_tensor)):
#            tdlig_tensor[i][0],tdlig_tensor[i][1],tdlig_tensor[i][2] = (0.0,0.0,0.0)
#
#        for i in range(len(_tdglo_tensor)):
#            tdglo_tensor[i][0],tdglo_tensor[i][1],tdglo_tensor[i][2] = td_rotation(tdglo_tensor[i][0],tdglo_tensor[i][1],tdglo_tensor[i][2],baisx,baisy,baisz,degree,degree,degree)
#
#        for i in range(len(_tdpkt_tensor)):
#            tdpkt_tensor[i][0],tdpkt_tensor[i][1],tdpkt_tensor[i][2] = td_rotation(tdpkt_tensor[i][0],tdpkt_tensor[i][1],tdpkt_tensor[i][2],baisx,baisy,baisz,degree,degree,degree)
#
#        for i in range(len(_tdlig_tensor)):
#            tdlig_tensor[i][0],tdlig_tensor[i][1],tdlig_tensor[i][2] = td_rotation(tdlig_tensor[i][0],tdlig_tensor[i][1],tdlig_tensor[i][2],baisx,baisy,baisz,degree,degree,degree)

        
        
        _seq_tensor = pd.read_csv(seq, index_col=0)
        if "idx" in _seq_tensor.columns:_seq_tensor = _seq_tensor.drop(['idx'], axis=1).values[:self.max_seq_len]      
        else:_seq_tensor = _seq_tensor.values[:self.max_seq_len] 
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
        seq_tensor[:len(_seq_tensor)] = _seq_tensor
        
        _pkt_tensor = pd.read_csv(pkt, index_col=0)
        if "idx" in _pkt_tensor.columns: _pkt_tensor = _pkt_tensor.drop(['idx'], axis=1).values[:self.max_pkt_len] 
        else:_pkt_tensor = _pkt_tensor.values[:self.max_pkt_len]  
            
        if self.pkt_window is not None and self.pkt_stride is not None:
            pkt_len = (int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride))
                       * self.pkt_stride
                       + self.pkt_window)
            pkt_tensor = np.zeros((pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            pkt_tensor = np.array(
                [pkt_tensor[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window]
                 for i in range(int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)))]
            )
        else:
            pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor

        return (seq_tensor.astype(np.float32),
                pkt_tensor.astype(np.float32),
                label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),
                np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32)
                )

    def __len__(self):
        return self.length
