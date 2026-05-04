import numpy as np
import deepsmiles
import torch
import numpy as np
import json
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#ligand_path = "kiba/ligands_can.txt"
#protein_path ="kiba/proteins.txt"
#affinity_path ="kiba/Y"
#motif_path ="kiba/motif2.txt"
#motif_path ="E:/data/data/kiba/motif2.txt"

ligand_path = "davis/ligands_can.txt"
protein_path ="davis/proteins.txt"
affinity_path ="davis/Y"
motif_path ="davis/motif2.txt"


#keys = ['O43741','O96017','P41279','P45983','P45984','P51955','P53779','P54619','P67870','Q15078','Q15118','Q8IW41','Q96KB5','Q9NWZ3','Q9UGI9','Q9UGJ0','Q9Y478']
keys=[13, 23, 85, 95, 96, 114, 121, 122, 125, 164, 165, 184, 189, 211, 219, 220, 226]
#keys=[13, 122, 125, 164, 189, 226]
#list(map(proteins.pop, keys))


def filter(x, l):
    pt = {}
    print(f"Filtering with length threshold {l}")
    for i, p in enumerate(x.values()):
        if len(p) <= l:
            pt[i] = p
    print(f"Number of entries after filtering: {len(pt)}")
    return pt

###########deepsml
###convert smiles to deep smiles
def deepsml(x):
    ldeep={}
    for i,m in (x.items()):
         converter = deepsmiles.Converter(rings=True, branches=True)
         ldeep[i]=converter.encode(m)
    return ldeep
#############makig ps
#making PS
def ps(x,word_len):
    d = {}
    for j, p in (x.items()):
        t=()
        for i in range(word_len):
            y = len(p)
            for m in range(i,y,word_len):
               k=p[m:m+word_len]
               if(len(k)==word_len):
                  t=t+(k,)
        d[j] = t
    return d
##onehot
def onehot(x):
    one_d={}
    ps1=list(x.values())
    p_set = set()
    lens_p = [len(p) for p in ps1]
    for p in ps1:
        p_set = p_set.union(set(p))
    char_to_int_p = dict((c, i) for i, c in enumerate(p_set))
    int_to_char_p = dict((i, c) for i, c in enumerate(p_set))
    #onehot_p = np.zeros((len(ps1), len(char_to_int_p), max(lens_p)))
    for i, p in enumerate(ps1):
        onehot_p = np.zeros((len(char_to_int_p), max(lens_p)))
        for j, char in enumerate(p):
            onehot_p[char_to_int_p[char], j] = 1.0
        one_d[i]=onehot_p
    return one_d
###making pair
def MPMy(mol,pro,moti,y):
    mpmy=[]
    for i,m in (mol.items()):
        for j,p,mp in zip(pro.keys(),pro.values(),moti.values()):
            try:
            #for j,mt in(moti.item()):
                mpmy.append(((torch.Tensor(m),torch.Tensor(p),torch.Tensor(mp)),y[i][j]))
            
            except IndexError as e:
                print(f"IndexError: {e} for ligand {i}, protein {j}")
            except Exception as e:
                print(f"Exception: {e}")
    print(f"Total pairs created: {len(mpmy)}")
    return mpmy


#custom dataset
import torch
from torch.utils.data import Dataset
class widedata(Dataset):
    def __init__(self, ligand_path, protein_path, keys, motif_path, affinity_path):
        with open(ligand_path) as ligand_data:
            ligands = json.load(ligand_data)
            #print(f"Number of ligands loaded: {len(ligands)}")  # Debug print
            
            # Print a few ligand samples to understand their lengths
            for i, (k, v) in enumerate(ligands.items()):
                #print(f"Ligand {i} key: {k}, length: {len(v)}, value (truncated): {v[:30]}")
                if i >= 5:  # Limit output to first few entries
                    break

            # Use the already read JSON data instead of re-reading the file
            #filtered_ligands = filter(ligands, 20)
            filtered_ligands = filter(ligands, 50)  # Filter function should use the ligands loaded above
            deep_smiles_ligands = deepsml(filtered_ligands)
            self.lig = ps(deep_smiles_ligands, 8)
            #print(f"Number of ligands after filtering and processing: {len(self.lig)}")
            self.lig2 = onehot(self.lig)

           #print(self.lig2)
            ##remain torch tnsr
        with open(protein_path) as protein_data:
           self.pro=filter(json.load(protein_data),600)
           #list(map(self.pro.pop, keys)) # Comment for Davis dataset and descomment for KIBA's
           self.pro=ps((self.pro),3)
           self.pro2=onehot(self.pro)
            ##torch tnsr remain
        with open(motif_path) as motif_data:
           self.motif =ps(json.load(motif_data),3)
           self.moti2 =onehot(self.motif)
           ##tnsr
        with open(affinity_path,'rb') as Y:
          self.y = torch.Tensor(np.nan_to_num(pickle.load(Y, encoding='latin1')))
          #self.y = np.nan_to_num(pickle.load(Y,encoding='latin1'))
          self.mpmy=MPMy(self.lig2,self.pro2,self.moti2,self.y)
    def __len__(self):

        return len(self.mpmy)

    def __getitem__(self, idx):
        try:
            return self.mpmy[idx]
        except IndexError as e:
            print(f"IndexError: {e} for index {idx}")
        except Exception as e:
            print(f"Exception: {e}")


#if __main__=="__main__":
#dataset = widedata(ligand_path, protein_path,keys,motif_path,affinity_path)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
##############################train,test loader

def load_splitset(dataset,test_split):
    #test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating  data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                sampler=test_sampler)
    return train_loader,test_loader
# train_loader, test_loader = load_splitset(dataset, .2)
# print(len(train_loader))
# print(len(test_loader))

