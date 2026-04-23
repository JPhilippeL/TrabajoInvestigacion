import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "MPro-URV_Version2")
file_path = os.path.join(DATASET_DIR, "pIC50.txt")

# Llegir els valors
labels = np.loadtxt(file_path, usecols=1)

# Mínim i màxim real
real_min = labels.min()
real_max = labels.max()

# Afegir marge
margin = 0.2
global_min = real_min - margin
global_max = real_max + margin

print("Global axis limits:", global_min, global_max)

# Guardar per utilitzar als scripts de predict
np.save("global_axis.npy", [global_min, global_max])
