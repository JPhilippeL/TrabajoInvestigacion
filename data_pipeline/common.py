import ast
import random

import numpy as np
import torch


def load_split_txt(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def escala_global(file_path):
    labels = np.loadtxt(file_path, usecols=1)

    real_min = labels.min()
    real_max = labels.max()

    margin = 0.2
    global_min = real_min - margin
    global_max = real_max + margin

    print("Global axis limits:", global_min, global_max)

    np.save("global_axis.npy", [global_min, global_max])

    return global_min, global_max
