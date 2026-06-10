ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]


def atom_type_onehot(symbol):
    vec = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec
