import numpy as np

ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]

ATOM_TYPES_DTA = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]


def atom_type_onehot(symbol):
    vec = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES)
        + one_of_k_encoding(atom.GetDegree(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(0, 11, 1)])
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )

    feature_sum = features.sum()

    if feature_sum != 0:
        features /= feature_sum

    return features


def atom_features_dta(atom):
    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES_DTA)
        + one_of_k_encoding(atom.GetDegree(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(0, 11, 1)])
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )

    feature_sum = features.sum()

    if feature_sum != 0:
        features /= feature_sum

    return features
