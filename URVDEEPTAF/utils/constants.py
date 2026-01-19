PROCESSED_DATA_DIR = "DEEPTAFDATA"

# -------------------------
# Feature groups and helper functions
# -------------------------
c1 = {
    'non_polar': ('G', 'A', 'V', 'L', 'I', 'M', 'F', 'P', 'W'),
    'polar': ('S', 'T', 'Y', 'C', 'Q', 'N'),
    'acidic': ('D', 'E'),
    'basic': ('K', 'R', 'H')
}
c2 = {
    1: ('A', 'G', 'V'),
    2: ('I', 'L', 'F', 'P'),
    3: ('Y', 'M', 'T', 'S'),
    4: ('H', 'N', 'Q', 'W'),
    5: ('R', 'K'),
    6: ('D', 'E'),
    7: ('C',)
}
structure_types = ('B', 'C', 'E', 'G', 'H', 'I', 'S', 'T')
amino_acids = ('G', 'A', 'V', 'L', 'I', 'M', 'F', 'P', 'W',
                'S', 'T', 'Y', 'C', 'Q', 'N', 'D', 'E', 'K', 'R', 'H', 'X')
