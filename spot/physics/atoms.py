# -*- coding: utf-8 -*-
"""
spot.physics.atoms — atomic data tables
========================================
Spot's atomic-data module: atomic weights and first/second ionization
potentials for the 92 elements, plus abundance management (the THEVENIN
table in the 12+log10(N/N_H) scale).

The abundance values themselves are loaded from the packaged
``spot/data/abundance.csv``; this module only supplies the element
names, weights and ionization potentials.
"""

import numpy as np

__all__ = [
    "ATOM_SYMBOLS", "ATOM_WEIGHTS", "IONIZATION_ENERGY_1", "IONIZATION_ENERGY_2",
    "element_symbol_to_number", "element_number_to_symbol",
    "AbundanceTable",
]

# Element symbols, Z = 1..92 (by atomic number)
ATOM_SYMBOLS = [
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
    "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
    "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "U",
]

# Atomic weights [g/mol] (Wittmann 1975)
ATOM_WEIGHTS = np.array([
    1.008, 4.003, 6.939, 9.012, 10.811, 12.011, 14.007, 16.0, 18.998,
    20.183, 22.99, 24.312, 26.982, 28.086, 30.974, 32.064, 35.453, 39.948,
    39.102, 40.08, 44.956, 47.90, 50.942, 51.996, 54.938, 55.847, 58.933,
    58.71, 63.54, 65.37, 69.72, 72.59, 74.92, 78.96, 79.91, 83.80, 85.47,
    87.62, 88.905, 91.22, 92.906, 95.94, 99.00, 101.07, 102.9, 106.4,
    107.87, 112.40, 114.82, 118.69, 121.75, 127.6, 126.9, 131.3, 132.9,
    137.34, 138.91, 140.12, 140.91, 144.24, 147.00, 150.35, 151.96, 157.25,
    158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95,
    183.85, 186.2, 190.2, 192.2, 195.09, 196.97, 200.59, 204.37, 207.19,
    208.98, 210.0, 211.0, 222.0, 223.0, 226.1, 227.1, 232.04, 231.0, 238.03,
])

# First ionization potentials [eV]
IONIZATION_ENERGY_1 = np.array([
    13.595, 24.58, 5.39, 9.32, 8.298, 11.256, 14.529, 13.614, 17.418,
    21.559, 5.138, 7.644, 5.984, 8.149, 10.474, 10.357, 13.012, 15.755,
    4.339, 6.111, 6.538, 6.825, 6.738, 6.763, 7.432, 7.896, 7.863, 7.633,
    7.724, 9.391, 5.997, 7.88, 9.81, 9.75, 11.840, 13.996, 4.176, 5.692,
    6.377, 6.838, 6.881, 7.10, 7.28, 7.36, 7.46, 8.33, 7.574, 8.991, 5.785,
    7.34, 8.64, 9.01, 10.454, 12.127, 3.893, 5.210, 5.577, 5.466, 5.422,
    5.489, 5.554, 5.631, 5.666, 6.141, 5.852, 5.927, 6.018, 6.101, 6.184,
    6.254, 5.426, 6.650, 7.879, 7.980, 7.870, 8.70, 9.10, 9.00, 9.22, 10.43,
    6.105, 7.415, 7.287, 8.43, 9.30, 10.745, 4.0, 5.276, 6.9, 6.0, 6.0, 6.0,
])

# Second ionization potentials [eV]
IONIZATION_ENERGY_2 = np.array([
    0.0, 54.403, 75.62, 18.21, 25.15, 24.376, 29.59, 35.11, 34.98, 41.07,
    47.290, 15.03, 18.823, 16.34, 19.72, 23.405, 23.798, 27.62, 31.81,
    11.868, 12.891, 13.63, 14.205, 16.493, 15.636, 16.178, 17.052, 18.15,
    20.286, 17.96, 20.509, 15.93, 18.63, 21.50, 21.60, 24.565, 27.50,
    11.027, 12.233, 13.13, 14.316, 16.15, 15.26, 16.76, 18.07, 19.42,
    21.48, 16.904, 18.86, 14.63, 16.50, 18.60, 19.09, 21.20, 25.10, 10.001,
    11.060, 10.850, 10.550, 10.730, 10.899, 11.069, 11.241, 12.090, 11.519,
    11.670, 11.800, 11.930, 12.050, 12.184, 13.900, 14.900, 16.2, 17.7,
    16.60, 17.00, 20.00, 18.56, 20.50, 18.75, 20.42, 15.03, 16.68, 19.0,
    20.0, 20.0, 22.0, 10.144, 12.1, 12.0, 12.0, 12.0,
])

_SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(ATOM_SYMBOLS)}


def element_symbol_to_number(symbol):
    """Atomic number of an element symbol (case-insensitive, e.g. 'Fe')."""
    key = symbol.strip().upper()
    if key not in _SYMBOL_TO_Z:
        raise ValueError(f"unknown element symbol {symbol!r}")
    return _SYMBOL_TO_Z[key]


def element_number_to_symbol(z):
    """Element symbol for an atomic number (1..92)."""
    if not 1 <= int(z) <= 92:
        raise ValueError(f"atomic number out of range: {z}")
    return ATOM_SYMBOLS[int(z) - 1]


class AbundanceTable:
    """
    Element abundances in the 12+log10(N/N_H) scale.

    Parameters
    ----------
    log12_abundance : array-like of length >= 92
        Abundances (the packaged ``abundance.csv`` column), indexed by
        atomic number minus one.
    """

    def __init__(self, log12_abundance):
        arr = np.asarray(log12_abundance, dtype=float)
        if arr.shape[0] < 92:
            raise ValueError(
                f"abundance table needs >= 92 entries, got {arr.shape[0]}")
        self.log12 = arr[:92]
        self.relative = np.power(10.0, self.log12 - 12.0)  # N(X)/N(H)

    def __getitem__(self, z):
        """Relative abundance N(X)/N(H) of atomic number ``z``."""
        return float(self.relative[int(z) - 1])

    @classmethod
    def from_default(cls):
        """Load the packaged THEVENIN abundance table."""
        from ..utils.data_io import load_abundance
        table = load_abundance("thevenin")
        return cls(table["abundance_log12"])
