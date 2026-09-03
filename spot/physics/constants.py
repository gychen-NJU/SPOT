# -*- coding: utf-8 -*-
"""
spot.physics.constants — physical constants (CGS)
==================================================
Spot's set of physical constants; the values follow the original
spectral-synthesis code so that numerical results are directly
comparable.
"""

# --- fundamental constants (CGS) -------------------------------------------
C_LIGHT = 2.99792458e10          # speed of light            [cm/s]
H_PLANCK = 6.6260755e-27         # Planck constant            [erg s]
K_BOLTZ = 1.3807e-16             # Boltzmann constant         [erg/K]
GAS = 8.31451e7                  # ideal gas constant         [erg/(mol K)]
AVOGADRO = 6.023e23              # Avogadro number            [1/mol]
ELECTRON_MASS = 9.1094e-28       # electron mass              [g]
PROTON_MASS = 1.6726e-24         # proton mass                [g]
ATOMIC_MASS_UNIT = 1.660540e-24  # atomic mass unit           [g]
GRAVITY_SUN = 2.7414e4           # solar surface gravity      [cm/s^2]
PI = 3.141592653589793

# --- derived quantities ----------------------------------------------------
HC_K = 1.43880                   # h c / k                    [cm K]
TWO_H_C2 = 1.1910627e-5          # 2 h c^2                    [erg cm^3/s]
THETA_FACTOR = 5040.0            # eV <-> K conversion factor (5040/T = eV/kT)
SIGMA_THOMSON = 6.653e-25        # Thomson scattering cross-section [cm^2]

# electron charge-related constant appearing in the line-opacity formula
# (pi e^2 / (m_e c)) in cm^2/s — folded into the 1.49736e-2 coefficient
# folded into the line-opacity coefficient eta00 = 1.49736e-2 * gf * abu * lambda_cm
E2_MC = 1.49736e-2               # (pi e^2/mc) * 1/(sqrt(pi)) style factor
