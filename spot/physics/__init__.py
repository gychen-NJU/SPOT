# -*- coding: utf-8 -*-
"""
spot.physics — vectorized physics core
=======================================
Pure PyTorch re-implementations of the spectral-synthesis physics
subroutines:

* ``constants``        — physical constants (cgs)
* ``atoms``            — atomic weights, ionization potentials, abundances
* ``partition``        — partition functions (temperature fits)
* ``ionization``       — Saha equation and molecular balance
* ``pressure``         — ionization-equilibrium partial pressures
* ``opacity``          — continuum opacity
* ``line``             — Zeeman patterns, line opacity, damping
* ``absorption``       — 4x4 propagation (absorption) matrix
* ``rte``              — RTE formal solvers: Hermitian (hermite, default),
                         DELO (delo_solve, Rees et al. 1989) and
                         A-stable Crank-Nicolson (cn_solve — ported from
                         pyPRT-dsh, which misnames it "rk4_solve"); plus
                         the propagation operator & variational solver
                         used by the response functions
* ``thermodynamics``   — Planck function, HSRA continuum, air refraction
* ``voigt``            — Voigt and Faraday profiles

All routines are batch-vectorized with torch broadcasting.
"""
