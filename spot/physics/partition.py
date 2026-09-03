# -*- coding: utf-8 -*-
"""
spot.physics.partition — atomic partition functions
=====================================================

Partition functions u0 (neutral), u1 (singly ionized) and u2 (doubly
ionized) as analytic fits of temperature, ported from the classical
Stokes inversion reference ``atmdatb.f`` / ``nelfctb`` data (polynomial
fits in T or in log(5040/T)).  These partition functions enter the Saha
equation, so the coefficients reproduce the reference fits to keep the
ionisation balances and the resulting line opacities consistent.

All functions are vectorized in torch: ``T`` may have any shape and the
returned tensors have that shape.
"""

import torch

__all__ = ["partition_functions"]

# Registry: atomic number -> function(T) -> (u0, u1, u2)
_HANDLERS = {}


def _register(z):
    def deco(fn):
        _HANDLERS[z] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# 1  H
# ---------------------------------------------------------------------------
@_register(1)
def _u_h(t):
    u0 = torch.full_like(t, 2.0)
    u0 = torch.where(t > 1.3e4, 1.51 + 3.8e-5 * t, u0)
    u0 = torch.where(t > 1.62e4, 11.41 + t * (-1.1428e-3 + t * 3.52e-8), u0)
    return u0, torch.ones_like(t), torch.zeros_like(t)


# ---------------------------------------------------------------------------
# 2  He
# ---------------------------------------------------------------------------
@_register(2)
def _u_he(t):
    u0 = torch.ones_like(t)
    u0 = torch.where(t > 3e4, 14.8 + t * (-9.4103e-4 + t * 1.6095e-8), u0)
    return u0, torch.full_like(t, 2.0), torch.ones_like(t)


# ---------------------------------------------------------------------------
# 3  Li
# ---------------------------------------------------------------------------
@_register(3)
def _u_li(t):
    t3 = 1e-3 * t
    u0 = 2.081 - t3 * (6.8926e-2 - t3 * 1.4081e-2)
    u0 = torch.where(t > 6e3, 3.4864 + t * (-7.3292e-4 + t * 8.5586e-8), u0)
    return u0, torch.ones_like(t), torch.full_like(t, 2.0)


# ---------------------------------------------------------------------------
# 4  Be
# ---------------------------------------------------------------------------
@_register(4)
def _u_be(t):
    return torch.clamp(0.631 + 7.032e-5 * t, min=1.0), \
        torch.full_like(t, 2.0), torch.ones_like(t)


# ---------------------------------------------------------------------------
# 5  B
# ---------------------------------------------------------------------------
@_register(5)
def _u_b(t):
    return 5.9351 + 1.0438e-2 * 1e-3 * t, torch.ones_like(t), \
        torch.full_like(t, 2.0)


# ---------------------------------------------------------------------------
# 6  C
# ---------------------------------------------------------------------------
@_register(6)
def _u_c(t):
    t3 = 1e-3 * t
    u0 = 8.6985 + t3 * (2.0485e-2 + t3 * (1.7629e-2 - 3.9091e-4 * t3))
    u0 = torch.where(t > 1.2e4, 12.54 + t * (-1.2933e-3 + t * 4.20e-8), u0)
    u1 = 5.838 + 1.6833e-5 * t
    u1 = torch.where(t > 2.4e4, 10.989 + t * (-6.9347e-4 + t * 2.0861e-8), u1)
    u2 = torch.ones_like(t)
    u2 = torch.where(t > 1.95e4, -0.555 + 8e-5 * t, u2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 7  N
# ---------------------------------------------------------------------------
@_register(7)
def _u_n(t):
    t3 = 1e-3 * t
    u0 = 3.9914 + t3 * (1.7491e-2 - t3 * (1.0148e-2 - t3 * 1.7138e-3))
    u0 = torch.where(t > 8800, 2.171 + 2.54e-4 * t, u0)
    u0 = torch.where(t > 1.8e4, 11.396 + t * (-1.7139e-3 + t * 8.633e-8), u0)
    u1 = 8.060 + 1.420e-4 * t
    u1 = torch.where(t > 3.3e4, 26.793 + t * (-1.8931e-3 + t * 4.4612e-8), u1)
    u2 = 5.9835 + t * (-2.6651e-5 + t * 1.8228e-9)
    u2 = torch.where(t < 7310.5, 5.89, u2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 8  O
# ---------------------------------------------------------------------------
@_register(8)
def _u_o(t):
    u0 = 8.29 + 1.10e-4 * t
    u0 = torch.where(t > 1.9e4, 66.81 + t * (-6.019e-3 + t * 1.657e-7), u0)
    u1 = torch.clamp(3.51 + 8e-5 * t, min=4.0)
    u1 = torch.where(t > 3.64e4, 68.7 + t * (-4.216e-3 + t * 6.885e-8), u1)
    return u0, u1, 7.865 + 1.1348e-4 * t


# ---------------------------------------------------------------------------
# 9  F
# ---------------------------------------------------------------------------
@_register(9)
def _u_f(t):
    t3 = 1e-3 * t
    u0 = 4.5832 + t3 * (0.77683 + t3 * (-0.20884 + t3 * (2.6771e-2 - 1.3035e-3 * t3)))
    u0 = torch.where(t > 8750, 5.9, u0)
    u0 = torch.where(t > 2e4, 15.16 + t * (-9.229e-4 + t * 2.312e-8), u0)
    return u0, 8.15 + 8.9e-5 * t, 2.315 + 1.38e-4 * t


# ---------------------------------------------------------------------------
# 10 Ne
# ---------------------------------------------------------------------------
@_register(10)
def _u_ne(t):
    u0 = torch.ones_like(t)
    u0 = torch.where(t < 2.69e4, 26.3 + t * (-2.113e-3 + t * 4.359e-8), u0)
    return u0, 5.4 + 4e-5 * t, 7.973 + 7.956e-5 * t


# ---------------------------------------------------------------------------
# 11 Na
# ---------------------------------------------------------------------------
@_register(11)
def _u_na(t):
    u0 = torch.clamp(1.72 + 9.3e-5 * t, min=2.0)
    u0 = torch.where(t > 5400, -0.83 + 5.66e-4 * t, u0)
    u0 = torch.where(t > 8.5e3, 4.5568 + t * (-1.2415e-3 + t * 1.3861e-7), u0)
    return u0, torch.ones_like(t), 5.69 + 5.69e-6 * t


# ---------------------------------------------------------------------------
# 12 Mg
# ---------------------------------------------------------------------------
@_register(12)
def _u_mg(t):
    x = torch.log(5040 / t)
    u0 = 1.0 + torch.exp(-4.027262 - x * (6.173172 + x * (2.889176 + x * (2.393895 + 0.784131 * x))))
    u0 = torch.where(t > 8e3, 2.757 + t * (-7.8909e-4 + t * 7.4531e-8), u0)
    u1 = 2.0 + torch.exp(-7.721172 - x * (7.600678 + x * (1.966097 + 0.212417 * x)))
    u1 = torch.where(t > 2e4, 7.1041 + t * (-1.0817e-3 + t * 4.7841e-8), u1)
    return u0, u1, torch.ones_like(t)


# ---------------------------------------------------------------------------
# 13 Al
# ---------------------------------------------------------------------------
@_register(13)
def _u_al(t):
    t3 = 1e-3 * t
    u0 = 5.2955 + t3 * (0.27833 - t3 * (4.7529e-2 - t3 * 3.0199e-3))
    u1 = torch.clamp(0.725 + 3.245e-5 * t, min=1.0)
    u1 = torch.where(t > 2.24e4, 61.06 + t * (-5.987e-3 + t * 1.485e-7), u1)
    u2 = torch.clamp(1.976 + 3.43e-6 * t, min=2.0)
    u2 = torch.where(t > 1.814e4, 3.522 + t * (-1.59e-4 + t * 4.382e-9), u2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 14 Si
# ---------------------------------------------------------------------------
@_register(14)
def _u_si(t):
    t3 = 1e-3 * t
    u0 = 6.7868 + t3 * (0.86319 + t3 * (-0.11622 + t3 * (0.013109 - 6.2013e-4 * t3)))
    u0 = torch.where(t > 1.04e4, 86.01 + t * (-1.465e-2 + t * 7.282e-7), u0)
    u1 = 5.470 + 4e-5 * t
    u1 = torch.where(t > 1.8e4, 26.44 + t * (-2.22e-3 + t * 6.188e-8), u1)
    u2 = torch.clamp(0.911 + 1.1e-5 * t, min=1.0)
    u2 = torch.where(t > 3.3e4, 19.14 + t * (-1.408e-3 + t * 2.617e-8), u2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 15 P
# ---------------------------------------------------------------------------
@_register(15)
def _u_p(t):
    t3 = 1e-3 * t
    u0 = 4.2251 + t3 * (-0.22476 + t3 * (0.057306 - t3 * 1.0381e-3))
    u0 = torch.where(t > 6e3, 1.56 + 5.2e-4 * t, u0)
    u1 = 4.4151 + t3 * (2.2494 + t3 * (-0.55371 + t3 * (0.071913 - t3 * 3.5156e-3)))
    u1 = torch.where(t > 7250, 4.62 + 5.38e-4 * t, u1)
    return u0, u1, 5.595 + 3.4e-5 * t


# ---------------------------------------------------------------------------
# 16 S
# ---------------------------------------------------------------------------
@_register(16)
def _u_s(t):
    u0 = 7.5 + 2.15e-4 * t
    u0 = torch.where(t > 1.16e4, 38.76 + t * (-4.906e-3 + t * 2.125e-7), u0)
    u1 = 2.845 + 2.43e-4 * t
    u1 = torch.where(t > 1.05e4, 6.406 + t * (-1.68e-4 + t * 1.323e-8), u1)
    return u0, u1, 7.38 + 1.88e-4 * t


# ---------------------------------------------------------------------------
# 17 Cl
# ---------------------------------------------------------------------------
@_register(17)
def _u_cl(t):
    u0 = 5.2 + 6e-5 * t
    u0 = torch.where(t > 1.84e4, -81.6 + 4.8e-3 * t, u0)
    return u0, 7.0 + 2.43e-4 * t, 2.2 + 2.62e-4 * t


# ---------------------------------------------------------------------------
# 18 Ar
# ---------------------------------------------------------------------------
@_register(18)
def _u_ar(t):
    return torch.ones_like(t), 5.20 + 3.8e-5 * t, 7.474 + 1.554e-4 * t


# ---------------------------------------------------------------------------
# 19 K
# ---------------------------------------------------------------------------
@_register(19)
def _u_k(t):
    t3 = 1e-3 * t
    u0 = 1.9909 + t3 * (0.023169 - t3 * (0.017432 - t3 * 4.0938e-3))
    u0 = torch.where(t > 5800, -9.93 + 2.124e-3 * t, u0)
    return u0, torch.ones_like(t), 5.304 + 1.93e-5 * t


# ---------------------------------------------------------------------------
# 20 Ca
# ---------------------------------------------------------------------------
@_register(20)
def _u_ca(t):
    x = torch.log(5040 / t)
    u0 = 1.0 + torch.exp(-1.731273 - x * (5.004556 + x * (1.645456 + x * (1.326861 + 0.508553 * x))))
    u1 = 2.0 + torch.exp(-1.582112 - x * (3.996089 + x * (1.890737 + 0.539672 * x)))
    return u0, u1, torch.ones_like(t)


# ---------------------------------------------------------------------------
# 21 Sc
# ---------------------------------------------------------------------------
@_register(21)
def _u_sc(t):
    x = torch.log(5040 / t)
    u0 = 4.0 + torch.exp(2.071563 + x * (-1.2392 + x * (1.173504 + 0.517796 * x)))
    u1 = 3.0 + torch.exp(2.988362 + x * (-0.596238 + 0.054658 * x))
    return u0, u1, torch.full_like(t, 10.0)


# ---------------------------------------------------------------------------
# 22 Ti
# ---------------------------------------------------------------------------
@_register(22)
def _u_ti(t):
    x = torch.log(5040 / t)
    u0 = 5.0 + torch.exp(3.200453 + x * (-1.227798 + x * (0.799613 + 0.278963 * x)))
    u0 = torch.where(t < 5.5e3, 16.37 + t * (-2.838e-4 + t * 5.819e-7), u0)
    u1 = 4.0 + torch.exp(3.94529 + x * (-0.551431 + 0.115693 * x))
    return u0, u1, 16.4 + 8.5e-4 * t


# ---------------------------------------------------------------------------
# 23 V
# ---------------------------------------------------------------------------
@_register(23)
def _u_v(t):
    x = torch.log(5040 / t)
    u0 = 4.0 + torch.exp(3.769611 + x * (-0.906352 + x * (0.724694 + 0.1622 * x)))
    u1 = 1.0 + torch.exp(3.755917 + x * (-0.757371 + 0.21043 * x))
    u2 = -18.0 + 1.03e-2 * t
    u2 = torch.where(t < 2.25e3, 2.4e-3 * t, u2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 24 Cr
# ---------------------------------------------------------------------------
@_register(24)
def _u_cr(t):
    x = torch.log(5040 / t)
    u0 = 7.0 + torch.exp(1.225042 + x * (-2.923459 + x * (0.154709 + 0.09527 * x)))
    u1 = 6.0 + torch.exp(0.128752 - x * (4.143973 + x * (1.096548 + 0.230073 * x)))
    return u0, u1, 10.4 + 2.1e-3 * t


# ---------------------------------------------------------------------------
# 25 Mn
# ---------------------------------------------------------------------------
@_register(25)
def _u_mn(t):
    x = torch.log(5040 / t)
    u0 = 6.0 + torch.exp(-0.86963 - x * (5.531252 + x * (2.13632 + x * (1.061055 + 0.265557 * x))))
    u1 = 7.0 + torch.exp(-0.282961 - x * (3.77279 + x * (0.814675 + 0.159822 * x)))
    return u0, u1, torch.full_like(t, 10.0)


# ---------------------------------------------------------------------------
# 26 Fe
# ---------------------------------------------------------------------------
@_register(26)
def _u_fe(t):
    x = torch.log(5040 / t)
    u0 = 9.0 + torch.exp(2.930047 + x * (-0.979745 + x * (0.76027 + 0.118218 * x)))
    u0 = torch.where(t < 4e3, 15.85 + t * (1.306e-3 + t * 2.04e-7), u0)
    u0 = torch.where(t > 9e3, 39.149 + t * (-9.5922e-3 + t * 1.2477e-6), u0)
    u1 = 10.0 + torch.exp(3.501597 + x * (-0.612094 + 0.280982 * x))
    u1 = torch.where(t > 1.8e4, 68.356 + t * (-6.1104e-3 + t * 5.1567e-7), u1)
    u2 = 17.336 + t * (5.5048e-4 + t * 5.7514e-8)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 27 Co
# ---------------------------------------------------------------------------
@_register(27)
def _u_co(t):
    return 8.65 + 4.9e-3 * t, 11.2 + 3.58e-3 * t, 15.0 + 1.42e-3 * t


# ---------------------------------------------------------------------------
# 28 Ni
# ---------------------------------------------------------------------------
@_register(28)
def _u_ni(t):
    x = torch.log(5040 / t)
    u0 = 9.0 + torch.exp(3.084552 + x * (-0.401323 + x * (0.077498 - 0.278468 * x)))
    u1 = 6.0 + torch.exp(1.593047 - x * (1.528966 + 0.115654 * x))
    return u0, u1, 13.3 + 6.9e-4 * t


# ---------------------------------------------------------------------------
# 29 Cu
# ---------------------------------------------------------------------------
@_register(29)
def _u_cu(t):
    u0 = torch.clamp(1.50 + 1.51e-4 * t, min=2.0)
    u0 = torch.where(t > 6250, -0.3 + 4.58e-4 * t, u0)
    u1 = torch.clamp(-0.3 + 4.58e-4 * t, min=1.0)
    return u0, u1, 8.025 + 9.4e-5 * t


# ---------------------------------------------------------------------------
# 30 Zn
# ---------------------------------------------------------------------------
@_register(30)
def _u_zn(t):
    return torch.clamp(0.632 + 5.11e-5 * t, min=1.0), \
        torch.full_like(t, 2.0), torch.ones_like(t)


# ---------------------------------------------------------------------------
# 31 Ga ... 92 U : default minimal fits; only elements used in the line
# list or in the pressure balance matter for the physics, the rest fall
# back to the statistical weight 2J+1.
# ---------------------------------------------------------------------------
@_register(31)
def _u_ga(t):
    t3 = 1e-3 * t
    u0 = 1.7931 + t3 * (1.9338 + t3 * (-0.4643 + t3 * (0.054876 - t3 * 2.5054e-3)))
    u0 = torch.where(t > 6e3, 4.18 + 2.03e-4 * t, u0)
    return u0, torch.ones_like(t), torch.full_like(t, 2.0)


@_register(32)
def _u_ge(t):
    return 6.12 + 4.08e-4 * t, 3.445 + 1.78e-4 * t, torch.full_like(t, 1.1)


@_register(33)
def _u_as(t):
    u0 = 2.65 + 3.65e-4 * t
    t3 = 1e-3 * t
    u1 = -0.25384 + t3 * (2.284 + t3 * (-0.33383 + t3 * (0.030408 - t3 * 1.1609e-3)))
    u1 = torch.where(t > 1.2e4, 8.0, u1)
    return u0, u1, torch.full_like(t, 8.0)


@_register(34)
def _u_se(t):
    t3 = 1e-3 * t
    u1 = 4.1786 + t3 * (-0.15392 + t3 * 3.2053e-2)
    return 6.34 + 1.71e-4 * t, u1, torch.full_like(t, 8.0)


@_register(35)
def _u_br(t):
    return 4.12 + 1.12e-4 * t, 5.22 + 3.08e-4 * t, 2.3 + 2.86e-4 * t


@_register(36)
def _u_kr(t):
    return torch.ones_like(t), 4.11 + 7.4e-5 * t, 5.35 + 2.23e-4 * t


@_register(37)
def _u_rb(t):
    u0 = torch.clamp(1.38 + 1.94e-4 * t, min=2.0)
    u0 = torch.where(t > 6250, -14.9 + 2.79e-3 * t, u0)
    return u0, torch.ones_like(t), 4.207 + 4.85e-5 * t


@_register(38)
def _u_sr(t):
    t3 = 1e-3 * t
    u0 = 0.87127 + t3 * (0.20148 + t3 * (-0.10746 + t3 * (0.021424 - t3 * 1.0231e-3)))
    u0 = torch.where(t > 6500, -6.12 + 1.224e-3 * t, u0)
    u1 = torch.clamp(0.84 + 2.6e-4 * t, min=2.0)
    return u0, u1, torch.ones_like(t)


@_register(39)
def _u_y(t):
    return 0.2 + 2.58e-3 * t, 7.15 + 1.855e-3 * t, 9.71 + 9.9e-5 * t


@_register(40)
def _u_zr(t):
    x = torch.log(5040 / t)
    u0 = 76.31 + t * (-1.866e-2 + t * 2.199e-6)
    u0 = torch.where(t > 6236, 6.8 + t * (2.806e-3 + t * 5.386e-7), u0)
    u1 = 4.0 + torch.exp(3.721329 - 0.906502 * x)
    return u0, u1, 12.3 + 1.385e-3 * t


@_register(41)
def _u_nb(t):
    return torch.clamp(-19.0 + 1.43e-2 * t, min=1.0), -4.0 + 1.015e-2 * t, \
        torch.full_like(t, 25.0)


@_register(42)
def _u_mo(t):
    u0 = torch.clamp(2.1 + 1.5e-3 * t, min=7.0)
    u0 = torch.where(t > 7e3, -38.1 + 7.28e-3 * t, u0)
    u1 = 1.25 + 1.17e-3 * t
    u1 = torch.where(t > 6900, -28.5 + 5.48e-3 * t, u1)
    return u0, u1, 24.04 + 1.464e-4 * t


@_register(43)
def _u_tc(t):
    t3 = 1e-3 * t
    u0 = 4.439 + t3 * (0.30648 + t3 * (1.6525 + t3 * (-0.4078 + t3 * (0.048401 - t3 * 2.1538e-3))))
    u0 = torch.where(t > 6e3, 24.0, u0)
    u1 = 8.1096 + t3 * (-2.963 + t3 * (2.369 + t3 * (-0.502 + t3 * (0.049656 - t3 * 1.9087e-3))))
    u1 = torch.where(t > 6e3, 17.0, u1)
    return u0, u1, torch.full_like(t, 220.0)


@_register(44)
def _u_ru(t):
    return -3.0 + 7.17e-3 * t, 3.0 + 4.26e-3 * t, torch.full_like(t, 22.0)


@_register(45)
def _u_rh(t):
    t3 = 1e-3 * t
    u0 = 6.9164 + t3 * (3.8468 + t3 * (0.043125 - t3 * (8.7907e-3 - t3 * 5.9589e-4)))
    u1 = 7.2902 + t3 * (1.7476 + t3 * (-0.038257 + t3 * (2.014e-3 + t3 * 2.1218e-4)))
    return u0, u1, torch.full_like(t, 30.0)


@_register(46)
def _u_pd(t):
    return torch.clamp(12.6 + 1.26e-3 * t, min=1.0), 5.60 + 3.62e-4 * t, \
        torch.full_like(t, 20.0)


@_register(47)
def _u_ag(t):
    return torch.clamp(1.537 + 7.88e-5 * t, min=2.0), \
        torch.clamp(0.73 + 3.4e-5 * t, min=1.0), 6.773 + 1.248e-4 * t


@_register(48)
def _u_cd(t):
    return torch.clamp(0.43 + 7.6e-5 * t, min=1.0), torch.full_like(t, 2.0), \
        torch.ones_like(t)


@_register(49)
def _u_in(t):
    return 2.16 + 3.92e-4 * t, torch.ones_like(t), torch.full_like(t, 2.0)


@_register(50)
def _u_sn(t):
    return 2.14 + 6.16e-4 * t, 2.06 + 2.27e-4 * t, torch.full_like(t, 1.05)


@_register(51)
def _u_sb(t):
    return 2.34 + 4.86e-4 * t, 0.69 + 5.36e-4 * t, torch.full_like(t, 3.5)


@_register(52)
def _u_te(t):
    t3 = 1e-3 * t
    u1 = 4.2555 + t3 * (-0.25894 + t3 * (0.06939 - t3 * 2.4271e-3))
    u1 = torch.where(t > 1.2e4, 7.0, u1)
    return 3.948 + 4.56e-4 * t, u1, torch.full_like(t, 5.0)


@_register(53)
def _u_i(t):
    return torch.clamp(3.8 + 9.5e-5 * t, min=4.0), 4.12 + 3e-4 * t, \
        torch.full_like(t, 7.0)


@_register(54)
def _u_xe(t):
    return torch.ones_like(t), 3.75 + 6.876e-5 * t, 4.121 + 2.323e-4 * t


@_register(55)
def _u_cs(t):
    u0 = torch.clamp(1.56 + 1.67e-4 * t, min=2.0)
    u0 = torch.where(t > 4850, -2.680 + 1.04e-3 * t, u0)
    return u0, torch.ones_like(t), 3.769 + 4.971e-5 * t


@_register(56)
def _u_ba(t):
    u0 = torch.clamp(-1.8 + 9.85e-4 * t, min=1.0)
    u0 = torch.where(t > 6850, -16.2 + 3.08e-3 * t, u0)
    return u0, 1.11 + 5.94e-4 * t, torch.ones_like(t)


@_register(57)
def _u_la(t):
    u0 = 15.42 + 9.5e-4 * t
    u0 = torch.where(t > 5060, 1.0 + 3.8e-3 * t, u0)
    return u0, 13.2 + 3.56e-3 * t, torch.full_like(t, 12.0)


@_register(58)
def _u_ce(t):
    x = torch.log(5040 / t)
    u0 = 9.0 + torch.exp(5.202903 + x * (-1.98399 + x * (0.119673 + 0.179675 * x)))
    u1 = 8.0 + torch.exp(5.634882 - x * (1.459196 + x * (0.310515 + 0.052221 * x)))
    return u0, u1, 9.0 + torch.exp(3.629123 - x * (1.340945 + x * (0.372409 + x * (0.03186 - 0.014676 * x))))


@_register(59)
def _u_pr(t):
    x = torch.log(5040 / t)
    u1 = 9.0 + torch.exp(4.32396 - x * (1.191467 + x * (0.149498 + 0.028999 * x)))
    u2 = 10.0 + torch.exp(3.206855 + x * (-1.614554 + x * (0.489574 + 0.277916 * x)))
    return u1, u1, u2


@_register(60)
def _u_nd(t):
    x = torch.log(5040 / t)
    u0 = 9.0 + torch.exp(4.456882 + x * (-2.779176 + x * (0.082258 + x * (0.50666 + 0.127326 * x))))
    u1 = 8.0 + torch.exp(4.689643 + x * (-2.039946 + x * (0.17193 + x * (0.26392 + 0.038225 * x))))
    return u0, u1, u1


@_register(61)
def _u_pm(t):
    return torch.full_like(t, 20.0), torch.full_like(t, 25.0), torch.full_like(t, 100.0)


@_register(62)
def _u_sm(t):
    x = torch.log(5040 / t)
    u0 = 1.0 + torch.exp(3.549595 + x * (-1.851549 + x * (0.9964 + 0.566263 * x)))
    u1 = 2.0 + torch.exp(4.052404 + x * (-1.418222 + x * (0.358695 + 0.161944 * x)))
    u2 = 1.0 + torch.exp(3.222807 - x * (0.699473 + x * (-0.056205 + x * (0.533833 + 0.251011 * x))))
    return u0, u1, u2


@_register(63)
def _u_eu(t):
    x = torch.log(5040 / t)
    u0 = 8.0 + torch.exp(1.024374 - x * (4.533653 + x * (1.540805 + x * (0.827789 + 0.286737 * x))))
    u1 = 9.0 + torch.exp(1.92776 + x * (-1.50646 + x * (0.379584 + 0.05684 * x)))
    return u0, u1, torch.full_like(t, 8.0)


@_register(64)
def _u_gd(t):
    x = torch.log(5040 / t)
    u0 = 5.0 + torch.exp(4.009587 + x * (-1.583513 + x * (0.800411 + 0.388845 * x)))
    u1 = 6.0 + torch.exp(4.362107 - x * (1.208124 + x * (-0.074813 + x * (0.076453 + 0.055475 * x))))
    u2 = 5.0 + torch.exp(3.412951 - x * (0.50271 + x * (0.042489 - 4.017e-3 * x)))
    return u0, u1, u2


@_register(65)
def _u_tb(t):
    x = torch.log(5040 / t)
    u0 = 16.0 + torch.exp(4.791661 + x * (-1.249355 + x * (0.570094 + 0.240203 * x)))
    u1 = 15.0 + torch.exp(4.472549 - x * (0.295965 + x * (5.88e-3 + 0.131631 * x)))
    return u0, u1, u1


@_register(66)
def _u_dy(t):
    x = torch.log(5040 / t)
    u0 = 17.0 + torch.exp(3.029646 - x * (3.121036 + x * (0.086671 - 0.216214 * x)))
    u1 = 18.0 + torch.exp(3.465323 - x * (1.27062 + x * (-0.382265 + x * (0.431447 + 0.303575 * x))))
    return u0, u1, u1


@_register(67)
def _u_ho(t):
    x = torch.log(5040 / t)
    u2 = 16.0 + torch.exp(1.610084 - x * (2.373926 + x * (0.133139 - 0.071196 * x)))
    return u2, u2, u2


@_register(68)
def _u_er(t):
    x = torch.log(5040 / t)
    u0 = 13.0 + torch.exp(2.895648 - x * (2.968603 + x * (0.561515 + x * (0.215267 + 0.095813 * x))))
    u1 = 14.0 + torch.exp(3.202542 - x * (0.852209 + x * (-0.226622 + x * (0.343738 + 0.186042 * x))))
    return u0, u1, u1


@_register(69)
def _u_tm(t):
    x = torch.log(5040 / t)
    u0 = 8.0 + torch.exp(1.021172 - x * (4.94757 + x * (1.081603 + 0.034811 * x)))
    u1 = 9.0 + torch.exp(2.173152 + x * (-1.295327 + x * (1.940395 + 0.813303 * x)))
    u2 = 8.0 + torch.exp(-0.567398 + x * (-3.383369 + x * (0.799911 + 0.554397 * x)))
    return u0, u1, u2


@_register(70)
def _u_yb(t):
    x = torch.log(5040 / t)
    u0 = 1.0 + torch.exp(-2.350549 - x * (6.688837 + x * (1.93869 + 0.269237 * x)))
    u1 = 2.0 + torch.exp(-3.047465 - x * (7.390444 + x * (2.355267 + 0.44757 * x)))
    u2 = 1.0 + torch.exp(-6.192056 - x * (10.560552 + x * (4.579385 + 0.940171 * x)))
    return u0, u1, u2


@_register(71)
def _u_lu(t):
    x = torch.log(5040 / t)
    u0 = 4.0 + torch.exp(1.537094 + x * (-1.140264 + x * (0.608536 + 0.193362 * x)))
    u1 = torch.clamp(0.66 + 1.52e-4 * t, min=1.0)
    u1 = torch.where(t > 5250, -1.09 + 4.86e-4 * t, u1)
    return u0, u1, torch.full_like(t, 5.0)


@_register(72)
def _u_hf(t):
    t3 = 1e-3 * t
    u0 = 4.1758 + t3 * (0.407 + t3 * (0.57862 - t3 * (0.072887 - t3 * 3.6848e-3)))
    return u0, -2.979 + 3.095e-3 * t, torch.full_like(t, 30.0)


@_register(73)
def _u_ta(t):
    t3 = 1e-3 * t
    u0 = 3.0679 + t3 * (0.81776 + t3 * (0.34936 + t3 * (7.4861e-3 + t3 * 3.0739e-4)))
    u1 = 1.6834 + t3 * (2.0103 + t3 * (0.56443 - t3 * (0.031036 - t3 * 8.9565e-4)))
    return u0, u1, torch.full_like(t, 15.0)


@_register(74)
def _u_w(t):
    t3 = 1e-3 * t
    u0 = 0.3951 + t3 * (-0.25057 + t3 * (1.4433 + t3 * (-0.34373 + t3 * (0.041924 - t3 * 1.84e-3))))
    u0 = torch.where(t > 1.2e4, 23.0, u0)
    u1 = 1.055 + t3 * (1.0396 + t3 * (0.3303 - t3 * (8.4971e-3 - t3 * 5.5794e-4)))
    return u0, u1, torch.full_like(t, 20.0)


@_register(75)
def _u_re(t):
    t3 = 1e-3 * t
    u0 = 5.5671 + t3 * (0.72721 + t3 * (-0.42096 + t3 * (0.09075 - t3 * 3.9331e-3)))
    u0 = torch.where(t > 1.2e4, 29.0, u0)
    u1 = 6.5699 + t3 * (0.59999 + t3 * (-0.28532 + t3 * (0.050724 - t3 * 1.8544e-3)))
    u1 = torch.where(t > 1.2e4, 22.0, u1)
    return u0, u1, torch.full_like(t, 20.0)


@_register(76)
def _u_os(t):
    t3 = 1e-3 * t
    u0 = 8.6643 + t3 * (-0.32516 + t3 * (0.68181 - t3 * (0.044252 - t3 * 1.9975e-3)))
    u1 = 9.7086 + t3 * (-0.3814 + t3 * (0.65292 - t3 * (0.064984 - t3 * 2.8792e-3)))
    return u0, u1, torch.full_like(t, 10.0)


@_register(77)
def _u_ir(t):
    t3 = 1e-3 * t
    u0 = 11.07 + t3 * (-2.412 + t3 * (1.9388 + t3 * (-0.34389 + t3 * (0.033511 - 1.3376e-3 * t3))))
    u0 = torch.where(t > 1.2e4, 30.0, u0)
    return u0, torch.full_like(t, 15.0), torch.full_like(t, 20.0)


@_register(78)
def _u_pt(t):
    t3 = 1e-3 * t
    u1 = 6.5712 + t3 * (-1.0363 + t3 * (0.57234 - t3 * (0.061219 - 2.6878e-3 * t3)))
    return 16.4 + 1.27e-3 * t, u1, torch.full_like(t, 15.0)


@_register(79)
def _u_au(t):
    t3 = 1e-3 * t
    u1 = 1.0546 + t3 * (-0.040809 + t3 * (2.8439e-3 + t3 * 1.6586e-3))
    return 1.24 + 2.79e-4 * t, u1, torch.full_like(t, 7.0)


@_register(80)
def _u_hg(t):
    return torch.ones_like(t), torch.full_like(t, 2.0), \
        torch.clamp(0.669 + 3.976e-5 * t, min=1.0)


@_register(81)
def _u_tl(t):
    return torch.clamp(0.63 + 3.35e-4 * t, min=2.0), torch.ones_like(t), \
        torch.full_like(t, 2.0)


@_register(82)
def _u_pb(t):
    u0 = torch.clamp(0.42 + 2.35e-4 * t, min=1.0)
    u0 = torch.where(t > 6125, -1.2 + 5e-4 * t, u0)
    u1 = torch.clamp(1.72 + 7.9e-5 * t, min=2.0)
    return u0, u1, torch.ones_like(t)


@_register(83)
def _u_bi(t):
    return 2.78 + 2.87e-4 * t, torch.clamp(0.37 + 1.41e-4 * t, min=1.0), \
        torch.full_like(t, 2.5)


def _fallback_u(t):
    """Default partition function for elements without a fit."""
    return torch.ones_like(t), torch.ones_like(t), torch.ones_like(t)


def partition_functions(atomic_number, temperature):
    """
    Partition functions of an element.

    Parameters
    ----------
    atomic_number : int
        Atomic number (1..92).
    temperature : torch.Tensor
        Temperature(s) [K], arbitrary shape.

    Returns
    -------
    u0, u1, u2 : torch.Tensor
        Partition functions of the neutral, singly-ionized and
        doubly-ionized species.
    """
    handler = _HANDLERS.get(int(atomic_number), _fallback_u)
    return handler(temperature)
