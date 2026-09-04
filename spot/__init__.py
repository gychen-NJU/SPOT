# -*- coding: utf-8 -*-
"""
spot — Stokes spectral synthesis and inversion in PyTorch
============================================================

spot is an integrated package for synthesizing and inverting polarized
(Stokes IQUV) spectral lines formed in LTE atmospheres, re-implementing
the reference physics in pure Python / PyTorch:

* fully vectorized: a batch of ``Nb`` atmospheres is synthesized in a
  single call with tensor broadcasting (no Python loop over the batch),
  on CPU or CUDA;
* batched inversion with a node-based Levenberg–Marquardt scheme and
  SVD (following a cycle / node philosophy that refines the model with
  an increasing number of free parameters);
* response functions ``dI/dx`` returned on demand (autograd or finite
  differences);
* a single user-facing configuration dictionary that is recursively
  merged over the built-in defaults (``spot.default``);
* packaged preset atmosphere models and the reference abundance table as
  CSV data files (resolved through ``importlib.resources``).

The physics lives under ``spot.physics``; ``spot.net`` provides the
neural-operator subpackage with data-driven alternatives for the
synthesis operator.

Author : Guoyin Chen
Email  : gychen@smail.nju.edu.cn
"""

from .config import merge_config, update_config
from .default import DEFAULT_CONFIG

__version__ = "1.1.0"
__author__ = "Guoyin Chen"
__email__ = "gychen@smail.nju.edu.cn"

__all__ = [
    "DEFAULT_CONFIG",
    "merge_config",
    "update_config",
    "Synthesis",
    "Inversion",
    "CmaesInversion",
    "load_atmosphere",
    "load_lines",
    "load_abundance",
    "plot_convergence",
]

# Lazy imports keep `import spot` light (torch is only needed when the
# classes are actually instantiated).
def __getattr__(name):
    if name == "Synthesis":
        from .synthesis.synthesis import Synthesis
        return Synthesis
    if name == "Inversion":
        from .inversion.inversion import Inversion
        return Inversion
    if name == "CmaesInversion":
        from .inversion.cmaes import CmaesInversion
        return CmaesInversion
    if name == "plot_convergence":
        from .visualization import plot_convergence
        return plot_convergence
    if name in ("load_atmosphere", "load_lines", "load_abundance"):
        from .utils.data_io import (load_abundance, load_atmosphere, load_lines)
        return {"load_atmosphere": load_atmosphere,
                "load_lines": load_lines,
                "load_abundance": load_abundance}[name]
    raise AttributeError(f"module 'spot' has no attribute {name!r}")
