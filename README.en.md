<div align="center">

# SPOT

**Stokes Profile Optimization Toolkit** — a PyTorch-based library for Stokes IQUV spectral synthesis and inversion (with a neural-operator subpackage)

[English](README.en.md) | [简体中文](README.md)

</div>

### Overview

`spot` is a Python library for solar Stokes (polarized) spectral synthesis
and inversion. It implements the LTE line-formation physics (opacity, Zeeman
line profiles, radiative-transfer solvers, etc.) in pure Python / PyTorch
(NumPy), and bundles several inversion strategies — node-parameterized
Levenberg–Marquardt, derivative-free CMA-ES, and a pretrained
neural-operator network — on CPU or CUDA.

Capabilities:

1. **Forward synthesis** of Stokes I, Q, U, V for arbitrary atmosphere
   models (packed presets or custom CSV), line lists and wavelength grids,
   vectorized over batches on CPU / CUDA (PyTorch);
2. **Response functions** dI/dx with several methods (autograd, finite
   differences, analytic chain rule); the default `fast` method is exact
   and efficient;
3. **Inversion**: node-parametrized atmosphere + Levenberg–Marquardt
   (damped SVD) multi-cycle inversion; plus **CMA-ES** (derivative-free)
   batched inversion for initial-guess search;
4. **Network inversion**: `spot.net.StokesInference` loads the pretrained
   neural operator (FNO + Transformer + DeepONet-style decoder) and
   inverts Hinode SP configured Stokes profiles in one call.

### Installation

```bash
cd SPOT
pip install -e .          # or pip install . (includes the spot.net pretrained model)
```

Dependencies: `numpy`, `torch>=2.0`, `matplotlib` (see `requirements.txt`).

### Quick start

#### 1. Forward synthesis (Hinode SP configuration)

```python
import numpy as np
import torch
from spot import Synthesis
from spot.utils.data_io import load_atmosphere

wavs = torch.tensor(6300.8840305 + 21.549e-3 * np.arange(112), dtype=torch.float64)

model = load_atmosphere("cool11")          # preset atmosphere (55 layers)
ltau = torch.tensor(model["ltau"], dtype=torch.float64)

nt = len(model["ltau"])
atmos = torch.tensor(np.concatenate(
    [model["T"], model["Pe"], model["B"], model["gamma"],
     model["phi"], model["vlos"], [np.mean(model["vmic"])], [2.0]])[None, :],
    dtype=torch.float64)                    # vmac = 2 km/s

syn = Synthesis({
    "device": "cuda", "dtype": "float64",
    "lines": [141, 142],                    # Fe I 6301.508 / 6302.499 A
    "synthesis": {"refractive_index": 1.0, "macroturbulence": True},
})
stokes = syn(wavs, ltau, atmos)             # (1, 112, 4): I, Q, U, V
```

#### 2. Response functions (default `fast`)

```python
stokes, rf = syn(wavs, ltau, atmos, return_rf=True)
print(rf.shape)   # (1, 112, 4, 6*Nt+2): dI/dx wrt the packed atmosphere
```

#### 3. Inversion (default recommended flow: network guess -> 4-cycle node LM)

```python
from spot import Inversion

# No inversion settings needed - the defaults are the recommended flow:
#   atmosphere='auto' (the spot.net hinode_sp network inverts the target
#   profiles first) + 4-cycle node schedule [2,3,4,auto] (T/B/gamma/phi/vlos)
#   + fast response functions + sigma='sir' (snr=1000) + weights 1:5:5:10
#   + no hse.  Hinode SP test case: chi2 ~ 0.05-0.06 in ~130 s
#   (vs ~2500 s / chi2 ~7e-2 for the traditional 4-cycle flow with a preset
#   initial guess and hse).
res = Inversion().invert(wavs, ltau, target)   # initial=None -> network guess
print(res.chi2, res.atmos)                     # (Nb,) chi2 + final atmosphere

# Traditional preset initial guess (e.g. hot11) or explicit initial:
res = Inversion().invert(wavs, ltau, target, initial="hot11")
```

For maximum accuracy see the variant matrix in demo/08_net_guess_search
(e.g. automatic nodes + hse=500: chi2 ~2e-2 in ~890 s).

#### 3b. Inversion (custom: CMA-ES warm start + LM with automatic node counts)

```python
from spot import CmaesInversion, Inversion

cfg = {...}   # same base configuration as the synthesis
cfg["inversion"] = {
    "nodes": {"T": 4, "Pe": 0, "B": 3, "gamma": 2, "phi": 2,
              "vlos": 3, "vmic": 1, "vmac": 1},   # coarse node layout
    "max_cycles": 1,
}
res_a = CmaesInversion(cfg).invert(wavs, ltau, target, initial="hot11")

cfg["inversion"] = {
    "nodes": {"T": "auto", "Pe": 0, "B": "auto", "gamma": "auto",
              "phi": "auto", "vlos": "auto", "vmic": 1, "vmac": 1},
    "max_cycles": 1, "max_iterations": 40,
}
res_b = Inversion(cfg).invert(wavs, ltau, target, initial=res_a.atmos)
```

#### 4. Network inversion (pretrained `hinode_sp`)

```python
from spot.net import StokesInference

infer = StokesInference(preset="hinode_sp", device="cuda")
out = infer.predict_numpy(stokes_phys)      # input must be physical intensity
# out: t,p,b,g,f,v (B,64) + m,M (B,); t [K], p [dyn/cm^2], b [G],
#      g [deg], f [deg] (-180,180], v/m/M [cm/s]
```

### Layout

```
spot/
├── __init__.py             # top-level exports (Synthesis, Inversion, CmaesInversion, ...)
├── config.py               # config merge/freeze utilities
├── default.py              # default configuration (units + all options)
├── visualization.py        # convergence plots
├── data/                   # preset atmospheres, line list, abundances, opacity tables
├── physics/                # atomic data, opacity, Zeeman line profiles, RTE solvers, thermodynamics, pressure...
├── synthesis/              # Synthesis: forward synthesis + response functions
├── inversion/              # Inversion (LM), CmaesInversion, marquardt, nodes
├── utils/                  # data_io (presets/files), interpolation
└── net/                    # neural-operator subpackage (StokesInference, training/eval...)
    └── models/hinode_sp/   # pretrained model (best_model.pt / config.json / norm_stats.pt)
```

### Key configuration (`spot.default.DEFAULT_CONFIG`)

* Units: wavelength [Angstrom], `ltau = log10(tau5000)`, T [K], Pe [dyn/cm^2],
  B [G], angles [deg], velocities [km/s];
* Atmosphere vector: `[T, Pe, B, gamma, phi, vlos (per layer), vmic (scalar),
  vmac (scalar)]`, `Nx = 6*Nt + 2`;
* Synthesis: `solver` (hermitian/cn/delo), `continuum_opacity`
  (mihalas/atlas/opacity_project), `macroturbulence`, `normalize_continuum`,
  `rf_method` (fast/autograd/analytic/analytic_chain/finite_diff);
* Inversion: `atmosphere='auto'` (default: the `spot.net` operator
  (`network_preset`) inverts the target profiles first for the initial
  guess) + `nodes` (nodes per quantity; list = one value per cycle,
  `"auto"` = automatic node counts), `max_cycles=4`, `max_iterations=80`,
  `sigma='sir'`+`snr=1000`+`stokes_weights=[1,5,5,10]`, `hse_pg0=0`
  (the default recommended flow, see quick start 3),
  `lambda0/lambda_factor` (LM damping), `svd_tolerance`;
* CMA-ES: `population`, `sigma0`, `scale`, `bounds`, `covariance_mode`,
  `stall`, `hse_refresh`.

### Data and presets

* Atmosphere presets: `hot11`, `cool11`, `falc11`, `falf11`, `hsra11`,
  `valc11`, `granulebbr`, ... (`spot/data/models/*.csv`);
* Line list: `spot/data/lines.csv` (includes Fe I 6301.508/6302.499);
* Abundance table: `thevenin` (`spot/data/abundance.csv`);
* Continuum opacity: `mihalas` (default) / `atlas` (ATLAS solar ODF) /
  `opacity_project`;
* Network preset: `spot.net` bundles `hinode_sp` (fine-tuned on Hinode SP
  Fe I 6301.5/6302.5 profiles).

### Demos and validation

The `demo/` directory contains end-to-end demos (forward synthesis, response
functions, node-based inversion with LM / CMA-ES, network inversion, and
combined flows), each producing figures and a detailed report under
`demo/reports/`. See `demo/README.md`.

### License

MIT (author Guoyin Chen, gychen@smail.nju.edu.cn).
