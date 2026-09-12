# Changelog

## v1.2.0 — 2026-09-12

**New bundled network model (`hinode_sp` preset).**

* The pretrained `spot.net` model is replaced by **v3**, trained **from scratch**
  (random initialisation) on 2,361,889 BIFROST `en024048_hion` 1-D atmospheres
  (2,300,000 train / 61,889 held-out test) plus 519,466 Hinode SP profiles whose
  atmospheres come from a SIR inversion of the 2017 full map, forward-synthesised
  with SPOT itself. 8,554,249 parameters, 50 epochs in three cosine-annealing cycles
  (peak lr 5e-4 → 2e-4 → 1e-4), batch 1152, 27 h on one Tesla T4; best validation
  loss 0.69899.
* Every physical parameter improves substantially on the 20,000-profile hold-out
  (MAE, new vs previous model): t 128.7 vs 587.5 K, p 230.0 vs 1179 dyn/cm²,
  b 32.97 vs 141.5 G, g 10.50 vs 24.18°, f 30.47 vs 43.40° (azimuth r²_corr 0.26 vs
  0.00 — the previous model never learned the azimuth), v 3.10e4 vs 1.45e5,
  m 1.21e4 vs 4.72e5, M 7.16e3 vs 3.62e5 cm/s. End-to-end (invert → forward-synthesise
  → compare) median Stokes I residual 6.84e-03 vs 1.47e-01.
* **No API change**: `StokesInference(preset="hinode_sp")` and the default inversion
  flow (`network_preset="hinode_sp"`, `atmosphere="auto"`) now load the new weights;
  the input convention (physical intensity divided by I_c,ref = 8.257986e14) and the
  depth convention (index 0 = shallowest, log τ5000 = −4) are unchanged, so existing
  scripts work as before.
* Model card: `spot/net/models/hinode_sp/MODEL_CARD.md` (training data, metrics,
  provenance, limitations — notably a systematic **under-estimate of |B| in strong
  umbrae**, B ≳ 1.5 kG, inherited from the training sample's median field of 15 G).
* The previous fine-tuned checkpoint (md5 `76e898d8c898f780eb48aa8b64242c0c`) is
  **retained but deprecated** under `spot/net/models/deprecated/hinode_sp_v1/`: no
  preset points there, no API exposes it, and it is excluded from the wheel
  (`setup.py` `package_data`). It stays in the repository for provenance only.
* `spot/net/models/hinode_sp/*.md` is now shipped with the package.

## v1.1.1

* Version bump and release housekeeping (see the repository history for details).

## v1.1.0 and earlier

* Integrated Stokes synthesis + inversion library: forward synthesis and response
  functions, node-based Levenberg-Marquardt and CMA-ES inversion, the `spot.net`
  neural-operator subpackage with a bundled Hinode SP model, and the demo suite.
