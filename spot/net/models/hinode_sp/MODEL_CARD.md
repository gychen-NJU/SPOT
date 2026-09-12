# Model card — `hinode_sp` preset (SPOT network v3, 50 epochs)

**Status: current / recommended.** This is the model loaded by
`StokesInference(preset="hinode_sp")` and by the default inversion flow
(`network_preset="hinode_sp"`, i.e. `atmosphere="auto"` in `Inversion`).

## What it is

A `StokesPHNO` neural operator (8,554,249 parameters) mapping continuum-normalised
Stokes IQUV profiles (112 wavelength points, Fe I 6301.5 / 6302.5 Å) to a 1-D
atmosphere: `t, p, b, g, f, v` on 64 optical-depth layers plus the scalars
`m` (vmic) and `M` (vmac). The azimuth is encoded as sin/cos (`out_dims={"f": 2}`).

* **Trained from scratch** (random initialisation) — it is *not* a fine-tune of the
  previously bundled checkpoint.
* Layer order: **index 0 = shallowest (log τ₅₀₀₀ = −4) … index 63 = deepest (+2)**,
  the same convention as `spot.net`'s `NET_LTAU` and as the training labels.
* Output units: `t` [K], `p` [dyn/cm²], `b` [G], `g`/`f` [deg] (`f` in (−180, 180]),
  `v`/`m`/`M` [cm/s].

## Training data (2,819,466 profiles)

| source | profiles | role |
|---|---|---|
| BIFROST `en024048_hion` 3-D MHD snapshots → 1-D columns after quality cuts | 2,361,889 | 2,300,000 train / 61,889 held-out test |
| Hinode SP 2017 full map, SIR-inverted atmospheres re-synthesised with SPOT | 519,466 | test |

Every input is a SPOT forward synthesis on a common 64-point log τ₅₀₀₀ ∈ [−4, 2] grid,
so input/label pairs are exactly self-consistent. Azimuth labels are folded to
[0, 180) because the forward model is invariant under χ → χ + 180.

## Training

AdamW (β = 0.9/0.95, wd = 1e-4) + cosine annealing with a 0.5-epoch warmup, fp16 AMP,
batch 1152, gradient clip 1.0, dropout 0.1. Three cosine cycles with peak learning
rates 5e-4 (10 epochs) → 2e-4 (→ 30) → 1e-4 (→ 50); 27 h in total on one Tesla T4.
Best validation loss **0.69899** (global epoch 49, BIFROST hold-out).

## Accuracy (20,000 held-out BIFROST profiles)

| | t [K] | p [dyn/cm²] | b [G] | g [°] | f [°] | v [cm/s] | m [cm/s] | M [cm/s] |
|---|---|---|---|---|---|---|---|---|
| **this model (50 ep)** | 128.7 | 230.0 | 32.97 | 10.50 | 30.47 | 3.10e4 | 1.21e4 | 7.16e3 |
| previously bundled model | 587.5 | 1179.1 | 141.5 | 24.18 | 43.40 | 1.45e5 | 4.72e5 | 3.62e5 |

End-to-end check (invert → forward-synthesise → compare with the input spectrum):
median Stokes I residual **6.84e-03** (previous model 1.47e-01).

## Input convention

`predict()` expects **physical intensity units** and divides by
I_c,ref = 8.25798607858631e14 automatically (the preset's `input_scale`); the network
then applies the continuum-normalisation statistics stored in `norm_stats.pt`.
If your profiles are already continuum-normalised (I_c = 1), pass `input_scale=None`
instead — the two paths agree to ~1e-5 relative.

## Provenance

* Training/evaluation code, dataset builders, plots and the full report live in the
  accompanying StokesInversion project: `Data/05_post_training/04_v3_from_scratch.md`
  (§4 10-epoch, §7 30-epoch, §8 50-epoch results and the five-way comparison).
* Checkpoint md5: `90b280ff78a32e055431a4fef0865e2d` (`config.json`
  `daf6a31789a665fd6de9f0b827945d99`, `norm_stats.pt` `037d6b4ce58c3f6bc1fb21bddcf5ef52`).
* The preset→weights wiring is verified by
  `Data/05_post_training/scripts/v3_14_verify_spot_preset.py` (prints the resolved
  paths/md5, checks the two loading paths agree, and re-checks the depth direction).

## Known limitations

* Trained on quiet-Sun-like and sunspot spectra; **strong-field umbrae are
  under-estimated** (|B| biased low for B ≳ 1.5 kG — the BIFROST sample has a median
  field of only 15 G; add strong-field / low-continuum samples to fix this);
* `M` (vmac) is the least constrained output;
* Out-of-distribution continua are not reproduced (e.g. cool11, I_c ≈ 0.11).
