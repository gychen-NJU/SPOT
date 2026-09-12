# Deprecated: the original `hinode_sp` network (v1)

This directory keeps the checkpoint that SPOT bundled up to **v1.1.1**.

* files: `best_model.pt` (md5 `76e898d8c898f780eb48aa8b64242c0c`), `config.json`,
  `norm_stats.pt`
* what it was: a `StokesPHNO` fine-tune on Hinode SP Fe I 6301.5/6302.5 profiles with
  labels from a classical Stokes inversion code. The stored checkpoint is `epoch=9`
  (val_loss 2.72) even though its config declared 300 epochs — i.e. it is a
  half-finished training run.
* why deprecated: superseded on **2026-09-12** by the current model in `../hinode_sp/`
  (SPOT network v3, trained from scratch for 50 epochs), which is better on every
  physical parameter — see `../hinode_sp/MODEL_CARD.md`.

**Retained for provenance and reproducibility only.** No `spot.net` preset points
here, no API exposes it, and `setup.py` `package_data` excludes this directory (so it
is not shipped in the wheel). To use it you must load the files by explicit path:

```python
from spot.net import StokesInference
D = ".../spot/net/models/deprecated/hinode_sp_v1"
infer = StokesInference(model_path=f"{D}/best_model.pt", norm_path=f"{D}/norm_stats.pt",
                        config_path=f"{D}/config.json")   # input_scale defaults to None
```

It shares the current model's depth convention (index 0 = shallowest, log τ₅₀₀₀ = −4)
and its `I_c,ref = 8.257986e14` input scaling, so results are directly comparable.
