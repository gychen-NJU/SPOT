<div align="center">

# SPOT

**Spectrapolarimetric Operator Transformer** — 面向太阳偏振光谱的 Stokes IQUV 谱线合成与反演库（含神经算子子包）

[English](README.md) | [简体中文](README.zh.md)

</div>

### 概述

`spot` 是一个基于 PyTorch 的太阳偏振光谱（Stokes IQUV）合成与反演库。它以纯 Python / PyTorch（NumPy）实现 LTE 大气中谱线形成的完整物理链路（不透明度、Zeeman 线型、辐射转移方程求解等），并内置节点化反演、无导数优化与神经算子网络反演等多种方法，支持 CPU / GPU。

核心功能：

1. **正向合成**：任意大气模型（内置预设 + 自定义 CSV）、任意谱线列表、任意波长网格下，批量（batch）合成 Stokes I、Q、U、V 谱线，支持 CPU / GPU（PyTorch 向量化）；
2. **响应函数**：dI/dx（对大气参数）自动微分 / 有限差分 / 解析链式法则多种方法，默认 `fast` 方法精确且高效；
3. **反演**：节点化大气参数化 + Levenberg–Marquardt（阻尼 SVD）多循环反演；另有 CMA-ES（无导数）批量反演用于初值搜索；
4. **网络反演**：`spot.net.StokesInference` 加载预训练神经算子（FNO + Transformer + DeepONet 风格解码器），一键对 Hinode SP 配置的 Stokes 谱进行反演。

### 安装

```bash
cd SPOT
pip install -e .          # 或 pip install .（完整安装，含 spot.net 预训练模型）
```

依赖：`numpy`、`torch>=2.0`、`matplotlib`（见 `requirements.txt`）。

### 快速上手

#### 1. 正向合成（Hinode SP 配置示例）

```python
import numpy as np
import torch
from spot import Synthesis
from spot.utils.data_io import load_atmosphere

# Hinode SP 窗口：112 个采样，6300.8840305 .. 6303.2759695 A，步长 21.549 mA
wavs = torch.tensor(6300.8840305 + 21.549e-3 * np.arange(112), dtype=torch.float64)

model = load_atmosphere("cool11")          # 预设大气（55 层，log tau 递减）
ltau = torch.tensor(model["ltau"], dtype=torch.float64)

# 打包大气向量：[T, Pe, B, gamma, phi, vlos] x Nt + [vmic, vmac]（km/s）
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

#### 2. 响应函数（默认 `fast`）

```python
stokes, rf = syn(wavs, ltau, atmos, return_rf=True)
print(rf.shape)   # (1, 112, 4, 6*Nt+2)，dI/dx 对打包大气向量
```

#### 3. 反演（默认推荐流程：网络初猜 → 4 轮节点 LM，一键调用）

```python
from spot import Inversion

# 不传任何 inversion 设置即可使用默认推荐配置：
#   atmosphere='auto'（先用 spot.net 的 hinode_sp 网络反演目标谱线得到初猜）
#   + 4 轮节点表 [2,3,4,auto]（T/B/gamma/phi/vlos）
#   + fast 响应函数 + sigma='sir' (snr=1000) + 权重 1:5:5:10 + 无 hse
# Hinode SP 测试算例上：chi2 ≈ 0.05-0.06，约 130 s
# （对照：hot11 初猜 + 老默认 + hse 的完整 4 轮反演 ~2500 s，chi2 ~7e-2）
res = Inversion().invert(wavs, ltau, target)   # initial=None -> 网络初猜
print(res.chi2, res.atmos)                     # (Nb,) chi2 与最终大气

# 想回到传统初猜（如 hot11）或指定别的初猜：
res = Inversion().invert(wavs, ltau, target, initial="hot11")
```

如果需要更极致精度：`config["inversion"] = {"nodes": {..., "auto"}, "hse_pg0": 500}` 等变体
可在 demo/08_net_guess_search 中对比参考（chi2 ~2e-2，~890 s）。

#### 3b. 反演（自定义：CMA-ES 初值搜索 + LM 自动节点数）

```python
from spot import CmaesInversion, Inversion

cfg = {...}   # 与合成相同的基础配置
cfg["inversion"] = {
    "nodes": {"T": 4, "Pe": 0, "B": 3, "gamma": 2, "phi": 2,
              "vlos": 3, "vmic": 1, "vmac": 1},   # 少量节点
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

#### 4. 网络反演（预训练 `hinode_sp`）

```python
from spot.net import StokesInference

infer = StokesInference(preset="hinode_sp", device="cuda")
out = infer.predict_numpy(stokes_phys)      # 输入需为物理单位强度
# out: t,p,b,g,f,v (B,64) + m,M (B,)；t [K], p [dyn/cm^2], b [G],
#      g [deg], f [deg](-180,180], v/m/M [cm/s]
```

### 目录结构

```
spot/
├── __init__.py             # 顶层导出（Synthesis, Inversion, CmaesInversion, ...）
├── config.py               # 配置合并/冻结
├── default.py              # 默认配置（单位说明 + 所有可选项）
├── visualization.py        # 收敛曲线绘图
├── data/                   # 预设大气模型、谱线表、丰度表、不透明度表
├── physics/                # 原子数据、不透明度、Zeeman 线型、RTE 求解器、热力学、压力平衡...
├── synthesis/              # Synthesis：正向合成 + 响应函数
├── inversion/              # Inversion (LM), CmaesInversion, marquardt, nodes
├── utils/                  # data_io（预设/文件加载）、interpolation
└── net/                    # 神经算子子包（StokesInference, 训练/评估...）
    └── models/hinode_sp/   # 预训练模型（best_model.pt / config.json / norm_stats.pt）
```

### 配置要点（`spot.default.DEFAULT_CONFIG`）

* 单位：波长 [Angstrom]，`ltau = log10(tau5000)`，T [K]，Pe [dyn/cm²]，B [G]，角度 [deg]，速度 [km/s]；
* 大气向量：`[T, Pe, B, gamma, phi, vlos (每层) , vmic(标量), vmac(标量)]`，`Nx = 6*Nt + 2`；
* 合成：`solver`（hermitian/cn/delo）、`continuum_opacity`（mihalas/atlas/opacity_project）、`macroturbulence`、`normalize_continuum`、`rf_method`（fast/autograd/analytic/analytic_chain/finite_diff）；
* 反演：`atmosphere='auto'`（默认：先调 `spot.net` 网络（`network_preset`）对目标谱线反演出初猜）＋ `nodes`（每量的节点数，list = 每循环一个值，`"auto"` = 自动节点数）、`max_cycles=4`、`max_iterations=80`、`sigma='sir'`+`snr=1000`+`stokes_weights=[1,5,5,10]`、`hse_pg0=0`（默认推荐流程，详见快速上手 3）、`lambda0/lambda_factor`（LM 阻尼）、`svd_tolerance`；
* CMA-ES：`population`、`sigma0`、`scale`、`bounds`、`covariance_mode`、`stall`、`hse_refresh`。

### 数据与预设

* 大气预设：`hot11`、`cool11`、`falc11`、`falf11`、`hsra11`、`valc11`、`granulebbr` 等（`spot/data/models/*.csv`）；
* 谱线表：`spot/data/lines.csv`（含 Fe I 6301.508/6302.499 等）；
* 丰度表：`thevenin`（`spot/data/abundance.csv`）；
* 不透明度：`mihalas`（默认）／`atlas`（ATLAS 太阳 ODF）／`opacity_project`；
* 网络预设：`spot.net` 自带 `hinode_sp`（Hinode SP Fe I 6301.5/6302.5 微调模型）。

### 演示与验证

`demo/` 目录包含端到端演示（正向合成、响应函数、节点化反演（LM / CMA-ES）、网络反演及组合流程），每个演示生成图与详细报告（`demo/reports/`）。详见 `demo/README.md`。

### 许可

MIT（作者 Guoyin Chen，gychen@smail.nju.edu.cn）。
