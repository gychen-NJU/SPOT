"""spot.net.model — StokesPHNO, a parallel hybrid neural operator for Stokes inversion.

Architecture (faithful re-implementation of the proven v7 design):
  1. input projection + sinusoidal positional encoding
  2. FNO (Fourier Neural Operator) blocks: learnable spectral filter in the
     frequency domain, capturing quasi-periodic spectro-line structure
  3. Transformer encoder with Squeeze-and-Excitation channel attention:
     long-range dependencies between wavelength samples
  4. DeepONet-style cross-attention decoder: NeRF-like Fourier embeddings of
     the depth grid act as queries that attend to the encoded Stokes sequence;
     a pooled global feature vector drives the scalar heads
  5. parameter-specific MLP heads (one per quantity)

Input : stokes (B, n_wavelength, 4)  — I, Q, U, V
Output: dict of tensors
  depth params: (B, n_depth)                     [t, p, b, g, f, v]
  scalar params: (B,)                            [m, M]
'f' (azimuth) may use a 2-D output (sin/cos encoding) when out_dims={'f': 2}.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class SinusoidalPosEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the wavelength axis."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0.0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2.0) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SqueezeExcitation(nn.Module):
    """Channel attention: global-average pooling -> FC -> ReLU -> FC -> sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        y = self.fc(x.mean(dim=-1))
        return x * y.view(b, c, 1)


class FNO1DLayer(nn.Module):
    """1-D Fourier layer: RFFT -> learnable complex filter -> IRFFT + skip.

    The complex filter is stored as two real parameters (re/im) so that AMP
    (GradScaler) and standard optimisers work unchanged.
    """

    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (channels * channels)
        self.w_re = nn.Parameter(self.scale * torch.randn(channels, channels, modes))
        self.w_im = nn.Parameter(self.scale * torch.randn(channels, channels, modes))
        self.linear = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT is run in float32 (no fp16 CUDA kernel); result cast back afterwards.
        dt = x.dtype if x.dtype.is_floating_point else torch.float32
        xf = x.to(torch.float32)
        B, C, L = xf.shape
        x_fft = torch.fft.rfft(xf, n=L, dim=-1)
        k = min(self.modes, x_fft.shape[-1])
        w = torch.complex(self.w_re, self.w_im)[:, :, :k]
        filt = torch.einsum("cok,bil->bcl", w, x_fft[:, :, :k].contiguous())
        x_fno = torch.fft.irfft(
            torch.cat([filt, x_fft[:, :, k:]], dim=-1), n=L, dim=-1
        )
        x_skip = self.linear(xf.transpose(1, 2)).transpose(1, 2)
        return (x_fno + x_skip).to(dt)


class FNOBlock(nn.Module):
    """Residual FNO layer: FNO -> GELU -> skip add -> LayerNorm."""

    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.fno = FNO1DLayer(channels, modes)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        h = self.act(self.fno(x_t)).transpose(1, 2)
        return self.norm(x + h)


class TransformerEncoderLayerSE(nn.Module):
    """Transformer encoder layer with SE-enhanced FFN block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.se = SqueezeExcitation(d_model, se_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        x = x + self.dropout(self.self_attn(q, q, q)[0])
        h = self.ffn(self.norm2(x))
        h = self.se(h.transpose(1, 2)).transpose(1, 2)
        return x + h


class DepthFourierEmbedding(nn.Module):
    """NeRF-style Fourier feature embedding of the depth grid positions."""

    def __init__(self, n_depth: int = 64, d_model: int = 256, num_frequencies: int = 8):
        super().__init__()
        depth_pos = torch.linspace(0, 1, n_depth).unsqueeze(-1)
        freqs = 2 ** torch.arange(num_frequencies) * math.pi
        enc = []
        for f in freqs:
            enc.append(torch.sin(depth_pos * f))
            enc.append(torch.cos(depth_pos * f))
        self.register_buffer("fourier_feats", torch.cat(enc, dim=-1))
        self.proj = nn.Sequential(
            nn.Linear(2 * num_frequencies, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, batch_size: Optional[int] = None) -> torch.Tensor:
        emb = self.proj(self.fourier_feats).unsqueeze(0)
        if batch_size is not None:
            emb = emb.expand(batch_size, -1, -1)
        return emb


class CrossAttentionDecoder(nn.Module):
    """DeepONet-style decoder: depth queries cross-attend to the encoded spectrum.

    Returns (depth_out (B, n_depth, d_model), global_feats (B, d_model)).
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_depth: int = 64, dropout: float = 0.1):
        super().__init__()
        self.depth_embed = DepthFourierEmbedding(n_depth, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.se = SqueezeExcitation(d_model)
        self.norm_post = nn.LayerNorm(d_model)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, encoded_stokes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = encoded_stokes.shape[0]
        depth_q = self.depth_embed(B)
        depth_feats, _ = self.cross_attn(
            depth_q, encoded_stokes, encoded_stokes, need_weights=False
        )
        depth_feats = self.norm_ca(depth_feats + depth_q)
        h = self.ffn(depth_feats)
        h = self.se(h.transpose(1, 2)).transpose(1, 2)
        depth_out = self.norm_post(depth_feats + h)
        global_feats = self.global_pool(encoded_stokes.transpose(1, 2))
        return depth_out, global_feats


class ParameterHead(nn.Module):
    """Per-parameter MLP head; out_dim > 1 for encodings (e.g. sin/cos azimuth)."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, out_dim: int = 1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(d_model // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class StokesPHNO(nn.Module):
    """Parallel hybrid neural operator mapping Stokes spectra to atmospheric parameters.

    Args:
      d_model: hidden dimension of the sequence encoder.
      n_fno_layers / n_transformer_layers: depth of the two encoder stages.
      n_heads / d_ff: attention heads and FFN width of the transformer.
      dropout / se_reduction: regularisation and SE-channel compression ratio.
      fno_modes: number of retained Fourier modes in the FNO filter.
      n_wavelength: number of spectral points (input grid length).
      n_depth: number of atmospheric depth points (output grid length).
      depth_params: names of depth-dependent outputs.
      scalar_params: names of scalar outputs.
      out_dims: optional per-parameter output dimensionality (e.g. {'f': 2}).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_fno_layers: int = 2,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        se_reduction: int = 16,
        fno_modes: int = 16,
        n_wavelength: int = 112,
        n_depth: int = 64,
        depth_params: Sequence[str] = ("t", "p", "b", "g", "f", "v"),
        scalar_params: Sequence[str] = ("m", "M"),
        out_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_depth = n_depth
        self.n_wavelength = n_wavelength
        self.depth_params = tuple(depth_params)
        self.scalar_params = tuple(scalar_params)
        self.out_dims = dict(out_dims or {})

        self.input_proj = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPosEncoding(d_model, n_wavelength)
        self.fno_layers = nn.ModuleList([FNOBlock(d_model, fno_modes) for _ in range(n_fno_layers)])
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerSE(d_model, n_heads, d_ff, dropout, se_reduction)
                for _ in range(n_transformer_layers)
            ]
        )
        self.decoder = CrossAttentionDecoder(d_model, n_heads, n_depth, dropout)

        hidden = max(d_model // 2, 64)
        self.depth_heads = nn.ModuleDict(
            {
                name: ParameterHead(d_model, hidden, self.out_dims.get(name, 1))
                for name in self.depth_params
            }
        )
        self.scalar_heads = nn.ModuleDict(
            {
                name: ParameterHead(d_model, hidden, self.out_dims.get(name, 1))
                for name in self.scalar_params
            }
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, stokes: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(stokes)
        x = self.pos_enc(x)
        for fno in self.fno_layers:
            x = fno(x)
        for enc in self.transformer_layers:
            x = enc(x)
        depth_feats, global_feats = self.decoder(x)

        out: Dict[str, torch.Tensor] = {}
        for name, head in self.depth_heads.items():
            out[name] = head(depth_feats)
            if out[name].shape[-1] == 1:
                out[name] = out[name].squeeze(-1)
        for name, head in self.scalar_heads.items():
            y = head(global_feats)
            out[name] = y if y.shape[-1] > 1 else y.squeeze(-1)
        return out

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
