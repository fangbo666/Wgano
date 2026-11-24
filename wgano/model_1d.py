"""
1D WGANO Model for Irregular Mesh Problems (e.g., Elasticity)

Uses DWT_1D for wavelet decomposition along the sequence dimension.
Designed for problems with irregular/unstructured grids where points are
represented as 1D sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from torch_wavelets import DWT_1D, IDWT_1D
except ImportError:
    raise ImportError("torch_wavelets is required. Install with: pip install pytorch-wavelets")


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
}


class HFGuidedWaveletAttn1D(nn.Module):
    """
    1D version of HF-guided wavelet attention.

    Applies DWT_1D to decompose sequence into LL and HF components,
    uses HF to gate the attention on LL.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        C = self.C = hidden_size
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.scale = self.head_dim ** -0.5

        # 1D wavelet transforms
        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')

        # HF processing
        self.hf_dw = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, groups=C),
            nn.BatchNorm1d(C),
        )
        self.hf_pw = nn.Sequential(
            nn.Conv1d(C, C, 1),
            nn.BatchNorm1d(C),
        )
        self.hf_refine = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, groups=C),
            nn.BatchNorm1d(C),
        )

        # QKV projections
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] input sequence

        Returns:
            out: [B, N, C] output sequence
        """
        B, N, C = x.shape

        # Reshape for conv: [B, N, C] -> [B, C, N]
        xc = x.transpose(1, 2).contiguous()

        # Pad to even length
        pad = N % 2
        if pad:
            xc = F.pad(xc, [0, pad])

        # DWT_1D decomposition
        dwt_out = self.dwt(xc)  # [B, 2C, N/2]
        Nd = dwt_out.shape[-1]
        LL = dwt_out[:, :C]    # [B, C, N/2]
        HF = dwt_out[:, C:]    # [B, C, N/2]

        # HF gating
        hf_feat = self.hf_dw(HF)
        gate = torch.sigmoid(self.hf_pw(hf_feat))  # [B, C, Nd]
        hf_refined = self.hf_refine(HF)

        # Attention on LL with HF gating
        LL_flat = LL.transpose(1, 2)  # [B, Nd, C]

        qkv = self.qkv(LL_flat).reshape(B, Nd, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, Nd, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Gate V with HF
        gate_flat = gate.transpose(1, 2).unsqueeze(1)  # [B, 1, Nd, C]
        v_guided = v * gate_flat.expand(-1, self.num_heads, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out_attn = (attn @ v_guided).transpose(1, 2).reshape(B, Nd, C)
        out_attn = self.proj(out_attn)

        # Reshape back to conv format
        out_LL = out_attn.transpose(1, 2)  # [B, C, Nd]

        # IDWT reconstruction
        rec_in = torch.cat([out_LL, hf_refined], dim=1)  # [B, 2C, Nd]
        rec_out = self.idwt(rec_in)  # [B, C, N_pad]

        # Remove padding
        if pad:
            rec_out = rec_out[:, :, :-pad]

        # Back to [B, N, C]
        out = rec_out.transpose(1, 2).contiguous()

        return out


class Block1D(nn.Module):
    """1D Transformer block with wavelet attention."""
    def __init__(self, n_hidden, n_head, mlp_ratio=1, act='gelu',
                 last_layer=False, out_dim=1):
        super().__init__()
        self.last_layer = last_layer

        self.attn = HFGuidedWaveletAttn1D(n_hidden, n_head)
        self.norm1 = nn.LayerNorm(n_hidden)

        mlp_hidden = int(n_hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, mlp_hidden),
            ACTIVATION[act](),
            nn.Linear(mlp_hidden, n_hidden),
        )
        self.norm2 = nn.LayerNorm(n_hidden)

        if last_layer:
            self.head = nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)

        if self.last_layer:
            x = self.head(x)

        return x


class Model1D(nn.Module):
    """
    1D WGANO model for irregular mesh problems.

    Args:
        in_dim: input dimension (e.g., 2 for coordinates)
        n_layers: number of transformer blocks
        n_hidden: hidden dimension
        n_head: number of attention heads
        out_dim: output dimension
        mlp_ratio: MLP expansion ratio
        act: activation function
    """
    def __init__(self, in_dim=2, n_layers=4, n_hidden=64, n_head=4,
                 out_dim=1, mlp_ratio=1, act='gelu'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        # Input preprocessing
        self.preprocess = nn.Linear(in_dim, n_hidden)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block1D(n_hidden, n_head, mlp_ratio, act,
                   last_layer=(i == n_layers - 1), out_dim=out_dim)
            for i in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, N, in_dim] input sequence

        Returns:
            out: [B, N, out_dim] output
        """
        x = self.preprocess(x)

        for block in self.blocks:
            x = block(x)

        return x
