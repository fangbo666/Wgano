"""
WGANO: Wavelet-Guided Attention Neural Operator

This module implements the core WGANO model with high-frequency guided
wavelet attention mechanism for solving PDEs.

Architecture:
- HFGuidedWaveletAttn: Wavelet-based attention with HF gating
- N1Block: Transformer block with attention + MLP
- Model: Complete WGANO with configurable layers

Author: Bo Fang
Institution: China University of Petroleum (East China)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from torch_wavelets import DWT_2D, IDWT_2D
except ImportError:
    raise ImportError("torch_wavelets is required. Install with: pip install pytorch-wavelets")


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
}


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for temporal data.

    Args:
        timesteps: [B, 1] tensor of timestep indices
        dim: embedding dimension
        max_period: controls minimum frequency

    Returns:
        [B, dim] tensor of embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class HFGuidedWaveletAttn(nn.Module):
    """
    High-Frequency Guided Wavelet Attention.

    Uses single-level DWT to decompose features into LL (low-freq) and HF (high-freq).
    HF components guide the attention on LL through a gating mechanism.

    Args:
        hidden_size: channel dimension
        num_heads: number of attention heads
        use_qk_norm: whether to apply LayerNorm to Q and K (default: True)
    """
    def __init__(self, hidden_size, num_heads, use_qk_norm=True):
        super().__init__()
        C = self.C = hidden_size
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm

        # Wavelet transforms
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        # HF processing: DW-Conv → gate features, PW-Conv → gate
        self.hf_dw = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, 3, padding=1, groups=3 * C),
            nn.BatchNorm2d(3 * C),
        )
        self.hf_pw = nn.Sequential(
            nn.Conv2d(3 * C, C, 1),
            nn.BatchNorm2d(C),
        )
        # HF refinement for IDWT reconstruction
        self.hf_refine = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, 3, padding=1, groups=3 * C),
            nn.BatchNorm2d(3 * C),
        )

        # QKV projections for LL
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)

        # Optional QK LayerNorm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, pos_emb=None, ext_Ud=None, ext_Us=None):
        """
        Args:
            x: [B, H, W, C] input features
            pos_emb: optional positional embedding
            ext_Ud: [B, 3C, Hd, Wd] external coupling for HF (for multi-physics)
            ext_Us: [B, C, Hd, Wd] external coupling for LL (for multi-physics)

        Returns:
            out: [B, H, W, C] output features
            Ud: [B, 3C, Hd, Wd] HF features for coupling
            Us: [B, C, Hd, Wd] LL features for coupling
        """
        B, H, W, C = x.shape

        # Reshape for convolution: [B, H, W, C] → [B, C, H, W]
        xc = x.permute(0, 3, 1, 2).contiguous()

        # Pad to even dimensions for DWT
        ph = (2 - H % 2) % 2
        pw = (2 - W % 2) % 2
        if ph > 0 or pw > 0:
            xc = F.pad(xc, [0, pw, 0, ph])

        # Wavelet decomposition
        dwt_out = self.dwt(xc)  # [B, 4C, H/2, W/2]
        Hd, Wd = dwt_out.shape[-2], dwt_out.shape[-1]
        LL = dwt_out[:, :C]      # [B, C, Hd, Wd]
        HF = dwt_out[:, C:]      # [B, 3C, Hd, Wd]

        # ── HF gating ──
        hf_feat = self.hf_dw(HF)                      # [B, 3C, Hd, Wd]
        gate = torch.sigmoid(self.hf_pw(hf_feat))     # [B, C, Hd, Wd]
        hf_refined = self.hf_refine(HF)               # [B, 3C, Hd, Wd]

        # External coupling injection (for multi-physics)
        if ext_Ud is not None and ext_Ud.shape == hf_refined.shape:
            hf_refined = hf_refined + ext_Ud
        if ext_Us is not None and ext_Us.shape == LL.shape:
            LL = LL + ext_Us

        # ── LL attention ──
        LL_flat = LL.flatten(2).transpose(1, 2)  # [B, Hd*Wd, C]
        N = LL_flat.shape[1]

        qkv = self.qkv(LL_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK normalization if enabled
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention with HF-guided V
        gate_flat = gate.flatten(2).transpose(1, 2).unsqueeze(1)  # [B, 1, N, C]
        v_guided = v * gate_flat.expand(-1, self.num_heads, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out_attn = (attn @ v_guided).transpose(1, 2).reshape(B, N, C)
        out_attn = self.proj(out_attn)

        # Reshape back to spatial
        out_LL = out_attn.transpose(1, 2).reshape(B, C, Hd, Wd)

        # ── IDWT reconstruction ──
        rec_in = torch.cat([out_LL, hf_refined], dim=1)  # [B, 4C, Hd, Wd]
        rec_out = self.idwt(rec_in)  # [B, C, H_pad, W_pad]

        # Remove padding
        if ph > 0 or pw > 0:
            rec_out = rec_out[:, :, :H, :W]

        # Back to [B, H, W, C]
        out = rec_out.permute(0, 2, 3, 1).contiguous()

        return out, hf_refined, out_LL


class N1Block(nn.Module):
    """
    WGANO Transformer Block.

    Architecture: Attention → Residual → LayerNorm → MLP → Residual → LayerNorm

    Args:
        n_hidden: hidden dimension
        n_head: number of attention heads
        mlp_ratio: MLP hidden dimension multiplier
        act: activation function name
        last_layer: if True, adds output projection head
        out_dim: output dimension (only used if last_layer=True)
        use_qk_norm: whether to use QK normalization in attention
    """
    def __init__(self, n_hidden, n_head, mlp_ratio=1, act='gelu',
                 last_layer=False, out_dim=1, use_qk_norm=True):
        super().__init__()
        self.last_layer = last_layer

        self.attn = HFGuidedWaveletAttn(n_hidden, n_head, use_qk_norm=use_qk_norm)
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

    def forward(self, x, pos_emb=None, ext_Ud=None, ext_Us=None):
        """
        Args:
            x: [B, H, W, C] input
            pos_emb: optional positional embedding
            ext_Ud, ext_Us: external coupling features

        Returns:
            out: [B, H, W, C] or [B, H, W, out_dim] if last_layer
            Ud, Us: coupling features
        """
        # Attention with residual
        attn_out, Ud, Us = self.attn(x, pos_emb, ext_Ud, ext_Us)
        x = x + attn_out
        x = self.norm1(x)

        # MLP with residual
        x = x + self.mlp(x)
        x = self.norm2(x)

        if self.last_layer:
            x = self.head(x)

        return x, Ud, Us


class Model(nn.Module):
    """
    Complete WGANO model.

    Args:
        space_dim: spatial dimension (2 for 2D problems)
        n_layers: number of transformer blocks
        n_hidden: hidden dimension
        n_head: number of attention heads
        fun_dim: input function dimension (0 if no auxiliary function)
        out_dim: output dimension
        unified_pos: whether to use unified positional encoding
        ref: reference scale for positional encoding
        mlp_ratio: MLP expansion ratio
        H, W: spatial resolution
        act: activation function
        use_qk_norm: whether to use QK normalization (True for n1qk, False for n1)
    """
    def __init__(self, space_dim=2, n_layers=4, n_hidden=64, n_head=4,
                 fun_dim=0, out_dim=1, unified_pos=True, ref=8, mlp_ratio=1,
                 H=128, W=128, act='gelu', use_qk_norm=True):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.H = H
        self.W = W
        self.unified_pos = unified_pos
        self.ref = ref

        # Padding for odd dimensions
        self.pad_h = (2 - H % 2) % 2
        self.pad_w = (2 - W % 2) % 2

        # Input preprocessing
        in_dim = space_dim + fun_dim
        self.preprocess = nn.Linear(in_dim, n_hidden)

        # Optional timestep embedding
        self.time_fc = nn.Linear(n_hidden, n_hidden)

        # Positional encoding
        if unified_pos:
            pe = self._get_unified_pos_enc(H, W, n_hidden, ref)
            self.register_buffer('pe', pe)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            N1Block(n_hidden, n_head, mlp_ratio, act,
                    last_layer=(i == n_layers - 1), out_dim=out_dim,
                    use_qk_norm=use_qk_norm)
            for i in range(n_layers)
        ])

        self._init_weights()

    def _get_unified_pos_enc(self, H, W, C, ref):
        """Generate unified sinusoidal positional encoding."""
        y_grid = torch.linspace(0, 1, H).reshape(H, 1, 1).repeat(1, W, 1)
        x_grid = torch.linspace(0, 1, W).reshape(1, W, 1).repeat(H, 1, 1)
        grid = torch.cat([x_grid, y_grid], dim=-1)  # [H, W, 2]

        pe_list = []
        for i in range(C // 4):
            freq = ref ** i
            pe_list.append(torch.sin(2 * math.pi * freq * grid[..., 0:1]))
            pe_list.append(torch.cos(2 * math.pi * freq * grid[..., 0:1]))
            pe_list.append(torch.sin(2 * math.pi * freq * grid[..., 1:2]))
            pe_list.append(torch.cos(2 * math.pi * freq * grid[..., 1:2]))

        pe = torch.cat(pe_list[:C], dim=-1)  # [H, W, C]
        return pe.unsqueeze(0)  # [1, H, W, C]

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, fx=None, T=None, ext_Ud_list=None, ext_Us_list=None):
        """
        Forward pass.

        Args:
            x: [B, H, W, space_dim] spatial coordinates
            fx: [B, H, W, fun_dim] optional auxiliary function input
            T: [B, 1] optional timestep
            ext_Ud_list: list of external HF coupling features (for multi-physics)
            ext_Us_list: list of external LL coupling features (for multi-physics)

        Returns:
            fx: [B, H, W, out_dim] model output
            all_Ud: list of HF features from each layer
            all_Us: list of LL features from each layer
        """
        # Add positional encoding
        if self.unified_pos:
            x = x + self.pe[:, :x.shape[1], :x.shape[2]]

        # Padding
        if self.pad_h or self.pad_w:
            x = F.pad(x, [0, 0, 0, self.pad_w, 0, self.pad_h])
            if fx is not None:
                fx = F.pad(fx, [0, 0, 0, self.pad_w, 0, self.pad_h])

        # Preprocess input
        if fx is not None:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            fx = fx.reshape(fx.shape[0], -1, fx.shape[-1])
            x = self.preprocess(torch.cat([x, fx], dim=-1))
            x = x.reshape(x.shape[0], self.H + self.pad_h, self.W + self.pad_w, -1)
        else:
            x = self.preprocess(x)

        # Optional timestep embedding
        if T is not None:
            te = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            x = x + self.time_fc(te)

        # Forward through blocks
        all_Ud, all_Us = [], []
        for i, block in enumerate(self.blocks):
            ext_Ud = ext_Ud_list[i] if ext_Ud_list is not None else None
            ext_Us = ext_Us_list[i] if ext_Us_list is not None else None
            x, Ud, Us = block(x, None, ext_Ud, ext_Us)
            all_Ud.append(Ud)
            all_Us.append(Us)

        # Remove padding
        if self.pad_h:
            x = x[:, :-self.pad_h]
        if self.pad_w:
            x = x[:, :, :-self.pad_w]

        return x, all_Ud, all_Us
