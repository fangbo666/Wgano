"""
WGANO: Wavelet-Guided Attention Neural Operator

Extracted from original training code (WGANO/n1/model_n1.py).
All core algorithms preserved exactly as trained.

CRITICAL:
- V-gating: v = v_base * gate + v_base (additive residual)
- Pre-LayerNorm structure: fx = fx + attn(ln(fx))

Author: Bo Fang
Institution: China University of Petroleum (East China)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from torch_wavelets import DWT_2D, IDWT_2D
except ImportError:
    raise ImportError("torch_wavelets required. Install: pip install pytorch-wavelets")


ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'silu': nn.SiLU,
}


class HFGuidedWaveletAttn(nn.Module):
    """
    HF-Guided Wavelet Attention.
    Exact implementation from training code.
    """
    def __init__(self, hidden_size, num_heads, use_qk_norm=True):
        super().__init__()
        C = self.C = hidden_size
        self.num_heads = num_heads
        self.head_dim  = C // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm

        self.dwt  = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        # HF branch
        self.hf_dw = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, 3, padding=1, groups=3 * C),
            nn.BatchNorm2d(3 * C),
        )
        self.hf_pw = nn.Sequential(
            nn.Conv2d(3 * C, C, 1),
            nn.BatchNorm2d(C),
        )
        self.hf_refine = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, 3, padding=1, groups=3 * C),
            nn.BatchNorm2d(3 * C),
        )

        # LL SoftmaxAttn
        self.ll_norm = nn.LayerNorm(C)
        self.ll_qkv  = nn.Linear(C, C * 3)
        self.ll_proj = nn.Linear(C, C)
        self.q_norm  = nn.LayerNorm(self.head_dim) if use_qk_norm else None
        self.k_norm  = nn.LayerNorm(self.head_dim) if use_qk_norm else None

        # Output projection
        self.out_proj = nn.Linear(C + C, C)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x_ori, H, W, coord_map=None, ext_Ud=None, ext_Us=None):
        """
        EXACT implementation from training code.
        Args:
            x_ori: [B,N,C]
        Returns:
            out: [B,N,C], Ud, Us
        """
        B, N, C = x_ori.shape

        # DWT decomposition
        xc = x_ori.view(B, H, W, C).permute(0, 3, 1, 2)   # [B,C,H,W]
        ph = H % 2; pw = W % 2
        if ph or pw:
            xc = F.pad(xc, [0, pw, 0, ph])
        dwt_out = self.dwt(xc)                              # [B,4C,H/2,W/2]
        Hd, Wd  = dwt_out.shape[-2], dwt_out.shape[-1]
        LL = dwt_out[:, :C]                                 # [B,C,Hd,Wd]
        HF = dwt_out[:, C:]                                 # [B,3C,Hd,Wd]

        # HF gate
        hf_feat    = self.hf_dw(HF)                         # [B,3C,Hd,Wd]
        gate       = torch.sigmoid(self.hf_pw(hf_feat))     # [B,C,Hd,Wd]
        hf_refined = self.hf_refine(HF)                     # [B,3C,Hd,Wd]

        # External coupling injection
        if ext_Ud is not None and ext_Ud.shape == hf_refined.shape:
            hf_refined = hf_refined + ext_Ud
        if ext_Us is not None and ext_Us.shape == LL.shape:
            LL = LL + ext_Us

        # LL attention with HF-guided V
        Nd = Hd * Wd
        ll_tokens = LL.reshape(B, C, Nd).permute(0, 2, 1)  # [B,Nd,C]
        ll_tokens = self.ll_norm(ll_tokens)
        qkv = self.ll_qkv(ll_tokens).reshape(B, Nd, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v_base = qkv[0], qkv[1], qkv[2]              # [B,heads,Nd,head_dim]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # CRITICAL: V-gating with additive residual
        # Formula: v = v_base * gate + v_base
        # This is the CORE innovation - copied exactly from training code
        gate_flat = gate.reshape(B, C, Nd).permute(0, 2, 1)            # [B,Nd,C]
        gate_flat = gate_flat.reshape(B, Nd, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v_base * gate_flat + v_base  # Line 115 from original code

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B,heads,Nd,Nd]
        attn = attn.softmax(dim=-1)
        ll_attn = (attn @ v).transpose(1, 2).reshape(B, Nd, C)
        ll_attn = self.ll_proj(ll_attn)                     # [B,Nd,C]
        LL_out  = ll_attn.permute(0, 2, 1).reshape(B, C, Hd, Wd)

        # IDWT reconstruction
        idwt_in = torch.cat([LL_out, hf_refined], dim=1)   # [B,4C,Hd,Wd]
        x_idwt  = self.idwt(idwt_in)                       # [B,C,H(+pad),W(+pad)]
        x_idwt  = x_idwt[:, :, :H, :W]
        x_idwt_flat = x_idwt.permute(0, 2, 3, 1).reshape(B, N, C)

        out = self.out_proj(torch.cat([x_ori, x_idwt_flat], dim=-1))
        return out, hf_refined, LL_out


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        self.res = res
        self.linear_pre  = nn.Sequential(nn.Linear(n_input, n_hidden), ACTIVATION[act]())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), ACTIVATION[act]()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class N1Block(nn.Module):
    """
    CRITICAL: Pre-LayerNorm structure (line 165 from original code)
    Structure: fx = fx + attn(ln_1(fx))
    """
    def __init__(self, num_heads, hidden_dim, act='gelu', mlp_ratio=2,
                 last_layer=False, out_dim=1, H=85, W=85, use_qk_norm=True):
        super().__init__()
        self.last_layer = last_layer
        self.H, self.W  = H, W
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = HFGuidedWaveletAttn(hidden_dim, num_heads, use_qk_norm)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp  = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, coord_map=None, ext_Ud=None, ext_Us=None):
        # Pre-LayerNorm: ln BEFORE attention (line 165 from original code)
        attn_out, Ud, Us = self.attn(self.ln_1(fx), self.H, self.W, coord_map, ext_Ud, ext_Us)
        fx = attn_out + fx
        # Pre-LayerNorm: ln BEFORE mlp (line 167 from original code)
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.head(self.ln_3(fx)), Ud, Us
        return fx, Ud, Us


class Model(nn.Module):
    """WGANO Model - exact copy from training code."""
    def __init__(self, space_dim=1, n_layers=8, n_hidden=156, dropout=0.0,
                 n_head=4, Time_Input=False, act='gelu', mlp_ratio=2,
                 fun_dim=1, out_dim=1, slice_num=32, ref=8,
                 unified_pos=True, H=85, W=85, n_levels=1, use_qk_norm=True):
        super().__init__()
        self.__name__ = 'N1'
        self.H, self.W = H, W
        self.pad_h = self.pad_w = 0
        if H % 2 == 1: self.H += 1; self.pad_h = 1
        if W % 2 == 1: self.W += 1; self.pad_w = 1

        self.ref = ref
        self.unified_pos = unified_pos
        self.space_dim = space_dim
        if unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + ref * ref, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden   = n_hidden
        if Time_Input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([
            N1Block(num_heads=n_head, hidden_dim=n_hidden, act=act,
                    mlp_ratio=mlp_ratio, out_dim=out_dim,
                    H=self.H, W=self.W, use_qk_norm=use_qk_norm,
                    last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        H, W = self.H, self.W
        gx = torch.linspace(0, 1, H).reshape(1, H, 1, 1).repeat(batchsize, 1, W, 1)
        gy = torch.linspace(0, 1, W).reshape(1, 1, W, 1).repeat(batchsize, H, 1, 1)
        grid = torch.cat([gx, gy], dim=-1)
        rx = torch.linspace(0, 1, self.ref).reshape(1, self.ref, 1, 1).repeat(batchsize, 1, self.ref, 1)
        ry = torch.linspace(0, 1, self.ref).reshape(1, 1, self.ref, 1).repeat(batchsize, self.ref, 1, 1)
        grid_ref = torch.cat([rx, ry], dim=-1)
        pos = torch.sqrt(torch.sum(
            (grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1
        )).reshape(batchsize, H, W, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None, ext_Ud_list=None, ext_Us_list=None):
        x = F.pad(x, [0, 0, 0, self.pad_w, 0, self.pad_h])
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(
                x.shape[0], self.H * self.W, self.ref * self.ref)

        if fx is not None:
            fx = F.pad(fx, [0, 0, 0, self.pad_w, 0, self.pad_h])
            fx = fx.reshape(fx.shape[0], -1, fx.shape[-1])
            fx = self.preprocess(torch.cat([x, fx], dim=-1))
        else:
            fx = self.preprocess(x)

        if T is not None:
            from .utils import timestep_embedding
            te = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            fx = fx + self.time_fc(te)

        all_Ud, all_Us = [], []
        for i, block in enumerate(self.blocks):
            ext_Ud = ext_Ud_list[i] if ext_Ud_list is not None else None
            ext_Us = ext_Us_list[i] if ext_Us_list is not None else None
            fx, Ud, Us = block(fx, None, ext_Ud, ext_Us)
            all_Ud.append(Ud)
            all_Us.append(Us)

        fx = fx.reshape(fx.shape[0], self.H, self.W, fx.shape[-1])
        if self.pad_h: fx = fx[:, :-self.pad_h]
        if self.pad_w: fx = fx[:, :, :-self.pad_w]
        return fx, all_Ud, all_Us


def timestep_embedding(timesteps, dim, max_period=10000):
    """Timestep embedding from SAOT."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
