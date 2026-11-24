"""
C-WGANO: Coupled WGANO for Multi-Physics Problems

This module implements the physics-coupled decoder for solving multi-physics PDEs
with bidirectional gating between multiple physical fields.

Architecture:
- WGANO backbone for feature extraction
- Task-specific projections for each physical field
- Bidirectional gating with shared parameters across iterations
- K iterations of coupling refinement

Author: Bo Fang
Institution: China University of Petroleum (East China)
"""

import torch
import torch.nn as nn
from .model import Model


class PhysicsCoupledHeadModel(nn.Module):
    """
    Physics-Coupled WGANO with iterative bidirectional gating.

    This model handles multi-physics problems by coupling multiple physical fields
    through shared gating mechanisms. Parameters are shared across iterations for
    efficiency.

    Args:
        n_hidden: hidden dimension of backbone
        n_layers: number of transformer layers in backbone
        n_head: number of attention heads
        head_hidden: hidden dimension of task-specific heads
        out_dims: list of output dimensions for each physical field
                 e.g., [3, 2] for MHD (3-channel EM + 2-channel flow)
        coupling_iter: number of coupling iterations (K)
        H, W: spatial resolution
        use_qk_norm: whether to use QK normalization in backbone
    """
    def __init__(self, n_hidden, n_layers, n_head, head_hidden, out_dims,
                 coupling_iter=2, H=128, W=128, use_qk_norm=True):
        super().__init__()
        self.coupling_iter = coupling_iter
        self.n_fields = len(out_dims)
        self.out_dims = out_dims

        # WGANO backbone
        self.backbone = Model(
            space_dim=2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            fun_dim=1,           # Single input channel
            out_dim=n_hidden,    # Output feature dimension
            unified_pos=True,
            ref=8,
            mlp_ratio=1,
            H=H,
            W=W,
            use_qk_norm=use_qk_norm,
        )

        # Task-specific projections
        self.projections = nn.ModuleList([
            nn.Linear(n_hidden, head_hidden) for _ in range(self.n_fields)
        ])

        self.act = nn.GELU()

        # Bidirectional coupling gates (shared across K iterations)
        # For n fields, we have n*(n-1) directed gates
        self.gates = nn.ModuleDict()
        for i in range(self.n_fields):
            for j in range(self.n_fields):
                if i != j:
                    self.gates[f'gate_{i}_from_{j}'] = nn.Linear(head_hidden, head_hidden)

        # Output projections
        self.output_heads = nn.ModuleList([
            nn.Linear(head_hidden, out_dims[i]) for i in range(self.n_fields)
        ])

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [B, H, W, 2] input coordinates (or [B, H, W, 3] with input channel)

        Returns:
            outputs: list of [B, H, W, out_dim_i] for each physical field
        """
        # Extract features from backbone
        feat, _, _ = self.backbone(x, x)  # [B, H, W, n_hidden]

        # Project to task-specific representations
        h_fields = [self.act(proj(feat)) for proj in self.projections]

        # K-round iterative coupling with bidirectional gating
        for _ in range(self.coupling_iter):
            new_h_fields = []
            for i in range(self.n_fields):
                # Aggregate information from all other fields
                coupling = h_fields[i].clone()
                for j in range(self.n_fields):
                    if i != j:
                        gate = torch.sigmoid(self.gates[f'gate_{i}_from_{j}'](h_fields[j]))
                        coupling = coupling + gate * h_fields[j]
                new_h_fields.append(coupling)
            h_fields = new_h_fields

        # Output predictions for each field
        outputs = [head(h) for h, head in zip(h_fields, self.output_heads)]

        return outputs


class DualPhysicsCoupledModel(nn.Module):
    """
    Simplified C-WGANO for two-field coupling (most common case).

    This is a specialized version optimized for dual-field problems like:
    - MHD: electromagnetic + flow
    - E_flow: potential + velocity
    - NS_heat: velocity + temperature
    - VA: acoustic + structural

    Args:
        n_hidden: hidden dimension of backbone
        n_layers: number of transformer layers
        n_head: number of attention heads
        head_hidden: hidden dimension of task heads
        out_dim_1: output dimension of first field
        out_dim_2: output dimension of second field
        coupling_iter: number of coupling iterations (K=2 by default)
        H, W: spatial resolution
        use_qk_norm: whether to use QK normalization
    """
    def __init__(self, n_hidden, n_layers, n_head, head_hidden,
                 out_dim_1, out_dim_2, coupling_iter=2, H=128, W=128,
                 use_qk_norm=True):
        super().__init__()
        self.coupling_iter = coupling_iter

        # WGANO backbone
        self.backbone = Model(
            space_dim=2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            fun_dim=1,
            out_dim=n_hidden,
            unified_pos=True,
            ref=8,
            mlp_ratio=1,
            H=H,
            W=W,
            use_qk_norm=use_qk_norm,
        )

        # Task-specific projections
        self.proj_field1 = nn.Linear(n_hidden, head_hidden)
        self.proj_field2 = nn.Linear(n_hidden, head_hidden)
        self.act = nn.GELU()

        # Bidirectional coupling gates (shared across iterations)
        self.gate_1_from_2 = nn.Linear(head_hidden, head_hidden)
        self.gate_2_from_1 = nn.Linear(head_hidden, head_hidden)

        # Output projections
        self.out_field1 = nn.Linear(head_hidden, out_dim_1)
        self.out_field2 = nn.Linear(head_hidden, out_dim_2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [B, H, W, 2 or 3] input

        Returns:
            pred_field1: [B, H, W, out_dim_1]
            pred_field2: [B, H, W, out_dim_2]
        """
        # Extract features
        feat, _, _ = self.backbone(x, x)  # [B, H, W, n_hidden]

        # Project to task-specific representations
        h1 = self.act(self.proj_field1(feat))
        h2 = self.act(self.proj_field2(feat))

        # K-round iterative coupling with shared gate parameters
        for _ in range(self.coupling_iter):
            g1 = torch.sigmoid(self.gate_1_from_2(h2))
            g2 = torch.sigmoid(self.gate_2_from_1(h1))
            h1 = h1 + g1 * h2
            h2 = h2 + g2 * h1

        # Output predictions
        return self.out_field1(h1), self.out_field2(h2)


# Convenient aliases for specific problems
class MHDModel(DualPhysicsCoupledModel):
    """C-WGANO for Magnetohydrodynamics (3-channel EM + 2-channel flow)."""
    def __init__(self, n_hidden=64, n_layers=4, n_head=4, head_hidden=64,
                 coupling_iter=2, H=128, W=128, use_qk_norm=True):
        super().__init__(
            n_hidden, n_layers, n_head, head_hidden,
            out_dim_1=3,  # Jx, Jy, Jz
            out_dim_2=2,  # u, v
            coupling_iter=coupling_iter,
            H=H, W=W, use_qk_norm=use_qk_norm
        )


class EFlowModel(DualPhysicsCoupledModel):
    """C-WGANO for Electroosmotic Flow (1-channel potential + 2-channel velocity)."""
    def __init__(self, n_hidden=64, n_layers=4, n_head=4, head_hidden=64,
                 coupling_iter=2, H=128, W=128, use_qk_norm=True):
        super().__init__(
            n_hidden, n_layers, n_head, head_hidden,
            out_dim_1=1,  # Potential
            out_dim_2=2,  # u, v
            coupling_iter=coupling_iter,
            H=H, W=W, use_qk_norm=use_qk_norm
        )


class NSHeatModel(DualPhysicsCoupledModel):
    """C-WGANO for Thermal Convection (2-channel velocity + 1-channel temperature)."""
    def __init__(self, n_hidden=64, n_layers=4, n_head=4, head_hidden=64,
                 coupling_iter=2, H=128, W=128, use_qk_norm=True):
        super().__init__(
            n_hidden, n_layers, n_head, head_hidden,
            out_dim_1=2,  # u, v
            out_dim_2=1,  # Temperature
            coupling_iter=coupling_iter,
            H=H, W=W, use_qk_norm=use_qk_norm
        )


class VAModel(DualPhysicsCoupledModel):
    """C-WGANO for Vibroacoustics (6-channel acoustic + 6-channel structural)."""
    def __init__(self, n_hidden=64, n_layers=4, n_head=4, head_hidden=64,
                 coupling_iter=2, H=128, W=128, use_qk_norm=True):
        super().__init__(
            n_hidden, n_layers, n_head, head_hidden,
            out_dim_1=6,  # 3 complex acoustic (real + imag)
            out_dim_2=6,  # 3 complex structural (real + imag)
            coupling_iter=coupling_iter,
            H=H, W=W, use_qk_norm=use_qk_norm
        )
