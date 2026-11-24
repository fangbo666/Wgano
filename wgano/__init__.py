"""
WGANO: Wavelet-Guided Attention Neural Operator

Official implementation for solving single-physics and multi-physics PDEs.

Author: Bo Fang
Institution: China University of Petroleum (East China)
"""

from .model import Model, HFGuidedWaveletAttn, N1Block, timestep_embedding
from .model_1d import Model1D, HFGuidedWaveletAttn1D, Block1D
from .coupled_model import (
    PhysicsCoupledHeadModel,
    DualPhysicsCoupledModel,
    MHDModel,
    EFlowModel,
    NSHeatModel,
    VAModel,
)

__version__ = '1.0.0'
__author__ = 'Bo Fang'

__all__ = [
    # Single-physics 2D model
    'Model',
    'HFGuidedWaveletAttn',
    'N1Block',
    'timestep_embedding',

    # Single-physics 1D model (for irregular mesh)
    'Model1D',
    'HFGuidedWaveletAttn1D',
    'Block1D',

    # Multi-physics models
    'PhysicsCoupledHeadModel',
    'DualPhysicsCoupledModel',
    'MHDModel',
    'EFlowModel',
    'NSHeatModel',
    'VAModel',
]
