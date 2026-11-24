"""
Utility modules for WGANO training and evaluation.

Includes:
- normalizer: Data normalization transformers
- testloss: Relative L2 error calculation
"""

from .normalizer import (
    IdentityTransformer,
    UnitTransformer,
    UnitGaussianNormalizer,
)
from .testloss import TestLoss

__all__ = [
    'IdentityTransformer',
    'UnitTransformer',
    'UnitGaussianNormalizer',
    'TestLoss',
]
