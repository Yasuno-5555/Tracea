"""
Tracea Python Integration

Zero-copy PyTorch backend for Stable Diffusion / diffusers
"""

import os
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(os.path.dirname(__file__))
    except OSError: pass

from . import ops
from .tracea import __version__

__all__ = ["ops", "__version__", "patch_conv2d", "TraceaConv2d"]


def patch_conv2d():
    """
    Monkey patch torch.nn.Conv2d to use Tracea backend.
    
    Unsupported configurations automatically fall back to PyTorch.
    """
    import torch
    torch.nn.Conv2d = TraceaConv2d


class TraceaConv2d(torch.nn.Conv2d):
    """
    Drop-in replacement for torch.nn.Conv2d using Tracea backend.
    
    Phase 7.0 Constraints:
    - Supported: groups=1, dilation=(1,1), padding_mode="zeros"
    - All others: automatic fallback to PyTorch
    """
    
    def forward(self, x):
        # Check for unsupported configurations
        if (
            self.groups != 1 or
            self.dilation != (1, 1) or
            self.padding_mode != "zeros"
        ):
            return super().forward(x)
        
        # Use Tracea backend
        return ops.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
