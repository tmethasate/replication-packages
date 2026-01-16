# src/uo2009/dgp/__init__.py
"""
Data generating processes (DGPs) for Ubukata & Oya (2009).

Public API:
- Section 4.1.1 simulator(s): observed prices with MA(2) noise
- Core efficient price simulator: latent prices + SV + stochastic correlation
- Parameter dataclasses for configuring the DGP
"""

from __future__ import annotations

# ---- Parameters ----
from .params import EfficientPriceParams, NoiseParamsMA2

# ---- Core latent process ----
from .efficient_price import (
    EfficientPricePath,
    simulate_efficient_price,
    integrated_variance,
)

# ---- Section 4 simulation glue ----
from .simulate import (
    DGPOutput,
    simulate_section4_1_1,
    simulate_section4_1_1_grid,
)

__all__ = [
    # params
    "EfficientPriceParams",
    "NoiseParamsMA2",
    # latent process
    "EfficientPricePath",
    "simulate_efficient_price",
    "integrated_variance",
    # section 4 outputs / simulators
    "DGPOutput",
    "simulate_section4_1_1",
    "simulate_section4_1_1_grid",
]
