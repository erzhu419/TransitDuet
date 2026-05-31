"""Causal spectral encoders for exogenous time-series streams."""

from .base import CausalSpectralEncoder, ExogenousStreamAdapter
from .causal_ema import CausalEMAEncoder
from .causal_fourier import CausalFourierEncoder
from .causal_state_space import CausalStateSpaceEncoder
from .causal_wavelet import CausalHaarWaveletEncoder, CausalWaveletEncoder

__all__ = [
    "CausalEMAEncoder",
    "CausalFourierEncoder",
    "CausalHaarWaveletEncoder",
    "CausalSpectralEncoder",
    "CausalStateSpaceEncoder",
    "CausalWaveletEncoder",
    "ExogenousStreamAdapter",
]
