from __future__ import annotations
import torch
from jaxtyping import Float
from pathlib import Path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config"

BOLTZMANN_CONST = 1.380649e-23  # K_b
ELEMENTARY_CHARGE = 1.60217663e-19

def dB_to_linear(
    dB: float | Float[torch.Tensor, "N"],
) -> float | Float[torch.Tensor, "N"]:
    """
    Convert a decibel (dB) value to a linear scale factor.

    For a loss L in dB, the linear efficiency factor is:
       factor = 10^(-L/10)
    """
    return 10 ** (-dB / 10)
