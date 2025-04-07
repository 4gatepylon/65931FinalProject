import torch
import torch.nn as nn
import typing as t
import math
import numpy as np
from jaxtyping import Float, Bool

BOLTZMANN_CONST = 1.380649e-23  # K_b
ELEMENTARY_CHARGE = 1.60217663e-19


def dB_to_linear(dB):
    """
    Convert a decibel (dB) value to a linear scale factor.

    For a loss L in dB, the linear efficiency factor is:
       factor = 10^(-L/10)
    """
    return 10 ** (-dB / 10)


# TODO(From Dylan): Implement electrical cross talk
class DAC(nn.Module):
    def __init__(
        self,
        quantization_bitwidth: int = 8,
        voltage_min: int = 0,
        voltage_max: int = 255,
    ):
        super().__init__()
        self.quantization_bitwidth = quantization_bitwidth
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        self.max_q_val = 2**quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = torch.round(tensor.clamp(0, self.max_q_val)) / self.max_q_val
        return self.voltage_min + (self.voltage_max - self.voltage_min) * tensor


# TODO(From Dylan): Implement electrical cross talk
class ADC(nn.Module):
    def __init__(
        self,
        quantization_bitwidth: int = 8,
        voltage_min: int = 0,
        voltage_max: int = 255,
    ):
        super().__init__()
        self.quantization_bitwidth = quantization_bitwidth
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        self.max_q_val = 2**quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = (tensor - self.voltage_min) / (self.voltage_max - self.voltage_min)
        tensor = torch.round(tensor.clamp(0, 1) * self.max_q_val)
        return tensor


class Laser(nn.Module):
    """
    Returns the power of the wave
    """

    def __init__(
        self,
        optical_gain: float = 0.1,  # What the voltage is multiplied by to get the optical power.
    ):
        super().__init__()
        self.optical_gain = optical_gain

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        return tensor * self.optical_gain


class MZM(nn.Module):
    """Mach-Zehnder Modulator."""

    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        voltage_min: float = 0,
        voltage_max: float = 255,
        mzm_loss_DB: float = 0,
        y_branch_loss_DB: float = 0,
    ):
        super().__init__()
        self.weights = weights
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        self.mzm_loss = dB_to_linear(mzm_loss_DB)
        self.y_branch_loss = dB_to_linear(y_branch_loss_DB)

    def forward(self, tensor):
        ideal = (
            tensor
            * (self.weights - self.voltage_min)
            / (self.voltage_max - self.voltage_min)
        )
        return ideal * self.y_branch_loss * self.mzm_loss


class MRR(nn.Module):
    """Micro-ring resonator."""

    def __init__(
        self,
        weights_positive_mask: Bool[torch.Tensor, "N"],
        mrr_k2: float = 0.03,  # This is k^2 i.e. squared
        mrr_fsr_nm: float = 16.1,
        mrr_loss_dB: float = 0,
    ):
        super().__init__()
        self.weights_positive_mask = weights_positive_mask.float()
        self.mrr_loss = dB_to_linear(mrr_loss_dB)
        self.mrr_k2 = mrr_k2
        self.mrr_fsr_nm = mrr_fsr_nm

    def forward(self, tensor):
        stacked = torch.stack(
            [
                tensor * self.weights_positive_mask,
                tensor * (1 - self.weights_positive_mask),
            ],
            dim=0,
        )

        stacked *= self.mrr_loss
        # TODO(From Dylan): Implement optical cross talk (We still have powers seperated by wavelength at this point)

        ret = stacked.sum(dim=1)
        return ret


class PD(nn.Module):
    """Photo-diode"""

    def __init__(
        self,
        pd_rin_DBCHZ: float = 0,
        pd_GHZ: float = 5,
        pd_T: float = 300,  # Temperature in Kelvin.
        pd_responsivity: float = 1.0,  # In A/W.
        pd_dark_current_pA: float = 0,  # In pA @ 1V.
        pd_resistance: float = 50,  # In Ohm. TODO: Not specified anywhere in the paper.
    ):
        super().__init__()
        self.pd_resistance = pd_resistance
        self.pd_responsivity = pd_responsivity
        self.pd_dark_current_pA = pd_dark_current_pA
        self.pd_rin_DBCHZ = pd_rin_DBCHZ
        self.pd_HZ = pd_GHZ * 1e9
        self.pd_T = pd_T

    def forward(self, tensor):
        tensor = tensor * self.pd_responsivity
        tensor = tensor + self.pd_dark_current_pA

        noise_thermal = torch.randn_like(tensor) * (
            4 * BOLTZMANN_CONST * self.pd_T * self.pd_HZ / self.pd_resistance
        )
        tensor = tensor + noise_thermal

        noise_shot = torch.randn_like(tensor) * (2 * ELEMENTARY_CHARGE * self.pd_HZ)
        tensor = tensor * (1 + noise_shot)
        return tensor


class OpticalDotProduct(nn.Module):
    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        weight_quantization_bitwidth: int = 8,
        input_quantization_bitwidth: int = 8,
        output_quantization_bitwidth: int = 10,
        tia_gain: float = 1,
    ):
        super().__init__()
        # Software Implemented transformation
        self.weights_normalization = torch.max(torch.abs(weights)).item()
        if self.weights_normalization <= 1e-9:
            self.weights_normalization = 1
        weights = torch.round(
            (weights / self.weights_normalization)
            * (2**weight_quantization_bitwidth - 1)
        )
        #

        self.input_DAC = DAC(input_quantization_bitwidth)
        self.weight_DAC = DAC(weight_quantization_bitwidth)
        self.weight_tensor = self.weight_DAC(torch.abs(weights))

        self.laser = Laser()
        self.mzm = MZM(self.weight_tensor)

        self.mrr = MRR(weights >= 0)
        self.pd_positive = PD()
        self.pd_negative = PD()

        self.adc = ADC(output_quantization_bitwidth)

        self.tia_gain = tia_gain

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "1"]:
        input_tensor = self.laser(self.input_DAC(tensor))
        multiplied = self.mzm(input_tensor)
        accumulated = self.mrr(multiplied)
        output_current = self.pd_positive(accumulated[0]) - self.pd_positive(
            accumulated[1]
        )
        output_voltage = output_current * self.tia_gain
        output = self.adc(output_voltage)

        # Software Implemented transformation
        output *= self.weights_normalization
        output /= 2 ** (
            self.adc.quantization_bitwidth - self.input_DAC.quantization_bitwidth
        )
        output /= (
            self.laser.optical_gain
            * self.mrr.mrr_loss
            * self.mzm.y_branch_loss
            * self.mzm.mzm_loss
        )
        output /= (
            self.pd_positive.pd_responsivity + self.pd_negative.pd_responsivity
        ) / 2
        #
        return output
