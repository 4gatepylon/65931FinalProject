from __future__ import annotations
import torch
import torch.nn as nn
import typing as t
import math
import numpy as np
from jaxtyping import Float, Bool
import pydantic
from typing import Optional

# Our imports
from src.utils import dB_to_linear, BOLTZMANN_CONST, ELEMENTARY_CHARGE


class DACConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255


# TODO(From Dylan): Implement electrical cross talk
class DAC(nn.Module):
    def __init__(
        self,
        configuration: DACConfiguration,
    ):
        super().__init__()
        self.quantization_bitwidth = configuration.quantization_bitwidth
        self.voltage_min = configuration.voltage_min
        self.voltage_max = configuration.voltage_max

        self.max_q_val = 2**configuration.quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = torch.round(tensor.clamp(0, self.max_q_val)) / self.max_q_val
        return self.voltage_min + (self.voltage_max - self.voltage_min) * tensor


class ADCConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255


# TODO(From Dylan): Implement electrical cross talk
class ADC(nn.Module):
    def __init__(
        self,
        configuration: ADCConfiguration,
    ):
        super().__init__()
        self.quantization_bitwidth = configuration.quantization_bitwidth
        self.voltage_min = configuration.voltage_min
        self.voltage_max = configuration.voltage_max

        self.max_q_val = 2**configuration.quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = (tensor - self.voltage_min) / (self.voltage_max - self.voltage_min)
        tensor = torch.round(tensor.clamp(0, 1) * self.max_q_val)
        return tensor


class LaserConfiguration(pydantic.BaseModel):
    # What the voltage is multiplied by to get the optical power.
    optical_gain: float = 0.1


class Laser(nn.Module):
    """
    Returns the power of the wave
    """

    def __init__(
        self,
        configuration: LaserConfiguration,
    ):
        super().__init__()
        self.optical_gain = configuration.optical_gain

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        return tensor * self.optical_gain


class MZMConfiguration(pydantic.BaseModel):
    voltage_min: float = 0
    voltage_max: float = 255
    y_branch_loss_DB: float = 0
    mzm_loss_DB: float = 0


class MZM(nn.Module):
    """Mach-Zehnder Modulator."""

    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        configuration: MZMConfiguration,
    ):
        super().__init__()
        self.weights = weights
        self.voltage_min = configuration.voltage_min
        self.voltage_max = configuration.voltage_max

        self.mzm_loss = dB_to_linear(configuration.mzm_loss_DB)
        self.y_branch_loss = dB_to_linear(configuration.y_branch_loss_DB)

    def forward(self, tensor):
        ideal = (
            tensor
            * (self.weights - self.voltage_min)
            / (self.voltage_max - self.voltage_min)
        )
        return ideal * self.y_branch_loss * self.mzm_loss


class MRRConfiguration(pydantic.BaseModel):
    mrr_k2: float = 0.03
    mrr_fsr_nm: float = 16.1
    mrr_loss_dB: float = 0


class MRR(nn.Module):
    """Micro-ring resonator."""

    def __init__(
        self,
        weights_positive_mask: Bool[torch.Tensor, "N"],
        configuration: MRRConfiguration,
    ):
        super().__init__()
        self.weights_positive_mask = weights_positive_mask.float()
        self.mrr_loss = dB_to_linear(configuration.mrr_loss_dB)
        self.mrr_k2 = configuration.mrr_k2
        self.mrr_fsr_nm = configuration.mrr_fsr_nm

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


class PDConfiguration(pydantic.BaseModel):
    pd_rin_DBCHZ: float = 0
    pd_GHZ: float = 5
    pd_T: float = 300  # Temperature in Kelvin.
    pd_responsivity: float = 1.0  # In A/W.
    pd_dark_current_pA: float = 0  # In pA @ 1V.
    pd_resistance: float = 50  # In Ohm. TODO: Not specified anywhere in the paper.


class PD(nn.Module):
    """Photo-diode"""

    def __init__(
        self,
        configuration: PDConfiguration,
    ):
        super().__init__()
        self.pd_resistance = configuration.pd_resistance
        self.pd_responsivity = configuration.pd_responsivity
        self.pd_dark_current_pA = configuration.pd_dark_current_pA
        self.pd_rin_DBCHZ = configuration.pd_rin_DBCHZ
        self.pd_HZ = configuration.pd_GHZ * 1e9
        self.pd_T = configuration.pd_T

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


class OpticalDotProductConfiguration(pydantic.BaseModel):
    # Set these to override the defaults or whatever was set below in the DAC/ADC/etc...
    # configurations
    input_quantization_bitwidth: Optional[int] = 8
    weight_quantization_bitwidth: Optional[int] = 8
    output_adc_quantization_bitwidth: Optional[int] = 10
    tia_gain: Optional[float] = 1

    # Components that compose into the optical dot product for configuration...
    input_dac_conifiguration: DACConfiguration = DACConfiguration()
    weight_dac_configuration: DACConfiguration = DACConfiguration()
    laser_configuration: LaserConfiguration = LaserConfiguration()
    mzm_configuration: MZMConfiguration = MZMConfiguration()
    mrr_configuration: MRRConfiguration = MRRConfiguration()
    pd_configuration: PDConfiguration = PDConfiguration()
    adc_configuration: ADCConfiguration = ADCConfiguration()


class OpticalDotProduct(nn.Module):
    @staticmethod
    def software_implemented_transformation(
        weights: Float[torch.Tensor, "N"],
        weight_quantization_bitwidth: int,
    ):
        weights_normalization = torch.max(torch.abs(weights)).item()
        if weights_normalization <= 1e-9:
            weights_normalization = 1
        weights = torch.round(
            (weights / weights_normalization) * (2**weight_quantization_bitwidth - 1)
        )
        return weights_normalization, weights

    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        configuration: OpticalDotProductConfiguration,
    ):
        super().__init__()
        # Override the defaults
        if configuration.input_quantization_bitwidth is not None:
            configuration.input_dac_conifiguration.quantization_bitwidth = configuration.input_quantization_bitwidth # fmt: skip
        if configuration.weight_quantization_bitwidth is not None:
            configuration.weight_dac_configuration.quantization_bitwidth = configuration.weight_quantization_bitwidth # fmt: skip
        if configuration.output_adc_quantization_bitwidth is not None:
            configuration.adc_configuration.quantization_bitwidth = configuration.output_adc_quantization_bitwidth # fmt: skip

        # Software Implemented transformation
        self.weights_normalization, weights_transformed = (
            self.software_implemented_transformation(
                weights, configuration.weight_dac_configuration.quantization_bitwidth
            )
        )

        self.input_DAC = DAC(configuration.input_dac_conifiguration)
        self.weight_DAC = DAC(configuration.weight_dac_configuration)
        self.weight_tensor = self.weight_DAC(torch.abs(weights_transformed))

        self.laser = Laser(configuration.laser_configuration)
        self.mzm = MZM(self.weight_tensor, configuration.mzm_configuration)

        self.mrr = MRR(weights_transformed >= 0, configuration.mrr_configuration)
        self.pd_positive = PD(configuration.pd_configuration)
        self.pd_negative = PD(configuration.pd_configuration)

        self.adc = ADC(configuration.adc_configuration)

        self.tia_gain = configuration.tia_gain

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
