from __future__ import annotations
import os
from pathlib import Path
import torch
import pydantic_yaml
import torch.nn as nn
import torch.nn.functional as F
import typing as t
import math
import numpy as np
from jaxtyping import Float, Bool
import pydantic
from typing import Optional
import dotenv
from src.kernels.configurations import (
    OpticalDotProductConfiguration,
    DACConfiguration,
    ADCConfiguration,
    LaserConfiguration,
    MZMConfiguration,
    MRRConfiguration,
    PDConfiguration, TIAConfiguration
)

# Our imports
from src.kernels.utils import (
    loss_dB_to_linear,
    BOLTZMANN_CONST,
    ELEMENTARY_CHARGE,
)

class DAC(nn.Module):
    def __init__(
        self,
        cfg: DACConfiguration,
        is_weight: bool = False,
    ):
        super().__init__()
        self.quantization_bitwidth = cfg.quantization_bitwidth
        self.voltage_min = cfg.voltage_min
        self.voltage_max = cfg.voltage_max
        self.is_weight = is_weight
        self.max_q_val = 2**cfg.quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        if self.is_weight:
            return tensor # Keep in [-1, 1]
        # Assumes input is already quantized.
        tensor = torch.round(tensor.clamp(0, self.max_q_val)) / self.max_q_val
        return self.voltage_min + (self.voltage_max - self.voltage_min) * tensor




class ADC(nn.Module):
    def __init__(
        self,
        cfg: ADCConfiguration,
    ):
        super().__init__()
        self.quantization_bitwidth = cfg.quantization_bitwidth
        self.voltage_min = cfg.voltage_min
        self.voltage_max = cfg.voltage_max

        self.max_q_val = 2**cfg.quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        """
        Simulate precision loss in ADC.
        Caused by clamping to the voltage range, then rounding for quantization.
        """
        # Assume voltage_min is simply 0.
        tensor = tensor.clamp(0, self.voltage_max)
        scaling_factor = (self.max_q_val / self.voltage_max)
        tensor = torch.round(tensor * scaling_factor) / scaling_factor
        return tensor

class Laser(nn.Module):
    """
    Returns the power of the wave
    """

    def __init__(
        self,
        cfg: LaserConfiguration,
        device: Optional[str | torch.device] = "cpu",
    ):
        super().__init__()
        self.optical_gain = cfg.optical_gain
        self.awg_cross_talk_rate = cfg.awg_cross_talk_rate
        # X = X + sum(surrounding X * cross_talk_rate) 
        self.cross_talk_kernel = torch.tensor([
           self.awg_cross_talk_rate/4, self.awg_cross_talk_rate, 1.0, self.awg_cross_talk_rate, self.awg_cross_talk_rate/4
        ], device=device).view(1, 1, -1)

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = tensor * self.optical_gain
        # Tensor is NxM
        # Each of the N rows receives the 1D convolution
        # with the cross talk kernel
        tensor = tensor.unsqueeze(1)
        tensor = F.conv1d(
            tensor,
            self.cross_talk_kernel,
            padding='same',
            groups=1,
        )
        tensor = tensor.squeeze(1)
        return tensor
    

class MZM(nn.Module):
    """Mach-Zehnder Modulator."""

    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        cfg: MZMConfiguration,
    ):
        super().__init__()
        self.weights = weights
        self.voltage_min = cfg.voltage_min
        self.voltage_max = cfg.voltage_max

        self.mzm_loss = loss_dB_to_linear(cfg.mzm_loss_DB)
        self.y_branch_loss = loss_dB_to_linear(cfg.y_branch_loss_DB)

    def forward(self, tensor):
        ideal = (
            tensor
            * self.weights
            / (self.voltage_max - self.voltage_min)
        )
        return ideal * self.y_branch_loss * self.mzm_loss

class MRR(nn.Module):
    """Micro-ring resonator."""

    def __init__(
        self,
        weights_positive_mask: Bool[torch.Tensor, "N"],
        cfg: MRRConfiguration,
        device: Optional[str | torch.device] = "cpu",
    ):
        super().__init__()
        self.weights_positive_mask = weights_positive_mask
        self.mrr_loss = loss_dB_to_linear(cfg.mrr_loss_dB)
        self.mrr_cross_talk_rate = cfg.mrr_cross_talk_rate
        self.cross_talk_kernel = torch.tensor([
            self.mrr_cross_talk_rate/4, self.mrr_cross_talk_rate, 1.0, self.mrr_cross_talk_rate, self.mrr_cross_talk_rate/4
        ], device=device).view(1, 1, -1)

    def forward(self, tensor, tensor_positive_mask):
        positive_mask = (self.weights_positive_mask^tensor_positive_mask^1).float()
        positive_tensor = tensor * positive_mask
        negative_tensor = tensor * (1 - positive_mask)
        stacked = torch.stack(
            [
                positive_tensor,
                negative_tensor,
            ],
            dim=0,
        )
        # Apply cross talk
        B, R, C = stacked.shape
        stacked = stacked.reshape(B * R, 1, C)
        stacked = F.conv1d(
            stacked,
            self.cross_talk_kernel,
            padding='same',
            groups=1,
        )
        stacked = stacked.squeeze(1).reshape(B, R, C)
        # Apply loss
        stacked *= self.mrr_loss
        ret = stacked.sum(dim=2)
        return ret

class PD(nn.Module):
    """Photo-diode"""

    def __init__(
        self,
        cfg: PDConfiguration,
    ):
        super().__init__()
        self.pd_resistance = cfg.pd_resistance
        self.pd_responsivity = cfg.pd_responsivity
        self.pd_dark_current_pA = cfg.pd_dark_current_pA
        self.pd_rin_DBCHZ = cfg.pd_rin_DBCHZ
        self.pd_HZ = cfg.pd_GHZ * 1e9
        self.pd_T = cfg.pd_T

    def forward(self, tensor):
        tensor = tensor * self.pd_responsivity
        tensor = tensor + self.pd_dark_current_pA

        noise_thermal = torch.randn_like(tensor) * 4 * BOLTZMANN_CONST * self.pd_T * self.pd_HZ / self.pd_resistance # fmt: skip
        tensor = tensor + noise_thermal
        noise_shot = torch.randn_like(tensor) * (2 * ELEMENTARY_CHARGE * self.pd_HZ)
        tensor = tensor * (1 + noise_shot)
        return tensor
    
class TIA(nn.Module):
    """TIA"""

    def __init__(
        self,
        cfg: TIAConfiguration,
    ):
        super().__init__()
        self.gain=cfg.gain

    def forward(self, tensor):
        return torch.abs(tensor*self.gain), (tensor>=0).float()

class OpticalDotProduct(nn.Module):
    @staticmethod
    def software_implemented_transformation(
        weights: Float[torch.Tensor, "N"],
        weight_quantization_bitwidth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Set weight normalization
        weights_normalization = torch.max(torch.abs(weights)).item()
        if weights_normalization <= 1e-9:
            weights_normalization = 1
        # Update/transform the weights
        # weights = torch.round(
        #     (weights / weights_normalization) * (2**weight_quantization_bitwidth - 1)
        # )
        weights = weights / weights_normalization
        # Return these settings
        weights_normalization = torch.tensor(weights_normalization)
        return weights_normalization, weights

    def __init__(
        self,
        weights: Float[torch.Tensor, "N"],
        cfg: OpticalDotProductConfiguration,
        # Override options
        # NOTE: this is technically not something you should use because
        # it will lead to confusion later if you save your configuration
        # and did not save this (i.e. this will modify configuration objects
        # but you will need to get the updated version from the OpticalDotProduct)
        input_quantization_bitwidth: Optional[int] = None,
        weight_quantization_bitwidth: Optional[int] = None,
        output_adc_quantization_bitwidth: Optional[int] = None,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = weights.device
        # Override the defaults
        if input_quantization_bitwidth is not None:
            cfg.input_dac_cfg.quantization_bitwidth = input_quantization_bitwidth # fmt: skip
            print(f"Overwriting input quantization bitwidth, from {cfg.input_dac_cfg.quantization_bitwidth} to {input_quantization_bitwidth}") # fmt: skip
        if weight_quantization_bitwidth is not None:
            cfg.weight_dac_cfg.quantization_bitwidth = weight_quantization_bitwidth # fmt: skip
            print(f"Overwriting weight quantization bitwidth, from {cfg.weight_dac_cfg.quantization_bitwidth} to {weight_quantization_bitwidth}") # fmt: skip
        if output_adc_quantization_bitwidth is not None:
            cfg.adc_cfg.quantization_bitwidth = output_adc_quantization_bitwidth # fmt: skip
            print(f"Overwriting output ADC quantization bitwidth, from {cfg.adc_cfg.quantization_bitwidth} to {output_adc_quantization_bitwidth}") # fmt: skip
        # Software Implemented transformation
        self.weights_normalization, weights_transformed = self.software_implemented_transformation(weights, cfg.weight_dac_cfg.quantization_bitwidth) # fmt: skip
        self.weights_normalization = self.weights_normalization.to(self.device)
        self.weights_transformed = weights_transformed.to(self.device)

        self.input_DAC = DAC(cfg=cfg.input_dac_cfg)
        self.weight_DAC = DAC(cfg=cfg.weight_dac_cfg, is_weight=True)
        self.weight_tensor = self.weight_DAC(torch.abs(weights_transformed))

        self.laser = Laser(cfg=cfg.laser_cfg, device=self.device)
        self.mzm = MZM(self.weight_tensor, cfg=cfg.mzm_cfg)

        self.mrr = MRR(weights_transformed >= 0, cfg=cfg.mrr_cfg, device=self.device)
        self.pd_positive = PD(cfg=cfg.pd_cfg)
        self.pd_negative = PD(cfg=cfg.pd_cfg)

        self.tia = TIA(cfg=cfg.tia_cfg)

        self.adc = ADC(cfg=cfg.adc_cfg)

        self.cfg = cfg  # Save to deal with overrides
        

    def forward(self, tensor: Float[torch.Tensor, "B N"]) -> Float[torch.Tensor, "B"]:
        if tensor.ndim != 2:
            raise ValueError(f"Tensor must have 2 dimensions, got shape {tensor.shape}")
        tensor_device = tensor.device
        tensor = tensor.to(self.device)

        #Software Implemented transformation
        tensor_positive_mask = tensor > 0
        tensor=torch.abs(tensor)
        input_normalization =torch.max(tensor, dim=1).values
        assert input_normalization.ndim == 1
        input_normalization[input_normalization <= 1e-9] = 1.0
        input_normalization=input_normalization.unsqueeze(1)
        assert input_normalization.ndim == 2
        tensor=torch.round((tensor/input_normalization)*(2**self.input_DAC.quantization_bitwidth - 1))
        #

        input_tensor=self.laser(self.input_DAC(tensor))
        multiplied = self.mzm(input_tensor)
        accumulated = self.mrr(multiplied, tensor_positive_mask)
        output_current = self.pd_positive(accumulated[0])-self.pd_positive(accumulated[1])
        output_voltage, negatives = self.tia(output_current)
        output = self.adc(output_voltage)*(negatives*2-1)
        
        #Software Implemented transformation
        scale=torch.ones((1, 1), device=self.device)
        scale*=self.weights_normalization
        scale/=2**(self.adc.quantization_bitwidth-self.input_DAC.quantization_bitwidth)
        scale/=self.laser.optical_gain*self.tia.gain * self.mrr.mrr_loss * self.mzm.y_branch_loss*self.mzm.mzm_loss
        scale/=(self.pd_positive.pd_responsivity+self.pd_negative.pd_responsivity)/2
        # scale/=(2**self.input_DAC.quantization_bitwidth - 1)
        scale*=input_normalization
        scale=scale.T[0]

        output=output*scale
        #
        return output.to(tensor_device)
    
    def get_active_configuration(self) -> OpticalDotProductConfiguration:
        return self.cfg