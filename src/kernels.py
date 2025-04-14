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
from src.configurations import *

dotenv.load_dotenv()

# Our imports
from src.utils import (
    dB_to_linear,
    BOLTZMANN_CONST,
    ELEMENTARY_CHARGE,
    DEFAULT_CONFIG_PATH,
)


class DAC(nn.Module):
    def __init__(
        self,
        cfg: DACConfiguration,
    ):
        super().__init__()
        self.quantization_bitwidth = cfg.quantization_bitwidth
        self.voltage_min = cfg.voltage_min
        self.voltage_max = cfg.voltage_max

        self.max_q_val = 2**cfg.quantization_bitwidth - 1

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        tensor = torch.round(tensor.clamp(0, self.max_q_val)) / self.max_q_val
        return self.voltage_min + (self.voltage_max - self.voltage_min) * tensor




# TODO(From Dylan): Implement electrical cross talk
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
        tensor = (tensor - self.voltage_min) / (self.voltage_max - self.voltage_min)
        tensor = torch.round(tensor.clamp(0, 1) * self.max_q_val)
        return tensor

class Laser(nn.Module):
    """
    Returns the power of the wave
    """

    def __init__(
        self,
        cfg: LaserConfiguration,
    ):
        super().__init__()
        self.optical_gain = cfg.optical_gain

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "N"]:
        return tensor * self.optical_gain

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

        self.mzm_loss = dB_to_linear(cfg.mzm_loss_DB)
        self.y_branch_loss = dB_to_linear(cfg.y_branch_loss_DB)

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
        cfg: MRRConfiguration,
    ):
        super().__init__()
        self.weights_positive_mask = weights_positive_mask.float()
        self.mrr_loss = dB_to_linear(cfg.mrr_loss_dB)
        self.mrr_k2 = cfg.mrr_k2
        self.mrr_fsr_nm = cfg.mrr_fsr_nm

    def forward(self, tensor):
        stacked = torch.stack(
            [
                tensor * self.weights_positive_mask,
                tensor * (1 - self.weights_positive_mask),
            ],
            dim=0,
        )

        stacked *= self.mrr_loss
        #TODO(From Dylan): Implement optical cross talk (We still have powers seperated by wavelength at this point)
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

class OpticalDotProduct(nn.Module):
    @staticmethod
    def software_implemented_transformation(
        weights: Float[torch.Tensor, "N"],
        weight_quantization_bitwidth: int,
    ):
        # Set weight normalization
        weights_normalization = torch.max(torch.abs(weights)).item()
        if weights_normalization <= 1e-9:
            weights_normalization = 1
        # Update/transform the weights
        weights = torch.round(
            (weights / weights_normalization) * (2**weight_quantization_bitwidth - 1)
        )
        # Return these settings
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
    ):
        super().__init__()
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

        self.input_DAC = DAC(cfg=cfg.input_dac_cfg)
        self.weight_DAC = DAC(cfg=cfg.weight_dac_cfg)
        self.weight_tensor = self.weight_DAC(torch.abs(weights_transformed))

        self.laser = Laser(cfg=cfg.laser_cfg)
        self.mzm = MZM(self.weight_tensor, cfg=cfg.mzm_cfg)

        self.mrr = MRR(weights_transformed >= 0, cfg=cfg.mrr_cfg)
        self.pd_positive = PD(cfg=cfg.pd_cfg)
        self.pd_negative = PD(cfg=cfg.pd_cfg)

        self.adc = ADC(cfg=cfg.adc_cfg)

        self.tia_gain = cfg.tia_gain
        self.cfg = cfg  # Save to deal with overrides

    def forward(self, tensor):
        #Software Implemented transformation
        tensor=torch.clamp(tensor, 0)
        input_normalization =torch.max(tensor, dim=1).values
        input_normalization[input_normalization <= 1e-9] = 1.0
        input_normalization=input_normalization.unsqueeze(1)
        tensor=torch.round((tensor/input_normalization)*(2**self.input_DAC.quantization_bitwidth - 1))
        #

        input_tensor=self.laser(self.input_DAC(tensor))
        multiplied = self.mzm(input_tensor)
        accumulated = self.mrr(multiplied)
        output_current = self.pd_positive(accumulated[0])-self.pd_positive(accumulated[1])
        output_voltage=output_current*self.tia_gain
        output = self.adc(output_voltage)
        
        #Software Implemented transformation
        scale=1
        scale*=self.weights_normalization
        scale/=2**(self.adc.quantization_bitwidth-self.input_DAC.quantization_bitwidth)
        scale/=self.laser.optical_gain * self.mrr.mrr_loss * self.mzm.y_branch_loss*self.mzm.mzm_loss
        scale/=(self.pd_positive.pd_responsivity+self.pd_negative.pd_responsivity)/2
        scale/=(2**self.input_DAC.quantization_bitwidth - 1)
        scale*=input_normalization
        scale=scale.T[0]

        output=output*scale
        #
        return output
    def get_active_configuration(self) -> OpticalDotProductConfiguration:
        return self.cfg

def calculate_conv2d_output_size(input_height, input_width, kernel_height, kernel_width, stride, padding, dilation):
    output_height = (input_height - dilation * (kernel_height - 1) - 1 + 2 * padding) // stride + 1
    output_width = (input_width - dilation * (kernel_width - 1) - 1 + 2 * padding) // stride + 1
    return output_height, output_width

#weights = (Channels OUT, Channels IN, Kernel Y, Kernel X)
#inputs = (Batch, Channels IN, Y, X)
#output = (Batch, Channels OUT, Y OUT, X OUT)
class OpticalConvolution(nn.Module):
    def __init__(
        self,
        weights,
        cfg: OpticalDotProductConfiguration,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        if(bias==None):
            bias = torch.zeros((weights.shape[0],))
        self.kernel_size = weights.shape
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.plcus = []

        weights = F.unfold(weights.float(), kernel_size=1)
        for i in range(weights.shape[0]):
            weight=torch.cat([torch.flatten(weights[i]), torch.tensor([bias[i]])])
            self.plcus.append(OpticalDotProduct(weight, cfg))

    def forward(self, tensor):
        shape = tensor.shape
        conv_dims=calculate_conv2d_output_size(shape[2], shape[3], self.kernel_size[2], self.kernel_size[3], self.stride, self.padding, self.dilation)
        ret_shape=(shape[0], self.kernel_size[0], conv_dims[0], conv_dims[1])
        ret = torch.zeros(ret_shape)
        tensor=tensor.float()

        for i in range(shape[0]):
            batch = tensor[i]
            channel_in=F.unfold(batch, self.kernel_size[2:4], self.dilation, self.padding, self.stride).T
            channel_in=torch.cat([channel_in, torch.ones((channel_in.shape[0], 1))], dim=1)
            for j in range(self.kernel_size[0]):
                res = self.plcus[j](channel_in)
                res = F.fold(res.unsqueeze(0), output_size=ret_shape[2:], kernel_size=1, stride=1)[0]
                ret[i][j] = res
        return ret
    
class OpticalFC(nn.Module):
    def __init__(
        self,
        weights,
        biases,
        cfg: OpticalDotProductConfiguration
    ):
        super().__init__()
        weights = torch.cat([weights.float(), biases.unsqueeze(1).float()], dim=1)
        self.kernel_size = weights.shape
        self.plcus = []
        for i in weights:
            self.plcus.append(OpticalDotProduct(i, cfg))

    def forward(self, tensor):
        shape = tensor.shape
        ret_shape=(shape[0], self.kernel_size[0])
        ret = torch.zeros(ret_shape)
        tensor = torch.cat([tensor.float(), torch.ones([shape[0], 1]).float()], dim=1)
        for i in range(ret_shape[1]):
            ret[:,i] = self.plcus[i](tensor)
        return ret