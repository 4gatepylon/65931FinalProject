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
from .configurations import *

dotenv.load_dotenv()

# Our imports
from .utils import (
    loss_dB_to_linear,
    BOLTZMANN_CONST,
    ELEMENTARY_CHARGE,
    DEFAULT_CONFIG_PATH,
)


def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = best_device()


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

        noise_thermal = torch.randn_like(tensor, device=device) * 4 * BOLTZMANN_CONST * self.pd_T * self.pd_HZ / self.pd_resistance # fmt: skip
        tensor = tensor + noise_thermal
        noise_shot = torch.randn_like(tensor, device=device) * (2 * ELEMENTARY_CHARGE * self.pd_HZ)
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
    ):
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
        self.weight_DAC = DAC(cfg=cfg.weight_dac_cfg, is_weight=True)
        self.weight_tensor = self.weight_DAC(torch.abs(weights_transformed))

        self.laser = Laser(cfg=cfg.laser_cfg)
        self.mzm = MZM(self.weight_tensor, cfg=cfg.mzm_cfg)

        self.mrr = MRR(weights_transformed >= 0, cfg=cfg.mrr_cfg)
        self.pd_positive = PD(cfg=cfg.pd_cfg)
        self.pd_negative = PD(cfg=cfg.pd_cfg)

        self.tia = TIA(cfg=cfg.tia_cfg)

        self.adc = ADC(cfg=cfg.adc_cfg)

        self.cfg = cfg  # Save to deal with overrides

    def forward(self, tensor):
        #Software Implemented transformation
        tensor_positive_mask = tensor > 0
        tensor=torch.abs(tensor)
        input_normalization =torch.max(tensor, dim=1).values
        input_normalization[input_normalization <= 1e-9] = 1.0
        input_normalization=input_normalization.unsqueeze(1)
        tensor=torch.round((tensor/input_normalization)*(2**self.input_DAC.quantization_bitwidth - 1))
        #

        input_tensor=self.laser(self.input_DAC(tensor))
        multiplied = self.mzm(input_tensor)
        accumulated = self.mrr(multiplied, tensor_positive_mask)
        output_current = self.pd_positive(accumulated[0])-self.pd_positive(accumulated[1])
        output_voltage, negatives = self.tia(output_current)
        output = self.adc(output_voltage)*(negatives*2-1)
        
        #Software Implemented transformation
        scale=1
        scale*=self.weights_normalization
        scale/=2**(self.adc.quantization_bitwidth-self.input_DAC.quantization_bitwidth)
        scale/=self.laser.optical_gain*self.tia.gain * self.mrr.mrr_loss * self.mzm.y_branch_loss*self.mzm.mzm_loss
        scale/=(self.pd_positive.pd_responsivity+self.pd_negative.pd_responsivity)/2
        # scale/=(2**self.input_DAC.quantization_bitwidth - 1)
        scale*=input_normalization
        scale=scale.T[0]

        output=output*scale
        #
        return output
    def get_active_configuration(self) -> OpticalDotProductConfiguration:
        return self.cfg

def calculate_conv2d_output_size(input_height, input_width, kernel_height, kernel_width, stride, padding, dilation):
    if(isinstance(padding, int)):
        padding = (padding, padding)
    if(isinstance(stride, int)):
        stride = (stride, stride)
    if(isinstance(dilation, int)):
        dilation = (dilation, dilation)
    output_height = (input_height - dilation[0] * (kernel_height - 1) - 1 + 2 * padding[0]) // stride[0] + 1
    output_width = (input_width - dilation[1] * (kernel_width - 1) - 1 + 2 * padding[1]) // stride[1] + 1
    return output_height, output_width

#weights = (Channels OUT, Channels IN, Kernel Y, Kernel X)
#inputs = (Batch, Channels IN, Y, X)
#output = (Batch, Channels OUT, Y OUT, X OUT)
class BackupOpticalConvolution(nn.Module):
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
            bias = torch.zeros((weights.shape[0],), device=device)
        self.kernel_size = weights.shape
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.plcus = []

        weights = F.unfold(weights.float(), kernel_size=1)
        for i in range(weights.shape[0]):
            weight=torch.cat([torch.flatten(weights[i]), torch.tensor([bias[i]], device=device)])
            self.plcus.append(OpticalDotProduct(weight, cfg))

    def forward(self, tensor):
        shape = tensor.shape
        conv_dims=calculate_conv2d_output_size(shape[2], shape[3], self.kernel_size[2], self.kernel_size[3], self.stride, self.padding, self.dilation)
        ret_shape=(shape[0], self.kernel_size[0], conv_dims[0], conv_dims[1])
        ret = torch.zeros(ret_shape, device=device)
        tensor=tensor.float()

        for i in range(shape[0]):
            batch = tensor[i]
            channel_in=F.unfold(batch, self.kernel_size[2:4], self.dilation, self.padding, self.stride).T
            channel_in=torch.cat([channel_in, torch.ones((channel_in.shape[0], 1), device=device)], dim=1)
            for j in range(self.kernel_size[0]):
                res = self.plcus[j](channel_in)
                res = F.fold(res.unsqueeze(0), output_size=ret_shape[2:], kernel_size=1, stride=1)[0]
                ret[i][j] = res
        return ret
    
class BackupOpticalFC(nn.Module):
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
        ret = torch.zeros(ret_shape, device=device)
        tensor = torch.cat([tensor.float(), torch.ones([shape[0], 1], device=device).float()], dim=1)
        for i in range(ret_shape[1]):
            ret[:,i] = self.plcus[i](tensor)
        return ret
    



class OpticalConvolution(nn.Module):
    def __init__(
        self,
        weights: Float[torch.Tensor, "Out Cin Kh Kw"], # Standard Conv weights
        cfg: OpticalDotProductConfiguration,
        bias: Optional[Float[torch.Tensor, "Out"]] = None,
        stride: t.Union[int, t.Tuple[int, int]] = 1,
        padding: t.Union[int, t.Tuple[int, int]] = 0,
        dilation: t.Union[int, t.Tuple[int, int]] = 1,
    ):
        super().__init__()
        self.out_channels, self.in_channels, self.kernel_h, self.kernel_w = weights.shape
        self.kernel_size = (self.kernel_h, self.kernel_w)
        self.stride = stride # Store for fold calculation if needed, pass to Unfold
        self.padding = padding
        self.dilation = dilation
        self.cfg = cfg

        if bias is None:
            bias = torch.zeros(self.out_channels, device=weights.device) # Ensure bias exists

        # 1. Prepare weights for the equivalent FC layer
        # Flatten C_in, KH, KW dimensions
        weights_flat = weights.view(self.out_channels, -1) # Shape: (Out, Cin*Kh*Kw)

        # 2. Instantiate the optimized OpticalFC layer
        # OpticalFC handles augmenting weights with bias internally
        self.optical_fc = OpticalFC(weights=weights_flat, biases=bias, cfg=cfg)

        # 3. Store parameters for Unfold and Fold
        # Unfold extracts patches based on conv parameters
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def forward(self, tensor: Float[torch.Tensor, "B Cin Hin Win"]):
        # Inference only optimization
        B, C_in, H_in, W_in = tensor.shape
        if C_in != self.in_channels:
                raise ValueError(f"Input channels {C_in} != expected {self.in_channels}")

        # Calculate output spatial dimensions needed for Fold
        # Note: Ensure padding is handled correctly (tuple vs int)
        padding_tuple = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        stride_tuple = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        dilation_tuple = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)

        H_out = math.floor((H_in + 2 * padding_tuple[0] - dilation_tuple[0] * (self.kernel_h - 1) - 1) / stride_tuple[0] + 1)
        W_out = math.floor((W_in + 2 * padding_tuple[1] - dilation_tuple[1] * (self.kernel_w - 1) - 1) / stride_tuple[1] + 1)
        output_size = (H_out, W_out)

        # 1. Unfold: Extract patches (im2col)
        # Input: (B, Cin, Hin, Win)
        # Output: (B, Cin*Kh*Kw, L), where L = Hout * Wout
        patches = self.unfold(tensor.float())
        L = patches.shape[2] # Number of patches

        # 2. Prepare for OpticalFC:
        # Transpose and reshape patches: (B, Cin*Kh*Kw, L) -> (B, L, Cin*Kh*Kw) -> (B*L, Cin*Kh*Kw)
        in_features_fc = patches.shape[1] # Cin*Kh*Kw
        patches_fc_input = patches.transpose(1, 2).reshape(-1, in_features_fc)

        # 3. Apply OpticalFC
        # Input: (B*L, Cin*Kh*Kw) -> OpticalFC augments with bias internally
        # Output: (B*L, Cout)
        output_fc = self.optical_fc(patches_fc_input) # OpticalFC already handles augmented input/weights

        # 4. Fold: Reshape output back to spatial format (col2im)
        # Need to reshape output_fc (B*L, Cout) -> (B, L, Cout) -> (B, Cout, L)
        output_fc_reshaped = output_fc.view(B, L, self.out_channels).transpose(1, 2) # Shape: (B, Cout, L)

        # Fold requires input (B, Cout*Kh*Kw_fold, L). Our "fold kernel" is 1x1.
        # So input shape (B, Cout, L) is correct.
        # Fold requires kernel_size matching the *original* convolution for mapping locations correctly.
        # However, F.fold seems designed to reverse F.unfold directly. Let's test with kernel_size=1 for fold.
        # According to docs, fold sums overlapping blocks. unfold -> mm -> fold is the standard way.
        # Let's use kernel_size=(1,1) for fold as we are placing single values (Cout dims) at each location L.

        # output = F.fold(output_fc_reshaped, output_size=output_size, kernel_size=(1, 1), stride=(1,1)) # Does fold need stride? yes.
        # F.fold might not be the right tool here if the goal is just reshaping.
        # Let's manually reshape and permute like in the first example solution.

        # Reshape from (B*L, Cout) -> (B, L, Cout) -> (B, Hout, Wout, Cout)
        output_spatial = output_fc.view(B, H_out, W_out, self.out_channels)
        # Permute to (B, Cout, Hout, Wout)
        output = output_spatial.permute(0, 3, 1, 2).contiguous()

        return output



class OpticalFC(nn.Module):
    def __init__(
        self,
        weights: Float[torch.Tensor, "Out In"],
        biases: Float[torch.Tensor, "Out"],
        cfg: OpticalDotProductConfiguration
    ):
        super().__init__()
        self.out_features, self.in_features = weights.shape
        self.cfg = cfg

        # Augment weights with biases for combined dot product
        # Each OpticalDotProduct unit will handle one augmented row
        augmented_weights = torch.cat([weights.float(), biases.unsqueeze(1).float()], dim=1) # Shape: (Out, In + 1)

        # Create a ModuleList of OpticalDotProduct units, one for each output feature
        self.optical_dots = nn.ModuleList(
            [OpticalDotProduct(augmented_weights[i], cfg) for i in range(self.out_features)]
        )

    def forward(self, tensor: Float[torch.Tensor, "Batch In"]):
        # Inference only optimization
        with torch.no_grad():
            # Augment input tensor with ones for bias calculation
            ones = torch.ones(tensor.shape[0], 1, device=tensor.device, dtype=tensor.dtype)
            augmented_tensor = torch.cat([tensor.float(), ones], dim=1) # Shape: (Batch, In + 1)

            # Apply each OpticalDotProduct unit to the augmented input batch
            # Use list comprehension and torch.stack for efficiency
            # Each dot_product(augmented_tensor) returns shape (Batch,)
            outputs = [dot_product(augmented_tensor) for dot_product in self.optical_dots]

            # Stack results along the feature dimension
            output_tensor = torch.stack(outputs, dim=-1) # Shape: (Batch, Out)

        return output_tensor