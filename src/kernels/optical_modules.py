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
from src.kernels.configurations import OpticalDotProductConfiguration
from src.kernels.optical_dotproduct import OpticalDotProduct

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
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if(bias==None):
            bias = torch.zeros((weights.shape[0],), device=device)
        self.kernel_size = weights.shape
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.plcus = []
        self.device = device

        weights = F.unfold(weights.float(), kernel_size=1)
        for i in range(weights.shape[0]):
            weight=torch.cat([torch.flatten(weights[i]), torch.tensor([bias[i]], device=device)])
            self.plcus.append(OpticalDotProduct(weight, cfg))

    def forward(self, tensor):
        shape = tensor.shape
        conv_dims=calculate_conv2d_output_size(shape[2], shape[3], self.kernel_size[2], self.kernel_size[3], self.stride, self.padding, self.dilation)
        ret_shape=(shape[0], self.kernel_size[0], conv_dims[0], conv_dims[1])
        ret = torch.zeros(ret_shape, device=self.device)
        tensor=tensor.float()

        for i in range(shape[0]):
            batch = tensor[i]
            channel_in=F.unfold(batch, self.kernel_size[2:4], self.dilation, self.padding, self.stride).T
            channel_in=torch.cat([channel_in, torch.ones((channel_in.shape[0], 1), device=self.device)], dim=1)
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
        cfg: OpticalDotProductConfiguration,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        weights = torch.cat([weights.float(), biases.unsqueeze(1).float()], dim=1)
        self.kernel_size = weights.shape
        self.plcus = []
        self.device = device
        for i in weights:
            self.plcus.append(OpticalDotProduct(i, cfg))

    def forward(self, tensor):
        shape = tensor.shape
        ret_shape=(shape[0], self.kernel_size[0])
        ret = torch.zeros(ret_shape, device=self.device)
        tensor = torch.cat([tensor.float(), torch.ones([shape[0], 1], device=self.device).float()], dim=1)
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
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.out_channels, self.in_channels, self.kernel_h, self.kernel_w = weights.shape
        self.kernel_size = (self.kernel_h, self.kernel_w)
        self.stride = stride # Store for fold calculation if needed, pass to Unfold
        self.padding = padding
        self.dilation = dilation
        self.cfg = cfg
        self.device = device
        if self.device is None:
            self.device = weights.device
        if bias is None:
            bias = torch.zeros(self.out_channels, device=self.device) # Ensure bias exists

        # 1. Prepare weights for the equivalent FC layer
        # Flatten C_in, KH, KW dimensions
        weights_flat = weights.view(self.out_channels, -1).to(self.device) # Shape: (Out, Cin*Kh*Kw)

        # 2. Instantiate the optimized OpticalFC layer
        # OpticalFC handles augmenting weights with bias internally
        self.optical_fc = OpticalFC(weights=weights_flat, biases=bias, cfg=cfg, device=self.device)

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
        tensor_device = tensor.device
        tensor = tensor.to(self.device)
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
        output = output.to(tensor_device)
        return output



class OpticalFC(nn.Module):
    def __init__(
        self,
        weights: Float[torch.Tensor, "Out In"],
        biases: Float[torch.Tensor, "Out"],
        cfg: OpticalDotProductConfiguration,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.out_features, self.in_features = weights.shape
        self.cfg = cfg
        self.device = device
        if self.device is None:
            self.device = weights.device

        # Augment weights with biases for combined dot product
        # Each OpticalDotProduct unit will handle one augmented row
        augmented_weights = torch.cat([weights.float(), biases.unsqueeze(1).float()], dim=1) # Shape: (Out, In + 1)

        # Create a ModuleList of OpticalDotProduct units, one for each output feature
        self.optical_dots = nn.ModuleList(
            [OpticalDotProduct(augmented_weights[i], cfg, device=self.device) for i in range(self.out_features)]
        )

    def forward(self, tensor: Float[torch.Tensor, "Batch In"]):
        # Inference only optimization
        with torch.no_grad():
            original_shape = tensor.shape[:-1]
            in_features = tensor.shape[-1]
            tensor = tensor.view(-1, in_features)
            # Augment input tensor with ones for bias calculation
            ones = torch.ones(tensor.shape[0], 1, device=self.device, dtype=tensor.dtype)
            augmented_tensor = torch.cat([tensor.float(), ones], dim=1) # Shape: (Batch, In + 1)

            # Apply each OpticalDotProduct unit to the augmented input batch
            # Use list comprehension and torch.stack for efficiency
            # Each dot_product(augmented_tensor) returns shape (Batch,)
            outputs = [dot_product(augmented_tensor) for dot_product in self.optical_dots]

            # Stack results along the feature dimension
            output_tensor = torch.stack(outputs, dim=-1) # Shape: (Batch, Out)
            output_tensor = output_tensor.view(*original_shape, -1)

        return output_tensor