import torch
import torch.nn as nn
import typing as t
import math
import numpy as np
BOLTZMANN_CONST = 1.380649e-23
ELEMENTARY_CHARGE = 1.60217663e-19

def dB_to_linear(dB):
    """
    Convert a decibel (dB) value to a linear scale factor.

    For a loss L in dB, the linear efficiency factor is:
       factor = 10^(-L/10)
    """
    return 10 ** (-dB / 10)

#TODO(From Dylan): Implement electrical cross talk
class DAC(nn.Module):
    def __init__(self, 
        quantization_bitwidth=8,
        voltage_min=0,
        voltage_max=255):
        super().__init__()
        self.quantization_bitwidth=quantization_bitwidth
        self.voltage_min=voltage_min
        self.voltage_max=voltage_max
        
        self.max_q_val = 2**quantization_bitwidth - 1

    def forward(self, tensor):
        tensor = torch.round(tensor.clamp(0, self.max_q_val))/self.max_q_val
        return self.voltage_min + (self.voltage_max - self.voltage_min) * tensor
#TODO(From Dylan): Implement electrical cross talk
class ADC(nn.Module):
    def __init__(self, 
        quantization_bitwidth=8,
        voltage_min=0,
        voltage_max=255):
        super().__init__()
        self.quantization_bitwidth=quantization_bitwidth
        self.voltage_min=voltage_min
        self.voltage_max=voltage_max
        
        self.max_q_val = 2**quantization_bitwidth - 1

    def forward(self, tensor):
        tensor = (tensor - self.voltage_min)/(self.voltage_max-self.voltage_min)
        tensor = torch.round(tensor.clamp(0, 1)*self.max_q_val)
        return tensor

#Returns the power of the wave
class Laser(nn.Module):

    def __init__(
        self,
        optical_gain=1, # What the voltage is multiplied by to get the optical power.
    ):
        self.optical_gain = optical_gain


class MZM(nn.Module):

    def __init__(
        self,
        weights,
        voltage_min=0,
        voltage_max=255,
        mzm_loss_DB = 0,
        y_branch_loss_DB = 0,
    ):
        self.weights=weights
        self.voltage_min=voltage_min
        self.voltage_max=voltage_max
        
        self.mzm_loss = dB_to_linear(mzm_loss_DB)
        self.y_branch_loss = dB_to_linear(y_branch_loss_DB)
    def forward(self, tensor):
        ideal = tensor*(self.weights-self.voltage_min)/(self.voltage_max-self.voltage_min)
        return ideal * self.y_branch_loss * self.mzm_loss

class MRR(nn.Module):

    def __init__(
        self,
        weights_positive_mask,
        mrr_k2 = 0.03,
        mrr_fsr_nm = 16.1,
        mrr_loss_dB = 0,
    ):
        self.weights_positive_mask=weights_positive_mask
        self.mrr_loss = dB_to_linear(mrr_loss_dB)
        self.mrr_k2 = mrr_k2
        self.mrr_fsr_nm = mrr_fsr_nm
    def forward(self, tensor):
        stacked = torch.stack([tensor*self.weights_positive_mask, tensor*(1-self.weights_positive_mask)], dim=0)
        
        stacked *= self.mrr_loss
        #TODO(From Dylan): Implement optical cross talk (We still have powers seperated by wavelength at this point)

        ret = stacked.sum(dim=1)
        return ret

class PD(nn.Module):
    def __init__(
        self,
        pd_rin_DBCHZ = 0,
        pd_GHZ = 5,
        pd_T = 300, # Temperature in Kelvin.
        pd_responsivity = 1.0, # In A/W.
        pd_dark_current_pA = 0, # In pA @ 1V.
        pd_resistance = 50, # In Ohm. TODO: Not specified anywhere in the paper.
    ):
        self.pd_resistance = pd_resistance
        self.pd_responsivity = pd_responsivity
        self.pd_dark_current_pA = pd_dark_current_pA
        self.pd_rin_DBCHZ = pd_rin_DBCHZ
        self.pd_HZ = pd_GHZ*1e9
        self.pd_T = pd_T
    def forward(self, tensor):
        tensor = tensor * self.pd_responsivity
        tensor = tensor + self.pd_dark_current_pA

        noise_thermal = torch.randn_like(tensor)*(4*BOLTZMANN_CONST*self.pd_T*self.pd_HZ/self.pd_resistance)
        tensor = tensor + noise_thermal

        noise_shot = torch.randn_like(tensor)*(2*ELEMENTARY_CHARGE*self.pd_HZ)
        tensor = tensor * (1+noise_shot)
        return tensor
class OpticalDotProduct(nn.Module):
    def __init__(
        self,
        weights,
        tia_gain=1
    ):
        #Software Implemented transformation
        self.weights_normalization = torch.max(torch.abs(weights))[0]
        if(self.weights_normalization<=1e-9):
            self.weights_normalization=1
        weights=weights/self.weights_normalization
        #

        self.input_DAC=DAC()
        self.weight_DAC=DAC()
        self.weight_tensor=self.weight_DAC(self.weights)

        self.laser = Laser()
        self.mzm = MZM(torch.abs(self.weight_tensor))

        self.mrr = MRR(weights>=0)
        self.pd_positive=PD()
        self.pd_negative=PD()

        self.adc = ADC()

        self.tia_gain=tia_gain

    def forward(self, tensor):
        input_tensor=self.laser(self.input_DAC(tensor))
        multiplied = self.mzm(input_tensor)
        accumulated = self.mrr(multiplied)
        output_current = self.pd_positive(accumulated[0])-self.pd_positive(accumulated[1])

        output_voltage=output_current*self.tia_gain
        output = self.adc(output_voltage)
        return output



