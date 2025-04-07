from __future__ import annotations
import os
from pathlib import Path
import torch
import pydantic_yaml
import torch.nn as nn
import typing as t
import math
import numpy as np
from jaxtyping import Float, Bool
import pydantic
from typing import Optional
import dotenv

dotenv.load_dotenv()

# Our imports
from src.utils import (
    dB_to_linear,
    BOLTZMANN_CONST,
    ELEMENTARY_CHARGE,
    DEFAULT_CONFIG_PATH,
)


class DACConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255


# TODO(From Dylan): Implement electrical cross talk
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


# TODO(Adriano) we might want to move these to their own file some day
class ADCConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255


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


class LaserConfiguration(pydantic.BaseModel):
    # What the voltage is multiplied by to get the optical power.
    optical_gain: float = 0.1


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


class MRRConfiguration(pydantic.BaseModel):
    mrr_k2: float = 0.03
    mrr_fsr_nm: float = 16.1
    mrr_loss_dB: float = 0


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


class OpticalDotProductConfiguration(pydantic.BaseModel):
    tia_gain: Optional[float] = 1

    # Components that compose into the optical dot product for configuration...
    input_dac_cfg: DACConfiguration = DACConfiguration()
    weight_dac_cfg: DACConfiguration = DACConfiguration()
    laser_cfg: LaserConfiguration = LaserConfiguration()
    mzm_cfg: MZMConfiguration = MZMConfiguration()
    mrr_cfg: MRRConfiguration = MRRConfiguration()
    pd_cfg: PDConfiguration = PDConfiguration()
    adc_cfg: ADCConfiguration = ADCConfiguration()

    # TODO(Adriano) move all this configuration load/store stuff to a base class so we can, for example, have different
    # types of configs in different files (this is something that is low priority, so do it eventually when we know
    # what will be most convenient)
    @staticmethod
    def configs_path() -> Path:
        path = DEFAULT_CONFIG_PATH if os.environ.get("ALBIREO_CONFIG_PATH", None) is None else Path(os.environ["ALBIREO_CONFIG_PATH"]) # fmt: skip
        if not path.exists():
            raise FileNotFoundError(f"Config path {path} does not exist")
        return path

    #### Helper methods to save the config (you should always save to the `config/` folder) ####
    def save_to_path(self, path: Path | str):
        pydantic_yaml.to_yaml_file(path, self)

    def save(self, name: str):
        save_path = self.configs_path() / f"{name}.yaml"
        if save_path.exists():
            raise FileExistsError(f"Config file {save_path} already exists")
        self.save_to_path(save_path)

    #### Helper methods to load common configurations we have in the config folder ####
    @staticmethod
    def from_config_path(config_path: Path | str) -> OpticalDotProductConfiguration:
        return pydantic_yaml.parse_yaml_raw_as(OpticalDotProductConfiguration, Path(config_path).read_text()) # fmt: skip

    @staticmethod
    def from_config_name(config_name: str) -> OpticalDotProductConfiguration:
        configs_path = OpticalDotProduct.configs_path()
        config_path = configs_path / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist")
        return OpticalDotProductConfiguration.from_config_path(config_path)

    @staticmethod
    def from_default() -> OpticalDotProductConfiguration:
        return OpticalDotProductConfiguration.from_config_name("default")

    @staticmethod
    def from_zero_noise_high_precision() -> OpticalDotProductConfiguration:
        return OpticalDotProductConfiguration.from_config_name("zero_noise_high_precision") # fmt: skip


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

    def get_active_configuration(self) -> OpticalDotProductConfiguration:
        return self.cfg

    def forward(self, tensor: Float[torch.Tensor, "N"]) -> Float[torch.Tensor, "1"]:
        input_tensor=self.laser(self.input_DAC(tensor)) # fmt: skip
        multiplied = self.mzm(input_tensor) # fmt: skip
        accumulated = self.mrr(multiplied) # fmt: skip
        output_current = self.pd_positive(accumulated[0])-self.pd_positive(accumulated[1]) # fmt: skip
        output_voltage=output_current*self.tia_gain # fmt: skip
        output = self.adc(output_voltage) # fmt: skip

        # Software Implemented transformation
        output*=self.weights_normalization # fmt: skip
        output/=2**(self.adc.quantization_bitwidth-self.input_DAC.quantization_bitwidth) # fmt: skip
        output/=self.laser.optical_gain * self.mrr.mrr_loss * self.mzm.y_branch_loss*self.mzm.mzm_loss # fmt: skip
        output/=(self.pd_positive.pd_responsivity+self.pd_negative.pd_responsivity)/2 # fmt: skip
        return output
