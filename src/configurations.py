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

class DACConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255
    
class ADCConfiguration(pydantic.BaseModel):
    quantization_bitwidth: int = 8
    voltage_min: int = 0
    voltage_max: int = 255
    

class LaserConfiguration(pydantic.BaseModel):
    # What the voltage is multiplied by to get the optical power.
    optical_gain: float = 1.0
    awg_cross_talk_rate: float = 0.0 # NOTE: For inputs only; weights do not have cross-talk.
    
class MZMConfiguration(pydantic.BaseModel):
    voltage_min: float = 0
    voltage_max: float = 255
    y_branch_loss_DB: float = 0
    mzm_loss_DB: float = 0
    
class MRRConfiguration(pydantic.BaseModel):
    mrr_loss_dB: float = 0
    mrr_cross_talk_rate: float = 0.0

class PDConfiguration(pydantic.BaseModel):
    pd_rin_DBCHZ: float = 0
    pd_GHZ: float = 5
    pd_T: float = 300  # Temperature in Kelvin.
    pd_responsivity: float = 1.0  # In A/W.
    pd_dark_current_pA: float = 0  # In pA @ 1V.
    pd_resistance: float = 1.0  # In Ohm. TODO: Not specified anywhere in the paper.


class TIAConfiguration(pydantic.BaseModel):
    gain: Optional[float] = 1


    
class OpticalDotProductConfiguration(pydantic.BaseModel):
    # Components that compose into the optical dot product for configuration...
    input_dac_cfg: DACConfiguration = DACConfiguration()
    weight_dac_cfg: DACConfiguration = DACConfiguration()
    laser_cfg: LaserConfiguration = LaserConfiguration()
    mzm_cfg: MZMConfiguration = MZMConfiguration()
    mrr_cfg: MRRConfiguration = MRRConfiguration()
    pd_cfg: PDConfiguration = PDConfiguration()
    adc_cfg: ADCConfiguration = ADCConfiguration()
    tia_cfg: TIAConfiguration = TIAConfiguration()

    # TODO(Adriano) move all this configuration load/store stuff to a base class so we can and make it paramterizeable
    @staticmethod
    def configs_path() -> Path:
        if os.environ.get("ALBIREO_CONFIG_PATH", None) is not None:
            raise NotImplementedError("Config path is not implemented (would have issues with local path in testing scripts)") # fmt: skip
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config path {path} does not exist")
        return path

    #### Helper methods to save the config (you should always save to the `config/` folder) ####
    def save_to_path(self, path: Path | str):
        pydantic_yaml.to_yaml_file(path, self)

    def save(self, name: str):
        save_path = OpticalDotProductConfiguration.configs_path() / f"{name}.yaml"
        if save_path.exists():
            raise FileExistsError(f"Config file {save_path} already exists")
        self.save_to_path(save_path)

    #### Helper methods to load common configurations we have in the config folder ####
    @staticmethod
    def from_config_path(config_path: Path | str) -> OpticalDotProductConfiguration:
        return pydantic_yaml.parse_yaml_raw_as(OpticalDotProductConfiguration, Path(config_path).read_text()) # fmt: skip

    @staticmethod
    def from_config_name(config_name: str) -> OpticalDotProductConfiguration:
        configs_path = OpticalDotProductConfiguration.configs_path()
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