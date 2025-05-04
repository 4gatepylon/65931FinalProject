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
from configurations import *

for i in range(2, 16+1, 2):
    config = OpticalDotProductConfiguration()

    config.adc_cfg.quantization_bitwidth=8
    config.input_dac_cfg.quantization_bitwidth=8
    config.weight_dac_cfg.quantization_bitwidth=i
    config.save_to_path(f"config/scripted/WDAC{i}_IDAC8_ADC8.yaml")

for i in range(2, 16+1, 2):
    config = OpticalDotProductConfiguration()

    config.adc_cfg.quantization_bitwidth=8
    config.input_dac_cfg.quantization_bitwidth=i
    config.weight_dac_cfg.quantization_bitwidth=8
    config.save_to_path(f"config/scripted/WDAC8_IDAC{i}_ADC8.yaml")

for i in range(2, 16+2, 2):
    config = OpticalDotProductConfiguration()

    config.adc_cfg.quantization_bitwidth=i
    config.input_dac_cfg.quantization_bitwidth=8
    config.weight_dac_cfg.quantization_bitwidth=8
    config.save_to_path(f"config/scripted/WDAC8_IDAC8_ADC{i}.yaml")