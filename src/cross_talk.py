"""
Approximate cross-talk model.

We will assume that:
Adjacent Crosstalk = (N_distinct / (2 * finesse))^2
Non-adjacent crosstalk is negligible.
Adjacent crosstalk is a simple multiplicative factor.
Basically, X *= (1 + Adjacent_crosstalk)
NOTE: This should be X = X + adjacent_X * adjacent_crosstalk.


Then:
Based on the paper's default configuration, we will compute the finesse of the AWG and MRRs.
finesse = default_N / (2 * sqrt(default_crosstalk)) # From both chatgpt and gemini. Seems right.
or
finesse = pi * sqrt(1 - k^2) / (k^2) # formula from the paper.


Finally:
We will apply this finesse to the new configurations we experiment with.
"""

from .configurations import OpticalDotProductConfiguration
from enum import StrEnum, auto
from .utils import crosstalk_DB_to_linear
from math import pi, sqrt


def awg_paper_finesse(adjacent_crosstalk_db: float = -34) -> float:
    """ Approximate formula to get the finesse of the AWG in the paper. """
    crosstalk_rate = crosstalk_DB_to_linear(adjacent_crosstalk_db)
    paper_N = 63
    return paper_N / (2 * crosstalk_rate ** 0.5)

def mrr_paper_finesse(k_squared: float = 0.02) -> float:
    """ Approximate formula to get the finesse of the MRR in the paper. """
    return pi * sqrt(1 - k_squared) / (k_squared)


def compute_crosstalk(finesse: float, distinct_inputs: int):
    """ Compute the cross-talk based on the finesse and distinct inputs. """
    return (distinct_inputs / (2 * finesse)) ** 2


def make_configuration(
    n_columns: int,
    n_plcus: int,
    n_bits: int,
    noisy: bool = False,
):
    """
    Create a configuration for the cross-talk model.
    """
    # Set the base configuration
    config = OpticalDotProductConfiguration()
    config.input_dac_cfg.quantization_bitwidth = n_bits
    config.input_dac_cfg.voltage_max = 2 ** n_bits - 1
    config.weight_dac_cfg.quantization_bitwidth = n_bits
    config.weight_dac_cfg.voltage_max = 2 ** n_bits - 1
    config.adc_cfg.quantization_bitwidth = n_bits
    config.mzm_cfg.voltage_max = 2 ** n_bits - 1

    # Cross-talk specific configuration
    distinct_plcu_inputs = 3 * (3 + n_columns - 1)
    distinct_awg_inputs = n_plcus * distinct_plcu_inputs
    mrr_finesse = mrr_paper_finesse()
    awg_finesse = awg_paper_finesse()
    mrr_cross_talk = compute_crosstalk(mrr_finesse, distinct_plcu_inputs)
    awg_cross_talk = compute_crosstalk(awg_finesse, distinct_awg_inputs)
    if noisy:
        config.laser_cfg.awg_cross_talk_rate = awg_cross_talk
        config.mrr_cfg.mrr_cross_talk_rate = mrr_cross_talk
    else:
        config.laser_cfg.awg_cross_talk_rate = 0
        config.mrr_cfg.mrr_cross_talk_rate = 0
    return config


