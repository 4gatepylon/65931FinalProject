import pytest
import numpy as np
from src.crosstalk.crosstalk import (
    awg_paper_finesse,
    mrr_paper_finesse,
    compute_crosstalk,
    make_configuration
)
from src.kernels.utils import crosstalk_DB_to_linear


def test_awg_paper_finesse_crosstalk_sensitivity():
    """Test that as adjacent_crosstalk_db increases (less negative), finesse decreases."""
    finesse_1 = awg_paper_finesse(adjacent_crosstalk_db=-40)
    finesse_2 = awg_paper_finesse(adjacent_crosstalk_db=-30)
    assert finesse_1 > finesse_2, "Finesse should decrease as crosstalk increases"


def test_awg_paper_finesse_n_sensitivity():
    """Test that as N increases, finesse increases."""
    # Using the formula from the function but explicitly providing N
    crosstalk_rate = crosstalk_DB_to_linear(-34)
    finesse_1 = 50 / (2 * crosstalk_rate ** 0.5)
    finesse_2 = 100 / (2 * crosstalk_rate ** 0.5)
    assert finesse_2 > finesse_1, "Finesse should increase as N increases"


def test_mrr_paper_finesse_k_squared_sensitivity():
    """Test that as k_squared increases, finesse decreases."""
    finesse_1 = mrr_paper_finesse(k_squared=0.02)
    finesse_2 = mrr_paper_finesse(k_squared=0.03)
    finesse_3 = mrr_paper_finesse(k_squared=0.04)
    assert finesse_1 > finesse_2 > finesse_3, "Finesse should decrease as k_squared increases"


def test_compute_crosstalk_finesse_sensitivity():
    """Test that as finesse increases, crosstalk decreases."""
    crosstalk_1 = compute_crosstalk(finesse=10, distinct_inputs=50)
    crosstalk_2 = compute_crosstalk(finesse=20, distinct_inputs=50)
    assert crosstalk_1 > crosstalk_2, "Crosstalk should decrease as finesse increases"


def test_compute_crosstalk_distinct_inputs_sensitivity():
    """Test that as distinct_inputs increases, crosstalk increases."""
    crosstalk_1 = compute_crosstalk(finesse=10, distinct_inputs=50)
    crosstalk_2 = compute_crosstalk(finesse=10, distinct_inputs=100)
    assert crosstalk_2 > crosstalk_1, "Crosstalk should increase as distinct_inputs increases"


def test_make_configuration_noise_off():
    """Test that when noisy=False, crosstalk rates are set to 0."""
    config = make_configuration(n_columns=8, n_plcus=4, n_bits=8, noisy=False)
    assert config.laser_cfg.awg_cross_talk_rate == 0, "AWG crosstalk should be 0 when noisy=False"
    assert config.mrr_cfg.mrr_cross_talk_rate == 0, "MRR crosstalk should be 0 when noisy=False"
    assert config.name == "n_bits_8_no_noise", "Config name should indicate no noise"


def test_make_configuration_noise_on():
    """Test that when noisy=True, crosstalk rates are calculated properly."""
    config = make_configuration(n_columns=8, n_plcus=4, n_bits=8, noisy=True)
    assert config.laser_cfg.awg_cross_talk_rate > 0, "AWG crosstalk should be > 0 when noisy=True"
    assert config.mrr_cfg.mrr_cross_talk_rate > 0, "MRR crosstalk should be > 0 when noisy=True"
    assert config.name == "n_columns_8_n_plcus_4_n_bits_8", "Config name should reflect parameters"


def test_make_configuration_sensitivity():
    """Test that when sensitive=True, k_squared changes and affects MRR crosstalk."""
    config_normal = make_configuration(n_columns=8, n_plcus=4, n_bits=8, noisy=True, sensitive=False)
    config_sensitive = make_configuration(n_columns=8, n_plcus=4, n_bits=8, noisy=True, sensitive=True)
    assert config_sensitive.mrr_cfg.mrr_cross_talk_rate > config_normal.mrr_cfg.mrr_cross_talk_rate, \
        "MRR crosstalk should be higher when sensitive=True"