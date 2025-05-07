import pytest
import torch
from typing import List, Callable
import numpy as np
import itertools
from src.kernels.optical_dotproduct import OpticalDotProduct, ADC, DAC
from src.kernels.configurations import (
    OpticalDotProductConfiguration,
    ADCConfiguration,
    DACConfiguration,
)

# Set up deterministic tests
torch.manual_seed(42)
np.random.seed(42)

@pytest.fixture
def zero_noise_config():
    """Load the zero noise high precision configuration"""
    return OpticalDotProductConfiguration.from_zero_noise_high_precision()

class TestADCDAC:
    """
    Static class to test the ADC and DAC. It makes sure that:
    1. The DAC will bin things as expected (i.e. certain things will be rounded
        to the nearest integer in the proper range)
    2. The ADC will un-bin things as expected (i.e. precision will not be lost
        if you have that precision and it will if you don't)
    3. With matching precisions you can enc-dec with the ADC and DAC without
        loss.
    4. Without matching precisions you can't enc-dec without loss and the
        loss is basically as expected.
    
    TODO test non-matching precisions (4) and with weight
    and also compose with some noise and make sure it behaves
    as expected.
    """
    @staticmethod
    def helper_test_forall(
        n_bits: int,
        voltage_mins: List[float],
        voltage_maxs: List[float],
        # Function should make sure to return some number that is informed
        # by the range allowable
        input_generator: Callable[[int, int], torch.Tensor],
        min_err_fp: float,
        max_err_fp: float,
        min_err_int: float,
        max_err_int: float,
        rescale_down: bool,
    ):
        """Test that integers are properly quantized"""
        torch.manual_seed(4212312)
        for voltage_min, voltage_max in itertools.product(voltage_mins, voltage_maxs):
            if voltage_min >= voltage_max:
                continue
            # 1. Create ADC
            adc_cfg = ADCConfiguration(
                quantization_bitwidth=n_bits,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
            )
            adc = ADC(adc_cfg)
            # 2. Create DAC
            dac_cfg = DACConfiguration(
                quantization_bitwidth=n_bits,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
            )
            dac = DAC(dac_cfg)
            # 3. Test no error for integers
            left = 0
            right = 2**n_bits - 1
            input_integers = input_generator(left, right)
            input_fp = input_integers.float()
            output_fp = adc(dac(input_fp), rescale_down=rescale_down)
            output_integers = output_fp.round()
            
            # Check fp (max error)
            fp_err = (input_fp - output_fp).abs().max().item()
            assert fp_err >= min_err_fp, f"FP error: {fp_err}"
            assert fp_err <= max_err_fp, f"FP error: {fp_err}"
            # Check int (max error)
            int_err = (input_integers - output_integers).abs().max().item()
            assert int_err >= min_err_int, f"Int error: {int_err}"
            assert int_err <= max_err_int, f"Int error: {int_err}"

    @staticmethod
    def test_integers_in_range_no_error():
        """Integers are mapped bijectively to integers for any voltage range."""
        n_bits = 4
        voltage_mins = [-1.0, -1e-3, 0.0, 1e-3, 1.0]
        voltage_maxs = [0.0, 2e-3, 1.0]
        TestADCDAC.helper_test_forall(
            n_bits,
            voltage_mins,
            voltage_maxs,
            lambda left, right: torch.randint(left, right, (100,)),
            min_err_fp=0.0,
            max_err_fp=1e-1,
            min_err_int=0.0,
            max_err_int=0.0,
            rescale_down=False,
        )
    
    @staticmethod
    def test_reals_in_range_small_error():
        """Reals are mapped to the nearest integer basically for any voltage range."""
        n_bits = 4
        voltage_mins = [-1.0, -1e-3, 0.0, 1e-3, 1.0]
        voltage_maxs = [0.0, 2e-3, 1.0]
        TestADCDAC.helper_test_forall(
            n_bits,
            voltage_mins,
            voltage_maxs,
            lambda left, right: torch.rand(100) * (right - left) + left,
            min_err_fp=0.0,
            max_err_fp=0.5,
            min_err_int=0.0,
            max_err_int=0.5,
            rescale_down=False,
        )
    
    @staticmethod
    def test_values_outside_range_are_clamped():
        """Test that values outside the voltage range are properly clamped + make sure scaling is OK."""
        n_bits = 8
        assert 2**8-1 == 255
        voltage_mins = [-99, -1, -1e-10, 0.0, 1e-10, 1, 99]
        voltage_maxs = [-98, -1e-10, 0.0, 1e-10, 1, 98]
        for voltage_min, voltage_max in itertools.product(voltage_mins, voltage_maxs):
            if voltage_min >= voltage_max:
                continue
            
            # Create ADC
            adc_cfg = ADCConfiguration(
                quantization_bitwidth=n_bits,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
            )
            dac_cfg = DACConfiguration(
                quantization_bitwidth=n_bits,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
            )
            adc = ADC(adc_cfg)
            dac = DAC(dac_cfg)
            #
            voltage_max = torch.tensor([voltage_max], dtype=torch.float)
            voltage_min = torch.tensor([voltage_min], dtype=torch.float)
            # Test values below minimum
            # NOTE minimum is 0 and maximum is 2^8-1 = 255
            below_min = torch.tensor([-2.0, -1.0, -0.5, -0.1])
            below_min_result = dac(below_min) # clamp to zero and then go to voltage min
            assert torch.allclose(below_min_result, voltage_min), f"Values below min should be clamped to {voltage_min}, got {below_min_result}"
            
            # Test values above maximum
            above_max = torch.tensor([256, 257, 258, 259]).float()
            above_max_result = dac(above_max) # clamp to 255 and then go to voltage max
            assert torch.allclose(above_max_result, voltage_max), f"Values above max should be clamped to {voltage_max}, got {above_max_result}"
            
            # Test mixed values
            # first one should return to 0, next 3 should be 0, 1, 2, last 4 should be 255
            mixed_values = torch.tensor([-1.0, 0.3, 0.7, 2.0, 256, 257, 258, 259]).float()
            mixed_expected = torch.tensor([0, 0, 1, 2, 255, 255, 255, 255]).float()
            mixed_result_voltage_scale = adc(dac(mixed_values), rescale_down=True).float()
            mixed_result_number_scale = adc(dac(mixed_values), rescale_down=False).float()
            delta = (mixed_result_number_scale - mixed_expected).round()
            # NOTE if you flip 0 to 1 you get an error
            assert torch.all(delta == 0), f"Mixed values not properly clamped. Expected {mixed_expected}, got {mixed_result_number_scale}"
            # TODO(Adriano) this is a wierd specification but ok
            dv = voltage_max - voltage_min
            assert mixed_result_voltage_scale.min() >= 0, f"Mixed values not properly clamped. Expected {voltage_min}, got {mixed_result_voltage_scale.min()}"
            assert mixed_result_voltage_scale.max() <= dv, f"Mixed values not properly clamped. Expected {voltage_max}, got {mixed_result_voltage_scale.max()}"

        

class TestNoisers:
    """
    TODO(Adriano) we will want to do this at some point but
    I think it doesn't make sense to do it now, it's too
    late :/
    """

class TestNoiseless:
    """
    Static class to test 
    """
    @staticmethod
    def get_test_device():
        return "cpu" # Always use CPU for THESE tests

    @staticmethod
    def test_basic_ones_vector(zero_noise_config):
        """Test with simple vectors of ones, matching the example from the notebook"""
        device = TestNoiseless.get_test_device()
        weights = torch.ones(4).unsqueeze(0).to(device)
        vector = torch.ones(4).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        # Based on the example output, there might be some scaling
        assert torch.isclose(result, expected), f"Expected {expected}, got {result}"

    @staticmethod
    def test_random_vectors_with_seed(zero_noise_config):
        """Test with random vectors using a fixed seed"""
        torch.manual_seed(123)
        device = TestNoiseless.get_test_device()
        weights = torch.randn(10).unsqueeze(0).to(device)
        vector = torch.randn(10).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-06), f"Expected {expected}, got {result}"

    @staticmethod
    def test_large_values(zero_noise_config):
        """Test with large values"""
        device = TestNoiseless.get_test_device()
        weights = (torch.ones(4) * 100).unsqueeze(0).to(device)
        vector = (torch.ones(4) * 100).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-04), f"Expected {expected}, got {result}"

    @staticmethod
    def test_small_values(zero_noise_config):
        """Test with small values"""
        device = TestNoiseless.get_test_device()
        weights = (torch.ones(4) * 0.001).unsqueeze(0).to(device)
        vector = (torch.ones(4) * 0.001).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-09), f"Expected {expected}, got {result}"

    @staticmethod
    def test_mixed_positive_negative(zero_noise_config):
        """Test with mixed positive and negative values"""
        device = TestNoiseless.get_test_device()
        weights = torch.tensor([1.0, -2.0, 3.0, -4.0]).unsqueeze(0).to(device)
        vector = torch.tensor([-5.0, 6.0, -7.0, 8.0]).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-06), f"Expected {expected}, got {result}"

    @staticmethod
    def test_zeros_and_non_zeros(zero_noise_config):
        """Test with zeros and non-zeros"""
        device = TestNoiseless.get_test_device()
        weights = torch.tensor([0.0, 2.0, 0.0, 4.0]).unsqueeze(0).to(device)
        vector = torch.tensor([1.0, 0.0, 3.0, 0.0]).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-06), f"Expected {expected}, got {result}"

    @staticmethod
    def test_all_zeros(zero_noise_config):
        """Test with all zeros"""
        device = TestNoiseless.get_test_device()
        weights = torch.zeros(4).unsqueeze(0).to(device)
        vector = torch.zeros(4).unsqueeze(0).to(device)
        
        dotter = OpticalDotProduct(
            weights,
            cfg=zero_noise_config,
            device=device,
        )
        # Ensure we are using 64-bit precision
        assert dotter.input_DAC.quantization_bitwidth == 64
        assert dotter.weight_DAC.quantization_bitwidth == 64
        assert dotter.adc.quantization_bitwidth == 64
        
        result = dotter(vector)
        expected = torch.sum(weights * vector).reshape((1,)) # Batch only
        
        assert torch.isclose(result, expected, rtol=1e-06), f"Expected {expected}, got {result}"

    @staticmethod
    def test_different_dimensions(zero_noise_config):
        """Test with different dimensions"""
        for dim in [2, 8, 16, 32]:
            # These are pretty small so small rtol makes sense
            device = TestNoiseless.get_test_device()
            weights = torch.randn(dim).unsqueeze(0).to(device)
            vector = torch.randn(dim).unsqueeze(0).to(device)
            
            dotter = OpticalDotProduct(
                weights,
                cfg=zero_noise_config,
                device=device,
            )
            # Ensure we are using 64-bit precision
            assert dotter.input_DAC.quantization_bitwidth == 64
            assert dotter.weight_DAC.quantization_bitwidth == 64
            assert dotter.adc.quantization_bitwidth == 64
            
            result = dotter(vector)
            expected = torch.sum(weights * vector).reshape((1,)) # Batch only
            
            assert torch.isclose(result, expected, rtol=1e-06), f"Failed with dimension {dim}. Expected {expected}, got {result}"