import pytest
import torch
import numpy as np
from src.kernels.optical_dotproduct import OpticalDotProduct
from src.kernels.configurations import OpticalDotProductConfiguration

# Set up deterministic tests
torch.manual_seed(42)
np.random.seed(42)

@pytest.fixture
def zero_noise_config():
    """Load the zero noise high precision configuration"""
    return OpticalDotProductConfiguration.from_zero_noise_high_precision()

class TestNoiseless:
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
            
            result = dotter(vector)
            expected = torch.sum(weights * vector).reshape((1,)) # Batch only
            
            assert torch.isclose(result, expected, rtol=1e-07), f"Failed with dimension {dim}. Expected {expected}, got {result}"