from src.kernels import OpticalFC, OpticalDotProductConfiguration
import torch
from torch import nn
from torch.nn import functional as F


def test_indiviual_dot_product():
    default_config = OpticalDotProductConfiguration()
    test_weight = torch.tensor([[
        1.0,
        2.0,
        3.0,
        -1.0,
        -2.0,
    ]])
    test_bias = torch.ones(1)
    test_input = torch.tensor([[
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]])
    print(f"Original PyTorch implementation: {F.linear(test_input, test_weight, test_bias)}")
    fc = OpticalFC(
        test_weight,
        test_bias,
        default_config
    )
    # Original PyTorch implementation
    print(f"OpticalFC implementation: {fc(test_input)}")


def test_end_to_end():
    from src.model_converter import Loader
    # Example usage
    dataset_name = "mini-imagenet"  # Change this to "tiny-imagenet" or "imagenet" as needed
    loader = Loader(dataset_name, max_num_data_points=20)
    loader.test_model_on_dataset(custom=True)
    loader.test_model_on_dataset(custom=False)
    # loader.test_model_on_dataset(custom=False)


if __name__ == "__main__":
    # test_indiviual_dot_product()
    test_end_to_end()