from src.kernels import OpticalFC, OpticalDotProductConfiguration, best_device
from src.cross_talk import make_configuration
import torch
from torch import nn
from torch.nn import functional as F

device = best_device()

def test_indiviual_dot_product():
    config = make_configuration(
        n_columns=5,
        n_plcus=3,
        n_bits=8,
        noisy=False
    )
    test_weights = torch.randn((3, 10), device=device)
    test_biases = torch.randn(3, device=device)
    test_inputs = torch.randn((3, 10), device=device)
    print(f"Original PyTorch implementation: {F.linear(test_inputs, test_weights, test_biases)}")
    fc = OpticalFC(
        test_weights,
        test_biases,
        config,
    )
    # Original PyTorch implementation
    print(f"OpticalFC implementation: {fc(test_inputs)}")
    return


def test_end_to_end():
    from src.model_converter import Loader
    # Example usage
    dataset_name = "mini-imagenet"  # Change this to "tiny-imagenet" or "imagenet" as needed
    config = make_configuration(
        n_columns=5,
        n_plcus=3,
        n_bits=6,
        noisy=False
    )
    loader = Loader(dataset_name, max_num_data_points=4, config=config)
    loader.test_model_on_dataset(custom=True)
    loader.test_model_on_dataset(custom=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the optical NN implementation.")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end test.")
    args = parser.parse_args()
    if args.e2e:
        test_end_to_end()
    else:
        test_indiviual_dot_product()