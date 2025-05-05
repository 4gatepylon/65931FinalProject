from src.kernels import BackupOpticalFC, OpticalFC, OpticalDotProduct, OpticalDotProductConfiguration, best_device
from src.cross_talk import make_configuration
import torch
from torch import nn
from torch.nn import functional as F

device = best_device()

def test_indiviual_dot_product():
    config = make_configuration(
        n_columns=5,
        n_plcus=3,
        n_bits=32,
        noisy=False
    )
    # test_weights = torch.randn((3, 10), device=device)
    # test_biases = torch.randn(3, device=device)
    # test_inputs = torch.randn((3, 10), device=device)
    test_weights = torch.tensor([[ 0.8111,  0.6688,  2.6989,  1.3934, -0.6673,  0.1056,  1.4883,  1.0756,
         1.7219, -0.2584, 1.1821]], device=device)
    test_inputs = torch.tensor([[-0.2083,  1.7521,  0.3294,  0.7586, -0.5800, -1.9319, -1.0867,  1.0023,
         0.0613,  0.1703, 1.0]], device=device)
    test_weights = torch.abs(test_weights)
    test_weights /= torch.max(test_weights)
    test_inputs = torch.abs(test_inputs)
    test_biases = torch.tensor([1.1821], device=device)
    test_biases *= 0.0 # Incorporated.
    print(f"Original PyTorch implementation: {F.linear(test_inputs, test_weights, test_biases)}")
    fc = OpticalFC(
        test_weights,
        test_biases,
        config,
    )
    # Original PyTorch implementation
    print(f"OpticalFC implementation: {fc(test_inputs)}")
    # FastOpticalFC implementation
    fast_fc = FastOpticalFC(
        test_weights,
        test_biases,
        config,
    )
    print(f"FastOpticalFC implementation: {fast_fc(test_inputs)}")
    return

def test_dumb_dot_product():
    # config = OpticalDotProductConfiguration()
    config = make_configuration(
        n_columns=5,
        n_plcus=3,
        n_bits=7,
        noisy=True,
    )
    # test_weights = torch.tensor([[1, 1, 1]], device=device)
    # test_inputs = torch.tensor([[1, 1, 1]], device=device)
    
    
    print(f"Original Implementation: {torch.sum(test_inputs * test_weights)}")

    mod = OpticalDotProduct(
        weights=test_weights,
        cfg=config,
    )
    print(f"OpticalDotProduct implementation: {mod(test_inputs)}")
    return



def test_end_to_end():
    from src.model_converter import Loader
    # Example usage
    dataset_name = "mini-imagenet"  # Change this to "tiny-imagenet" or "imagenet" as needed
    config = make_configuration(
        n_columns=5,
        n_plcus=3,
        n_bits=8,
        noisy=False
    )
    loader = Loader(dataset_name, max_num_data_points=20, config=config)
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
        # test_dumb_dot_product()


builtin_dot = lambda x, y, b: torch.sum(x * x) + b