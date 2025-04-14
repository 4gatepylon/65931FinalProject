"""
Used to replace a model's FC and Conv layers with custom ones during inference.
Current custom layers behave like the original ones.

TODO(From Amadou For Dylan and Adriano): Replace the custom layers with the Optical Ones.
"""



import requests
import json
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from datasets import load_dataset  # Hugging Face Datasets library
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

class CustomFC(nn.Module):
    """
    Custom Fully Connected Layer.
    For now, behave like an actual fully connected layer.
    """

    def __init__(
        self,
        original_fc: nn.Linear,
    ):
        super(CustomFC, self).__init__()
        self.in_features = original_fc.in_features
        self.out_features = original_fc.out_features
        self.weight = original_fc.weight
        self.bias = original_fc.bias
        assert not original_fc.training, "CustomFC should not be in training mode"

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom fully connected layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Perform the forward pass using the original weight and bias
        return F.linear(x, self.weight, self.bias)
    

    def extra_repr(self) -> str:
        """
        Extra representation of the custom fully connected layer.
        Returns:
            str: Extra representation string.
        """
        return f"CustomFC(in_features={self.in_features}, out_features={self.out_features})"
    

    def __repr__(self) -> str:
        """
        String representation of the custom fully connected layer.
        Returns:
            str: String representation.
        """
        return f"CustomFC(in_features={self.in_features}, out_features={self.out_features})"
    


class CustomConv2d(nn.Module):
    """
    Custom Convolutional Layer.
    For now, behave like an actual convolutional layer.
    """

    def __init__(
        self,
        original_conv: nn.Conv2d,
    ):
        super(CustomConv2d, self).__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.weight = original_conv.weight
        self.bias = original_conv.bias
        assert not original_conv.training, "CustomConv2d should not be in training mode"
        self.eval()  # Set the module to evaluation mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom convolutional layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Perform the forward pass using the original weight and bias
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

    def extra_repr(self) -> str:
        """
        Extra representation of the custom convolutional layer.
        Returns:
            str: Extra representation string.
        """
        return f"CustomConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
    
    def __repr__(self) -> str:
        """
        String representation of the custom convolutional layer.
        Returns:
            str: String representation.
        """
        return f"CustomConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


def get_image_net_mappings():
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        response = requests.get(labels_url)
        response.raise_for_status() # Raise an exception for bad status codes
        # This map is { "0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ... }
        imagenet1k_idx_to_classinfo = json.loads(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ImageNet mapping: {e}")
        exit()
    except json.JSONDecodeError:
        print("Error decoding ImageNet mapping JSON.")
        exit()
    print(list(imagenet1k_idx_to_classinfo.items())[:10])
    imagenet1k_idx_to_classinfo['12']
    ident_to_idx = {v[0]: int(k) for k, v in imagenet1k_idx_to_classinfo.items()}
    idx_to_ident = {int(k): v[0] for k, v in imagenet1k_idx_to_classinfo.items()}
    return ident_to_idx, idx_to_ident

def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")



def replace_layers(model: nn.Module) -> nn.Module:
    """
    Replace all the fully connected and convolutional layers in the model with custom ones.
    Args:
        model (nn.Module): The original model.
    Returns:
        nn.Module: The modified model with custom layers.
    """
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replacements[name] = CustomFC(module)
        elif isinstance(module, nn.Conv2d):
            replacements[name] = CustomConv2d(module)
    for name, replacement in replacements.items():
        # Replace the original module with the custom one
        parent_module = model
        for part in name.split('.')[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name.split('.')[-1], replacement)
        
    return model







class Loader:
    def __init__(self, dataset_name: str, max_num_data_points: int = 100):
        self.dataset_name = dataset_name.lower()
        self.max_num_data_points = max_num_data_points
        self.original_model = None
        self.test_dataset = None
        self.custom_model = None
        self.ident_to_full_idx, self.full_idx_to_ident = get_image_net_mappings()
        self.load_dataset(self.dataset_name, max_num_data_points)
        self.load_pretrained_model(self.dataset_name)
        self.make_custom_model()



    def test_model_on_dataset(self, custom=True):
        if custom:
            model = self.custom_model
        else:
            model = self.original_model
        # Send to GPU
        model.to(best_device())
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            batch_idx = 0
            num_batches = len(self.test_dataset)
            for batch in self.test_dataset:
                images = batch["image"].to(best_device())
                labels = batch["label"].to(best_device())
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if batch_idx % 10 == 0:
                    print(f"Correct: {correct}, Total: {total}, Current Acc: {100 * correct / total}, Batch: {batch_idx+1}/{num_batches}.")
                batch_idx += 1
        model_name = "custom" if custom else "original"
        print(f"Accuracy of the {model_name} model on the {self.dataset_name} test images: {100 * correct / total}%")
        return 100 * correct / total


    def load_pretrained_model(self, dataset_name: str):
        """
        Loads a pretrained ResNet18 model according to the dataset.
        
        - For "mini-imagenet" or "tiny-imagenet": Uses the standard torchvision pretrained ResNet18 (on ImageNet).
        """
        dataset_name = dataset_name.lower()
        self.original_model = models.resnet18(pretrained=True)
        self.original_model.eval()
        self.original_model.to(best_device())
        return


    def make_custom_model(self):
        """
        Replaces the original model's layers with custom ones.
        Returns:
            nn.Module: The modified model with custom layers.
        """
        from copy import deepcopy
        self.original_model.eval()
        self.custom_model = deepcopy(self.original_model)
        self.custom_model = replace_layers(self.custom_model)
        self.custom_model.eval()
        return


    @staticmethod
    def convert_img(img):
        """
        Directly return the image if it is already a PIL Image, converting to RGB if necessary.
        Otherwise, if the image is provided as a list or a numpy array, convert it to a PIL Image.
        """
        # If it's already a PIL image, ensure it's in RGB mode
        if isinstance(img, list) and isinstance(img[0], Image.Image):
            img = img[0]
        return img if img.mode == "RGB" else img.convert("RGB")


    def load_dataset(self, dataset_name, max_num_data_points):
        """
        Loads and preprocesses the test dataset for the given name.

        - For "mini-imagenet": Loads from https://huggingface.co/datasets/timm/mini-imagenet.
        - For "tiny-imagenet": Loads from https://huggingface.co/datasets/zh-plus/tiny-imagenet.
        - For "imagenet": Loads from https://huggingface.co/datasets/huggingface/datasets.

        All images are resized to 224x224 and normalized using ImageNet statistics.
        """
        dataset_name = dataset_name.lower()
        # Define common image transformation: resize, to tensor, and normalize
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        img_key = "image"
        dataset_paths = {
            "mini-imagenet": "timm/mini-imagenet",
            "tiny-imagenet": "zh-plus/tiny-imagenet",
            "imagenet": "timm/imagenet-12k-wds",
        }
        split_names = {
            "mini-imagenet": "validation",
            "tiny-imagenet": "valid",
            "imagenet": "validation",
        }

        ds = load_dataset(dataset_paths[dataset_name], split=f"{split_names[dataset_name]}[:{max_num_data_points}]")
        mini_idents = ds.features['label'].names
        mini_idx_to_full_idx = {
            i: self.ident_to_full_idx[ident] for i, ident in enumerate(mini_idents)
        }
        # Assuming the image field is "img" and label is "label"
        def transform_batch(batch):
            batch['image'] = [
                transform(self.convert_img(img))
                for img in batch['image']
            ]
            batch['label'] = [
                mini_idx_to_full_idx[label]
                for label in batch['label']
            ]
            return batch
        ds = ds.map(transform_batch, batched=True)
        ds.set_format(type='torch', columns=['image', 'label'])
        test_dataset = DataLoader(ds, batch_size=64, shuffle=False)
        print(f"Loaded {dataset_name} dataset with {len(test_dataset)} batches.")
        i = 0
        for batch in test_dataset:
            images = batch['image']
            labels = batch['label']
            print(f"Images shape: {images.shape}")
            print(f"Labels Shape: {labels.shape}")
            i += 1
            if i == 5:
                break
        self.test_dataset = test_dataset
        return


if __name__ == "__main__":
    # Example usage
    dataset_name = "mini-imagenet"  # Change this to "tiny-imagenet" or "imagenet" as needed
    loader = Loader(dataset_name, max_num_data_points=100)
    loader.test_model_on_dataset(custom=True)
    loader.test_model_on_dataset(custom=False)
    # loader.test_model_on_dataset(custom=False)