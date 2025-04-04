# 65931FinalProject
FinalProjectBenchmarkingOpticalSystems

Please install with `conda create -n 65931-final-project python=3.12` followed by `conda activate 65931-final-project` followed by `pip3 install -r requirements.txt`

Things to do as of 2025-04-04:
1. Have a set of `nn.Module` objects to describe the architecture as defined in the paper. These should be parameterizeable.
    - Matrix-Vector dot product is the bare minimum to be supported
    - They can be combined in `nn.Sequential` (or should be possible to combine into a larger architecture with this)
2. Define some unit and integration tests for our modules. Specifically, you should be able to run (with zero error, or some sort of expected error based on the parameters):
    - A matrix multiplication
    - A convolution
    - A FC layer
    - Possibly a full Resnet18, but that would be optional (looks like that's milestone 2)
3. Ideally be able to visualize the error/noise in the above for a given set of parameters