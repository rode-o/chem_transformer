import torch

# Check the CUDA version PyTorch was built with
print("PyTorch built with CUDA:", torch.version.cuda)

# Check if CUDA is available and can be used
print("Is CUDA available:", torch.cuda.is_available())

# If CUDA is available, check the device details
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No CUDA device available.")
