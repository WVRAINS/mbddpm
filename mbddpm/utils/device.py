import torch

def get_device(device_str=None):
    if device_str:
        return torch.device(device_str)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")