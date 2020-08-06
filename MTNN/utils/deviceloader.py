# Helper functions for GPU-usage
import torch 

def get_device(verbose=False):
    """
    Set to GPU if available, else defaults to CPU.
    Args: 
        verbose <bool>  Prints which device is used to stdout
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if verbose: 
        print(f"Running on {device.type}")

    return device

def load_data(data, device): 
    """
    Loads onto appropriate device (cpu or gpu)
    Args: 
        data: mini-batch data from dataloader 
        device: <torch.device> 
    Returns: 
        input_data 
        target_data
    """
    if device.type == 'cuda':
        input_data, target_data = data[0].to(device), data[1].to(device)
    else: 
        input_data, target_data = data[0], data[1]
    return input_data, target_data