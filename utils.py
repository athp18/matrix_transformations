import torch
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(image):
    """
    Transform an image to grayscale and normalize it to values between 0 and 1.
    
    Args:
    image (PIL.Image.Image or torch.Tensor): Image to be transformed.
    
    Returns:
    torch.Tensor: Transformed and normalized image.
    """
    if isinstance(image, Image.Image):
        # first convert to grayscale
        image = image.convert('L')
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        image /= 255.0
    else:
        raise TypeError("Input must be a PIL Image or a torch.Tensor")
    return image

def relu(x):
    """
    Compute the ReLU activation function.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Output tensor after applying ReLU.
    """
    zeros = torch.zeros_like(x)
    return torch.maximum(zeros, x)

def relu_derivative(x):
    """
    Compute the derivative of the ReLU function.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Derivative of the ReLU function.
    """
    return (x > 0).float()

def sigmoid(x):
    """
    Compute the sigmoid activation function.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Output tensor after applying sigmoid.
    """
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y):
    """
    Compute the derivative of the sigmoid function.
    
    Args:
    y (torch.Tensor): Output of sigmoid(x).
    
    Returns:
    torch.Tensor: Derivative of the sigmoid function.
    """
    return y * (1 - y)
