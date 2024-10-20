import torch
from PIL import Image
import numpy as np
#from torchvision.transforms.functional import pil_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_image(tensor, min_val=0, max_val=255):
    """
    Normalize a tensor to values between 0 and 1.
    
    Args:
    tensor (torch.Tensor): Image tensor to be normalized.
    min_val (float): Minimum value in the input range (default: 0).
    max_val (float): Maximum value in the input range (default: 255).
    
    Returns:
    torch.Tensor: Normalized image tensor.
    """
    return (tensor - min_val) / (max_val - min_val)


def transform(image):
    """
    Transform an image to grayscale and prepare it for further processing.
    
    Args:
    image (PIL.Image.Image or torch.Tensor): Image to be transformed.
    
    Returns:
    torch.Tensor: Transformed image.
    """
    if isinstance(image, Image.Image):
        # Convert to grayscale
        image = image.convert('L')
        #image = pil_to_tensor(image)
        # Note: to avoid overloading memory, I am not going to use pil_to_tensor, which performs a deep copy of the underlying array; 
        # instead, I'll do np.array(image) to functionally wrap the data as a tensor
        image = torch.tensor(np.array(image), dtype=torch.float32)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.size(0) in [3, 4]:  # RGB or RGBA
            image = image.mean(dim=0, keepdim=True)  # grayscale convert
        elif image.dim() == 4 and image.size(1) in [3, 4]:  # Batch of RGB or RGBA
            image = image.mean(dim=1, keepdim=True)  # grayscale conversion
        if image.dim() not in [2, 3]:
            raise ValueError("Input tensor must be 2D or 3D (with batch dimension)")
    else:
        raise TypeError("Input must be a PIL Image or a torch.Tensor")

    return _normalize_image(image)

def relu(x):
    """
    Compute the ReLU activation function.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Output tensor after applying ReLU.
    """
    #zeros = torch.zeros_like(x)
    #return torch.maximum(zeros, x)
    return torch.relu(x)

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
    return torch.sigmoid(x)
    #return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y):
    """
    Compute the derivative of the sigmoid function.
    
    Args:
    y (torch.Tensor): Output of sigmoid(x).
    
    Returns:
    torch.Tensor: Derivative of the sigmoid function.
    """
    return y * (1 - y)
