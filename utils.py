import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(image):
  """
  Transform an image and normalize it to grayscale so that its values are between 0 and 1.

  Args:
  image (PIL.Image): Image to be transformed.
  """
  image = torch.tensor(image, dtype=torch.float32) / 255.0
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
