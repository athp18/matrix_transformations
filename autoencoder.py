import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def relu(x):
    """
    Compute the ReLU activation function.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Output tensor after applying ReLU.
    """
    return torch.maximum(torch.zeros_like(x), x)

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

def initialize_weights(input_dim, hidden_dim1, hidden_dim2, code_dim):
    """
    Initialize the weights and biases for an autoencoder.
    
    Args:
    input_dim (int): Dimension of the input layer.
    hidden_dim1 (int): Dimension of the first hidden layer.
    hidden_dim2 (int): Dimension of the second hidden layer.
    code_dim (int): Dimension of the latent code layer.
    
    Returns:
    dict: Dictionary containing initialized weights and biases.
    """
    weights = {
        # encoder weights and biases
        'W1': torch.randn(input_dim, hidden_dim1, device=device) * 0.01,
        'b1': torch.zeros(1, hidden_dim1, device=device),
        'W2': torch.randn(hidden_dim1, hidden_dim2, device=device) * 0.01,
        'b2': torch.zeros(1, hidden_dim2, device=device),
        'W3': torch.randn(hidden_dim2, code_dim, device=device) * 0.01,
        'b3': torch.zeros(1, code_dim, device=device),
        
        # decoder weights and biases
        'W4': torch.randn(code_dim, hidden_dim2, device=device) * 0.01,
        'b4': torch.zeros(1, hidden_dim2, device=device),
        'W5': torch.randn(hidden_dim2, hidden_dim1, device=device) * 0.01,
        'b5': torch.zeros(1, hidden_dim1, device=device),
        'W6': torch.randn(hidden_dim1, input_dim, device=device) * 0.01,
        'b6': torch.zeros(1, input_dim, device=device),
    }
    return weights

class Autoencoder:
    """
    The Autoencoder class.
    
    Attributes:
    weights (dict): Dictionary containing weights and biases.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, code_dim):
        """
        Initialize the Autoencoder.
        
        Args:
        input_dim (int): Dimension of the input layer.
        hidden_dim1 (int): Dimension of the first hidden layer.
        hidden_dim2 (int): Dimension of the second hidden layer.
        code_dim (int): Dimension of the latent code layer.
        """
        self.weights = initialize_weights(input_dim, hidden_dim1, hidden_dim2, code_dim)
        # initialize the gradients
        self.grads = {key: torch.zeros_like(value) for key, value in self.weights.items()}
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Reconstructed output tensor.
        """
        # Encoder layer
        self.x = x
        self.z1 = torch.matmul(self.x, self.weights['W1']) + self.weights['b1']
        self.h1 = relu(self.z1)
        self.z2 = torch.matmul(self.h1, self.weights['W2']) + self.weights['b2']
        self.h2 = relu(self.z2)
        self.z3 = torch.matmul(self.h2, self.weights['W3']) + self.weights['b3']
        self.code = relu(self.z3)
        
        # Decoder layer
        self.z4 = torch.matmul(self.code, self.weights['W4']) + self.weights['b4']
        self.h3 = relu(self.z4)
        self.z5 = torch.matmul(self.h3, self.weights['W5']) + self.weights['b5']
        self.h4 = relu(self.z5)
        self.z6 = torch.matmul(self.h4, self.weights['W6']) + self.weights['b6']
        self.output = sigmoid(self.z6)
        
        return self.output
    
    def backward(self, loss_grad):
        """
        Backward pass through the autoencoder.
        
        Args:
        loss_grad (torch.Tensor): Gradient of the loss with respect to the output.
        """
        # Derivative at the output layer (sigmoid)
        d_z6 = loss_grad * sigmoid_derivative(self.output)
        self.grads['W6'] += torch.matmul(self.h4.t(), d_z6)
        self.grads['b6'] += torch.sum(d_z6, dim=0, keepdim=True)
        
        # Backprop through h4
        d_h4 = torch.matmul(d_z6, self.weights['W6'].t())
        d_z5 = d_h4 * relu_derivative(self.z5)
        self.grads['W5'] += torch.matmul(self.h3.t(), d_z5)
        self.grads['b5'] += torch.sum(d_z5, dim=0, keepdim=True)
        
        # Backprop through h3
        d_h3 = torch.matmul(d_z5, self.weights['W5'].t())
        d_z4 = d_h3 * relu_derivative(self.z4)
        self.grads['W4'] += torch.matmul(self.code.t(), d_z4)
        self.grads['b4'] += torch.sum(d_z4, dim=0, keepdim=True)
        
        # Backprop through code layer
        d_code = torch.matmul(d_z4, self.weights['W4'].t())
        d_z3 = d_code * relu_derivative(self.z3)
        self.grads['W3'] += torch.matmul(self.h2.t(), d_z3)
        self.grads['b3'] += torch.sum(d_z3, dim=0, keepdim=True)
        
        # Backprop through h2
        d_h2 = torch.matmul(d_z3, self.weights['W3'].t())
        d_z2 = d_h2 * relu_derivative(self.z2)
        self.grads['W2'] += torch.matmul(self.h1.t(), d_z2)
        self.grads['b2'] += torch.sum(d_z2, dim=0, keepdim=True)
        
        # Backprop through h1
        d_h1 = torch.matmul(d_z2, self.weights['W2'].t())
        d_z1 = d_h1 * relu_derivative(self.z1)
        self.grads['W1'] += torch.matmul(self.x.t(), d_z1)
        self.grads['b1'] += torch.sum(d_z1, dim=0, keepdim=True)
    
    def zero_grads(self):
        """
        Zero out the gradients.
        """
        for key in self.grads:
            self.grads[key].zero_()
    
    def update_params(self, lr):
        """
        Update the weights and biases using the computed gradients.
        
        Args:
        lr (float): Learning rate.
        """
        for key in self.weights:
            self.weights[key] -= lr * self.grads[key]

def load_mnist():
    """
    Load the MNIST dataset with custom preprocessing.
    
    Returns:
    torch DataLoader: DataLoader for the MNIST dataset.
    """
    # Custom transformation: Convert images to tensors and normalize to [0, 1]
    def transform(image):
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader

def mse_loss(output, target):
    """
    Compute the Mean Squared Error loss.
    
    Args:
    output (torch.Tensor): Predicted outputs.
    target (torch.Tensor): Ground truth inputs.
    
    Returns:
    torch.Tensor: Mean Squared Error loss.
    """
    return torch.mean((output - target) ** 2)

def mse_loss_derivative(output, target):
    """
    Compute the derivative of the Mean Squared Error loss with respect to the output.
    
    Args:
    output (torch.Tensor): Predicted outputs.
    target (torch.Tensor): Ground truth inputs.
    
    Returns:
    torch.Tensor: Derivative of the loss with respect to the output.
    """
    return (2 * (output - target)) / output.size(0)

def visualize_results(original_data, reconstructed_data, n=5):
    """
    Visualize original and reconstructed images.
    
    Args:
    original_data (torch.Tensor): Original input images.
    reconstructed_data (torch.Tensor): Reconstructed images from the autoencoder.
    n (int): Number of images to display.
    """
    original_images = original_data.view(-1, 28, 28).cpu().detach()
    reconstructed_images = reconstructed_data.view(-1, 28, 28).cpu().detach()
    fig, axes = plt.subplots(2, n, figsize=(10, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def train(autoencoder, data_loader, num_epochs=5, lr=0.01):
    """
    Train the Autoencoder model.
    
    Args:
    autoencoder (Autoencoder): The autoencoder model.
    data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    num_epochs (int): Number of training epochs.
    lr (float): Learning rate.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, _) in enumerate(data_loader):
            # Prepare input data
            inputs = images.view(-1, 28*28).to(device)
            
            # Forward pass
            outputs = autoencoder.forward(inputs)
            loss = mse_loss(outputs, inputs)
            epoch_loss += loss.item()
            
            # Backward pass
            autoencoder.zero_grads()
            loss_grad = mse_loss_derivative(outputs, inputs)
            autoencoder.backward(loss_grad)
            
            # parameter update
            autoencoder.update_params(lr)
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}")
        
        #visualize
        visualize_results(inputs, outputs)
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 784        # Input dimension (28x28 images flattened)
    hidden_dim1 = 128      # First hidden layer size
    hidden_dim2 = 64       # Second hidden layer size
    code_dim = 32          # Latent code dimension
    num_epochs = 5         # Number of training epochs
    learning_rate = 0.01   # Learning rate
    
    # Instantiate the Autoencoder
    autoencoder = Autoencoder(input_dim, hidden_dim1, hidden_dim2, code_dim)
    
    # important: move weights to device
    for key in autoencoder.weights:
        autoencoder.weights[key] = autoencoder.weights[key].to(device)
    for key in autoencoder.grads:
        autoencoder.grads[key] = autoencoder.grads[key].to(device)
    
    # Load data
    data_loader = load_mnist()
    
    # Train the Autoencoder
    train(autoencoder, data_loader, num_epochs=num_epochs, lr=learning_rate)
