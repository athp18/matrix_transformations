import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    """
    Compute the sigmoid of x.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Output tensor after applying the sigmoid function.
    """
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y):
    """
    Compute the derivative of the sigmoid function.

    Args:
    y (torch.Tensor): Output of sigmoid(x).

    Returns:
    torch.Tensor: Derivative of sigmoid function.
    """
    return y * (1 - y)

def tanh(x):
    """
    Compute the tanh of x.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Output tensor after applying the tanh function.
    """
    return torch.tanh(x)

def dtanh(y):
    """
    Compute the derivative of the tanh function.

    Args:
    y (torch.Tensor): Output of tanh(x).

    Returns:
    torch.Tensor: Derivative of the tanh function.
    """
    return 1 - y ** 2

def relu(x):
    """
    Compute the ReLU of x.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Output tensor after applying the ReLU function.
    """
    return torch.maximum(torch.zeros_like(x), x)

def drelu(x):
    """
    Compute the derivative of the ReLU function.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Derivative of the ReLU function.
    """
    return (x > 0).float()

def initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim):
    """
    Initialize the weights and biases for a neural network.

    Args:
    input_dim (int): Number of input features.
    hidden_dim1 (int): Number of units in the first hidden layer.
    hidden_dim2 (int): Number of units in the second hidden layer.
    output_dim (int): Number of output features.

    Returns:
    Tuple[torch.Tensor]: Initialized weights and biases for the neural network.
    """
    W1 = torch.randn(input_dim, hidden_dim1, device=device) * 0.01
    b1 = torch.zeros(1, hidden_dim1, device=device)
    W2 = torch.randn(hidden_dim1, hidden_dim2, device=device) * 0.01
    b2 = torch.zeros(1, hidden_dim2, device=device)
    W3 = torch.randn(hidden_dim2, hidden_dim2, device=device) * 0.01
    b3 = torch.zeros(1, hidden_dim2, device=device)
    W4 = torch.randn(hidden_dim2, output_dim, device=device) * 0.01
    b4 = torch.zeros(1, output_dim, device=device)

    return W1, b1, W2, b2, W3, b3, W4, b4

class Generator:
    """
    Generator class for GAN.

    Attributes:
    W1, W2, W3, W4 (torch.Tensor): Weights for each layer.
    b1, b2, b3, b4 (torch.Tensor): Biases for each layer.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Initialize the Generator.

        Args:
        input_dim (int): Number of input features (e.g., latent space dimension).
        hidden_dim1 (int): Number of units in the first hidden layer.
        hidden_dim2 (int): Number of units in the second hidden layer.
        output_dim (int): Number of output features (e.g., flattened image size).
        """
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
    
    def forward(self, z):
        """
        Forward pass for the Generator.

        Args:
        z (torch.Tensor): Input noise tensor (latent space).

        Returns:
        torch.Tensor: Output tensor representing generated data.
        """
        self.z = z
        self.h1 = relu(torch.matmul(self.z, self.W1) + self.b1)
        self.h2 = relu(torch.matmul(self.h1, self.W2) + self.b2)
        self.h3 = relu(torch.matmul(self.h2, self.W3) + self.b3)
        self.output = torch.matmul(self.h3, self.W4) + self.b4
        return self.output

class SGD:
    """
    Custom Stochastic Gradient Descent (SGD) optimizer.
    
    Attributes:
    params (List[torch.Tensor]): List of model parameters to update.
    lr (float): Learning rate for the optimizer.
    """
    def __init__(self, params, lr=0.01):
        """
        Initialize the SGD optimizer.

        Args:
        params (List[torch.Tensor]): List of model parameters to optimize.
        lr (float): Learning rate for gradient descent. Default is 0.01.
        """
        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Perform one optimization step (parameter update).
        """
        for param in self.params:
            for p in param:
                p.data -= self.lr * p.grad
                p.grad.zero_()

def normalize(data):
    """
    Normalize the data to the range [0, 1].

    Args:
    data (torch.Tensor): Input tensor representing the dataset.

    Returns:
    torch.Tensor: Normalized data tensor.
    """
    return data.float() / 255.0

def flatten(tensor):
    """
    Flatten the input tensor from 28x28 to a 784 vector.

    Args:
    tensor (torch.Tensor): Input tensor of shape (batch_size, 28, 28).

    Returns:
    torch.Tensor: Flattened tensor of shape (batch_size, 784).
    """
    return tensor.view(-1, 28 * 28)

def load_mnist():
    """
    Load the MNIST dataset with custom preprocessing.

    Returns:
    torch DataLoader: DataLoader for the MNIST dataset.
    """
    train_dataset = MNIST(root='./data', train=True, download=True, transform=normalize)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader

class Discriminator:
    """
    Discriminator class for GAN.

    Attributes:
    W1, W2, W3, W4 (torch.Tensor): Weights for each layer.
    b1, b2, b3, b4 (torch.Tensor): Biases for each layer.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Initialize the Discriminator.

        Args:
        input_dim (int): Number of input features (e.g., flattened image size).
        hidden_dim1 (int): Number of units in the first hidden layer.
        hidden_dim2 (int): Number of units in the second hidden layer.
        output_dim (int): Number of output features (usually 1 for binary classification).
        """
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
    
    def forward(self, x):
        """
        Forward pass for the Discriminator.

        Args:
        x (torch.Tensor): Input tensor (real or fake data).

        Returns:
        torch.Tensor: Output tensor representing the probability of real data.
        """
        self.x = x
        self.h1 = relu(torch.matmul(self.x, self.W1) + self.b1)
        self.h2 = relu(torch.matmul(self.h1, self.W2) + self.b2)
        self.h3 = relu(torch.matmul(self.h2, self.W3) + self.b3)
        self.output = torch.matmul(self.h3, self.W4) + self.b4
        return sigmoid(self.output)

def train(generator, discriminator, data_loader, num_epochs=1000, lr=0.01):
    """
    Train the GAN model with custom SGD optimizer.

    Args:
    generator (Generator): Generator model.
    discriminator (Discriminator): Discriminator model.
    data_loader (torch.utils.data.DataLoader): DataLoader for the real data.
    num_epochs (int): Number of training epochs.
    lr (float): Learning rate for the optimizers.
    """
    generator_params = [generator.W1, generator.b1, generator.W2, generator.b2, generator.W3, generator.b3, generator.W4, generator.b4]
    discriminator_params = [discriminator.W1, discriminator.b1, discriminator.W2, discriminator.b2, discriminator.W3, discriminator.b3, discriminator.W4, discriminator.b4]
    
    optimizer_g = SGD(generator_params, lr=lr)
    optimizer_d = SGD(discriminator_params, lr=lr)
    
    for epoch in range(num_epochs):
        for real_images, _ in data_loader:
            real_images = flatten(real_images.to(device))
            real_data = custom_preprocess(real_images)
            
            # Generate fake data
            z = torch.randn((real_data.size(0), generator.W1.size(0)), device=device)
            fake_data = generator.forward(z)
            
            # Train Discriminator on real and fake data
            real_output = discriminator.forward(real_data)
            fake_output = discriminator.forward(fake_data.detach())
            
            # Calculate Discriminator loss
            loss_d_real = -torch.mean(torch.log(real_output + 1e-8))
            loss_d_fake = -torch.mean(torch.log(1 - fake_output + 1e-8))
            loss_d = loss_d_real + loss_d_fake
            
            # Backprop for Discriminator
            loss_d.backward()
            optimizer_d.step()
            
            # Train Generator
            fake_output = discriminator.forward(fake_data)
            loss_g = -torch.mean(torch.log(fake_output + 1e-8))
            
            # Backprop for Generator
            loss_g.backward()
            optimizer_g.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")
            visualize_results(fake_data)

def visualize_results(fake_data):
    """
    Visualize the generated images.

    Args:
    fake_data (torch.Tensor): Tensor of generated fake data.
    """
    fake_images = fake_data.view(-1, 28, 28).cpu().data
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(fake_images[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    input_dim = 100
    hidden_dim1 = 256
    hidden_dim2 = 256
    output_dim = 28 * 28

    generator = Generator(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
    discriminator = Discriminator(output_dim, hidden_dim1, hidden_dim2, 1).to(device)

    data_loader = load_mnist()

    train(generator, discriminator, data_loader, num_epochs=1000, lr=0.01)
