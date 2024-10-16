import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Set device
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
    torch.Tensor: Derivative of the sigmoid function.
    """
    return y * (1 - y)

def tanh(x):
    """
    Compute the hyperbolic tangent of x.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Output tensor after applying the tanh function.
    """
    return torch.tanh(x)

def tanh_derivative(y):
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
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(
            input_dim, hidden_dim1, hidden_dim2, output_dim
        )
        # Initialize gradients
        self.d_W1 = torch.zeros_like(self.W1)
        self.d_b1 = torch.zeros_like(self.b1)
        self.d_W2 = torch.zeros_like(self.W2)
        self.d_b2 = torch.zeros_like(self.b2)
        self.d_W3 = torch.zeros_like(self.W3)
        self.d_b3 = torch.zeros_like(self.b3)
        self.d_W4 = torch.zeros_like(self.W4)
        self.d_b4 = torch.zeros_like(self.b4)

    def forward(self, z):
        """
        Forward pass for the Generator.

        Args:
        z (torch.Tensor): Input noise tensor (latent space).

        Returns:
        torch.Tensor: Output tensor representing generated data.
        """
        self.z = z
        self.a1 = torch.matmul(self.z, self.W1) + self.b1
        self.h1 = relu(self.a1)
        self.a2 = torch.matmul(self.h1, self.W2) + self.b2
        self.h2 = relu(self.a2)
        self.a3 = torch.matmul(self.h2, self.W3) + self.b3
        self.h3 = relu(self.a3)
        self.a4 = torch.matmul(self.h3, self.W4) + self.b4
        self.output = tanh(self.a4)  # Use tanh to bound outputs between -1 and 1
        return self.output

    def backward(self, d_output):
        """
        Backward pass for the Generator.

        Args:
        d_output (torch.Tensor): Gradient of the loss with respect to the output.
        """
        # Derivative of tanh activation
        d_a4 = d_output * tanh_derivative(self.output)
        d_W4 = torch.matmul(self.h3.t(), d_a4)
        d_b4 = torch.sum(d_a4, dim=0, keepdim=True)

        d_h3 = torch.matmul(d_a4, self.W4.t())
        d_a3 = d_h3 * relu_derivative(self.a3)
        d_W3 = torch.matmul(self.h2.t(), d_a3)
        d_b3 = torch.sum(d_a3, dim=0, keepdim=True)

        d_h2 = torch.matmul(d_a3, self.W3.t())
        d_a2 = d_h2 * relu_derivative(self.a2)
        d_W2 = torch.matmul(self.h1.t(), d_a2)
        d_b2 = torch.sum(d_a2, dim=0, keepdim=True)

        d_h1 = torch.matmul(d_a2, self.W2.t())
        d_a1 = d_h1 * relu_derivative(self.a1)
        d_W1 = torch.matmul(self.z.t(), d_a1)
        d_b1 = torch.sum(d_a1, dim=0, keepdim=True)

        # Accumulate gradients
        self.d_W1 += d_W1
        self.d_b1 += d_b1
        self.d_W2 += d_W2
        self.d_b2 += d_b2
        self.d_W3 += d_W3
        self.d_b3 += d_b3
        self.d_W4 += d_W4
        self.d_b4 += d_b4

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
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(
            input_dim, hidden_dim1, hidden_dim2, output_dim
        )
        # Initialize gradients
        self.d_W1 = torch.zeros_like(self.W1)
        self.d_b1 = torch.zeros_like(self.b1)
        self.d_W2 = torch.zeros_like(self.W2)
        self.d_b2 = torch.zeros_like(self.b2)
        self.d_W3 = torch.zeros_like(self.W3)
        self.d_b3 = torch.zeros_like(self.b3)
        self.d_W4 = torch.zeros_like(self.W4)
        self.d_b4 = torch.zeros_like(self.b4)

    def forward(self, x):
        """
        Forward pass for the Discriminator.

        Args:
        x (torch.Tensor): Input tensor (real or fake data).

        Returns:
        torch.Tensor: Output tensor representing the probability of real data.
        """
        self.x = x
        self.a1 = torch.matmul(self.x, self.W1) + self.b1
        self.h1 = relu(self.a1)
        self.a2 = torch.matmul(self.h1, self.W2) + self.b2
        self.h2 = relu(self.a2)
        self.a3 = torch.matmul(self.h2, self.W3) + self.b3
        self.h3 = relu(self.a3)
        self.a4 = torch.matmul(self.h3, self.W4) + self.b4
        self.output = sigmoid(self.a4)
        return self.output

    def backward(self, d_output):
        """
        Backward pass for the Discriminator.

        Args:
        d_output (torch.Tensor): Gradient of the loss with respect to the output.
        """
        # Derivative of sigmoid activation
        d_a4 = d_output * sigmoid_derivative(self.output)
        d_W4 = torch.matmul(self.h3.t(), d_a4)
        d_b4 = torch.sum(d_a4, dim=0, keepdim=True)

        d_h3 = torch.matmul(d_a4, self.W4.t())
        d_a3 = d_h3 * relu_derivative(self.a3)
        d_W3 = torch.matmul(self.h2.t(), d_a3)
        d_b3 = torch.sum(d_a3, dim=0, keepdim=True)

        d_h2 = torch.matmul(d_a3, self.W3.t())
        d_a2 = d_h2 * relu_derivative(self.a2)
        d_W2 = torch.matmul(self.h1.t(), d_a2)
        d_b2 = torch.sum(d_a2, dim=0, keepdim=True)

        d_h1 = torch.matmul(d_a2, self.W2.t())
        d_a1 = d_h1 * relu_derivative(self.a1)
        d_W1 = torch.matmul(self.x.t(), d_a1)
        d_b1 = torch.sum(d_a1, dim=0, keepdim=True)

        # Accumulate gradients
        self.d_W1 += d_W1
        self.d_b1 += d_b1
        self.d_W2 += d_W2
        self.d_b2 += d_b2
        self.d_W3 += d_W3
        self.d_b3 += d_b3
        self.d_W4 += d_W4
        self.d_b4 += d_b4

def compute_bce_loss(output, target):
    """
    Compute the binary cross-entropy loss.

    Args:
    output (torch.Tensor): Predicted probabilities.
    target (torch.Tensor): Ground truth labels.

    Returns:
    torch.Tensor: Binary cross-entropy loss.
    """
    epsilon = 1e-8  # Small value to avoid log(0)
    return -torch.mean(target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon))

def compute_bce_loss_derivative(output, target):
    """
    Compute the derivative of the binary cross-entropy loss with respect to the output.

    Args:
    output (torch.Tensor): Predicted probabilities.
    target (torch.Tensor): Ground truth labels.

    Returns:
    torch.Tensor: Derivative of the loss with respect to the output.
    """
    epsilon = 1e-8  # Small value to avoid division by zero
    return (output - target) / ((output + epsilon) * (1 - output + epsilon)) / output.size(0)

def update_params(params, grads, lr):
    """
    Update parameters using gradient descent.

    Args:
    params (List[torch.Tensor]): List of parameters to update.
    grads (List[torch.Tensor]): List of gradients for each parameter.
    lr (float): Learning rate.
    """
    for param, grad in zip(params, grads):
        param -= lr * grad

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

def visualize_results(fake_data):
    """
    Visualize the generated images.

    Args:
    fake_data (torch.Tensor): Tensor of generated fake data.
    """
    fake_images = fake_data.view(-1, 28, 28).cpu().detach()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(fake_images[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

def save_models(generator, discriminator, generator_path='generator.pth', discriminator_path='discriminator.pth'):
    # save generator params
    torch.save({
        'W1': generator.W1,
        'b1': generator.b1,
        'W2': generator.W2,
        'b2': generator.b2,
        'W3': generator.W3,
        'b3': generator.b3,
        'W4': generator.W4,
        'b4': generator.b4
    }, generator_path)
    
    # save discriminator params
    torch.save({
        'W1': discriminator.W1,
        'b1': discriminator.b1,
        'W2': discriminator.W2,
        'b2': discriminator.b2,
        'W3': discriminator.W3,
        'b3': discriminator.b3,
        'W4': discriminator.W4,
        'b4': discriminator.b4
    }, discriminator_path)
    
    print("Models have been saved.")

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
    for epoch in range(num_epochs):
        for real_images, _ in data_loader:
            # Prepare real data
            real_data = real_images.view(-1, 28*28).to(device)
            real_data = (real_data - 0.5) / 0.5  # Normalize to [-1, 1]
            batch_size = real_data.size(0)
            
            # Labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            ########## Train Discriminator ##########
            # Zero out the gradients
            discriminator.d_W1.zero_()
            discriminator.d_b1.zero_()
            discriminator.d_W2.zero_()
            discriminator.d_b2.zero_()
            discriminator.d_W3.zero_()
            discriminator.d_b3.zero_()
            discriminator.d_W4.zero_()
            discriminator.d_b4.zero_()
            
            # Forward pass on real data
            real_output = discriminator.forward(real_data)
            # Compute loss on real data
            loss_d_real = compute_bce_loss(real_output, real_labels)
            # Compute gradient of loss wrt discriminator output
            d_loss_d_output_real = compute_bce_loss_derivative(real_output, real_labels)
            # Backward pass on real data
            discriminator.backward(d_loss_d_output_real)
            
            # Forward pass fake data
            z = torch.randn(batch_size, generator.W1.size(0), device=device)
            fake_data = generator.forward(z)
            fake_output = discriminator.forward(fake_data.detach())
            # Compute loss on fake data
            loss_d_fake = compute_bce_loss(fake_output, fake_labels)
            # Compute gradient of loss wrt discriminator output
            d_loss_d_output_fake = compute_bce_loss_derivative(fake_output, fake_labels)
            # Backward pass on fake data
            discriminator.backward(d_loss_d_output_fake)
            
            # Update discriminator parameters
            update_params(
                [discriminator.W1, discriminator.b1, discriminator.W2, discriminator.b2,
                 discriminator.W3, discriminator.b3, discriminator.W4, discriminator.b4],
                [discriminator.d_W1, discriminator.d_b1, discriminator.d_W2, discriminator.d_b2,
                 discriminator.d_W3, discriminator.d_b3, discriminator.d_W4, discriminator.d_b4],
                lr
            )
            
            ########## Train Generator ##########
            # Zero out the gradients
            generator.d_W1.zero_()
            generator.d_b1.zero_()
            generator.d_W2.zero_()
            generator.d_b2.zero_()
            generator.d_W3.zero_()
            generator.d_b3.zero_()
            generator.d_W4.zero_()
            generator.d_b4.zero_()
            
            # Forward pass fake data through discriminator
            fake_output = discriminator.forward(fake_data)
            # Compute generator loss
            loss_g = compute_bce_loss(fake_output, real_labels)
            # Compute gradient of loss w.r.t. discriminator output
            d_loss_g_output = compute_bce_loss_derivative(fake_output, real_labels)
            # Backward pass through discriminator to get gradients w.r.t. fake data
            discriminator.d_W1.zero_()
            discriminator.d_b1.zero_()
            discriminator.d_W2.zero_()
            discriminator.d_b2.zero_()
            discriminator.d_W3.zero_()
            discriminator.d_b3.zero_()
            discriminator.d_W4.zero_()
            discriminator.d_b4.zero_()
            discriminator.backward(d_loss_g_output)
            d_fake_data = torch.matmul(d_loss_g_output * sigmoid_derivative(discriminator.output), discriminator.W1.t())
            d_fake_data = d_fake_data * relu_derivative(discriminator.a1)

            # Backward pass through generator
            generator.backward(d_fake_data)
            # Update generator parameters
            update_params(
                [generator.W1, generator.b1, generator.W2, generator.b2,
                 generator.W3, generator.b3, generator.W4, generator.b4],
                [generator.d_W1, generator.d_b1, generator.d_W2, generator.d_b2,
                 generator.d_W3, generator.d_b3, generator.d_W4, generator.d_b4],
                lr
            )
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss D: {loss_d_real.item() + loss_d_fake.item()}, Loss G: {loss_g.item()}")
            visualize_results(fake_data)

if __name__ == "__main__":
    input_dim = 100      # Latent space dimension
    hidden_dim1 = 256    # First hidden layer size
    hidden_dim2 = 256    # Second hidden layer size
    output_dim = 784     # Output dimension (flattened image size)

    # Instantiate Generator and Discriminator
    generator = Generator(input_dim, hidden_dim1, hidden_dim2, output_dim)
    discriminator = Discriminator(output_dim, hidden_dim1, hidden_dim2, 1)

    # Move models to device
    generator.W1 = generator.W1.to(device)
    generator.b1 = generator.b1.to(device)
    generator.W2 = generator.W2.to(device)
    generator.b2 = generator.b2.to(device)
    generator.W3 = generator.W3.to(device)
    generator.b3 = generator.b3.to(device)
    generator.W4 = generator.W4.to(device)
    generator.b4 = generator.b4.to(device)
    generator.d_W1 = generator.d_W1.to(device)
    generator.d_b1 = generator.d_b1.to(device)
    generator.d_W2 = generator.d_W2.to(device)
    generator.d_b2 = generator.d_b2.to(device)
    generator.d_W3 = generator.d_W3.to(device)
    generator.d_b3 = generator.d_b3.to(device)
    generator.d_W4 = generator.d_W4.to(device)
    generator.d_b4 = generator.d_b4.to(device)

    discriminator.W1 = discriminator.W1.to(device)
    discriminator.b1 = discriminator.b1.to(device)
    discriminator.W2 = discriminator.W2.to(device)
    discriminator.b2 = discriminator.b2.to(device)
    discriminator.W3 = discriminator.W3.to(device)
    discriminator.b3 = discriminator.b3.to(device)
    discriminator.W4 = discriminator.W4.to(device)
    discriminator.b4 = discriminator.b4.to(device)
    discriminator.d_W1 = discriminator.d_W1.to(device)
    discriminator.d_b1 = discriminator.d_b1.to(device)
    discriminator.d_W2 = discriminator.d_W2.to(device)
    discriminator.d_b2 = discriminator.d_b2.to(device)
    discriminator.d_W3 = discriminator.d_W3.to(device)
    discriminator.d_b3 = discriminator.d_b3.to(device)
    discriminator.d_W4 = discriminator.d_W4.to(device)
    discriminator.d_b4 = discriminator.d_b4.to(device)

    # Load data
    data_loader = load_mnist()

    # Train the GAN
    train(generator, discriminator, data_loader, num_epochs=1000, lr=0.01)
    save_models(generator, discriminator, data_loader)
