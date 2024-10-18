import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils import *

epsilon = 1e-8

def tanh(x):
    return torch.tanh(x)

def tanh_derivative(y):
    return 1 - y ** 2

def initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim):
    W1 = torch.randn(input_dim, hidden_dim1, device=device) * 0.01
    W1.requires_grad = False
    b1 = torch.zeros(1, hidden_dim1, device=device)
    b1.requires_grad = False
    W2 = torch.randn(hidden_dim1, hidden_dim2, device=device) * 0.01
    W2.requires_grad = False
    b2 = torch.zeros(1, hidden_dim2, device=device)
    b2.requires_grad = False
    W3 = torch.randn(hidden_dim2, hidden_dim2, device=device) * 0.01
    W3.requires_grad = False
    b3 = torch.zeros(1, hidden_dim2, device=device)
    b3.requires_grad = False
    W4 = torch.randn(hidden_dim2, output_dim, device=device) * 0.01
    W4.requires_grad = False
    b4 = torch.zeros(1, output_dim, device=device)
    b4.requires_grad = False

    return W1, b1, W2, b2, W3, b3, W4, b4

class Generator:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(
            input_dim, hidden_dim1, hidden_dim2, output_dim
        )
        # Initialize gradients
        self.grad_W1 = torch.zeros_like(self.W1)
        self.grad_b1 = torch.zeros_like(self.b1)
        self.grad_W2 = torch.zeros_like(self.W2)
        self.grad_b2 = torch.zeros_like(self.b2)
        self.grad_W3 = torch.zeros_like(self.W3)
        self.grad_b3 = torch.zeros_like(self.b3)
        self.grad_W4 = torch.zeros_like(self.W4)
        self.grad_b4 = torch.zeros_like(self.b4)

    def forward(self, z):
        self.z = z
        self.a1 = torch.matmul(self.z, self.W1) + self.b1
        self.h1 = relu(self.a1)
        self.a2 = torch.matmul(self.h1, self.W2) + self.b2
        self.h2 = relu(self.a2)
        self.a3 = torch.matmul(self.h2, self.W3) + self.b3
        self.h3 = relu(self.a3)
        self.a4 = torch.matmul(self.h3, self.W4) + self.b4
        self.output = tanh(self.a4)
        return self.output

    def backward(self, d_output):
        # Derivative of tanh activation
        d_a4 = d_output * tanh_derivative(self.output)
        self.grad_W4 += torch.matmul(self.h3.t(), d_a4)
        self.grad_b4 += torch.sum(d_a4, dim=0, keepdim=True)

        d_h3 = torch.matmul(d_a4, self.W4.t())
        d_a3 = d_h3 * relu_derivative(self.a3)
        self.grad_W3 += torch.matmul(self.h2.t(), d_a3)
        self.grad_b3 += torch.sum(d_a3, dim=0, keepdim=True)

        d_h2 = torch.matmul(d_a3, self.W3.t())
        d_a2 = d_h2 * relu_derivative(self.a2)
        self.grad_W2 += torch.matmul(self.h1.t(), d_a2)
        self.grad_b2 += torch.sum(d_a2, dim=0, keepdim=True)

        d_h1 = torch.matmul(d_a2, self.W2.t())
        d_a1 = d_h1 * relu_derivative(self.a1)
        self.grad_W1 += torch.matmul(self.z.t(), d_a1)
        self.grad_b1 += torch.sum(d_a1, dim=0, keepdim=True)

    def zero_grad(self):
        self.grad_W1.zero_()
        self.grad_b1.zero_()
        self.grad_W2.zero_()
        self.grad_b2.zero_()
        self.grad_W3.zero_()
        self.grad_b3.zero_()
        self.grad_W4.zero_()
        self.grad_b4.zero_()

class Discriminator:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = initialize_weights(
            input_dim, hidden_dim1, hidden_dim2, output_dim
        )
        #initialize the gradients
        self.grad_W1 = torch.zeros_like(self.W1)
        self.grad_b1 = torch.zeros_like(self.b1)
        self.grad_W2 = torch.zeros_like(self.W2)
        self.grad_b2 = torch.zeros_like(self.b2)
        self.grad_W3 = torch.zeros_like(self.W3)
        self.grad_b3 = torch.zeros_like(self.b3)
        self.grad_W4 = torch.zeros_like(self.W4)
        self.grad_b4 = torch.zeros_like(self.b4)

    def forward(self, x):
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
        # Derivative of sigmoid activation
        d_a4 = d_output * sigmoid_derivative(self.output)
        self.grad_W4 += torch.matmul(self.h3.t(), d_a4)
        self.grad_b4 += torch.sum(d_a4, dim=0, keepdim=True)

        d_h3 = torch.matmul(d_a4, self.W4.t())
        d_a3 = d_h3 * relu_derivative(self.a3)
        self.grad_W3 += torch.matmul(self.h2.t(), d_a3)
        self.grad_b3 += torch.sum(d_a3, dim=0, keepdim=True)

        d_h2 = torch.matmul(d_a3, self.W3.t())
        d_a2 = d_h2 * relu_derivative(self.a2)
        self.grad_W2 += torch.matmul(self.h1.t(), d_a2)
        self.grad_b2 += torch.sum(d_a2, dim=0, keepdim=True)

        d_h1 = torch.matmul(d_a2, self.W2.t())
        d_a1 = d_h1 * relu_derivative(self.a1)
        self.grad_W1 += torch.matmul(self.x.t(), d_a1)
        self.grad_b1 += torch.sum(d_a1, dim=0, keepdim=True)

    def zero_grad(self):
        self.grad_W1.zero_()
        self.grad_b1.zero_()
        self.grad_W2.zero_()
        self.grad_b2.zero_()
        self.grad_W3.zero_()
        self.grad_b3.zero_()
        self.grad_W4.zero_()
        self.grad_b4.zero_()

def compute_bce_loss(output, target):
    return -torch.mean(target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon))

def compute_bce_loss_derivative(output, target):
    return (output - target) / ((output + epsilon) * (1 - output + epsilon)) / output.size(0)

def load_mnist():
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader

def visualize_results(fake_data):
    fake_images = fake_data.view(-1, 28, 28).cpu().detach()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(fake_images[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

def save_models(generator, discriminator, generator_path='generator.pth', discriminator_path='discriminator.pth'):
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

class Adam:
    def __init__(self, params, lr=0.0002, betas=(0.5, 0.999)):
        """
        Initialize the Adam optimizer.
        We use Adam here as opposed to SGD for a couple of reasons:
        - SGD is primarily interested in finding the mode of a distribution by finding the parameters that minimize the loss function, rather than modeling the distribution itself. For a generative model like a GAN, this is not useful.
        - SGD also tends to introduce some noise -- because it computes the gradient based on a random subset of the data at each step, it won't cover the full distribution, since each batch is a subset!
        - Adam uses adaptive learning rates for each parameter, adjusting them based on the first and second moments (mean and variance) of the gradients. It also maintains momentum, which is important in generative models like GANs, that have complex loss functions.

        Args:
            params (list): List of dictionaries containing 'params' (parameter tensors) and 'grads'.
            lr (float): Learning rate.
            betas (tuple): Coefficients for computing running averages.
            epsilon (float): Small value to prevent division by zero.

        Note that this implementation is heavily adapted from https://github.com/thetechdude124/Adam-Optimization-From-Scratch/blob/master/CustomAdam.py and my own research :)
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.t = 0  

        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self, params, grads):
        """
        Perform a single optimization step.

        Args:
            params (list): List of parameters to update.
            grads (list): List of gradients for each parameter.
        """
        self.t += 1
        lr_t = self.lr * (torch.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        for i in range(len(params)):
            if grads[i] is None:
                continue

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= lr_t * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self, generator, discriminator):
        """
        Reset gradients for all parameters.

        Args:
            generator (Generator): Generator model.
            discriminator (Discriminator): Discriminator model.
        """
        generator.zero_grad()
        discriminator.zero_grad()

def train(generator, discriminator, data_loader, num_epochs=1000, lr=0.0002):
    """
    Train the GAN model with custom Adam optimizer.

    Args:
        generator (Generator): Generator model.
        discriminator (Discriminator): Discriminator model.
        data_loader (DataLoader): DataLoader for the real data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizers.
    """
    # Collect all generator and discriminator parameters
    gen_params = [generator.W1, generator.b1, generator.W2, generator.b2,
                  generator.W3, generator.b3, generator.W4, generator.b4]
    disc_params = [discriminator.W1, discriminator.b1, discriminator.W2, discriminator.b2,
                  discriminator.W3, discriminator.b3, discriminator.W4, discriminator.b4]

    # Initialize Adam optimizers for generator and discriminator
    optimizer_g = Adam(gen_params, lr=lr)
    optimizer_d = Adam(disc_params, lr=lr)

    for epoch in range(1, num_epochs + 1):
        for real_images, _ in data_loader:
            # Prepare real data
            real_data = real_images.view(-1, 28*28).to(device)
            real_data = (real_data - 0.5) / 0.5  # Normalize to [-1, 1]
            batch_size = real_data.size(0)

            # Labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            ########## Train Discriminator ##########
            # Forward pass on real data
            real_output = discriminator.forward(real_data)
            loss_d_real = compute_bce_loss(real_output, real_labels)
            d_loss_d_output_real = compute_bce_loss_derivative(real_output, real_labels)
            discriminator.backward(d_loss_d_output_real)

            # Forward pass fake data
            z = torch.randn(batch_size, generator.W1.size(0), device=device)
            fake_data = generator.forward(z)
            fake_output = discriminator.forward(fake_data.detach())
            loss_d_fake = compute_bce_loss(fake_output, fake_labels)
            d_loss_d_output_fake = compute_bce_loss_derivative(fake_output, fake_labels)
            discriminator.backward(d_loss_d_output_fake)

            # Collect discriminator gradients
            grad_disc = [discriminator.grad_W1, discriminator.grad_b1,
                         discriminator.grad_W2, discriminator.grad_b2,
                         discriminator.grad_W3, discriminator.grad_b3,
                         discriminator.grad_W4, discriminator.grad_b4]

            # Update discriminator parameters
            optimizer_d.step(disc_params, grad_disc)

            # Zero discriminator gradients
            discriminator.zero_grad()

            ########## Train Generator ##########
            # Forward pass fake data through discriminator
            fake_output = discriminator.forward(fake_data)
            loss_g = compute_bce_loss(fake_output, real_labels)
            d_loss_g_output = compute_bce_loss_derivative(fake_output, real_labels)
            discriminator.backward(d_loss_g_output)
            # Note: Here we need to backpropagate through the discriminator to the generator

            # Collect generator gradients
            grad_gen = [generator.grad_W1, generator.grad_b1,
                        generator.grad_W2, generator.grad_b2,
                        generator.grad_W3, generator.grad_b3,
                        generator.grad_W4, generator.grad_b4]

            # Update generator parameters
            optimizer_g.step(gen_params, grad_gen)

            # Zero generator gradients
            generator.zero_grad()

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss D: {loss_d_real.item() + loss_d_fake.item():.4f}, Loss G: {loss_g.item():.4f}")
            visualize_results(fake_data)

    save_models(generator, discriminator)

if __name__ == "__main__":
    input_dim = 100      # Latent space dimension
    hidden_dim1 = 256    # First hidden layer size
    hidden_dim2 = 256    # Second hidden layer size
    output_dim = 784     # Output dimension (flattened image size)

    # Instantiate Generator and Discriminator
    generator = Generator(input_dim, hidden_dim1, hidden_dim2, output_dim)
    discriminator = Discriminator(output_dim, hidden_dim1, hidden_dim2, 1)

    # Load data
    data_loader = load_mnist()

    # Train the GAN
    train(generator, discriminator, data_loader, num_epochs=1000, lr=0.0002)
