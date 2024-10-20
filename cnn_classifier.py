from utils import *
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)

def softmax(x):
    exps = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exps / torch.sum(exps, dim=1, keepdim=True)

def cross_entropy_loss(predictions, targets):
    epsilon = 1e-12
    predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
    return -torch.mean(torch.sum(targets * torch.log(predictions), dim=1))

def cross_entropy_derivative(predictions, targets):
    return (predictions - targets) / predictions.shape[0]

def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes, device=device)
    one_hot[torch.arange(len(labels)), labels] = 1.0
    return one_hot

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Initialize weights and biases
        self.weights = torch.randn(out_channels, in_channels, *self.kernel_size, device=device) * torch.sqrt(2. / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.bias = torch.zeros(out_channels, device=device)

        # Gradients
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Calculate output dimensions
        out_height = (input_height - kernel_height + 2 * pad_h) // stride_h + 1
        out_width = (input_width - kernel_width + 2 * pad_w) // stride_w + 1

        # Apply padding
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h))
        else:
            x_padded = x

        # Initialize output
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=device)
      
        self.x_cols = self.im2col(x_padded, kernel_height, kernel_width, stride_h, stride_w)
        W_col = self.weights.view(self.out_channels, -1)
        out_cols = W_col @ self.x_cols + self.bias.unsqueeze(1)
        out = out_cols.view(self.out_channels, batch_size, out_height, out_width).permute(1, 0, 2, 3)

        return out

    def im2col(self, x_padded, kernel_h, kernel_w, stride_h, stride_w):
        """
        Rearrange image blocks into columns.
        """
        batch_size, channels, height, width = x_padded.shape
        out_h = (height - kernel_h) // stride_h + 1
        out_w = (width - kernel_w) // stride_w + 1

        i0 = torch.repeat_interleave(torch.arange(kernel_h, device=device), kernel_w)
        i0 = i0.view(-1, 1)
        i1 = torch.arange(out_h * stride_h, step=stride_h, device=device).repeat(kernel_h * kernel_w, out_w)
        j0 = torch.tile(torch.arange(kernel_w, device=device), (kernel_h, 1)).reshape(-1, 1)
        j1 = torch.arange(out_w * stride_w, step=stride_w, device=device).repeat(kernel_h * kernel_w, out_h).t().reshape(-1, 1)

        i = i0 + i1
        j = j0 + j1

        # Extract patches
        cols = x_padded[:, :, i, j]
        cols = cols.permute(1, 2, 0).reshape(x_padded.shape[1] * kernel_h * kernel_w, -1)
        return cols

    def backward(self, grad_output, learning_rate):
        """
        Backward pass of the Conv2D layer.
        """
        batch_size, out_channels, out_height, out_width = grad_output.shape
        grad_output_reshaped = grad_output.permute(1, 0, 2, 3).reshape(out_channels, -1)

        # Compute gradients wrt weights and biases
        self.grad_weights = grad_output_reshaped @ self.x_cols.t()
        self.grad_weights = self.grad_weights.view(self.weights.shape)
        self.grad_bias = grad_output_reshaped.sum(dim=1)

        # Update weights and biases
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

        # Compute gradient wrt input
        W_flat = self.weights.view(out_channels, -1)
        grad_input_cols = W_flat.t() @ grad_output_reshaped
        grad_input = self.col2im(grad_input_cols, self.input.shape, self.kernel_size, self.stride, self.padding)

        return grad_input

    def col2im(self, cols, input_shape, kernel_size, stride, padding):
        """
        Rearrange columns into image blocks.
        """
        batch_size, channels, height, width = input_shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        out_h = (height - kernel_h + 2 * pad_h) // stride_h + 1
        out_w = (width - kernel_w + 2 * pad_w) // stride_w + 1

        grad_input_padded = torch.zeros((batch_size, channels, height + 2 * pad_h, width + 2 * pad_w), device=device)
        W_flat = self.weights.view(self.out_channels, -1)

        grad_input_cols = cols @ W_flat
        grad_input_cols = grad_input_cols.view(self.in_channels, -1, batch_size, out_h, out_w)
        grad_input_cols = grad_input_cols.permute(2, 0, 3, 4, 1)

        for i in range(out_h):
            for j in range(out_w):
                grad_input_padded[:, :, i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w] += grad_input_cols[:, :, i, j, :].view(batch_size, channels, kernel_h, kernel_w)

        if pad_h > 0 or pad_w > 0:
            grad_input = grad_input_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            grad_input = grad_input_padded

        return grad_input

class ReLU:
    def forward(self, x):
        """
        Forward pass of the ReLU activation.
        """
        self.input = x
        return relu(x)

    def backward(self, grad_output, learning_rate):
        """
        Backward pass of the ReLU activation.
        """
        return relu_derivative(self.input) * grad_output

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        """
        Initialize a MaxPool2D layer.
        """
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

    def forward(self, x):
        """
        Forward pass of the MaxPool2D layer.
        """
        self.input = x
        batch_size, channels, height, width = x.shape
        pool_h, pool_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_h = (height - pool_h) // stride_h + 1
        out_w = (width - pool_w) // stride_w + 1

        # Reshape input to columns
        x_reshaped = x.unfold(2, pool_h, stride_h).unfold(3, pool_w, stride_w)
        self.x_reshaped = x_reshaped.contiguous().view(batch_size, channels, out_h, out_w, pool_h * pool_w)

        # Get max values and their indices
        out, self.max_indices = torch.max(self.x_reshaped, dim=4)
        return out

    def backward(self, grad_output, learning_rate):
        """
        Backward pass of the MaxPool2D layer.
        """
        batch_size, channels, out_h, out_w = grad_output.shape
        pool_h, pool_w = self.kernel_size

        grad = torch.zeros_like(self.x_reshaped, device=device)
        grad.scatter_(4, self.max_indices.unsqueeze(-1), grad_output.unsqueeze(-1))
        grad = grad.view(batch_size, channels, out_h, out_w, pool_h, pool_w)
        grad_input = torch.zeros_like(self.input, device=device)

        for i in range(pool_h):
            for j in range(pool_w):
                grad_input[:, :, i* self.stride[0]:i* self.stride[0]+out_h * self.stride[0]:self.stride[0],
                           j* self.stride[1]:j* self.stride[1]+out_w * self.stride[1]:self.stride[1]] += grad[:, :, :, :, i, j]

        return grad_input

class Flatten:
    def forward(self, x):
        """
        Forward pass of the Flatten layer.
        """
        self.input_shape = x.shape
        return x.view(x.shape[0], -1)

    def backward(self, grad_output, learning_rate):
        """
        Backward pass of the Flatten layer.
        """
        return grad_output.view(self.input_shape)

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize a Linear (fully connected) layer.
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        self.weights = torch.randn(in_features, out_features, device=device) * torch.sqrt(2. / in_features)
        self.bias = torch.zeros(out_features, device=device)

        # gradient update
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def forward(self, x):
        """
        Forward pass of the Linear layer.
        """
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output, learning_rate):
        """
        Backward pass of the Linear layer.
        """
        self.grad_weights = self.input.t() @ grad_output
        self.grad_bias = torch.sum(grad_output, dim=0)

        # Update weights and biases
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

        return grad_output @ self.weights.t()
      
class CNN:
    def __init__(self, num_classes=10):
        """
        The CNN
        """
        # Define layers
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64 * 3 * 3, out_features=128)
        self.relu4 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass through all layers.
        """
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu4.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, grad_output, learning_rate):
        """
        Backward pass through all layers.
        """
        grad = self.fc2.backward(grad_output, learning_rate)
        grad = self.relu4.backward(grad, learning_rate)
        grad = self.fc1.backward(grad, learning_rate)
        grad = self.flatten.backward(grad, learning_rate)
        grad = self.pool3.backward(grad, learning_rate)
        grad = self.relu3.backward(grad, learning_rate)
        grad = self.conv3.backward(grad, learning_rate)
        grad = self.pool2.backward(grad, learning_rate)
        grad = self.relu2.backward(grad, learning_rate)
        grad = self.conv2.backward(grad, learning_rate)
        grad = self.pool1.backward(grad, learning_rate)
        grad = self.relu1.backward(grad, learning_rate)
        grad = self.conv1.backward(grad, learning_rate)

    def train_model(self, train_loader, epochs=10, learning_rate=0.01):
        """
        Train the CNN model.
        """
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in progress:
                # Prepare input data and move to device
                batch_x = batch_x.to(device).unsqueeze(1)  # Add channel dimension
                batch_y = one_hot_encode(batch_y, num_classes=10)

                # Forward pass
                logits = self.forward(batch_x)
                probs = softmax(logits)
                loss = cross_entropy_loss(probs, batch_y)
                epoch_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(probs, dim=1)
                _, targets = torch.max(batch_y, dim=1)
                correct += (predicted == targets).sum().item()
                total += batch_y.size(0)

                # Backward pass
                grad_loss = cross_entropy_derivative(probs, batch_y)
                self.backward(grad_loss, learning_rate)

                # Update progress bar
                progress.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    def evaluate_model(self, test_loader):
        """
        Evaluate the CNN model.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device).unsqueeze(1)  # Add channel dimension
                batch_y = one_hot_encode(batch_y, num_classes=10)

                logits = self.forward(batch_x)
                probs = softmax(logits)

                _, predicted = torch.max(probs, dim=1)
                _, targets = torch.max(batch_y, dim=1)
                correct += (predicted == targets).sum().item()
                total += batch_y.size(0)
        accuracy = 100. * correct / total
        print(f"Evaluation Accuracy: {accuracy:.2f}%")

# Data Loading
def load_mnist(batch_size=64):
    """
    Load the MNIST dataset with custom preprocessing.
    """
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Visualization Functions
def visualize_results(original_data, predicted_labels, true_labels, n=5):
    """
    Visualize original images with their predicted and true labels.
    """
    original_images = original_data.view(-1, 28, 28).cpu().detach()
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        axes[i].imshow(original_images[i], cmap='gray')
        axes[i].set_title(f"Pred: {predicted_labels[i].item()}\nTrue: {true_labels[i].item()}")
        axes[i].axis('off')
    plt.show()

# Main Function
if __name__ == "__main__":
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    num_classes = 10

    train_loader, test_loader = load_mnist(batch_size=batch_size)

    cnn = CNN(num_classes=num_classes)

    cnn.train_model(train_loader, epochs=epochs, learning_rate=learning_rate)

    # Evaluate the CNN
    cnn.evaluate_model(test_loader)

    # visualize the predictions
    test_iter = iter(test_loader)
    batch_x, batch_y = next(test_iter)
    batch_x = batch_x.to(device).unsqueeze(1)
    logits = cnn.forward(batch_x)
    probs = softmax(logits)
    _, predicted = torch.max(probs, dim=1)

    visualize_results(batch_x, predicted, batch_y, n=5)
