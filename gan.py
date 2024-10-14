import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(y):
    # Derivative of sigmoid assuming y = sigmoid(x)
    return y * (1 - y)

def tanh(x):
    return torch.tanh(x)

def dtanh(y):
    # Derivative of tanh assuming y = tanh(x)
    return 1 - y ** 2

def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

def drelu(x):
    # Derivative of ReLU
    return (x > 0).float()

def initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim, extra_layer=False):
    # Initialize weights with small random values
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
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        # Initialize weights and biases
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
        
    def forward(self, z):
        self.z = z  # Input noise
        self.h1 = relu(torch.matmul(z, self.W1) + self.b1)  # First hidden layer
        self.h2 = relu(torch.matmul(self.h1, self.W2) + self.b2)  # Second hidden layer
        self.h3 = relu(torch.matmul(self.h2, self.W3) + self.b3)  # Third hidden layer
        self.out = tanh(torch.matmul(self.h3, self.W4) + self.b4)  # Output layer
        return self.out
    
    def backward(self, d_out, lr):
        # d_out is the gradient of loss w.r.t self.out
        dtanh_out = d_out * dtanh(self.out)  # Gradient through tanh
        
        # Gradients for W4 and b4
        dW4 = torch.matmul(self.h3.t(), dtanh_out)
        db4 = torch.sum(dtanh_out, dim=0, keepdim=True)
        
        # Gradient w.r.t h3
        dh3 = torch.matmul(dtanh_out, self.W4.t())
        drelu_h3 = dh3 * drelu(self.h3)  # Gradient through ReLU
        
        # Gradients for W3 and b3
        dW3 = torch.matmul(self.h2.t(), drelu_h3)
        db3 = torch.sum(drelu_h3, dim=0, keepdim=True)
        
        # Gradient w.r.t h2
        dh2 = torch.matmul(drelu_h3, self.W3.t())
        drelu_h2 = dh2 * drelu(self.h2)  # Gradient through ReLU
        
        # Gradients for W2 and b2
        dW2 = torch.matmul(self.h1.t(), drelu_h2)
        db2 = torch.sum(drelu_h2, dim=0, keepdim=True)
        
        # Gradient w.r.t h1
        dh1 = torch.matmul(drelu_h2, self.W2.t())
        drelu_h1 = dh1 * drelu(self.h1)  # Gradient through ReLU
        
        # Gradients for W1 and b1
        dW1 = torch.matmul(self.z.t(), drelu_h1)
        db1 = torch.sum(drelu_h1, dim=0, keepdim=True)
        
        # Update weights and biases
        self.W4 -= lr * dW4
        self.b4 -= lr * db4
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

class Discriminator:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        # Initialize weights and biases
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
        
    def forward(self, x):
        self.x = x  # Input image
        self.h1 = relu(torch.matmul(x, self.W1) + self.b1)  # First hidden layer
        self.h2 = relu(torch.matmul(self.h1, self.W2) + self.b2)  # Second hidden layer
        self.h3 = relu(torch.matmul(self.h2, self.W3) + self.b3)  # Third hidden layer
        self.out = sigmoid(torch.matmul(self.h3, self.W4) + self.b4)  # Output layer
        return self.out
    
    def backward(self, d_out, lr):
        # d_out is the gradient of loss w.r.t self.out
        dsig_out = d_out * sigmoid_derivative(self.out)  
        
        # Gradients for W4 and b4
        dW4 = torch.matmul(self.h3.t(), dsig_out)
        db4 = torch.sum(dsig_out, dim=0, keepdim=True)
        
        # Gradient w.r.t h3
        dh3 = torch.matmul(dsig_out, self.W4.t())
        drelu_h3 = dh3 * drelu(self.h3)  # Gradient through ReLU
        
        # Gradients for W3 and b3
        dW3 = torch.matmul(self.h2.t(), drelu_h3)
        db3 = torch.sum(drelu_h3, dim=0, keepdim=True)
        
        # Gradient w.r.t h2
        dh2 = torch.matmul(drelu_h3, self.W3.t())
        drelu_h2 = dh2 * drelu(self.h2)  # Gradient through ReLU
        
        # Gradients for W2 and b2
        dW2 = torch.matmul(self.h1.t(), drelu_h2)
        db2 = torch.sum(drelu_h2, dim=0, keepdim=True)
        
        # Gradient w.r.t h1
        dh1 = torch.matmul(drelu_h2, self.W2.t())
        drelu_h1 = dh1 * drelu(self.h1)  # Gradient through ReLU
        
        # Gradients for W1 and b1
        dW1 = torch.matmul(self.x.t(), drelu_h1)
        db1 = torch.sum(drelu_h1, dim=0, keepdim=True)
        
        # Update weights and biases
        self.W4 -= lr * dW4
        self.b4 -= lr * db4
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-8  # To prevent log(0)
    return -torch.mean(y_true * torch.log(y_pred + epsilon) + (1 - y_true) * torch.log(1 - y_pred + epsilon))

def dbinary_cross_entropy(y_pred, y_true):
    epsilon = 1e-8  # To prevent division by zero
    return (y_pred - y_true) / ((y_pred + epsilon) * (1 - y_pred + epsilon))

def main():
    # -----------------------------
    # 1. Setup and Data Preparation
    # -----------------------------
    
    # Set device to GPU if available, else CPU
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transformations: Convert to tensor, normalize to [-1, 1], and flatten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])
    
    # Download and load the training dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    # -----------------------------
    # 2. Initialize Generator and Discriminator
    # -----------------------------
    
    # Hyperparameters
    z_dim = 100
    hidden_dim1 = 256
    hidden_dim2 = 256
    output_dim = 784  # 28x28 images
    lr = 0.0002
    epochs = 50
    batch_size = 64
    
    # Initialize generator and discriminator
    G = Generator(input_dim=z_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim).to(device)
    D = Discriminator(input_dim=output_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=1).to(device)
    
    # -----------------------------
    # 3. Training the GAN
    # -----------------------------
    
    for epoch in range(1, epochs + 1):
        for real_imgs, _ in train_loader:
            # Move real images to device
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Real images
            D_real = D.forward(real_imgs)
            real_labels = torch.ones(current_batch_size, 1, device=device)
            loss_real = binary_cross_entropy(D_real, real_labels)
            d_loss_real = dbinary_cross_entropy(D_real, real_labels)
            D.backward(d_loss_real, lr)
            
            # Fake images
            z = torch.randn(current_batch_size, z_dim, device=device)
            fake_imgs = G.forward(z)
            D_fake = D.forward(fake_imgs)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)
            loss_fake = binary_cross_entropy(D_fake, fake_labels)
            d_loss_fake = dbinary_cross_entropy(D_fake, fake_labels)
            D.backward(d_loss_fake, lr)
            
            # Total discriminator loss
            d_loss = loss_real + loss_fake
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate fake images
            z = torch.randn(current_batch_size, z_dim, device=device)
            fake_imgs = G.forward(z)
            D_fake = D.forward(fake_imgs)
            valid_labels = torch.ones(current_batch_size, 1, device=device)  # Generator tries to make D_fake as real
            
            # Generator loss
            g_loss = binary_cross_entropy(D_fake, valid_labels)
            d_loss_g = dbinary_cross_entropy(D_fake, valid_labels)
            
            # Backward pass for generator
            # Compute gradients w.r.t Generator's output
            dtanh_out = d_loss_g * dtanh(G.out)  # Gradient through tanh
            
            # Gradients for G's W3 and b3
            dW3_G = torch.matmul(G.h2.t(), dtanh_out)
            db3_G = torch.sum(dtanh_out, dim=0, keepdim=True)
            
            # Gradient w.r.t h2 in generator
            dh2_G = torch.matmul(dtanh_out, G.W3.t())
            drelu_h2_G = dh2_G * drelu(G.h2)  # Gradient through ReLU
            
            # Gradients for G's W2 and b2
            dW2_G = torch.matmul(G.h1.t(), drelu_h2_G)
            db2_G = torch.sum(drelu_h2_G, dim=0, keepdim=True)
            
            # Gradient w.r.t h1 in generator
            dh1_G = torch.matmul(drelu_h2_G, G.W2.t())
            drelu_h1_G = dh1_G * drelu(G.h1)  # Gradient through ReLU
            
            # Gradients for G's W1 and b1
            dW1_G = torch.matmul(G.z.t(), drelu_h1_G)
            db1_G = torch.sum(drelu_h1_G, dim=0, keepdim=True)
            
            # Update generator's weights and biases
            G.W3 -= lr * dW3_G
            G.b3 -= lr * db3_G
            G.W2 -= lr * dW2_G
            G.b2 -= lr * db2_G
            G.W1 -= lr * dW1_G
            G.b1 -= lr * db1_G
            
        # Print losses at the end of each epoch
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    # -----------------------------
    # 4. Generating and Displaying a Fake Image
    # -----------------------------
    
    # Generate a fake image after training
    with torch.no_grad():
        z = torch.randn(1, z_dim, device=device)
        fake_img = G.forward(z)
        fake_img = fake_img.view(28, 28).cpu().numpy()  # Reshape and move to CPU
    
    # Rescale from [-1, 1] to [0, 1] for visualization
    fake_img = (fake_img + 1) / 2
    
    # Display the image
    plt.imshow(fake_img, cmap='gray')
    plt.title("Generated Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
