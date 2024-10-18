import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from utils import *

class NaiveBayes:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.class_priors = torch.zeros(num_classes)
        self.feature_means = torch.zeros((num_classes, num_features))
        self.feature_vars = torch.zeros((num_classes, num_features))
    
    def fit(self, dataloader):
        class_counts = torch.zeros(self.num_classes)
        feature_sums = torch.zeros((self.num_classes, self.num_features))
        feature_squared_sums = torch.zeros((self.num_classes, self.num_features))
        total_samples = 0

        for images, labels in dataloader:
            images = images.view(images.size(0), -1)  
            for c in range(self.num_classes):
                class_mask = (labels == c)
                class_samples = images[class_mask]
                class_counts[c] += class_mask.sum()
                feature_sums[c] += class_samples.sum(dim=0)
                feature_squared_sums[c] += (class_samples ** 2).sum(dim=0)
            total_samples += images.size(0)

        self.class_priors = class_counts / total_samples
        self.feature_means = feature_sums / class_counts.unsqueeze(1)
        self.feature_vars = (feature_squared_sums / class_counts.unsqueeze(1)) - (self.feature_means ** 2)
        self.feature_vars += 1e-9  # Add small epsilon to prevent division by zero

    def predict(self, images):
        images = images.view(images.size(0), -1)  # Flatten the images
        log_probs = torch.log(self.class_priors).unsqueeze(0)

        for c in range(self.num_classes):
            class_log_probs = -0.5 * torch.sum(
                torch.log(2 * torch.pi * self.feature_vars[c]) + 
                ((images - self.feature_means[c]) ** 2) / self.feature_vars[c],
                dim=1
            )
            log_probs[:, c] += class_log_probs

        return torch.argmax(log_probs, dim=1)

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage with CIFAR-10
if __name__ == "__main__":
    cifar_train = CIFAR10(root='./data', train=True, download=True)
    cifar_test = CIFAR10(root='./data', train=False, download=True)

    train_dataset = ImageDataset(cifar_train, transform=transform)
    test_dataset = ImageDataset(cifar_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_features = 32 * 32 * 3  # CIFAR-10 images are 32x32x3
    num_classes = 10
    model = NaiveBayes(num_features, num_classes)
    model.fit(train_loader)

    # Evaluate the model
    correct = total = 0
    with torch.no_grad(): # don't track gradients
        for images, labels in test_loader:
            predictions = model.predict(images)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
