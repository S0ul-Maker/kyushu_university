import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)

batch_size = 64
learning_rate = 0.001
num_epochs = 5
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                                ])

train_dataset = torchvision.datasets.MNIST(
    root='../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    root='../data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def add_noise(image, noise_level=0.5):
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, -1, 1)
    return noisy_image


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train():
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            noisy_images = add_noise(images)
            outputs = autoencoder(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


def evaluate():
    autoencoder.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            noisy_images = add_noise(images)
            reconstructed_images = autoencoder(noisy_images).view(-1, 28, 28)

            plt.subplots_adjust(hspace=1.0)
            for i in range(5):
                plt.subplot(4, 5, i + 1)
                plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
                if i == 0:
                    plt.title("Original Images", loc='center')

                plt.subplot(4, 5, i + 6)
                plt.imshow(noisy_images[i].view(
                    28, 28).cpu().numpy(), cmap='gray')
                if i == 0:
                    plt.title("Images with Noise", loc='center')

                plt.subplot(4, 5, i + 11)
                plt.imshow(reconstructed_images[i].cpu().numpy(), cmap='gray')
                if i == 0:
                    plt.title("Reconstructed Images", loc='center')

                plt.subplot(4, 5, i + 16)
                plt.imshow(reconstructed_images[i].cpu().numpy(
                )-images[i].view(28, 28).cpu().numpy(), cmap='gray')
                if i == 0:
                    plt.title("Residual", loc='center')
            plt.show()


if __name__ == '__main__':
    autoencoder = ConvAutoencoder()
    print(autoencoder)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train()
    evaluate()
