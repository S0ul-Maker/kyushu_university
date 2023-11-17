import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def add_noise(images, noise_level=0.2):
    noisy_images = images + noise_level * torch.randn_like(images)
    return torch.clamp(noisy_images, 0, 1)


if __name__ == '__main__':

    torch.manual_seed(42)
    writer = SummaryWriter()

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, transform=transform, download=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                net.eval()
                correct_train = 0
                total_train = 0

                with torch.no_grad():
                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_train += labels.shape[0]
                        correct_train += (predicted == labels).sum().item()

                accuracy_train = 100 * correct_train / total_train

                correct_test = 0
                total_test = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_test += labels.shape[0]
                        correct_test += (predicted == labels).sum().item()

                accuracy_test = 100 * correct_test / total_test

                print(f'Epoch [{epoch+1}/{num_epochs}] | Step [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Train Acc: {accuracy_train:.2f}% | Test Acc: {accuracy_test:.2f}%')
                writer.add_scalars('Clean MNIST Data', {
                    'Loss': loss.item(),
                    'Train Accuracy': accuracy_train,
                    'Test Accuracy': accuracy_test
                }, epoch+1)

                net.train()
    print('Finished Training on Clean Data')

    # -----------------------------------------------------------------------------

    noisy_trainset = [(add_noise(image, noise_level=0.4), label)
                      for image, label in train_dataset]
    noisy_testset = [(add_noise(image, noise_level=0.4), label)
                     for image, label in test_dataset]

    noisy_trainloader = DataLoader(
        noisy_trainset, batch_size=batch_size, shuffle=True)
    noisy_testloader = DataLoader(
        noisy_testset, batch_size=batch_size, shuffle=False)

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(noisy_trainloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                net.eval()
                correct_train = 0
                total_train = 0

                with torch.no_grad():
                    for images, labels in noisy_trainloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_train += labels.shape[0]
                        correct_train += (predicted == labels).sum().item()

                accuracy_train = 100 * correct_train / total_train

                correct_test = 0
                total_test = 0

                with torch.no_grad():
                    for images, labels in noisy_testloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total_test += labels.shape[0]
                        correct_test += (predicted == labels).sum().item()

                accuracy_test = 100 * correct_test / total_test

                print(f'Epoch [{epoch+1}/{num_epochs}] | Step [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Train Acc: {accuracy_train:.2f}% | Test Acc: {accuracy_test:.2f}%')
                writer.add_scalars('Noised MNIST Data', {
                    'Loss': loss.item(),
                    'Train Accuracy': accuracy_train,
                    'Test Accuracy': accuracy_test
                }, epoch+1)
                net.train()

    print('Finished Training on Noisy Data')
