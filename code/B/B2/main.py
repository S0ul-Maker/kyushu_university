
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx1):
        # 0 = same class, 1 = different class
        flag = np.random.randint(0, 2)
        img1, label1 = self.dataset[idx1]

        if flag == 0:
            # get two images with the same labels
            idx2 = np.random.choice(
                self.indices[self.dataset.targets == label1])
        else:
            # get images with different labels
            idx2 = np.random.choice(
                self.indices[self.dataset.targets != label1])

        img2, label2 = self.dataset[idx2]
        flag = torch.tensor(flag, dtype=torch.float32)

        return img1, label1, img2, label2, flag


class SiameseModel(torch.nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.drop1 = torch.nn.Dropout(0.25)
        self.drop2 = torch.nn.Dropout(0.50)

    def feed(self, x):
        z = torch.relu(self.conv1(x))
        z = self.pool1(z)
        z = self.drop1(z)
        z = torch.relu(self.conv2(z))
        z = self.pool2(z)

        z = z.reshape(-1, 1024)
        z = torch.relu(self.fc1(z))
        z = self.drop2(z)
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        return z

    def forward(self, x1, x2):
        oupt1 = self.feed(x1)
        oupt2 = self.feed(x2)
        return oupt1, oupt2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, y1, y2, flag):
        # flag = 0 means y1 and y2 are supposed to be same
        # flag = 1 means y1 and y2 are supposed to be different
        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)
        loss = torch.mean((1-flag) * torch.pow(euc_dist, 2) +
                          (flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))

        return loss


def calculate_similarity(siamese_model, image1, image2):
    image1 = image1.reshape(1, 1, 28, 28)
    image2 = image2.reshape(1, 1, 28, 28)
    with torch.no_grad():
        oupt1, oupt2 = siamese_model(image1, image2)
    return torch.nn.functional.pairwise_distance(oupt1, oupt2).cpu().item()


if __name__ == "__main__":
    random_seed = 3
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    batch_size = 256
    epochs = 50
    ep_log_interval = 5
    learning_rate = 0.001

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root='../data', train=True, transform=transform, download=True)
    siamese_train_dataset = SiameseDataset(train_dataset)

    test_dataset = datasets.MNIST(
        root='../data', train=False, transform=transform, download=True)
    siamese_test_dataset = SiameseDataset(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(siamese_train_dataset,
                                                   batch_size=batch_size, shuffle=True)

    net = SiameseModel().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()
    for epoch in range(epochs):
        loss = 0
        for batch in train_dataloader:
            X1, y1, X2, y2, flag = batch
            X1 = X1.to(device)
            X2 = X2.to(device)
            flag = flag.to(device)
            oupt1, oupt2 = net(X1, X2)

            optimizer.zero_grad()
            loss_val = criterion(oupt1, oupt2, flag)

            loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
        if epoch % ep_log_interval == 0:
            print("epoch = %4d,  loss = %10.4f" % (epoch, loss))

    torch.save(net.state_dict(), './model.pt')
    # net.load_state_dict(torch.load('./model.pt'))

# -----------------------------------------------------------
    test_subset = []
    i = 0
    for data in siamese_test_dataset:
        test_subset.append(data)
        if i == 9:
            break
        i += 1

    fig = plt.figure()
    i = 1
    for data in test_subset:
        X1, y1, X2, y2, flag = data
        X1 = X1.to(device)
        X2 = X2.to(device)
        sim = calculate_similarity(net, X1, X2)

        X1 = X1.cpu().numpy().reshape(28, 28)
        X2 = X2.cpu().numpy().reshape(28, 28)
        image = np.concatenate((X1, X2), axis=1)

        ax = fig.add_subplot(5, 4, i)
        ax.title.set_text(
            "label = {}\n sim = {:.4f}".format(flag.item(), sim))
        ax.imshow(image, cmap='gray')
        ax.set_axis_off()

        i += 2
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig("./Result.png")
