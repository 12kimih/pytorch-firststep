import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURR_DIR, 'MNIST')

batch_size = 256
learning_rate = 0.0002
num_epoch = 100

mnist_train = datasets.MNIST(DATASET_DIR, train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(DATASET_DIR, train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    loss_arr = []
    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print(loss)
                loss_arr.append(loss.cpu().detach().numpy())

    correct = 0
    total = 0
    with torch.no_grad():
        for [image, label] in test_loader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y).sum().float()

        print("Accuracy of Test Data: {}".format(100 * correct / total))
