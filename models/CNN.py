import sys; sys.path.append('..')
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data.chem_preprocess import PIDataset

def make_dataloader(base_structures, batch_size = 32):
    trans = lambda x: x.reshape(1, int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))

    dataset = PIDataset(root = '../data/xtb_data', base_structures = base_structures, transform = trans)

    splits = [int(0.7 * len(dataset)), int(0.3 * len(dataset))]

    train_data, test_data = torch.utils.data.random_split(dataset, [splits[0], splits[1] + (len(dataset) - sum(splits))])

    return DataLoader(train_data, batch_size = batch_size, shuffle=True), DataLoader(test_data, batch_size = len(test_data), shuffle=False)

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class CNN1(torch.nn.Module):
    def __init__(self, PI_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, padding = 'same')
        self.batchnorm = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding = 'same')
        self.fc1 = torch.nn.Linear(16 * PI_dim * PI_dim, 64)
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm(x)
        x = F.relu(self.conv2(x))
        x = torch.nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation on final

        return x

def train(train_dataloader, model, epochs = 100):
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for ep in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            X, y = data
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch: {ep}, Loss: {running_loss / 100}')

        # for X, y in test_dataloader:
            
def test(test_dataloader, model):

    criterion = torch.nn.L1Loss()

    total_loss = 0
    n = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            output = model(X)

            loss = criterion(output, y)

            total_loss += loss
            n += 1

    return total_loss / n

def get_embedding(loader, model):

    all_ys = []
    all_Xs = []

    with torch.no_grad():
        for X, y in loader:
            all_ys.append(y.detach())
            _ = model(X)
            all_Xs.append(activation['fc1'].detach())

    all_ys = torch.cat(all_ys, dim=0).tolist()
    all_Xs = torch.cat(all_Xs, dim = 0).numpy()

    xtsne = TSNE().fit_transform(all_Xs)
    plt.scatter(xtsne[:,0], xtsne[:,1], c = all_ys)
    plt.show()


if __name__ == '__main__':
    model = CNN1(PI_dim = 50)

    train_loader, test_loader = make_dataloader([1], batch_size = 64)

    train(train_loader, model, epochs = 50)

    print('MAE Test', test(test_loader, model))

    get_embedding(train_loader, model)