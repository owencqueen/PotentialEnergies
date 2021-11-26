import sys; sys.path.append('..')
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data.chem_preprocess import PIDataset

global_train_loss = []
global_test_loss = []

def make_dataloader(base_structures, batch_size = 32, device = None):
    trans = lambda x: x.reshape(1, int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))

    dataset = PIDataset(root = '../data/xtb_data', base_structures = base_structures, transform = trans, device = device)

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
        self.batchnorm1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding = 'same')
        self.batchnorm2 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(16 * PI_dim * PI_dim, 64)
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = torch.nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation on final

        return x

def train(train_dataloader, testloader, model, epochs = 100):
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
                train_loss = running_loss / (100 * int(100 / i)) if i > 0 else running_loss
                test_loss = test(testloader, model)
                global_train_loss.append(train_loss)
                global_test_loss.append(test_loss)
                print(f'Epoch: {ep}, Loss: {train_loss}, Test Loss: {test_loss}')

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

    assert len(sys.argv) == 2, "usage: python3 CNN.py <epochs>"
    epochs = int(sys.argv[1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_dataloader([1], batch_size = 8, device = device)

    train(train_loader, test_loader, model, epochs = epochs)

    print('MAE Test', test(test_loader, model))

    pickle.dump(global_train_loss, open('global_train_loss.pickle', 'wb'))
    pickle.dump(global_test_loss, open('global_test_loss.pickle', 'wb'))

    # plt.plot(global_train_loss, label = 'train')
    # plt.plot(global_test_loss, label = 'test')
    # plt.legend()
    # plt.show()

    #get_embedding(train_loader, model)