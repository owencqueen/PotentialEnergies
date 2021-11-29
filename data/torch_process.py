import torch, os


class PIDataset(torch.utils.data.Dataset):
    def __init__(self, root, base_structures, transform = None, device = None):

        Xlist = []
        ylist = []

        for i in base_structures:
            Xi = torch.load(os.path.join(root, f'base_{i}', 'X.pt'))
            yi = torch.load(os.path.join(root, f'base_{i}', 'Y.pt'))

            Xlist.append(Xi)
            ylist.append(yi)

        self.X = torch.cat(Xlist, dim=0)
        self.Y = torch.cat(ylist, dim=0)

        if transform is not None:
            # Apply transforms to self.X:
            Xnew = []
            for i in range(self.X.shape[0]):
                Xnew.append(transform(self.X[i]))

            self.X = torch.stack(Xnew)

        self.X.requires_grad = True # Require grad (for training)

        if device is not None:
            self.X.to(device)
            self.Y.to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]