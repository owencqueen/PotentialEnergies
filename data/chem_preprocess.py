import os, sys, glob
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange
from sklearn.model_selection import train_test_split

from .xyz2mol import read_xyz_file, xyz2mol

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from rdkit.Chem import Draw
from ChemTDA import VariancePersist

def convert_xyz_to_mol(filename):
    atoms, charge, xyz_coordinates = read_xyz_file(filename)

    mols = xyz2mol(
            atoms, 
            xyz_coordinates,
            charge = charge,
            use_graph=True,
            allow_charged_fragments=True,
            embed_chiral=False,
            use_huckel=False)

    # print(mols)

    # Draw.ShowMol(mols[0])

    return mols[0]

# Citation (C) = https://towardsdatascience.com/practical-graph-neural-networks-for-molecular-machine-learning-5e6dee7dc003
def get_atom_features(mol):
    '''
    Make atom features
        - Can be made more robust with background work
    '''
    # Cite: C
    atomic_number = []
    num_hs = []
    
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        
    return torch.tensor([atomic_number, num_hs]).t()

def get_edge_index(mol):
    '''
    Gets edge index for a molecule
    '''
    # Cite: C
    row, col = [], []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        
    return torch.tensor([row, col], dtype=torch.long)

def prepare_dataloader_graph(mol_list, Y = None):
    '''
    Prepares a dataloader given a list of molecules
    '''
    # Cite: C
    data_list = []

    for i, mol in enumerate(mol_list):

        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)

        if i == 0:
            print(edge_index)

        if Y is None:
            data = Data(x=x, edge_index=edge_index)
        else:
            data = Data(x=x, y=Y[i], edge_index=edge_index)
        data_list.append(data)

    return DataLoader(data_list, batch_size=3, shuffle=False), data_list

def get_graph_dataset(dir, Y, test_size = 0.25, seed = None):
    '''
    Make DataLoader from directory
    '''
    files = glob.glob(dir + '/*')

    all_mols = []
    for f in files:
        all_mols.append(convert_xyz_to_mol(f))

   # Split data:
    train_mask, Ytrain, test_mask, Ytest = train_test_split(list(range(len(all_mols))), Y, test_size = test_size)
    train_mols = [all_mols[i] for i in train_mask]
    train_loader = prepare_dataloader(train_mols, Ytrain)

    test_mols = [all_mols[i] for i in test_mask]
    test_loader = prepare_dataloader(test_mols, Ytest)

    #loader = prepare_dataloader(all_mols)
    return train_loader, test_loader

def show_networkx(data):
    '''
    For testing purposes, shows networkx graph
    '''
    G = to_networkx(data, to_undirected=True)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos)
    plt.show()

def PI_data(
    dir_list: list,
    split_by_mol = False,
    verbose: bool = False,
    pixelx: int = 50,
    pixely: int = 50,
    myspread: float = 2):

    MakePI = partial(VariancePersist, 
        pixelx = pixelx, 
        pixely = pixely,
        myspread = myspread,
        showplot = False,
        max_dim = 2,
        n_threads = 5)

    aggregate_X = []
    aggregate_y = []


    for i in range(len(dir_list)):

        dname = dir_list[i]

        # Get energies and filenames of xyz files:
        all_files = glob.glob(os.path.join(dname, 'STRUCTS', '*'))
        energies = pd.read_csv(os.path.join(dname, 'energies.csv'), index_col = 0)

        X = []
        y = energies.iloc[:,0].to_numpy() # To list so we can sort later
        keys = []

        # Load all PIs:
        if verbose: # Use tqdm

            for i in trange(len(all_files)):
                f = all_files[i]
                keys.append(int(os.path.basename(f)[7:-4])) # Gets the number of the file
                X.append(MakePI(f))

        else:
            for i in range(len(all_files)):
                f = all_files[i]
                keys.append(int(os.path.basename(f)[7:-4])) # Gets the number of the file
                X.append(MakePI(f))

        sorting_args = np.argsort(keys)
        X = [X[i] for i in sorting_args]
        keys = [keys[i] for i in sorting_args]

        if split_by_mol:
            aggregate_X.append(X) 
            aggregate_y.append(y) 
        else:
            aggregate_X += X
            aggregate_y += y.tolist()


    return aggregate_X, aggregate_y

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

def save_tensors():
    dir_list = [[os.path.join('xtb_data', n)] for n in os.listdir('xtb_data')]

    for d in dir_list:

        print(os.path.basename(d[0]))

        if os.path.basename(d[0]) == '.DS_Store':
            continue

        elif os.path.exists(os.path.join(d[0], 'X.pt')):
            continue

        agg_X, agg_y = PI_data(d, verbose = True)
        torch.save(torch.tensor(agg_X, dtype=torch.float), str(os.path.join(d[0], 'X.pt')))
        torch.save(torch.tensor(agg_y, dtype=torch.float), str(os.path.join(d[0], 'Y.pt')))

if __name__ == '__main__':
    # Testing conversion methods:
    # data, data_list = get_graph_dataset(dir = 'initial_data/STRUCTS')
    # print(data)

    # show_networkx(data_list[0])

    trans = lambda x: x.reshape(int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))

    dataset = PIDataset(root = 'xtb_data', base_structures = [1], transform = trans)
