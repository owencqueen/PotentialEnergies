import os, sys, glob
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from xyz2mol import read_xyz_file, xyz2mol

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from rdkit.Chem import Draw

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

def prepare_dataloader(mol_list, Y = None):
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

if __name__ == '__main__':
    # Testing conversion methods:
    data, data_list = get_graph_dataset(dir = 'initial_data/STRUCTS')
    print(data)

    show_networkx(data_list[0])
