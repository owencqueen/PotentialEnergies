import sys, os #sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt

from data.chem_preprocess import PI_data

color_list = ['b', 'g', 'r', 'yellow', 'purple']

def make_distribution(structures):
    #_, y = PI_data(structures, split_by_mol = True, verbose = True)

    #y = []

    for i in range(len(structures)):
        struct_num = int(structures[i][-1]) - 1
        y = pd.read_csv(os.path.join(structures[i], 'energies.csv'), index_col = 0)
        plt.hist(y.iloc[:,0], alpha = 0.5, color = color_list[struct_num], bins = 50, label = f'Molecule {i}')

    if len(structures) > 1:
        plt.title('Energy distribution')
        plt.legend()
    else:
        plt.title(f'Energy distribution - Molecule {structures[0][-1]}')
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':

    assert len(sys.argv) == 2, "Must give structure number as first argument"
    struct = int(sys.argv[1])

    base_str = 'data/xtb_data/base_'
    include_structs = [struct]
    make_distribution([base_str + str(i) for i in include_structs])