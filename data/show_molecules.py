import sys, os, random
from rdkit.Chem import Draw

from xyz2mol import read_xyz_file, xyz2mol

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

    #Draw.ShowMol(mols[0])

    im = Draw.MolToImage(mols[0], size = (1000, 1000))
    im.show()

# def show_molecule(file):
#     mol = convert_xyz_to_mol(file)
#     Draw.ShowMol(mol)

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Must give structure number as first argument"
    struct = int(sys.argv[1])

    base_str = 'xtb_data/base_' + str(struct)

    target_structs = os.path.join(base_str, 'STRUCTS')

    # Choose random structure to show:
    rand_struct = random.choice(list(range(len(os.listdir(target_structs)))))

    struct_path = os.path.join(target_structs, 'struct_{}.xyz'.format(rand_struct))

    convert_xyz_to_mol(struct_path)


