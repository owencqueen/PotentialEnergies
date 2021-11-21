import sys, os
import pandas as pd

def main1(fname, target_dir):
    flines = open(fname, 'r').readlines()

    max_lines = len(flines)
    i = 0
    count = 0

    while True:
        count += 1

        #print('flines', flines[i])

        num_atoms = int(flines[i])
        #print(num_atoms)
        next_spot = num_atoms + 2

        #print(''.join(flines[i:i + next_spot]))

        open(os.path.join(target_dir, f'struct_{count}.xyz'), 'w').write(''.join(flines[i:i + next_spot]))

        i += next_spot

        if i >= max_lines:
            break

def main2(fname, target_dir):

    flines = open(fname,'r').readlines()

    # Number of atoms should be the same for every molecule:
    num_atoms = int(flines[0])

    energy_list=[float(i.split()[1]) for i in flines if 'energy' in i]
    names=[f'struct_{i}' for i in range(len(energy_list))]
    filenames=[f'struct_{i}.xyz' for i in range(len(energy_list))]

    df=pd.DataFrame(energy_list,index=names,columns=['Energy (kcal/mol)'])

    # Put everything in one directory:
    df.to_csv(os.path.join(target_dir, 'energies.csv'))

    xyz=[flines[i:i + num_atoms + 2] for i in range(0, len(flines), num_atoms + 2)]
    for idx,i in enumerate(filenames):
        textfile = open(os.path.join(target_dir, 'STRUCTS',i), "w")
        textfile.write(''.join(xyz[idx]))
        textfile.close()


if __name__ == '__main__':
    option = int(sys.argv[1])

    assert (option == 1) or (option == 2)

    fname = sys.argv[2]
    target_dir = sys.argv[3]

    if option == 1:
        main1(fname, target_dir)
    elif option == 2:
        main2(fname, target_dir)

    

