#import sys; sys.path.append('..')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data.chem_preprocess import PI_data

def make_tsne(structures):
    if len(structures) == 1:
        X, y = PI_data(structures, split_by_mol=False)

        Xtsne = TSNE().fit_transform(X)

        plt.scatter(Xtsne[:,0], Xtsne[:,1], c = y)
        plt.title('TSNE of Base Structure 1')
        plt.xlabel('TSNE 1')
        plt.colorbar()
        plt.ylabel('TSNE 2')
        plt.show()
    else:
        X, y = PI_data(structures, split_by_mol=True)

        Xtot = []
        ytot = []

        start_spots = []

        for i in range(len(X)):
            start_spots.append(len(ytot))
            Xi, yi = X[i], [i] * len(X[i])
            Xtot += Xi
            ytot += yi

        Xtsne = TSNE().fit_transform(Xtot)
        for i in range(len(start_spots)):
            if i == len(start_spots) - 1:
                upper = -1
            else:
                upper = start_spots[i+1]

            lower = start_spots[i]

            plt.scatter(Xtsne[lower:upper,0], Xtsne[lower:upper,1], label = 'Molecule ' + str(i + 1))
            
        #plt.scatter(Xtsne[:,0], Xtsne[:,1], c = ytot, label = ytot)
        plt.title('Molecules {}'.format([s[-1:] for s in structures]))
        plt.legend()
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
        plt.show()


if __name__ == '__main__':
    base_str = 'data/xtb_data/base_'
    include_structs = [1, 2, 3, 4, 5]
    make_tsne([base_str + str(i) for i in include_structs])