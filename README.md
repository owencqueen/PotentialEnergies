# PotentialEnergies
CHEM 420 final project.

## Working with PI code
Install the package ChemTDA by using the `setup.py` file with pip:

```pip install -e .```

This will allow you to access the `ChemTDA` package, which is what we use to make the persistence images.

## How to run code

#### TSNE Plot
To make the TSNE plot for molecules 1-5, run:
```
python3 models/tsne.py
```

#### Distribution Plots
To make the plots of the distributions of energies, run
```
python3 models/energy_distribution.py 1
python3 models/energy_distribution.py 2
```
The command line argument specifies the base structure number for which to show the distribution.
