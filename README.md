# Potential Energy Prediction with Persistence Images
CHEM 420 final project: Owen Queen, Sai Thatigotla, and Henry Eigen

## Installing Dependencies
Install the package ChemTDA by using the `setup.py` file with pip:

```pip install -e .```

This will allow you to access the `ChemTDA` package, which is what we use to make the persistence images.

## How to run code
Code below is listed in order of appearance in the PowerPoint.

#### Generating Conformations
To generate the conformations 

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

#### Basic ML Models
To run the basic ML models, please run the Jupyter Notebook in.

#### Convolutional Neural Networks
To run the CNN, please execute the following command:
```
python3 models/CNN.py <epochs>
```
Where the epochs are the number of training iterations for which you want to train the CNN. We recommend 100, but this will take about 5 minutes to train.

*Note*: Running the CNN is highly stochastic, so results may not match what was presented in the PowerPoint.  
