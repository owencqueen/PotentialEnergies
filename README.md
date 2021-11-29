# Potential Energy Prediction with Persistence Images
CHEM 420 final project: Owen Queen, Sai Thatigotla, and Henry Eigen

## Installing Dependencies
Install the package ChemTDA by using the `setup.py` file with pip:

```
pip install -e .
```

This will allow you to access the `ChemTDA` package, which is what we use to make the persistence images. The next step is to install all dependencies:

```
pip install -r requirements.txt
```

#### XTB
To generate the conformations that we used, you will have to download (XTB)[https://xtb-docs.readthedocs.io/en/latest/contents.html].

## How to run code
Code below is listed in order of appearance in the PowerPoint.

#### Generating Conformations
To generate the conformations, first change the path within the (generate/md.sh)[https://github.com/owencqueen/PotentialEnergies/blob/main/generate/md.sh] to where you have the xtb executable downloaded.

Then, run the following:
```
cd generate
./md.sh <structure number>
```

"structure number" can be any integer between 1 and 40. This short shell script generates conformations with XTB and then parses and moves all files into the appropriate position in the `data` directory.

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
To run the basic ML models, please run the Jupyter Notebook [models/PI_basic_predictors.ipynb](https://github.com/owencqueen/PotentialEnergies/blob/main/models/PI_basic_predictors.ipynb).

#### Convolutional Neural Networks
To run the CNN, please execute the following command:
```
python3 models/CNN.py <epochs>
```
Where the epochs are the number of training iterations for which you want to train the CNN. We recommend 100, but this will take about 5 minutes to train.

*Note*: Running the CNN is highly stochastic, so results may not match what was presented in the PowerPoint.  
