from ripser import Rips
from ripser import ripser
from gph.python import ripser_parallel
rips = Rips(verbose=False)
from sklearn.base import TransformerMixin
import numpy as np
import collections
from itertools import product
import collections
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import scipy.spatial as spatial
import matplotlib.pyplot as plt

from .elements import ELEMENTS


def Makexyzdistance(t, generator = False):
    ''' Prepares a file with geometric chemical data for use in Vietoris-Rips filtration.

    Parameters
    ----------
    t: string
        - Name of source file for coordinate data for a compound
        - See documentation's specifications for file structure
        - File contains `(x, y, z)` coordinates for each atom in compound

    Returns
    -------
    Distance: Numpy array
        - Distance matrix for every atom in the compound
        - Used as input to Vietoris-Rips filtration function (`ripser`)
    element: string
        - Name of the element being read 

    Example
    -------
    Generating distance matrix for chemical compound stored in a file named `compound.xyz`::

        from PersistentImages_Chemistry.Element_PI import Makexyzdistance

        file_name = 'compound.xyz'
        dist_matrix, name = Makexyzdistance(file_name)
    '''

    if (generator): # Allows you to provide a generator to numpy.loadtxt (more general use)
        element = np.loadtxt(t(), dtype=str, usecols=(0,), skiprows=2)
        x = np.loadtxt(t(), dtype=float, usecols=(1), skiprows=2)
        y = np.loadtxt(t(), dtype=float, usecols=(2), skiprows=2)
        z = np.loadtxt(t(), dtype=float, usecols=(3), skiprows=2)     
    else:
        # Load in data
        element = np.loadtxt(t, dtype=str, usecols=(0,), skiprows=2)
        x = np.loadtxt(t, dtype=float, usecols=(1), skiprows=2)
        y = np.loadtxt(t, dtype=float, usecols=(2), skiprows=2)
        z = np.loadtxt(t, dtype=float, usecols=(3), skiprows=2)

    # Initialize distance matrix
    Distance = np.zeros(shape = (len(x),len(x)))

    for i in range(0, len(x)): # Fill distance matrix

       # Make an array for each atom
        for j in range(0, len(x)):

        #Calculate the distance between every atom
            Distance[i][j] = np.sqrt(  ((x[i]-x[j])**2)  + ((y[i]-y[j])**2)  \
                + ((z[i]-z[j]) **2)  )

    return [Distance, element]

__all__ = ["PersImage"]

class PersImage(TransformerMixin):
    """ Initialize a persistence image generator.

    Parameters
    -----------
    pixels: pair of ints like (int, int)
        - Tuple representing number of pixels in return image along x and y axis.
    spread: float
        - Standard deviation of gaussian kernel
    specs: dict
        - Parameters for shape of image with respect to diagram domain. 
        - Units are specified in Angstroms.
        - This is used if you would like images to have a particular range. Shaped like::
        
            {
                "maxBD": float,
                "minBD": float
            }

    kernel_type: string or ...
        - TODO: Implement this feature.
        - Determine which type of kernel used in the convolution, or pass in custom kernel. Currently only implements Gaussian.
    weighting_type: string or ...
        - TODO: Implement this feature.
        - Determine which type of weighting function used, or pass in custom weighting function.
        - Currently only implements linear weighting.

    """

    def __init__(
        self,
        pixels=(20, 20),
        spread=None,
        specs=None,
        kernel_type="gaussian",
        weighting_type="linear",
        verbose=True,
    ):

        self.specs = specs
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        self.nx, self.ny = pixels

        if verbose: # Prints parameters for user to see if verbose
            print(
                'PersImage(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                    pixels, spread, specs, kernel_type, weighting_type
                )
            )

    def transform(self, diagrams, scale = None):
        """ Convert diagram or list of diagrams to a persistence image.

        Parameters
        ----------
        diagrams: list of or singleton diagram, list of pairs. [(birth, death)]
            - Persistence diagrams to be converted to persistence images. 
            - It is assumed they are in (birth, death) format. 
            - Can input a list of diagrams or a single diagram.
        scale: float or None, optional
            - Default: None
            - Multiplies all holes with Betti > 0 to the value of scale
            - If None, has no effect

        Returns
        -------
        imgs: list or singular 
            - Persistence images converted from corresponding diagrams
        """

        # if diagram is empty, return empty image
        if len(diagrams) == 0:
            return np.zeros((self.nx, self.ny))

        # if first entry of first entry is not iterable, then diagrams is 
        #   singular and we need to make it a list of diagrams
        try:
            singular = not isinstance(diagrams[0][0], collections.Iterable)
        except IndexError:
            singular = False

        if singular: # Make diagrams into a list of diagrams
            diagrams = [diagrams]

        # Copy diagrams to avoid changing original input
        dgs = [np.copy(diagram) for diagram in diagrams]
        
        if scale is not None:
            # Vector determines which points in the PD will be scaled
            # Need a range to account for rounding errors:
            scale_vec = [(scale if (dgs[0][i,0] < 1e-7 and dgs[0][i,0] > -1e-7) else 1) for i in range(dgs[0].shape[0])]

        # Converts each diagram to langscapes
        landscapes = [PersImage.to_landscape(dg) for dg in dgs]

        if not self.specs: # Set specs for diagram if not given
            self.specs = {
                "maxBD": np.max([np.max(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
                "minBD": np.min([np.min(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
            }

        # Applies the kernel over the landscape to smooth
        if scale is not None:
            imgs = [self.__transform(dgm, scale_vec) for dgm in landscapes]
        else:
            imgs = [self.__transform(dgm) for dgm in landscapes]

        # Make sure we return one item.
        if singular:
            imgs = imgs[0]

        return imgs

    def __transform(self, landscape, scale_vec = None):
        """ Applies kernel over the landscape.

        Parameters
        ----------
        diagrams: list of or singleton diagram, list of pairs. [(birth, death)]
            - Persistence diagrams to be converted to persistence images. 
            - It is assumed they are in (birth, death) format. 
            - Can input a list of diagrams or a single diagram.
        scale_vec: list or ndarray or None, optional
            - Default: None
            - Element-wise factors to multiply each point by in the Gaussian transformation
            - If None, has no effect

        Returns
        -------
        imgs: list 
            - Persistence images converted from corresponding diagrams
        """

        # Define an NxN grid over our landscape
        maxBD = self.specs["maxBD"]
        minBD = min(self.specs["minBD"], 0)  # at least show 0, maybe lower

        # Same bins in x and y axis
        dx = maxBD / (self.ny)
        xs_lower = np.linspace(minBD, maxBD, self.nx)
        xs_upper = np.linspace(minBD, maxBD, self.nx) + dx

        ys_lower = np.linspace(0, maxBD, self.ny)
        ys_upper = np.linspace(0, maxBD, self.ny) + dx

        # Apply weighting function
        weighting = self.weighting(landscape)

        # Define zeros
        img = np.zeros((self.nx, self.ny))
        
        # Implement this as a `summed-area table` - it'll be way faster
        
        if np.size(landscape,1) == 2:
            
            spread = self.spread if self.spread else dx
            #for point in landscape:
            for i in range(len(landscape)):
                point = landscape[i]
                x_smooth = norm.cdf(xs_upper, point[0], spread) - norm.cdf(
                    xs_lower, point[0], spread
                )
                y_smooth = norm.cdf(ys_upper, point[1], spread) - norm.cdf(
                    ys_lower, point[1], spread
                )
                if scale_vec is None:
                    img += np.outer(x_smooth, y_smooth) * weighting(point)
                else: # Scales by the scaling vector
                    img += np.outer(x_smooth, y_smooth) * weighting(point) * scale_vec[i]
            img = img.T[::-1]
            return img
        else:
            spread = self.spread if self.spread else dx
            #for point in landscape:
            for i in range(len(landscape)):
                point = landscape[i]
                x_smooth = norm.cdf(xs_upper, point[0], point[2]*spread) - norm.cdf(
                    xs_lower, point[0], point[2]*spread
                )
                y_smooth = norm.cdf(ys_upper, point[1], point[2]*spread) - norm.cdf(
                    ys_lower, point[1], point[2]*spread
                )
                if scale_vec is None:
                    img += np.outer(x_smooth, y_smooth) * weighting(point)
                else:
                    #print('scale', scale_vec[i])
                    img += np.outer(x_smooth, y_smooth) * weighting(point) * (1 / scale_vec[i])
            img = img.T[::-1]
            return img

    def weighting(self, landscape=None):
        """ Define a weighting function.
            .. note:: For stability results to hold, the function must be 0 at y=0.

        Parameters
        ----------
        landscape: Numpy array
            - Converted diagram feature (see `diagram` argument in `tranform` function)
            - Note: diagram converted to landscape in to_landscape function

        Returns
        -------
        weighting_fn: function
            - The weighting function based on specifications in __init__
        """

        # TODO: Implement a logistic function
        # TODO: use self.weighting_type to choose function

        if landscape is not None:
            if len(landscape) > 0:
                maxy = np.max(landscape[:, 1])
            else: 
                maxy = 1

        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = 1
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d

        def pw_linear(interval):
            """ This is the function defined as w_b(t) in the original PI paper

                Take b to be maxy/self.ny to effectively zero out the bottom pixel row
            """

            t = interval[1]
            b = maxy / self.ny

            if t <= 0:
                return 0
            if 0 < t < b:
                return t / b
            if b <= t:
                return 1

        weighting_fn = linear

        return weighting_fn

    def kernel(self, spread=1):
        """ This will return whatever kind of kernel we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1.

        Parameters
        ----------
        spread: float, optional
            - Default: 1
            - Variance/covariance for the kernel

        Returns
        -------
        kernel_fn: function
            - Kernel function based on specification in `__init__`
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean = pixel, cov = spread)

        kernel_fn = gaussian

        return kernel_fn

    @staticmethod
    def to_landscape(diagram):
        """ Convert a diagram to a landscape (birth, death) -> (birth, death-birth)
    
        Parameters
        ----------
        diagram: list of pairs, [(birth, death)]
            - Persistence diagram to be converted to persistence image. 
            - It is assumed to be in (birth, death) format. 

        Returns
        -------
        diagram: list of pairs, [(birth, death)]
            - Converted persistence diagram with coordinates [(birth, death-birth)]
        """
        diagram[:, 1] -= diagram[:, 0]

        return diagram

    def show(self, imgs, ax=None):
        """ Visualize the persistence image.

        Parameters
        ----------
        imgs: Numpy array
            - Persistence images to show
            - Can be list of images or single image
        ax: Axes instance from `matplotlib.pyplot`, optional
            - Option to provide a plotting object for plotting PI

        Returns
        -------
        None: None
        No explicit return. Plots the PI on the given `Axes` (or a new one if not given).
        """

        ax = ax or plt.gca() # Get current axis if none is given

        # Need to convert imgs into a list if not already
        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")


def VariancePersist(Filename, generator = False, pixelx=100, pixely=100, myspread=2, 
                    myspecs={"maxBD": 2, "minBD":0}, showplot=True, max_dim = 2,
                    n_threads = 1):
    ''' Generate a persistence image given a file with coordinates of atoms.
        Includes difference in electronegativity.

    Parameters
    ----------
    Filename: string
        - Name of file with chemical data to read
    pixelx: int, optional
        - Default value = 100
        - Number of pixels on x-axis
    pixely: int, optional
        - Default value = 100
        - Number of pixels on y-axis
    myspread: int, optional
        - Default value = 2
        - Parameter for kernel
        - For Gaussian kernel, this specifies the variance
    myspecs: dictionary, optional
        - Default value = ``{"maxBD": 2, "minBD":0}``
        - Specifies boundary extent in Angstroms
        - Format::

            { 
                "maxBD": <float>,
                "minBD": <float>
            }

        - ``maxBD``: upper boundary of persistence image (in Angstroms)
        - ``minBD``: lower boundary of persistence image (in Angstroms)
    showplot: bool, optional
        - Default value = True
        - Options:
            - ``True``: plot the PI once generated
            - ``False``: do not plot the PI
    max_dim: int, optional
        - Default value = 2
        - Maximum dimension of holes to be computed by Vietoris-Rips Filtration
        - Must be >= 2
    n_threads: int, optional
        - Default value = 1
        - Number of threads to use in ripser computation (in giotto-ph)
        - -1 tries to use all available threads on machine

    Returns
    -------
    img_array: Numpy array
        - One-dimensional vector representation of a persistence image
    '''

    rips = Rips(maxdim = max_dim, verbose = False)
    
    #Generate distance matrix and elementlist
    D, elements = Makexyzdistance(Filename, generator)
   
    #Generate data for persistence diagram
    # maxdim=max_dim generates multi-dimensional holes
    a = ripser(D,distance_matrix=True, maxdim = max_dim)
    #a = ripser_parallel(D, maxdim = max_dim, metric = 'precomputed', n_threads = n_threads)
    # precomputed denotes that D is a distance matrix

    #Make the birth,death for h0 and h1
    points = (a['dgms'][0][0:-1,1])
    pointsh1 = (a['dgms'][1])
    diagrams = rips.fit_transform(D, distance_matrix=True)

    #Find pair electronegativities
    eleneg=list()

    # Iterate over connected components to assign difference in electronegativity
    for index in points:
        c = np.where(np.abs((index-a['dperm2all'])) < 1.5e-7)[0] # Decreased tolerance from original code
        #c = np.where(np.abs((index-D)) < 1.5e-7)[0]

        assert not (c.size == 0), "Cannot find matching pair"

        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
   
   
    #Original implementation of h0matrix:
    #   h0matrix = np.hstack(((diagrams[0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    h0matrix = np.hstack(((diagrams[0][0:-1,:], np.reshape((((np.array(eleneg)*1.05)+.01)/10 ), (np.size(eleneg),1)))))
    buffer = np.full((diagrams[1][:,0].size,1), 0.05)
    h1matrix = np.hstack((diagrams[1],buffer))

    # Expanding to more than 0, 1 dimensional holes:
    if max_dim >= 2:
        # If we are dealing with more than 2 dimensions
        additional_mats = [h0matrix, h1matrix]
        for i in range(2, max_dim + 1):
            buffer = np.full((diagrams[i][:,0].size,1), 0.05)
            matrix = np.hstack((diagrams[i],buffer))
            additional_mats.append(matrix)
        Totalmatrix = np.vstack(additional_mats)

    else:
        #combine them
        Totalmatrix = np.vstack((h0matrix,h1matrix))

    # CUTOFF --------------------------------------

    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def gen_Totalmatrix(Filename, generator = False, max_dim = 1, n_threads = 1):
    '''
    Splits Variance_Persist into two functions
        - It could now be defined as:
    ```
    def Variance_Persist(file_args, pi_args):
        return Variance_Persist_PD_direct( gen_Totalmatrix(file_args) , pi_args)
    ```
    where `pi_args` are the hyperparameters for building the PI
    `file_args` are the parameters for reading in the file/ building the PD

    Splitting the two up may be important if you want to test different hyperparameters 
        for your PI generation, i.e. storing the output of gen_Totalmatrix and reading
        it every time you wanted to try different hyperparameters

    Arguments:
    ----------
    Filename: string or file generator
        - If string, this should be a path to an xyz file to be read in by the
        - Note that the function does not perform parsing of conformers in the xyz file
        - Argument fed directly into Makexyzdistance
        - If file generator, should generate one xyz chunk
    generator: bool, optional
        - Default: False
        - If True, it is assumed that Filename is a file generator object (i.e. a function)
        - If False, it is assumed that Filename is a string, denoting the path to file that is to be read
    max_dim: int, optional
        - Default: 1
        - See VariancePersist function
    n_threads: int, optional
        - Default value = 1
        - Number of threads to use in ripser computation (in giotto-ph)
        - -1 tries to use all available threads on machine

    Returns:
    --------
    Totalmatrix: ndarray
        - Totalmatrix computed for this given xyz input
        - This is what should be saved if doing the TM -> PI intermediate method
    '''
    
    rips = Rips(maxdim = max_dim, verbose = False)
    
    #Generate distance matrix and elementlist
    D, elements = Makexyzdistance(Filename, generator)
   
    #Generate data for persistence diagram
    # maxdim=max_dim generates multi-dimensional holes
    #a = ripser(D,distance_matrix=True, maxdim = max_dim)
    a = ripser_parallel(D, maxdim = max_dim, metric = 'precomputed', n_threads = n_threads)

    #Make the birth,death for h0 and h1
    points = (a['dgms'][0][0:-1,1])
    pointsh1 = (a['dgms'][1])
    diagrams = rips.fit_transform(D, distance_matrix=True)

    #Find pair electronegativities
    eleneg=list()

    for index in points:
        #c = np.where(np.abs((index-a['dperm2all'])) < 1.5e-7)[0] # Decreased tolerance from original code
        c = np.where(np.abs((index-D)) < 1.5e-7)[0]

        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
   
   
    #Original implementation of h0matrix:
    #   h0matrix = np.hstack(((diagrams[0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    h0matrix = np.hstack(((diagrams[0][0:-1,:], np.reshape((((np.array(eleneg)*1.05)+.01)/10 ), (np.size(eleneg),1)))))
    buffer = np.full((diagrams[1][:,0].size,1), 0.05)
    h1matrix = np.hstack((diagrams[1],buffer))

    # Expanding to more than 0, 1 dimensional holes:
    if max_dim >= 2:
        # If we are dealing with more than 2 dimensions
        additional_mats = [h0matrix, h1matrix]
        for i in range(2, max_dim + 1):
            buffer = np.full((diagrams[i][:,0].size,1), 0.05)
            matrix = np.hstack((diagrams[i],buffer))
            additional_mats.append(matrix)
        Totalmatrix = np.vstack(additional_mats)

    else:
        #combine them
        Totalmatrix = np.vstack((h0matrix,h1matrix))

    return Totalmatrix


def VariancePersist_PD_direct(Totalmatrix, pixelx=100, pixely=100, myspread=2, 
                    myspecs={"maxBD": 2, "minBD":0}, showplot=True, scale = 0):
    '''
    Generates a persistence image directly from a generated TotalMatrix object
        - Should be computed in gen_Totalmatrix
    Arguments:
    ----------
    Totalmatrix:
    pixelx: int, optional
        - Default: 100
        - See VariancePersist function
    pixely: int, optional
        - Default: 100
        - See VariancePersist function
    myspread: float, optional
        - Default: 2
        - See VariancePersist function
    myspecs: dict, optional
        - Default: {"maxBD": 2, "minBD": 0}
        - See VariancePersist function
    showplot: bool, optional
        - Default: True
        - If True, shows a plot of the generated persistence image
    scale: float, optional
        - Default: 0
        - If scale != 0, holes with Betti > 0 are multiplied by its value
        - If scale == 0, argument has no effect

    Returns:
    --------
    pi_vec: ndarray of size (pixelx * pixely,)
        - 1D vector representation of a persistence image
    '''

    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    if scale != 0:
        imgs = pim.transform(Totalmatrix, scale)
    else:
        imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())    


def PersDiagram(xyz, lifetime=True, showplot = True, generator = False, make_plt = True):
    ''' Creates a visual representation for a persistence diagram

    Parameters
    ----------
    xyz: string or generator
        - Name for local file containing data on coordinates representing atoms in compound
        - If generator, must set argument ``generator = True``
    lifetime: bool, optional
        - Option to set the y-axis to lifetime value
        - Options:
            - ``True``: set coordinates to (birth, death - birth)
            - ``False``: set coordinates to (birth, death)
    showplot: bool, optional
        - Option to output PD plot automatically to screen or not
        - Options:
            - ``True``: show plot
            - ``False``: do not show plot
    generator: bool, optional
        - Default: False
        - Can provide file generator instead of a filename to the xyz argument
        - Options:
            - ``True``: xyz is a generator
            - ``False``: xyz is a filename

    Returns
    -------
    rips: `Rips` object from the ripser module
        - See `ripser documentation <https://ripser.scikit-tda.org/reference/stubs/ripser.Rips.html#>`_ for this return value.
        - This object has the data specified in `xyz` fit to it.

    .. note:: If ``showplot = True``, then a plot of the PD will be output to the screen.
    '''

    if (make_plt):
        plt.rcParams["font.family"] = "Times New Roman"

    D,elements = Makexyzdistance(xyz, generator = generator)
    data = ripser(D,distance_matrix=True)

    # Perform plotting with Rips() object
    rips = Rips(verbose = False)             # Create Rips object
    rips.transform(D, distance_matrix=True)
    rips.dgms_[0] = rips.dgms_[0][0:-1]

    if (make_plt):
        rips.plot(show = showplot, lifetime=lifetime, labels=['Connected Components','Holes'])

        L = plt.legend()
        plt.setp(L.texts, family="Times New Roman")
        plt.rcParams["font.family"] = "Times New Roman"

    # Return the Rips object fitted with our data.
    return rips

def GeneratePI(xyz, savefile=False, pixelx=100, pixely=100, myspread=2, bounds={"maxBD": 3, "minBD":-0.1}):
    ''' Outputs a visual representation of a persistence image based on file given.

    Parameters
    ----------
    xyz: string
        - Name for local file containing data on coordinates representing atoms in compound
    savefile: bool, optional
        - Default value = False
        - Options:
            - `True` = plot of PI is saved
            - `False` = plot is not saved
        - Saves file to: `<xyz>_img.png`
    pixelx: int, optional
        - Default value = 100
        - Number of pixels on x-axis
    pixely: int, optional
        - Default value = 100
        - Number of pixels on y-axis
    myspread: float, optional
        - Default value = 2
        - Parameter for kernel
        - For Gaussian kernel, this specifies the variance
    bounds: dictionary, optional
        - Default value = ``{"maxBD": 2, "minBD":0}``
        - Specifies boundary extent in Angstroms
        - Format::

            { 
                "maxBD": <float>,
                "minBD": <float>
            }

        - ``maxBD``: upper boundary of persistence image (in Angstroms)
        - ``minBD``: lower boundary of persistence image (in Angstroms)

    Returns
    -------
    None: none
    No explicit return value. Outputs the plot of the PI to the screen.
    '''
    X = VariancePersist(xyz, pixelx = pixelx, pixely = pixely, myspread = myspread, myspecs = bounds, showplot=True)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=False)

    img = X.reshape(pixelx,pixely)
    pim.show(img)

    if savefile==True: # Save file to specified output
        plt.imsave(xyz+'_img.png',img, cmap=plt.get_cmap("plasma"), dpi=200)

