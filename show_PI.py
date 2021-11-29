import sys
import numpy as np
import matplotlib.pyplot as plt
from ChemTDA import VariancePersist

assert len(sys.argv) == 2, 'usage: python3 show_PI.py <xyz path>'


PI = VariancePersist(
    Filename = sys.argv[1],
    pixelx = 10,
    pixely = 10,
    showplot=False)

PI = PI.reshape((10, 10))
plt.imshow(np.sqrt(PI))
plt.show()
