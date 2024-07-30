from dataclasses import dataclass
import numpy as np

@dataclass
class Field3dData:
    nx: np.int32
    ny: np.int32
    nz: np.int32
    px: np.float64
    py: np.float64
    pz: np.float64
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray

    @classmethod
    def from_fits(cls, path, a, b, alpha, z0, deltaz):
        # do some stuff here to get the class parameters

        return Field3dData(......)
    
    @classmethod
    def from_array(cls, arr):
        # stuff here

        return Field3dData(.....)
    

"""

data = Field3dData.from_fits("path/to/fits/file)

data = Field3dData.from_array(array)



"""