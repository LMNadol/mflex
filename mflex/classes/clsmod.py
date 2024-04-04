import numpy as np
from dataclasses import dataclass


@dataclass
class Data3D:
    data_x: np.ndarray[np.float64, np.dtype[np.float64]]
    data_y: np.ndarray[np.float64, np.dtype[np.float64]]
    data_z: np.ndarray[np.float64, np.dtype[np.float64]]
    nresol_x: int
    nresol_y: int
    nresol_z: int
    pixelsize_x: np.float64
    pixelsize_y: np.float64
    pixelsize_z: np.float64
    nf_max: int
    xmin: np.float64
    xmax: np.float64
    ymin: np.float64
    ymax: np.float64
    zmin: np.float64
    zmax: np.float64
    z0: np.float64


@dataclass
class DataBz:
    data_z: np.ndarray[np.float64, np.dtype[np.float64]]
    nresol_x: int
    nresol_y: int
    nresol_z: int
    pixelsize_x: np.float64
    pixelsize_y: np.float64
    pixelsize_z: np.float64
    nf_max: int
    xmin: np.float64
    xmax: np.float64
    ymin: np.float64
    ymax: np.float64
    zmin: np.float64
    zmax: np.float64
    z0: np.float64
