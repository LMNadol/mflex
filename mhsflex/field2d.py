from dataclasses import dataclass
import numpy as np

from astropy.io.fits import open as astroopen
from astropy.io.fits import getdata


@dataclass
class Field2dData:
    nx: np.int32
    ny: np.int32
    nz: np.int32
    nf: np.int32
    px: np.float64
    py: np.float64
    pz: np.float64
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray

    xmin, xmax, ymin, ymax, zmin, zmax = (
        x[0],
        x[-1],
        y[0],
        y[-1],
        z[0],
        z[-1],
    )

    @classmethod
    def from_fits(cls, path):

        with astroopen(path) as data:

            image = getdata(path, ext=False)
            hdr = data[0].header
            dist = hdr["DSUN_OBS"]
            px_unit = hdr["CUNIT1"]
            py_unit = hdr["CUNIT2"]
            px_arcsec = hdr["CDELT1"]
            py_arcsec = hdr["CDELT2"]

        stx = int(input("First pixel x axis: "))
        lstx = int(input("Last pixel x axis: "))
        sty = int(input("First pixel y axis: "))
        lsty = int(input("Last pixel y axis: "))

        image = image[sty:lsty, stx:lstx]

        nx = image.shape[1]
        ny = image.shape[0]

        nf = min(nx, ny)

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        pz = np.float64(90.0 * 10**-3)

        nz = np.int32(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        return Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, image)
