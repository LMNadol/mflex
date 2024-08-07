from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from astropy.io.fits import open as astroopen
from astropy.io.fits import getdata
from astropy.coordinates import SkyCoord
from astropy import units as u

import sunpy.map


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

    @classmethod
    def from_fits_SolOr(cls, path):

        with astroopen(path) as data:

            image = getdata(path, ext=False)

            hdr = data[0].header
            dist = hdr["DSUN_OBS"]
            px_unit = hdr["CUNIT1"]
            py_unit = hdr["CUNIT2"]
            px_arcsec = hdr["CDELT1"]
            py_arcsec = hdr["CDELT2"]

        stx = 400  # int(input("First pixel x axis: "))
        lstx = 1200  # int(input("Last pixel x axis: "))
        sty = 500  # int(input("First pixel y axis: "))
        lsty = 1000  # int(input("Last pixel y axis: "))

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

    @classmethod
    def from_fits_SDO(cls, path):

        hmi_image = sunpy.map.Map(path).rotate()

        hdr = hmi_image.fits_header

        sty = int(input("Lower boundary latitute: "))
        lsty = int(input("Upper boundary latitute:  "))
        stx = int(input("Lower boundary longitude: "))
        lstx = int(input("Upper boundary longitude: "))

        left_corner = SkyCoord(
            Tx=lsty * u.arcsec, Ty=sty * u.arcsec, frame=hmi_image.coordinate_frame
        )
        right_corner = SkyCoord(
            Tx=lstx * u.arcsec, Ty=stx * u.arcsec, frame=hmi_image.coordinate_frame
        )

        image = hmi_image.submap(left_corner, top_right=right_corner)

        dist = hdr["DSUN_OBS"]
        px_unit = hdr["CUNIT1"]
        py_unit = hdr["CUNIT2"]
        px_arcsec = hdr["CDELT1"]
        py_arcsec = hdr["CDELT2"]

        nx = image.data.shape[1]
        ny = image.data.shape[0]

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

        return Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, image.data)
