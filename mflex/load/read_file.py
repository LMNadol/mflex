from typing import Tuple, Dict
import scipy
import numpy as np
import math
from astropy.io.fits import open, getdata
from mflex.plot.plot_magnetogram import plot_magnetogram_boundary
from mflex.classes.clsmod import Data3D, DataBz


def read_issi_rmhd(path: str, L: np.float64) -> Data3D:
    """
    Returns dataclass Data3D extracted from Analytical_boundary_data.sav
    provided by ISSI Team.
    """
    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz = data["b2dz"]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx"]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy"]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    print(data["info_array"])

    nresol_x = data_bz.shape[1]
    nresol_y = data_bz.shape[0]

    pixelsize_y_Mm = 192.0 * 10**-3  # Pixelsize is given in km, so converted to Mm
    pixelsize_x_Mm = 192.0 * 10**-3
    pixelsize_z_Mm = 64.0 * 10**-3

    xmin = np.float64(0.0)  # Minimum value of x in data length scale, not in Mm
    ymin = np.float64(0.0)  # Minimum value of y in data length scale, not in Mm
    zmin = np.float64(0.0)  # Minimum value of z in data length scale, not in Mm

    xmax_Mm = nresol_x * pixelsize_x_Mm  # xmax in Mm
    ymax_Mm = nresol_y * pixelsize_y_Mm
    zmax_Mm = 41.6  # given zmax in Mm (provided by ISSI Team)

    nresol_z = int(np.floor(zmax_Mm / pixelsize_z_Mm))

    z0 = 2000.0 * 10**-3  # z0 at 2Mm

    nf_max = min(nresol_x, nresol_y)

    xmax = xmax_Mm / L  # Normalising Mm into length scale L
    ymax = ymax_Mm / L
    zmax = zmax_Mm / L
    z0 = z0 / L

    pixelsize_x = pixelsize_x_Mm / L  # Normalising Mm into length scale L
    pixelsize_y = pixelsize_y_Mm / L
    pixelsize_z = pixelsize_z_Mm / L

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_Mm, ymax_Mm, zmax_Mm", xmax_Mm, ymax_Mm, zmax_Mm)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_Mm", pixelsize_x, pixelsize_x_Mm)
    print("pixelsize_y, pixelsize_y_Mm", pixelsize_y, pixelsize_y_Mm)
    print("pixelsize_z, pixelsize_z_Mm", pixelsize_z, pixelsize_z_Mm)
    print("z0", z0)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_issi_analytical(path: str, L: np.float64) -> Data3D:
    """
    Returns dataclass Data3D extracted from Analytical_boundary_data.sav
    provided by ISSI Team.
    """
    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz = data["b2dz5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # scale_height = data['h3d']

    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    print(data["info_boundary"])
    print(data["info_array"])

    nresol_x = data_bz.shape[1]
    nresol_y = data_bz.shape[0]

    pixelsize_z_Mm = 40.0 * 10**-3  # Convert pixelsize from km into Mm
    pixelsize_x_Mm = 40.0 * 10**-3
    pixelsize_y_Mm = 40.0 * 10**-3

    xmin = np.float64(0.0)  # Minimum value of x in data length scale, not in Mm
    ymin = np.float64(0.0)  # Minimum value of y in data length scale, not in Mm
    zmin = np.float64(0.0)  # Minimum value of z in data length scale, not in Mm

    xmax_Mm = nresol_x * pixelsize_x_Mm
    ymax_Mm = nresol_y * pixelsize_y_Mm
    zmax_Mm = 2000.0 * 10**-3

    nresol_z = int(np.floor(zmax_Mm / pixelsize_z_Mm))

    z0 = 2000.0 * 10**-3

    nf_max = min(nresol_x, nresol_y)

    xmax = xmax_Mm / L  # Convert from Mm into length scale
    ymax = ymax_Mm / L
    zmax = zmax_Mm / L
    z0 = z0 / L

    pixelsize_x = pixelsize_x_Mm / L
    pixelsize_y = pixelsize_y_Mm / L
    pixelsize_z = pixelsize_z_Mm / L

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_Mm, ymax_Mm, zmax_Mm", xmax_Mm, ymax_Mm, zmax_Mm)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_Mm", pixelsize_x, pixelsize_x_Mm)
    print("pixelsize_y, pixelsize_y_Mm", pixelsize_y, pixelsize_y_Mm)
    print("pixelsize_z, pixelsize_z_Mm", pixelsize_z, pixelsize_z_Mm)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_fits_soar(
    path: str, L: np.float64, header: bool = False, cmap: str = "bone"
) -> DataBz:
    """
    Returns dataclass DataBz extracted from _blos.fits file
    previously downloaded from Solar Orbiter Archive.
    """

    with open(path) as data:
        # data.info()
        image = getdata(path, ext=False)
        x_len = image.shape[0]
        y_len = image.shape[1]
        plot_magnetogram_boundary(image, x_len, y_len, cmap=cmap)

        # x_start = int(input("First pixel x axis: "))
        # x_last = int(input("Last pixel x axis: "))
        # y_start = int(input("First pixel y axis: "))
        # y_last = int(input("Last pixel y axis: "))
        x_start = 400
        x_last = 1200
        y_start = 500
        y_last = 1000
        cut_image = image[y_start:y_last, x_start:x_last]  # type: ignore
        # plot_magnetogram_boundary(cut_image, x_last - x_start, y_last - y_start)
        # if header == True:
        #    with open(
        #        "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
        #        "w",
        #    ) as f:
        #        for d in data:
        #            f.write(repr(d.header))
        #    print("File header has been printed to Desktop/SOAR/obs")
        hdr = data[0].header  # type: ignore
        # the primary HDU header
        dist = hdr["DSUN_OBS"]
        pixelsize_x_unit = hdr["CUNIT1"]
        pixelsize_y_unit = hdr["CUNIT2"]
        pixelsize_x_arcsec = hdr["CDELT1"]
        pixelsize_y_arcsec = hdr["CDELT2"]

        if not pixelsize_x_unit == pixelsize_y_unit:
            print("Pixelsize units not matchy-matchy")
            raise ValueError
        if not pixelsize_x_arcsec == pixelsize_y_arcsec:
            print("Data pixelsizes in x and y direction not matchy-matchy")
            raise ValueError
        else:
            pixelsize_radians = pixelsize_x_arcsec / 206265.0

    dist_km = dist / 1000.0
    pixelsize_Mm = np.floor(pixelsize_radians * dist_km) * 10**-3

    nresol_x = cut_image.shape[1]
    nresol_y = cut_image.shape[0]

    xmax_Mm = nresol_x * pixelsize_Mm
    ymax_Mm = nresol_y * pixelsize_Mm
    pixelsize_z_Mm = 90.0 * 10**-3

    zmax_Mm = 20000.0 * 10**-3

    xmin = np.float64(0.0)  # Minimum value of x in data length scale, not in Mm
    ymin = np.float64(0.0)  # Minimum value of y in data length scale, not in Mm
    zmin = np.float64(0.0)  # Minimum value of z in data length scale, not in Mm

    nresol_z = int(np.floor(zmax_Mm / pixelsize_z_Mm))

    z0 = 2000.0 * 10**-3

    nf_max = min(nresol_x, nresol_y)

    xmax = xmax_Mm / L
    ymax = ymax_Mm / L
    zmax = zmax_Mm / L
    z0 = z0 / L

    pixelsize_x = pixelsize_Mm / L
    pixelsize_y = pixelsize_Mm / L
    pixelsize_z = pixelsize_Mm / L
    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_Mm, ymax_Mm, zmax_Mm", xmax_Mm, ymax_Mm, zmax_Mm)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_Mm", pixelsize_x, pixelsize_Mm)
    print("pixelsize_y, pixelsize_y_Mm", pixelsize_y, pixelsize_Mm)
    print("pixelsize_z, pixelsize_z_Mm", pixelsize_z, pixelsize_Mm)

    databz = DataBz(
        cut_image,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )

    return databz
