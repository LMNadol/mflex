#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import numpy as np
import math
from astropy.io.fits import open, getdata
from mflex.plot.plot_magnetogram import plot_magnetogram_boundary
from mflex.classes.clsmod import Data3D, DataBz

# TO DO
# Need to split def prep_ISSI_data from get_magnetogram or sth like that
# Need to decide on general format we want data to be given, probably as
# Bx = np.array(ny, nx) and By = np.array(ny, nx) at z = 0 (Photosphere)
# Additionally need pixelsize in km


def read_issi_rmhd(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from RMHD_boundary_data.sav
    provided by ISSI Team.
    """

    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = data[
        "b2dz"
    ]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx: np.ndarray[np.float64, np.dtype[np.float64]] = data[
        "b2dx"
    ]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by: np.ndarray[np.float64, np.dtype[np.float64]] = data[
        "b2dy"
    ]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    # print(data["info_unit"])
    # print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"])

    pixelsize: np.float64 = np.float64(input("Pixelsize in km?"))

    # bx_xlen: np.int16 = data_bx.shape[1]
    # bx_ylen: np.int16 = data_bx.shape[0]
    # by_xlen: np.int16 = data_by.shape[1]
    # by_ylen: np.int16 = data_bx.shape[0]
    bz_xlen: int = data_bz.shape[1]
    bz_ylen: int = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x: int = bz_xlen
    nresol_y: int = bz_ylen
    L: np.float64 = np.float64(1.0)

    xmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of x in data length scale, not in Mm
    ymin: np.float64 = np.float64(
        0.0
    )  # Minimum value of y in data length scale, not in Mm
    zmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of z in data length scale, not in Mm
    xmax: np.float64 = np.float64(
        0.0
    )  # Minimum value of x in data length scale, not in Mm
    ymax: np.float64 = np.float64(
        0.0
    )  # Minimum value of y in data length scale, not in Mm
    zmax: np.float64 = np.float64(0.0)

    if nresol_x < nresol_y:
        xmax: np.float64 = L  # Maximum value of x in data length scale, not in Mm
        ymax: np.float64 = np.float64(
            nresol_y / nresol_x
        )  # Maximum value of y in data length scale, not in Mm
    if nresol_y < nresol_x:
        ymax: np.float64 = L
        xmax: np.float64 = np.float64(nresol_x / nresol_y)
    if nresol_y == nresol_x:
        xmax: np.float64 = L
        ymax: np.float64 = L

    pixelsize_x: np.float64 = (
        abs(xmax - xmin) / nresol_x
    )  # Data pixel size in x direction
    pixelsize_y: np.float64 = (
        abs(ymax - ymin) / nresol_y
    )  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        print("directional pixel sizes of data do not match")
        raise ValueError

    nresol_z: int = math.floor(
        10000.0 / pixelsize
    )  # Artifical upper boundary at 10Mm outside of corona
    z0_index: int = math.floor(2000.0 / pixelsize)  # Height of Transition Region at 2Mm

    if xmax == L:
        zmax: np.float64 = np.float64(nresol_z / nresol_x)
        z0: np.float64 = np.float64(z0_index / nresol_x)
    if ymax == L:
        zmax: np.float64 = np.float64(nresol_z / nresol_y)
        z0: np.float64 = np.float64(z0_index / nresol_y)

    pixelsize_z: np.float64 = (
        abs(zmax - zmin) / nresol_z
    )  # Data pixel size in z direction

    if pixelsize_z != pixelsize_x:
        print("nresol_z and zmax do not match")
        raise ValueError

    nf_max: int = min(nresol_x, nresol_y)

    data3d: Data3D = Data3D(
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

    return data3d


def read_issi_analytical(path: str) -> Data3D:
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

    # print(data["info_unit"])
    print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"])

    pixelsize: np.float64 = np.float64(input("Pixelsize in km?"))

    # bx_xlen: np.int16 = data_bx.shape[1]
    # bx_ylen: np.int16 = data_bx.shape[0]
    # by_xlen: np.int16 = data_by.shape[1]
    # by_ylen: np.int16 = data_bx.shape[0]
    bz_xlen: int = data_bz.shape[1]
    bz_ylen: int = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x: int = bz_xlen
    nresol_y: int = bz_ylen
    L: np.float64 = np.float64(1.0)

    xmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of x in data length scale, not in Mm
    ymin: np.float64 = np.float64(
        0.0
    )  # Minimum value of y in data length scale, not in Mm
    zmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of z in data length scale, not in Mm
    xmax: np.float64 = np.float64(
        0.0
    )  # Minimum value of x in data length scale, not in Mm
    ymax: np.float64 = np.float64(
        0.0
    )  # Minimum value of y in data length scale, not in Mm
    zmax: np.float64 = np.float64(0.0)

    if nresol_x < nresol_y:
        xmax: np.float64 = L  # Maximum value of x in data length scale, not in Mm
        ymax: np.float64 = np.float64(
            nresol_y / nresol_x
        )  # Maximum value of y in data length scale, not in Mm
    if nresol_y < nresol_x:
        ymax: np.float64 = L
        xmax: np.float64 = np.float64(nresol_x / nresol_y)
    if nresol_y == nresol_x:
        xmax: np.float64 = L
        ymax: np.float64 = L

    pixelsize_x: np.float64 = (
        abs(xmax - xmin) / nresol_x
    )  # Data pixel size in x direction
    pixelsize_y: np.float64 = (
        abs(ymax - ymin) / nresol_y
    )  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        print("directional pixel sizes of data do not match")
        raise ValueError

    nresol_z: int = math.floor(
        10000.0 / pixelsize
    )  # Artifical upper boundary at 10Mm outside of corona
    z0_index: int = math.floor(2000.0 / pixelsize)  # Height of Transition Region at 2Mm

    if xmax == L:
        zmax: np.float64 = np.float64(nresol_z / nresol_x)
        z0: np.float64 = np.float64(z0_index / nresol_x)
    if ymax == L:
        zmax: np.float64 = np.float64(nresol_z / nresol_y)
        z0: np.float64 = np.float64(z0_index / nresol_y)

    pixelsize_z: np.float64 = (
        abs(zmax - zmin) / nresol_z
    )  # Data pixel size in z direction

    if pixelsize_z != pixelsize_x:
        print("nresol_z and zmax do not match")
        raise ValueError

    nf_max: int = min(nresol_x, nresol_y)

    data3d: Data3D = Data3D(
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

    return data3d


def read_fits_soar(path: str, header: bool = False) -> DataBz:
    """
    Returns dataclass DataBz extracted from _blos.fits file
    previously downloaded from Solar Orbiter Archive.
    """

    with open(path) as data:
        # data.info()
        image: np.ndarray[np.float64, np.dtype[np.int64]] = getdata(path, ext=False)
        x_len: int = image.shape[0]
        y_len: int = image.shape[1]
        plot_magnetogram_boundary(image, x_len, y_len)
        x_start: int = int(input("First pixel x axis: "))
        x_last: int = int(input("Last pixel x axis: "))
        y_start: int = int(input("First pixel y axis: "))
        y_last: int = int(input("Last pixel y axis: "))
        x_start: int = 400
        x_last: int = 1200
        y_start: int = 500
        y_last: int = 1000
        cut_image: np.ndarray[np.float64, np.dtype[np.float64]] = image[
            y_start:y_last, x_start:x_last
        ]
        # plot_magnetogram_boundary(cut_image, x_last - x_start, y_last - y_start)
        # if header == True:
        #    with open(
        #        "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
        #        "w",
        #    ) as f:
        #        for d in data:
        #            f.write(repr(d.header))
        #    print("File header has been printed to Desktop/SOAR/obs")
        hdr = data[0].header  # the primary HDU header
        dist = hdr["DSUN_OBS"]
        pixelsize_x_unit: str = hdr["CUNIT1"]
        pixelsize_y_unit: str = hdr["CUNIT2"]
        pixelsize_x_arcsec: np.float64 = hdr["CDELT1"]
        pixelsize_y_arcsec: np.float64 = hdr["CDELT2"]

        if not pixelsize_x_unit == pixelsize_y_unit:
            print("Pixelsize units not matchy-matchy")
            raise ValueError
        if not pixelsize_x_arcsec == pixelsize_y_arcsec:
            print("Data pixelsizes in x and y direction not matchy-matchy")
            raise ValueError
        else:
            pixelsize_radians: np.float64 = pixelsize_x_arcsec / 206265.0

    dist_km: np.float64 = np.float64(dist / 1000.0)
    pixelsize_km: np.float64 = np.floor(pixelsize_radians * dist_km)

    nresol_x: int = cut_image.shape[1]
    nresol_y: int = cut_image.shape[0]
    length_scale: np.float64 = np.float64(1.0)  # L

    xmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of x in data length scale, not in Mm
    ymin: np.float64 = np.float64(
        0.0
    )  # Minimum value of y in data length scale, not in Mm
    zmin: np.float64 = np.float64(
        0.0
    )  # Minimum value of z in data length scale, not in Mm

    if nresol_x < nresol_y:
        xmax: np.float64 = (
            length_scale  # Maximum value of x in data length scale, not in Mm
        )
        ymax: np.float64 = np.float64(
            nresol_y / nresol_x
        )  # Maximum value of y in data length scale, not in Mm
    if nresol_y < nresol_x:
        ymax: np.float64 = length_scale
        xmax: np.float64 = np.float64(nresol_x / nresol_y)
    if nresol_y == nresol_x:
        xmax: np.float64 = length_scale
        ymax: np.float64 = length_scale

    pixelsize_x: np.float64 = (
        abs(xmax - xmin) / nresol_x
    )  # Data pixel size in x direction in relation to xmin and xmax
    pixelsize_y: np.float64 = (
        abs(ymax - ymin) / nresol_y
    )  # Data pixel size in y direction in relation to ymin and ymax

    if pixelsize_x != pixelsize_y:
        print("Directional pixel sizes of data do not match")
        raise ValueError

    nresol_z: int = math.floor(
        10000.0 / (pixelsize_km)
    )  # Artifical upper boundary at 10Mm from photosphere
    z0_index: int = math.floor(
        2000.0 / (pixelsize_km)
    )  # Centre of region over which transition from NFF to FF takes place at 2Mm from photosphere

    if xmax == length_scale:
        zmax: np.float64 = np.float64(nresol_z / nresol_x)
        z0: np.float64 = np.float64(z0_index / nresol_x)
    if ymax == length_scale:
        zmax: np.float64 = np.float64(nresol_z / nresol_y)
        z0: np.float64 = np.float64(z0_index / nresol_y)

    pixelsize_z: np.float64 = (
        abs(zmax - zmin) / nresol_z
    )  # Data pixel size in z direction in relation to zmin and zmax

    # Calulate parameters in Mm for checking purposes

    # xmax_Mm = nresol_x * pixelsize_km / 1000.0
    # ymax_Mm = nresol_y * pixelsize_km / 1000.0
    # zmax_Mm = nresol_z * pixelsize_km / 1000.0

    # print("Mm", xmax_Mm, ymax_Mm, zmax_Mm)

    # if xmax == length_scale:
    # ratio_Mm_xy = ymax_Mm / xmax_Mm
    # ratio_Mm_xz = zmax_Mm / xmax_Mm
    # if ymax == length_scale:
    # ratio_Mm_xy = xmax_Mm / ymax_Mm
    # ratio_Mm_xz = zmax_Mm / ymax_Mm

    # print("ratio Mm", ratio_Mm_xy, ratio_Mm_xz)
    # pixelsize_z = pixelsize_x * pixelsize_z_km / pixelsize_km
    # if pixelsize_z != pixelsize_x:
    #    print("nresol_z and zmax do not match")
    #    raise ValueError

    nf_max: int = min(nresol_x, nresol_y)

    databz: DataBz = DataBz(
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
