from typing import Tuple
import numpy as np
import math
from mflex.plot.linetracer.fieldline3D import fieldline3d
from scipy.stats import pearsonr


def vec_corr_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Vector Correlation metric of B : B_ref and b : B_rec.
    """

    return np.sum(np.multiply(B, b)) / (
        np.sqrt(np.sum(np.multiply(B, B)) * np.sum(np.multiply(b, b)))
    )


def cau_Schw_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Cauchy Schwarz metric of B : B_ref and b : B_rec.
    """

    N = np.size(B)
    num = np.multiply(B, b)
    div = np.reciprocal(np.multiply(abs(B), abs(b)))
    return np.sum(np.multiply(num, div)) / N


def norm_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Normalised Vector Error metric of B : B_ref and b : B_rec.
    """

    return np.sum(abs(np.subtract(B, b))) / np.sum(np.abs(B))


def mean_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Mean Vector Error metric of B : B_ref and b : B_rec.
    """

    N = np.size(B)
    num = abs(np.subtract(B, b))
    div = abs(np.reciprocal(B))

    return np.sum(np.multiply(num, div)) / N


def mag_ener_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Magnetic Energy metric of B : B_ref and b : B_rec.
    """

    Bx = B[:, :, :, 1][0, 0]
    By = B[:, :, :, 0][0, 0]
    Bz = B[:, :, :, 2][0, 0]
    bx = b[:, :, :, 1][0, 0]
    by = b[:, :, :, 0][0, 0]
    bz = b[:, :, :, 2][0, 0]

    num = np.sqrt(np.dot(bx, bx) + np.dot(by, by) + np.dot(bz, bz))
    div = np.sqrt(np.dot(Bx, Bx) + np.dot(By, By) + np.dot(Bz, Bz))

    return num / div


def field_div_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
    h1: float,
    hmin: float,
    hmax: float,
    eps: float,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    stepsize,
) -> np.float64:
    """
    Returns Field Line Divergence metric of B : B_ref and b : B_rec.
    xmin, ymin, ymax as when using plot_fieldline_grid.
    """
    x_arr = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    y_arr = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = 0.0
    boxedges[1, 0] = ymax
    boxedges[0, 1] = 0.0
    boxedges[1, 1] = xmax
    boxedges[0, 2] = 0.0
    boxedges[1, 2] = zmax

    h1_ref = h1
    h1_rec = h1

    count = 0
    count_closed = 0
    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if B[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref

            if b[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline_ref = fieldline3d(
                ystart,
                B,
                y_arr,
                x_arr,
                z_arr,
                h1_ref,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            fieldline_rec = fieldline3d(
                ystart,
                b,
                y_arr,
                x_arr,
                z_arr,
                h1_rec,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )

            len_ref = len(fieldline_ref)
            len_rec = len(fieldline_rec)
            fieldline_x_ref = np.zeros(len_ref)
            fieldline_y_ref = np.zeros(len_ref)
            fieldline_z_ref = np.zeros(len_ref)

            fieldline_x_rec = np.zeros(len_rec)
            fieldline_y_rec = np.zeros(len_rec)
            fieldline_z_rec = np.zeros(len_rec)

            fieldline_x_ref = fieldline_ref[:, 0]
            fieldline_y_ref = fieldline_ref[:, 1]
            fieldline_z_ref = fieldline_ref[:, 2]

            fieldline_x_rec = fieldline_rec[:, 0]
            fieldline_y_rec = fieldline_rec[:, 1]
            fieldline_z_rec = fieldline_rec[:, 2]

            if (0.0 <= fieldline_x_ref[len_ref - 1]) and (
                fieldline_x_ref[len_ref - 1] <= xmax
            ):
                if (0.0 <= fieldline_y_ref[len_ref - 1]) and (
                    fieldline_y_ref[len_ref - 1] <= ymax
                ):
                    if fieldline_z_ref[len_ref - 1] <= 0.00001:
                        if (0.0 <= fieldline_x_rec[len_rec - 1]) and (
                            fieldline_x_rec[len_rec - 1] <= xmax
                        ):
                            if (0.0 <= fieldline_y_rec[len_rec - 1]) and (
                                fieldline_y_rec[len_rec - 1] <= ymax
                            ):
                                if fieldline_z_rec[len_rec - 1] <= 0.00001:
                                    count_closed = count_closed + 1
                                    num = np.sqrt(
                                        (
                                            fieldline_x_rec[len_rec - 1]
                                            - fieldline_x_ref[len_ref - 1]
                                        )
                                        ** 2.0
                                        + (
                                            fieldline_y_rec[len_rec - 1]
                                            - fieldline_y_ref[len_ref - 1]
                                        )
                                        ** 2.0
                                        + (
                                            fieldline_z_rec[len_rec - 1]
                                            - fieldline_z_ref[len_ref - 1]
                                        )
                                        ** 2.0
                                    )

                                    div = 0.0
                                    for i in range(0, len_ref - 1):
                                        div = div + np.sqrt(
                                            (
                                                fieldline_x_ref[i]
                                                - fieldline_x_ref[i + 1]
                                            )
                                            ** 2.0
                                            + (
                                                fieldline_y_ref[i]
                                                - fieldline_y_ref[i + 1]
                                            )
                                            ** 2.0
                                            + (
                                                fieldline_z_ref[i]
                                                - fieldline_z_ref[i + 1]
                                            )
                                            ** 2.0
                                        )

                                    temp = num / div
                                    if temp <= 0.1:
                                        count = count + 1

    return count / (nlinesmaxx * nlinesmaxy)


def pearson_corr_coeff(
    pres_3d_ref: np.ndarray[np.float64, np.dtype[np.float64]],
    den_3d_ref: np.ndarray[np.float64, np.dtype[np.float64]],
    pres_3d_rec: np.ndarray[np.float64, np.dtype[np.float64]],
    den_3d_rec: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    zmin: np.float64,
    zmax: np.float64,
) -> Tuple[np.ndarray[np.float64, np.dtype[np.float64]], ...]:

    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    pres_surface_ref = np.zeros((nresol_y, nresol_x))
    den_surface_ref = np.zeros((nresol_y, nresol_x))
    pres_surface_rec = np.zeros((nresol_y, nresol_x))
    den_surface_rec = np.zeros((nresol_y, nresol_x))

    for ix in range(nresol_x):
        for iy in range(nresol_y):
            pres_surface_ref[iy, ix] = np.trapz(pres_3d_ref[:, iy, ix], z_arr)
            den_surface_ref[iy, ix] = np.trapz(den_3d_ref[:, iy, ix], z_arr)
            pres_surface_rec[iy, ix] = np.trapz(pres_3d_rec[:, iy, ix], z_arr)
            den_surface_rec[iy, ix] = np.trapz(den_3d_rec[:, iy, ix], z_arr)

    print(
        "Pearson Ref Pres",
        pearsonr(pres_surface_ref.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Ref Den",
        pearsonr(den_surface_ref.flatten(), den_surface_ref.flatten()),
    )
    print(
        "Pearson Pres", pearsonr(pres_surface_rec.flatten(), pres_surface_ref.flatten())
    )
    print("Pearson Den", pearsonr(den_surface_rec.flatten(), den_surface_ref.flatten()))

    return pres_surface_ref, den_surface_ref, pres_surface_rec, den_surface_rec
