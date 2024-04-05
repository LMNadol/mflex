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

    Tracing field lines from footpoints on a grid on the bottom boundary in both reference model
    and reconstruction. If both field lines end again on the bottom boundary, a score is assigned:
    the distance between the two endpoints divided by the length of the field line in the reference model.
    Then the metric score can be obtained by the fraction of the initial footpoints in which these scores
    are less than 10%.
    """
    x_arr = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    y_arr = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    # First startpoint close to origin
    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8

    # Grid stepping size for footpoints and number of overall footpoints in x- and y-direction
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

    # set fieldline3D initial stepsize
    h1_ref = h1
    h1_rec = h1

    # counter for number of footpoints that have an error in the endpoints smaller than 10 percent of field line length in reference model
    count = 0

    # counter for number of field lines that are closed within box
    count_closed = 0

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):

            # Footpoint
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy
            ystart = [y_start, x_start, 0.0]

            # decide direction of fieldline for reference model
            if B[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_ref = -h1_ref
            # decide direction of fieldline for reconstruction model
            if b[int(y_start), int(x_start), 0, 2] < 0.0:
                h1_rec = -h1_rec

            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X

            # Get fieldline coordinates for reference model
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

            # Get fieldline coordinates for reconstruction model
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

            valid_fieldline = True

            # Check if field lines end on bottom boundary
            if not np.isclose(fieldline_ref[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not np.isclose(fieldline_rec[len_ref - 1, 2], 0.0):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_ref - 1, 1] <= xmax):
                valid_fieldline = False
            if not (0.0 <= fieldline_ref[len_ref - 1, 0] <= ymax):
                valid_fieldline = False
            if not (0.0 <= fieldline_rec[len_ref - 1, 0] <= ymax):
                valid_fieldline = False

            if valid_fieldline:

                # add to counter if field line is closed field line within box
                count_closed = count_closed + 1

                # calculate distance between the endpoints of reference field line and of reconstructed field line
                num = np.sqrt(
                    (fieldline_rec[len_rec - 1, 1] - fieldline_ref[len_ref - 1, 1])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 0] - fieldline_ref[len_ref - 1, 0])
                    ** 2.0
                    + (fieldline_rec[len_rec - 1, 2] - fieldline_ref[len_ref - 1, 2])
                    ** 2.0
                )

                # calculate length of reference field line
                div = 0.0
                for i in range(0, len_ref - 1):
                    div = div + np.sqrt(
                        (fieldline_ref[i, 1] - fieldline_ref[i + 1, 1]) ** 2.0
                        + (fieldline_ref[i, 0] - fieldline_ref[i + 1, 0]) ** 2.0
                        + (fieldline_ref[i, 2] - fieldline_ref[i + 1, 2]) ** 2.0
                    )

                # divide distance between endpoints by length of reference field line
                # gives error between endpoints as percentage of length of reference field line

                temp = num / div

                # add to counter if error is smaller than 10 percent
                if temp <= 0.1:
                    count = count + 1

    # return number of footpoints with error smaller than 10 percent as percentage of all footpoints
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
    """
    Returns line-of-sigth integration (using composite trapezoidal rule) with respect to
    the z-direction for pressure and density for two given magnetic field models
    (reference model and reconstructed model) in the order:
        (1) Pressure surface data for reference field
        (2) Density surface data for reference field
        (3) Pressure surface data for reconstructed field
        (4) Density surface data for reconstructed field
    Also, prints the Pearson Correlation Coefficient (reference and actual) for the line-of-sight integration
    for both pressure and density between the reference and the recreated model.
    """

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
        "Pearson Correlation reference value for pressure",
        pearsonr(pres_surface_ref.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation reference value for density",
        pearsonr(den_surface_ref.flatten(), den_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for pressure",
        pearsonr(pres_surface_rec.flatten(), pres_surface_ref.flatten()),
    )
    print(
        "Pearson Correlation actual value for density",
        pearsonr(den_surface_rec.flatten(), den_surface_ref.flatten()),
    )

    return pres_surface_ref, den_surface_ref, pres_surface_rec, den_surface_rec
