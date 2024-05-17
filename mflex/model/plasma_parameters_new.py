import numpy as np
from mflex.model.field.utility.height_profile import f, dfdz


@np.vectorize
def btemp(z, z0, deltaz, t0, t1):
    return t0 + t1 * np.tanh((z - z0) / deltaz)


@np.vectorize
def bpressure(z, z0, deltaz, h0, t0, t1):
    q1 = deltaz / (2.0 * h0 * (1.0 + t1 / t0))
    q2 = deltaz / (2.0 * h0 * (1.0 - t1 / t0))
    q3 = deltaz * (t1 / t0) / (h0 * (1.0 - (t1 / t0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (z - z0) / deltaz)
        / (1.0 + np.exp(-2.0 * (z - z0) / deltaz))
        / (1.0 + np.tanh(z0 / deltaz))
    )
    p2 = (1.0 - np.tanh(z0 / deltaz)) / (1.0 + np.tanh((z - z0) / deltaz))
    p3 = (1.0 + t1 / t0 * np.tanh((z - z0) / deltaz)) / (
        1.0 - t1 / t0 * np.tanh(z0 / deltaz)
    )

    return (p1**q1) * (p2**q2) * (p3**q3)


@np.vectorize
def bdensity(z, z0, deltaz, h0, t0, t1, t_photo):
    dummypres = bpressure(z, z0, deltaz, h0, t0, t1)
    dummytemp = btemp(z, z0, deltaz, t0, t1)
    return dummypres / dummytemp * t_photo


def btemp_linear(z, temps, heights):
    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    h_index = 0

    for i in range(0, len(heights) - 1):
        if z >= heights[i] and z <= heights[i + 1]:
            h_index = i

    return temps[h_index] + (temps[h_index + 1] - temps[h_index]) / (
        heights[h_index + 1] - heights[h_index]
    ) * (z - heights[h_index])


def bpressure_linear(z, temps, heights, t0, h0):

    h_index = 0

    for i in range(0, len(heights) - 1):
        if heights[i] <= z <= heights[i + 1]:
            h_index = i

    pro = 1.0
    for j in range(0, h_index):
        qj = (temps[j + 1] - temps[j]) / (heights[j + 1] - heights[j])
        expj = -t0 / (h0 * qj)
        tempj = (abs(temps[j] + qj * (heights[j + 1] - heights[j])) / temps[j]) ** expj
        pro = pro * tempj

    q = (temps[h_index + 1] - temps[h_index]) / (
        heights[h_index + 1] - heights[h_index]
    )
    tempz = (abs(temps[h_index] + q * (z - heights[h_index])) / temps[h_index]) ** (
        -t0 / (h0 * q)
    )
    return pro * tempz


def bdensity_linear(z, temps, heights, t0, h0, t_photo):
    dummypres = bpressure_linear(z, temps, heights, t0, h0)
    dummytemp = btemp_linear(z, temps, heights)
    return dummypres / dummytemp * t_photo


@np.vectorize
def deltapres(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
) -> np.float64:
    """
    Returns variation of pressure with height z at given x and y.
    """

    return -f(z, z0, deltaz, a, b) * bz**2.0 / 2.0


@np.vectorize
def deltaden(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
    bzdotgradbz: np.float64,
    l: float,
) -> np.float64:
    """
    Returns variation of density with height z at given x and y.
    """

    return (
        dfdz(z, z0, deltaz, a, b) * bz**2.0 / 2.0 + f(z, z0, deltaz, a, b) * bzdotgradbz
    )
