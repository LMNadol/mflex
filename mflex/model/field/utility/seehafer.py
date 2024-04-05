import numpy as np


def mirror_magnetogram(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    nresol_x: int,
    nresol_y: int,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Given the photospheric magnetic field data_bz,
    returns Seehafer-mirrored Bz field vector.
    Four times the size of original photospheric Bz vector.
    """

    b_arr = np.zeros((2 * nresol_y, 2 * nresol_x))  # [0:2*nresol_y,0:2*nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    if xmin != 0.0 or ymin != 0.0 or not (xmax > 0.0 or ymax > 0.0):
        print("Magneotgram not centred at origin and in correct quadrant")
        raise ValueError

    # Seehafer mirroring

    for ix in range(0, nresol_x):
        for iy in range(0, nresol_y):
            b_arr[nresol_y + iy, nresol_x + ix] = data_bz[iy, ix]
            b_arr[nresol_y + iy, ix] = -data_bz[iy, nresol_x - 1 - ix]
            b_arr[iy, nresol_x + ix] = -data_bz[nresol_y - 1 - iy, ix]
            b_arr[iy, ix] = data_bz[nresol_y - 1 - iy, nresol_x - 1 - ix]

    return b_arr
