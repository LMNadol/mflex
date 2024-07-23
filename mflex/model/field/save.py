import numpy as np


def save_field(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    data_db: np.ndarray[np.float64, np.dtype[np.float64]],
    path: str,
) -> None:

    np.save("/Users/lilli/Desktop/Paper/" + path + "_bfield3d", data_b)
    np.save("/Users/lilli/Desktop/Paper/" + path + "_dbzdxdydz3d", data_db)


def msatformat(
    nx: np.int32,
    ny: np.int32,
    nz: np.int32,
    bx: np.ndarray[np.float64, np.dtype[np.float64]],
    by: np.ndarray[np.float64, np.dtype[np.float64]],
    bz: np.ndarray[np.float64, np.dtype[np.float64]],
    x: np.ndarray[np.float64, np.dtype[np.float64]],
    y: np.ndarray[np.float64, np.dtype[np.float64]],
    z: np.ndarray[np.float64, np.dtype[np.float64]],
    path: str,
) -> None:

    with open(path, "wb") as file:

        np.array([nx, ny, nz], dtype=np.int32).tofile(file)
        bx.T.tofile(file)  # transpose is required
        by.T.tofile(file)  # for memory order
        bz.T.tofile(file)
        x.tofile(file)
        y.tofile(file)
        z.tofile(file)
