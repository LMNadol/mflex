import numpy as np


def save_field(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    data_db: np.ndarray[np.float64, np.dtype[np.float64]],
    path: str,
) -> None:

    np.save("/Users/lilli/Desktop/mflex/results/" + path + "_bfield3d", data_b)
    np.save("/Users/lilli/Desktop/mflex/results/" + path + "_dbzdxdydz3d", data_db)
