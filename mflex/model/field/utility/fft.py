import numpy as np


def fft_coeff_seehafer(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    nf_max: int,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns coefficients anm for series expansion of 3D magnetic field.
    """

    anm = np.zeros((nf_max, nf_max))

    nresol_y = int(data_bz.shape[0])
    nresol_x = int(data_bz.shape[1])

    signal = np.fft.fftshift(np.fft.fft2(data_bz) / nresol_x / nresol_y)

    for ix in range(0, nresol_x, 2):
        for iy in range(1, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    for ix in range(1, nresol_x, 2):
        for iy in range(0, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    if nresol_x % 2 == 0:
        centre_x = int(nresol_x / 2)
    else:
        centre_x = int((nresol_x + 1) / 2)
    if nresol_y % 2 == 0:
        centre_y = int(nresol_y / 2)
    else:
        centre_y = int((nresol_y + 1) / 2)

    for ix in range(nf_max):
        for iy in range(nf_max):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ).real

    return anm
