import numpy as np
from mhsflex.poloidal import phi, dphidz, phi_hypgeo, phi_low, dphidz_hypgeo, dphidz_low


def mirror(
    field: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Given the photospheric magnetic field data_bz,
    returns Seehafer-mirrored Bz field vector.
    Four times the size of original photospheric Bz vector.
    """

    nx = field.shape[0]
    ny = field.shape[1]

    field_big = np.array((2 * nx, 2 * ny))

    for ix in range(nx):
        for iy in range(ny):
            field_big[ny + iy, nx + ix] = field[iy, ix]
            field_big[ny + iy, ix] = -field[iy, nx - 1 - ix]
            field_big[iy, nx + ix] = -field[ny - 1 - iy, ix]
            field_big[iy, ix] = field[ny - 1 - iy, nx - 1 - ix]

    return field_big


def fftcoeff(
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


def get_phi_dphi(
    z_arr: np.ndarray[np.float64, np.dtype[np.float64]],
    q_arr: np.ndarray[np.float64, np.dtype[np.float64]],
    p_arr: np.ndarray[np.float64, np.dtype[np.float64]],
    nf_max: int,
    nresol_z: int,
    z0: np.float64 | None = None,
    deltaz: np.float64 | None = None,
    kappa: np.float64 | None = None,
    solution: str = "Asym",
):
    phi_arr = np.zeros((nf_max, nf_max, nresol_z))
    dphidz_arr = np.zeros((nf_max, nf_max, nresol_z))

    if solution == "Asym":
        assert z0 is not None and deltaz is not None

        for iz, z in enumerate(z_arr):
            phi_arr[:, :, iz] = phi(z, p_arr, q_arr, z0, deltaz)
            dphidz_arr[:, :, iz] = dphidz(z, p_arr, q_arr, z0, deltaz)

    elif solution == "Hypergeo":

        assert z0 is not None and deltaz is not None

        for iz, z in enumerate(z_arr):
            phi_arr[:, :, iz] = phi_hypgeo(z, p_arr, q_arr, z0, deltaz)
            dphidz_arr[:, :, iz] = dphidz_hypgeo(z, p_arr, q_arr, z0, deltaz)

    elif solution == "Exp":

        assert kappa is not None
        for iy in range(0, int(nf_max)):
            for ix in range(0, int(nf_max)):
                q = q_arr[iy, ix]
                p = p_arr[iy, ix]
                for iz in range(0, int(nresol_z)):
                    z = z_arr[iz]
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, kappa)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, kappa)

    return phi_arr, dphidz_arr
