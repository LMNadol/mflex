import numpy as np
from mhsflex.poloidal import phi, dphidz, phi_hypgeo, phi_low, dphidz_hypgeo, dphidz_low

from mhsflex.field2d import Field2dData
from typing import Tuple


def mirror(
    field: np.ndarray,
) -> np.ndarray:
    """
    Given the photospheric magnetic field data_bz,
    returns Seehafer-mirrored Bz field vector.
    Four times the size of original photospheric Bz vector.
    """

    nx = field.shape[1]
    ny = field.shape[0]

    field_big = np.zeros((2 * ny, 2 * nx))

    for ix in range(nx):
        for iy in range(ny):
            field_big[ny + iy, nx + ix] = field[iy, ix]
            field_big[ny + iy, ix] = -field[iy, nx - 1 - ix]
            field_big[iy, nx + ix] = -field[ny - 1 - iy, ix]
            field_big[iy, ix] = field[ny - 1 - iy, nx - 1 - ix]

    return field_big


def fftcoeff(
    data_bz: np.ndarray,
    nf_max: np.int32,
) -> np.ndarray:
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
    z_arr: np.ndarray,
    q_arr: np.ndarray,
    p_arr: np.ndarray,
    nf_max: np.int32,
    nresol_z: np.int32,
    z0: np.float64 | None = None,
    deltaz: np.float64 | None = None,
    kappa: float | None = None,
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


def b3d(
    field: Field2dData,
    a: float,
    b: float,
    alpha: float,
    z0: np.float64,
    deltaz: np.float64,
) -> Tuple:
    # Calculate 3d magnetic field data using N+N(2024)]

    xmin, xmax, ymin, ymax, zmin, zmax = (
        field.x[0],
        field.x[-1],
        field.y[0],
        field.y[-1],
        field.z[0],
        field.z[-1],
    )

    l = 2.0
    lx = field.nx * field.px * l
    ly = field.ny * field.py * l
    lxn = lx / l
    lyn = ly / l

    # print(self.px, self.py, self.nx, self.ny)

    # print("length scale", l)
    # print("length scale x", lx)
    # print("length scale y", lx)
    # print("length scale x norm", lxn)
    # print("length scale y norm", lxn)

    kx = np.arange(field.nf) * np.pi / lxn
    ky = np.arange(field.nf) * np.pi / lyn
    ones = 0.0 * np.arange(field.nf) + 1.0

    ky_grid = np.outer(ky, ones)
    kx_grid = np.outer(ones, kx)

    k2 = np.outer(ky**2, ones) + np.outer(ones, kx**2)
    k2[0, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2

    p = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a - a * b) - alpha**2)
    q = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a + a * b) - alpha**2)

    seehafer = mirror(field.bz)

    anm = np.divide(fftcoeff(seehafer, field.nf), k2)

    phi, dphi = get_phi_dphi(field.z, q, p, field.nf, field.nz, z0, deltaz)

    bfield = np.zeros((2 * field.ny, 2 * field.nx, field.nz, 3))
    dbz = np.zeros((2 * field.ny, 2 * field.nx, field.nz, 3))

    x_big = np.arange(2.0 * field.nx) * 2.0 * xmax / (2.0 * field.nx - 1) - xmax
    y_big = np.arange(2.0 * field.ny) * 2.0 * ymax / (2.0 * field.ny - 1) - ymax

    sin_x = np.sin(np.outer(kx, x_big))
    sin_y = np.sin(np.outer(ky, y_big))
    cos_x = np.cos(np.outer(kx, x_big))
    cos_y = np.cos(np.outer(ky, y_big))

    # print("k2", k2.shape)
    # print("phi", phi.shape)
    # print("anm", anm.shape)
    # print("siny", sin_y.shape)
    # print("sinx", sin_x.shape)
    # print("x big", self.x_big.shape)
    # print("y big", self.y_big.shape)
    # print("x", self.x.shape)

    # print("b", b.shape)

    for iz in range(0, field.nz):
        coeffs = np.multiply(np.multiply(k2, phi[:, :, iz]), anm)
        bfield[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))

        coeffs1 = np.multiply(np.multiply(anm, dphi[:, :, iz]), ky_grid)
        coeffs2 = alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), kx_grid)
        bfield[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs1, sin_x)) - np.matmul(
            sin_y.T, np.matmul(coeffs2, cos_x)
        )

        coeffs3 = np.multiply(np.multiply(anm, dphi[:, :, iz]), kx_grid)
        coeffs4 = alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), ky_grid)
        bfield[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs3, cos_x)) + np.matmul(
            cos_y.T, np.matmul(coeffs4, sin_x)
        )

        coeffs5 = np.multiply(np.multiply(k2, dphi[:, :, iz]), anm)
        dbz[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs5, sin_x))

        coeffs6 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), anm), kx_grid)
        dbz[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs6, cos_x))

        coeffs7 = np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), anm),
            ky_grid,
        )
        dbz[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs7, sin_x))

    return bfield, dbz