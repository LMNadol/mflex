import numpy as np


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


# @njit
def b3d(
    bc,
    x,
    y,
    a,
    b,
    alpha,
    z0=2000.0 * 10**-3,
    deltaz=200.0 * 10**-3,
    px=90.0 * 10**-3,
    zmax=20000.0 * 10**-3,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns 3D magnetic field vector [By, Bx, Bz] calculated from
    series expansion using anm, phi and dphidz.
    """

    bc_big = mirror(bc)
    nx = bc.shape[0]
    ny = bc.shape[1]
    nz = np.floor(zmax / nx)

    nf = min(nx, ny)

    xmin, xmax, ymin, ymax = 0.0, x[-1], 0.0, y[-1]

    px = xmax / nx
    py = ymax / ny

    l = 2.0  # length scale

    lx = 2 * nx * px * l
    ly = 2 * ny * py * l
    lxn = lx / l
    lyn = ly / l

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnetogram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetogram in wrong quadrant of Seehafer mirroring")

    x_big = np.arange(2.0 * nx) * 2.0 * xmax / (2.0 * nx - 1) - xmax
    y_big = np.arange(2.0 * ny) * 2.0 * ymax / (2.0 * ny - 1) - ymax
    z = np.arange(nz) * zmax / (nz - 1)

    alpha = l * alpha

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf) * np.pi / lxn  # [0:nf_max]
    ky_arr = np.arange(nf) * np.pi / lyn  # [0:nf_max]
    one_arr = 0.0 * np.arange(nf) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2

    p = 0.5 * deltaz * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q = 0.5 * deltaz * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    anm = np.divide(fftcoeff(bc_big, nf), k2_arr)

    phi_arr, dphidz_arr = get_phi_dphi(
        z_arr,
        q,
        p,
        nf_max,
        nresol_z,
        z0=z0,
        deltaz=deltaz,
        solution="Asym",
    )

    b_arr = np.zeros((2 * nresol_y, 2 * nresol_x, nresol_z, 3))

    bz_derivs = np.zeros((2 * nresol_y, 2 * nresol_x, nresol_z, 3))

    sin_x = np.sin(np.outer(kx_arr, x_arr))
    sin_y = np.sin(np.outer(ky_arr, y_arr))
    cos_x = np.cos(np.outer(kx_arr, x_arr))
    cos_y = np.cos(np.outer(ky_arr, y_arr))

    for iz in range(0, nresol_z):
        coeffs = np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        coeffs1 = np.multiply(np.multiply(anm, dphidz_arr[:, :, iz]), ky_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs2 = alpha * np.multiply(np.multiply(anm, phi_arr[:, :, iz]), kx_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs1, sin_x)) - np.matmul(
            sin_y.T, np.matmul(coeffs2, cos_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        # Fieldline3D program was written for order B = [Bx,By,Bz] with indexing [ix,iy,iz] but here we have indexing [iy,ix,iz]
        # so in order to be consistent we have to switch to order B = [Bx,By,Bz] such that fieldline3D program treats X and Y as Y and X consistently

        coeffs3 = np.multiply(np.multiply(anm, dphidz_arr[:, :, iz]), kx_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs4 = alpha * np.multiply(np.multiply(anm, phi_arr[:, :, iz]), ky_grid)
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs3, cos_x)) + np.matmul(
            cos_y.T, np.matmul(coeffs4, sin_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]
        coeffs5 = np.multiply(np.multiply(k2_arr, dphidz_arr[:, :, iz]), anm)
        bz_derivs[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs5, sin_x))

        coeffs6 = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm), kx_grid
        )
        bz_derivs[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs6, cos_x))

        coeffs7 = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm),
            ky_grid,
        )
        bz_derivs[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs7, sin_x))

    return b_arr, bz_derivs
