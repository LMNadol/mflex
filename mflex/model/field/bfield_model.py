import numpy as np
from mflex.model.field.utility.seehafer import mirror_magnetogram
from mflex.model.field.utility.fft import fft_coeff_seehafer
from mflex.model.field.utility.poloidal import (
    phi,
    phi_low,
    dphidz,
    dphidz_low,
    phi_hypgeo,
    dphidz_hypgeo,
)
from mflex.plot.plot_magnetogram import plot_magnetogram_boundary
from numba import njit
from typing import Tuple

"""
# Old version of magnetic field code NOT including normalising length scale L 

def magnetic_field(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    
    # Given the Seehafer-mirrored photospheric magnetic field data_bz,
    # returns 3D magnetic field vector [By, Bx, Bz] calculated from
    # series expansion using anm, phi and dphidz.

    # L = 1.0
    length_scale = 2.0  # * 1.0  Normalising length scale for Seehafer
    length_scale_x = 2.0 * nresol_x * pixelsize_x  # * 1.0
    # Length scale in x direction for Seehafer
    length_scale_y = 2.0 * nresol_y * pixelsize_y  # * 1.0
    # Length scale in y direction for Seehafer
    length_scale_x_norm = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    print("length scale", length_scale)
    print("length scale x", length_scale_x)
    print("length scale y", length_scale_y)
    print("length scale x norm", length_scale_x_norm)
    print("length scale y norm", length_scale_y_norm)

    print("length scale", length_scale)
    print("length scale x", length_scale_x)
    print("length scale y", length_scale_y)
    print("length scale x norm", length_scale_x_norm)
    print("length scale y norm", length_scale_y_norm)

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnetogram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetogram in wrong quadrant of Seehafer mirroring")

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzls = deltaz  # Normalised deltaz

    # lengthscale = 2.0 * L 
    # alpha = alpha * lengthscale / L
    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]

    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]

    one_arr = 0.0 * np.arange(nf_max) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm = fft_coeff_seehafer(
        data_bz_seehafer, k2_arr, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    phi_arr = np.zeros((nf_max, nf_max, nresol_z))
    # [0:nf_max,0:nf_max, 0:nresol_z]
    dphidz_arr = np.zeros((nf_max, nf_max, nresol_z))  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, int(nf_max)):
        for ix in range(0, int(nf_max)):
            q = q_arr[iy, ix]
            p = p_arr[iy, ix]
            for iz in range(0, int(nresol_z)):
                z = z_arr[iz]
                phi_arr[iy, ix, iz] = phi(z, p, q, z0, deltaz)
                dphidz_arr[iy, ix, iz] = dphidz(z, p, q, z0, deltaz)

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

    
    # Using Einstein Summation to get rid of z loop
    # This did not yield any time advantage therefore isn't in use 

    #coeffs = (k2_arr[:, :, np.newaxis] * phi_arr) * anm[:, :, np.newaxis]
    ## m_mat = np.einsum("nmk,mj->jnk", coeffs, sin_x)
    ## b_arr[:, :, :, 2] = np.einsum("jnk,ni->ijk", m_mat, sin_y)
    #b_arr[:, :, :, 2] = np.einsum(
    #    "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs, sin_x), sin_y
    #)

    #coeffs1 = (anm[:, :, np.newaxis] * dphidz_arr) * ky_grid[:, :, np.newaxis]
    #coeffs2 = alpha * (anm[:, :, np.newaxis] * phi_arr) * kx_grid[:, :, np.newaxis]
    ## m_mat1 = np.einsum("nmk,mj->jnk", coeffs1, sin_x)
    ## m_mat2 = np.einsum("nmk,mj->jnk", coeffs2, cos_x)
    ## b_arr[:, :, :, 0] = np.einsum("jnk,ni->ijk", m_mat1, cos_y) - np.einsum(
    ##    "jnk,ni->ijk", m_mat2, sin_y
    ## )
    #b_arr[:, :, :, 0] = np.einsum(
    #    "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs1, sin_x), cos_y
    #) - np.einsum("jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs2, cos_x), sin_y)

    #coeffs3 = (anm[:, :, np.newaxis] * dphidz_arr) * kx_grid[:, :, np.newaxis]
    #coeffs4 = alpha * (anm[:, :, np.newaxis] * phi_arr) * ky_grid[:, :, np.newaxis]
    ## m_mat3 = np.einsum("nmk,mj->jnk", coeffs3, cos_x)
    ## m_mat4 = np.einsum("nmk,mj->jnk", coeffs4, sin_x)
    ## b_arr[:, :, :, 1] = np.einsum("jnk,ni->ijk", m_mat3, sin_y) + np.einsum(
    ##    "jnk,ni->ijk", m_mat4, cos_y
    ## )
    # b_arr[:, :, :, 1] = np.einsum(
    #    "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs3, cos_x), sin_y
    #) + np.einsum("jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs4, sin_x), cos_y)
    
    return b_arr, bz_derivs
"""


# @njit
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
        # Repeat changes for this branch

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
def magnetic_field(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
    L: np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns 3D magnetic field vector [By, Bx, Bz] calculated from
    series expansion using anm, phi and dphidz.
    """

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )

    length_scale = float(2.0 * L)  # Normalising length scale for Seehafer

    length_scale_x = 2.0 * nresol_x * pixelsize_x * L
    length_scale_y = 2.0 * nresol_y * pixelsize_y * L

    length_scale_x_norm = length_scale_x / length_scale
    length_scale_y_norm = length_scale_y / length_scale

    print("length scale", length_scale)
    print("length scale x", length_scale_x)
    print("length scale y", length_scale_y)
    print("length scale x norm", length_scale_x_norm)
    print("length scale y norm", length_scale_y_norm)

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnetogram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetogram in wrong quadrant of Seehafer mirroring")

    print("xmin, xmax, ymin, ymax, zmin, zmax ", xmin, xmax, ymin, ymax, zmin, zmax)

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzls = deltaz  # Normalised deltaz

    alpha = float(alpha * length_scale / L)

    # alpha_issi / L_issi = alpha_lilli / L_lilli,See = alpha_lilli / (2*L_lilli)
    # means that we have alpha_lilli = 2* alpha_issi * L_lilli / L_issi
    # with hopefully L_lilli / L_issi = length_scale / (2 L) = 1.0

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]
    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]
    one_arr = 0.0 * np.arange(nf_max) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale_x_norm) ** 2 + (
        np.pi / length_scale_y_norm
    ) ** 2

    p_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    anm = np.divide(fft_coeff_seehafer(data_bz_seehafer, nf_max), k2_arr)

    phi_arr, dphidz_arr = get_phi_dphi(
        z_arr,
        q_arr,
        p_arr,
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


# @njit
def magnetic_field_low(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    kappa: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
    L: np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns 3D magnetic field vector [By, Bx, Bz] calculated from
    series expansion using anm, phi and dphidz.
    """

    length_scale = 2.0 * L  # Normalising length scale for Seehafer

    length_scale_x = 2.0 * nresol_x * pixelsize_x * L
    length_scale_y = 2.0 * nresol_y * pixelsize_y * L

    length_scale_x_norm = length_scale_x / length_scale
    length_scale_y_norm = length_scale_y / length_scale

    print("length scale", length_scale)
    print("length scale x", length_scale_x)
    print("length scale y", length_scale_y)
    print("length scale x norm", length_scale_x_norm)
    print("length scale y norm", length_scale_y_norm)

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnetogram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetogram in wrong quadrant of Seehafer mirroring")

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]
    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]
    one_arr = 0.0 * np.arange(nf_max) + 1.0
    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale_x_norm) ** 2 + (
        np.pi / length_scale_y_norm
    ) ** 2

    p_arr = 2.0 / kappa * np.sqrt(k2_arr - alpha**2)
    q_arr = 2.0 / kappa * np.sqrt(k2_arr * a)

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm = np.divide(fft_coeff_seehafer(data_bz_seehafer, nf_max), k2_arr)

    phi_arr, dphidz_arr = get_phi_dphi(
        z_arr, q_arr, p_arr, nf_max, nresol_z, kappa=kappa, solution="Exp"
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


# @njit
def magnetic_field_hypergeo(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
    L: np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns 3D magnetic field vector [By, Bx, Bz] calculated from
    series expansion using anm, phi and dphidz.
    """

    length_scale = 2.0 * L  # Normalising length scale for Seehafer

    length_scale_x = 2.0 * nresol_x * pixelsize_x * L
    # Length scale in x direction for Seehafer
    length_scale_y = 2.0 * nresol_y * pixelsize_y * L
    # Length scale in y direction for Seehafer
    length_scale_x_norm = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnotgram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetrogram in wrong quadrant of Seehafer mirroring")

    x_arr = np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    y_arr = np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin

    ratiodzls = deltaz  # Normalised deltaz

    alpha = float(alpha * length_scale / L)  # 2*alpha for Seehafer

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr = np.arange(nf_max) * np.pi / length_scale_x_norm  # [0:nf_max]

    ky_arr = np.arange(nf_max) * np.pi / length_scale_y_norm  # [0:nf_max]

    one_arr = 0.0 * np.arange(nf_max) + 1.0

    ky_grid = np.outer(ky_arr, one_arr)  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(one_arr, kx_arr)  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)

    data_bz_seehafer = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm = np.divide(fft_coeff_seehafer(data_bz_seehafer, nf_max), k2_arr)

    phi_arr, dphidz_arr = get_phi_dphi(
        z_arr, q_arr, p_arr, nf_max, nresol_z, z0=z0, deltaz=deltaz, solution="Hypergeo"
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
