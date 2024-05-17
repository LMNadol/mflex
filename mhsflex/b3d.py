import numpy as np
from typing import Tuple
from numba import njit
from mhsflex.Magfield import Magfield


@njit
def odesol(
    z,
    p,
    q,
    z0,
    deltaz,
):

    rplus = p / deltaz
    rminus = q / deltaz
    r = rminus / rplus

    d = np.cosh(2.0 * rplus * z0) + np.multiply(r, np.sinh(2.0 * rplus * z0))

    if z - z0 < 0.0:
        phi = np.cosh(2.0 * rplus * (z0 - z)) + r * np.sinh(2.0 * rplus * (z0 - z))
        dphi = (
            -2.0
            * rplus
            * (np.sinh(2.0 * rplus * (z0 - z)) + r * np.cosh(2.0 * rplus * (z0 - z)))
        )
    else:
        phi = np.exp(-2.0 * rminus * (z - z0))
        dphi = -2.0 * np.multiply(rminus, np.exp(-2.0 * rminus * (z - z0)))

    return phi / d, dphi / d


def extend(bzp):

    bzpb = np.ndarray((2 * bzp.shape[0], 2 * bzp.shape[1]))

    bzpb[bzp.shape[0] : 2 * bzp.shape[0], bzp.shape[1] : 2 * bzp.shape[1]] = bzp[:, :]
    bzpb[0 : bzp.shape[0], 0 : bzp.shape[1]] = np.flip(np.flip(bzp, axis=0), axis=1)
    bzpb[bzp.shape[0] : 2 * bzp.shape[0], 0 : bzp.shape[1]] = -np.flip(bzp, axis=1)
    bzpb[0 : bzp.shape[0], bzp.shape[1] : 2 * bzp.shape[1]] = -np.flip(bzp, axis=0)

    return bzpb


def fftc(bzp):

    signal = np.fft.fftshift(np.fft.fft2(bzp) / bzp.shape[0] / bzp.shape[1])

    ones = np.array(
        [
            [1 if (i + j) % 2 == 0 else -1 for j in range(signal.shape[0])]
            for i in range(signal.shape[1])
        ]
    )

    signal = np.multiply(signal, ones)

    quad1 = -signal[
        int(bzp.shape[0] / 2) : bzp.shape[0], int(bzp.shape[1] / 2) : bzp.shape[1]
    ]
    quad2 = np.flip(
        signal[int(bzp.shape[0] / 2) : bzp.shape[0], 1 : int(bzp.shape[1] / 2) + 1],
        axis=1,
    )
    quad3 = np.flip(
        signal[1 : int(bzp.shape[0] / 2) + 1, int(bzp.shape[1] / 2) : bzp.shape[1]],
        axis=0,
    )
    quad4 = -np.flip(
        np.flip(
            signal[1 : int(bzp.shape[0] / 2) + 1, 1 : int(bzp.shape[1] / 2) + 1], axis=1
        ),
        axis=0,
    )

    return (quad1 + quad2 + quad3 + quad4).real


def b3d(bfield: Magfield, lx, ly, deltaz, z0, a, b, alpha):

    phi = np.zeros((bfield.ny, bfield.nx, bfield.nz))
    dphidz = np.zeros((bfield.ny, bfield.nx, bfield.nz))

    x = (
        np.arange(2.0 * bfield.nx) * 2.0 * bfield.x[-1] / (2.0 * bfield.nx - 1)
        - bfield.x[-1]
    )
    y = (
        np.arange(2.0 * bfield.ny) * 2.0 * bfield.y[-1] / (2.0 * bfield.ny - 1)
        - bfield.y[-1]
    )

    kx = np.arange(bfield.nx) * np.pi / lx  # [0:nf_max]
    ky = np.arange(bfield.ny) * np.pi / ly
    k2 = np.outer(ky**2, np.ones_like(kx)) + np.outer(np.ones_like(ky), kx**2)
    k2[0, 0] = (np.pi / lx) ** 2 + (np.pi / ly) ** 2

    ky_grid = np.outer(ky, np.ones_like(kx))  # [0:nf_max, 0:nf_max]
    kx_grid = np.outer(np.ones_like(ky), kx)

    p = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a - a * b) - alpha**2)
    q = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a + a * b) - alpha**2)

    for iz, z in enumerate(bfield.z):
        phi[:, :, iz], dphidz[:, :, iz] = odesol(z, p, q, z0, deltaz)

    anm = (
        fftc(extend(bfield.bz[bfield.ny : 2 * bfield.ny, bfield.nx : 2 * bfield.nx, 0]))
        / k2
    )

    sinx = np.sin(np.outer(kx, x))
    siny = np.sin(np.outer(ky, y))
    cosx = np.cos(np.outer(kx, x))
    cosy = np.cos(np.outer(ky, y))

    for iz in range(bfield.nz):
        coeffs = k2 * phi[:, :, iz] * anm
        bfield.bz[:, :, iz] = siny.T @ coeffs @ sinx
        coeffs1 = anm * dphidz[:, :, iz] * ky_grid
        coeffs2 = alpha * anm * phi[:, :, iz] * kx_grid
        bfield.by[:, :, iz] = cosy.T @ (coeffs1 @ sinx) - siny.T @ (coeffs2 @ cosx)
        coeffs3 = anm * dphidz[:, :, iz] * kx_grid
        coeffs4 = alpha * anm * phi[:, :, iz] * ky_grid
        bfield.bx[:, :, iz] = siny.T @ (coeffs3 @ cosx) + cosy.T @ (coeffs4 @ sinx)

        coeffs5 = k2 @ dphidz[:, :, iz] @ anm
        bfield.bzdz[:, :, iz] = siny.T @ (coeffs5 @ sinx)
        coeffs6 = k2 * phi[:, :, iz] * anm * kx_grid
        bfield.bzdx[:, :, iz] = siny.T @ (coeffs6 @ cosx)
        coeffs7 = k2 * phi[:, :, iz] * anm * ky_grid
        bfield.bzdy[:, :, iz] = cosy.T @ (coeffs7 @ sinx)
