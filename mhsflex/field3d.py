import numpy as np
import matplotlib.pyplot as plt

from mhsflex.bfieldmodel import mirror, fftcoeff, get_phi_dphi


class Field3d:

    def __init__(
        self,
        path2file: str,
        a,
        b,
        alpha,
        z0,
        deltaz,
        model: str = "",
        figure=True,
        fieldlines="all",
        footpoints="grid",
    ):

        with open(path2file, "rb") as file:
            shape = np.fromfile(file, count=3, dtype=np.int32)
            self.nx, self.ny, self.nz = (int(n) for n in shape)
            count = self.nx * self.ny * self.nz
            pixel = np.fromfile(file, count=3, dtype=np.float64)
            self.px, self.py, self.pz = (float(p) for p in pixel)
            self.bz = np.fromfile(file, count=count, dtype=np.float64).reshape(shape)
            self.x = np.fromfile(file, count=self.nx, dtype=np.float64)
            self.y = np.fromfile(file, count=self.ny, dtype=np.float64)
            self.z = np.fromfile(file, count=self.nz, dtype=np.float64)

        self.nf = min(self.nx, self.ny)

        self.model = model
        self.a = a
        self.b = b
        self.alpha = alpha
        self.z0 = z0
        self.deltaz = deltaz

        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = (
            self.x[0],
            self.x[-1],
            self.y[0],
            self.y[-1],
            self.z[0],
            self.z[-1],
        )

        if self.xmin != 0.0 or self.ymin != 0.0 or self.zmin != 0.0:
            raise ValueError("Magnetogram not centred at origin.")
        if not (self.xmax > 0.0 or self.ymax > 0.0 or self.zmax > 0.0):
            raise ValueError("Magnetogram in wrong quadrant for Seehafer mirroring.")

    def b3d(self, *args, **kwargs):
        # Calculate 3d magnetic field data using N+N(2024)

        photosphere = mirror(self.bz)

        l = 2.0
        lx = 2.0 * self.nx * self.px * l
        ly = 2.0 * self.ny * self.px * l
        lxn = lx / l
        lyn = ly / l

        x = np.arange(2.0 * self.nx) * 2.0 * self.xmax / (2.0 * self.nx - 1) - self.xmax
        y = np.arange(2.0 * self.ny) * 2.0 * self.ymax / (2.0 * self.ny - 1) - self.ymax
        z = np.arange(self.nz) * self.zmax / (self.nz - 1)

        kx = np.arange(self.nf) * np.pi / lxn
        ky = np.arange(self.nf) * np.pi / lyn
        ones = 0.0 * np.arange(self.nf) + 1.0

        kx_grid = np.outer(ones, kx)
        ky_grid = np.outer(ky, ones)

        k2 = np.outer(ky**2, ones) + np.outer(ones, kx**2)
        k2[0, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2

        p = (
            0.5
            * self.deltaz
            * np.sqrt(k2 * (1.0 - self.a - self.a * self.b) - self.alpha**2)
        )
        q = (
            0.5
            * self.deltaz
            * np.sqrt(k2 * (1.0 - self.a + self.a * self.b) - self.alpha**2)
        )

        anm = np.divide(fftcoeff(photosphere, self.nf), k2)

        phi, dphi = get_phi_dphi(
            z, q, p, self.nf, self.z0, self.deltaz, solution=self.model
        )

        b = np.zeros((2 * self.ny, 2 * self.nx, self.nz, 3))
        dbz = np.zeros((2 * self.ny, 2 * self.nx, self.nz, 3))

        sin_x = np.sin(np.outer(kx, x))
        sin_y = np.sin(np.outer(ky, y))
        cos_x = np.cos(np.outer(kx, x))
        cos_y = np.cos(np.outer(ky, y))

        for iz in range(0, self.nz):
            coeffs = np.multiply(np.multiply(k2, phi[:, :, iz]), anm)
            b[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))

            coeffs1 = np.multiply(np.multiply(anm, dphi[:, :, iz]), ky_grid)
            coeffs2 = self.alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), kx_grid)
            b[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs1, sin_x)) - np.matmul(
                sin_y.T, np.matmul(coeffs2, cos_x)
            )

            coeffs3 = np.multiply(np.multiply(anm, dphi[:, :, iz]), kx_grid)
            coeffs4 = self.alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), ky_grid)
            b[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs3, cos_x)) + np.matmul(
                cos_y.T, np.matmul(coeffs4, sin_x)
            )

            coeffs5 = np.multiply(np.multiply(k2, dphi[:, :, iz]), anm)
            dbz[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs5, sin_x))

            coeffs6 = np.multiply(
                np.multiply(np.multiply(k2, phi[:, :, iz]), anm), kx_grid
            )
            dbz[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs6, cos_x))

            coeffs7 = np.multiply(
                np.multiply(np.multiply(k2, phi[:, :, iz]), anm),
                ky_grid,
            )
            dbz[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs7, sin_x))

        return b, dbz
