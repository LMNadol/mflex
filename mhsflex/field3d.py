import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colors
import math
from mhsflex.bfieldmodel import mirror, fftcoeff, get_phi_dphi
from msat.pyvis.fieldline3d import fieldline3d

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

cmap = colors.LinearSegmentedColormap.from_list(
    "cmap",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#cmap=000000-A8A8A8-FFFFFF
        (0.000, (0.000, 0.000, 0.000)),
        (0.500, (0.659, 0.659, 0.659)),
        (1.000, (1.000, 1.000, 1.000)),
    ),
)


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
            pixel = np.fromfile(file, count=3, dtype=np.float64)
            self.px, self.py, self.pz = (float(p) for p in pixel)
            self.bz = np.fromfile(
                file, count=self.nx * self.ny, dtype=np.float64
            ).reshape((self.ny, self.nx))
            self.x = np.fromfile(file, count=self.nx, dtype=np.float64)
            self.y = np.fromfile(file, count=self.ny, dtype=np.float64)
            self.z = np.fromfile(file, count=self.nz, dtype=np.float64)

        self.nf = min(self.nx, self.ny)

        # print(self.nx)
        # print(self.ny)
        # print(self.nz)
        # print(self.px)
        # print(self.py)
        # print(self.pz)
        # print(self.bz)

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
            raise ValueError("Magnetogram not centered at origin.")
        if not (self.xmax > 0.0 or self.ymax > 0.0 or self.zmax > 0.0):
            raise ValueError("Magnetogram in wrong quadrant for Seehafer mirroring.")

        self.field = np.zeros((2 * self.nx, 2 * self.ny, self.nz, 3))
        self.dfield = np.zeros((2 * self.nx, 2 * self.ny, self.nz, 3))
        self.b3d()

        if figure is True:

            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.plot_magnetogram()
            self.plot_fieldlines()

        plt.show()

    def b3d(self, *args, **kwargs):
        # Calculate 3d magnetic field data using N+N(2024)

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

        seehafer = mirror(self.bz)

        anm = np.divide(fftcoeff(seehafer, self.nf), k2)

        phi, dphi = get_phi_dphi(z, q, p, self.nf, self.nz, self.z0, self.deltaz)

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

        self.field = b
        self.dfield = dbz

    def plot_magnetogram(self):

        x_grid, y_grid = np.meshgrid(self.x, self.y)

        self.ax.contourf(x_grid, y_grid, self.bz, 1000, cmap=cmap, offset=0.0)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")  # type: ignore
        self.ax.grid(False)
        self.ax.set_zlim(self.zmin, self.zmax)  # type: ignore
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)

        self.ax.xaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
        self.ax.xaxis._axinfo["tick"]["outward_factor"] = -0.2  # type : ignore
        self.ax.yaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
        self.ax.yaxis._axinfo["tick"]["outward_factor"] = -0.2  # type : ignore
        self.ax.zaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
        self.ax.zaxis._axinfo["tick"]["outward_factor"] = -0.2  # type : ignore

        self.ax.xaxis.pane.fill = False  # type : ignore
        self.ax.yaxis.pane.fill = False  # type : ignore
        self.ax.zaxis.pane.fill = False  # type : ignore

        [t.set_va("center") for t in self.ax.get_yticklabels()]
        [t.set_ha("center") for t in self.ax.get_yticklabels()]

        [t.set_va("top") for t in self.ax.get_xticklabels()]
        [t.set_ha("center") for t in self.ax.get_xticklabels()]

        [t.set_va("center") for t in self.ax.get_zticklabels()]
        [t.set_ha("center") for t in self.ax.get_zticklabels()]

    def plot_fieldlines(self):

        x_0 = 1.0 * 10**-8
        y_0 = 1.0 * 10**-8
        dx = 0.1
        dy = 0.1
        nlinesmaxx = math.floor(self.xmax / dx)
        nlinesmaxy = math.floor(self.ymax / dy)

        h1 = 2.0 / 100.0  # Initial step length for fieldline3D
        eps = 1.0e-8
        # Tolerance to which we require point on field line known for fieldline3D
        hmin = 0.0  # Minimum step length for fieldline3D
        hmax = 2.0  # Maximum step length for fieldline3D

        # Limit fieldline plot to original data size (rather than Seehafer size)
        boxedges = np.zeros((2, 3))

        # # Y boundaries must come first, X second due to switched order explained above
        boxedges[0, 0] = self.ymin
        boxedges[1, 0] = self.ymax
        boxedges[0, 1] = self.xmin
        boxedges[1, 1] = self.xmax
        boxedges[0, 2] = self.zmin
        boxedges[1, 2] = self.zmax

        x = np.arange(self.nx) * (self.xmax - self.xmin) / (self.nx - 1) + self.xmin
        y = np.arange(self.ny) * (self.ymax - self.ymin) / (self.ny - 1) + self.ymin
        z = np.arange(self.nz) * (self.zmax - self.zmin) / (self.nz - 1) + self.zmin

        # print(nlinesmaxx, nlinesmaxy)
        # for ilinesx in range(0, nlinesmaxx):
        #     for ilinesy in range(0, nlinesmaxy):
        #         x_start = x_0 + dx * ilinesx
        #         y_start = y_0 + dy * ilinesy

        #         if self.bz[int(y_start), int(x_start)] < 0.0:
        #             h1 = -h1

        #         ystart = [y_start, x_start, 0.0]
        #         # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
        #         fieldline = fieldline3d(
        #             ystart,
        #             self.field,
        #             y,
        #             x,
        #             z,
        #             h1,
        #             hmin,
        #             hmax,
        #             eps,
        #             oneway=False,
        #             boxedge=boxedges,
        #             gridcoord=False,
        #             coordsystem="cartesian",
        #         )  # , periodicity='xy')

        #         # Plot fieldlines
        #         fieldline_x = np.zeros(len(fieldline))
        #         fieldline_y = np.zeros(len(fieldline))
        #         fieldline_z = np.zeros(len(fieldline))
        #         fieldline_x[:] = fieldline[:, 0]
        #         fieldline_y[:] = fieldline[:, 1]
        #         fieldline_z[:] = fieldline[:, 2]

        #         if np.isclose(fieldline_z[-1], 0.0) and np.isclose(fieldline_z[0], 0.0):
        #             # Need to give row direction first/ Y, then column direction/ X
        #             self.ax.plot(
        #                 fieldline_y,
        #                 fieldline_x,
        #                 fieldline_z,
        #                 color=(0.420, 0.502, 1.000),
        #                 linewidth=0.5,
        #                 zorder=4000,
        #             )
        #         else:
        #             self.ax.plot(
        #                 fieldline_y,
        #                 fieldline_x,
        #                 fieldline_z,
        #                 color=(0.420, 0.502, 1.000),
        #                 linewidth=0.5,
        #                 zorder=4000,
        #             )

        nlinesmaxr = 2
        nlinesmaxphi = 5
        # x_0 = -1.2 / np.pi + 1.0
        # y_0 = -1.2 / np.pi + 1.0
        dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
        dphi = 2.0 * np.pi / nlinesmaxphi

        list = [
            (1.0, -1.0),
            (-1.2, -1.2),
            (-2.4, 1.9),
            (2.1, -1.6),
            (-1.5, 1.2),
            (2.5, 0.0),
            (0.0, -2.0),
            (-1.0, -2.4),
            (-1.0, 2.4),
        ]

        for xx, yy in list:
            y_0 = yy / np.pi + 1.0
            x_0 = xx / np.pi + 1.0
            for ilinesr in range(0, nlinesmaxr):
                for ilinesphi in range(0, nlinesmaxphi):
                    x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
                    y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

                    if self.bz[int(y_start), int(x_start)] < 0.0:
                        h1 = -h1

                    ystart = [y_start, x_start, 0.0]
                    # ax.scatter(y_start, x_start, 0.0, s=0.5)
                    # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
                    fieldline = fieldline3d(
                        ystart,
                        self.field,
                        y,
                        x,
                        z,
                        h1,
                        hmin,
                        hmax,
                        eps,
                        oneway=False,
                        boxedge=boxedges,
                        gridcoord=False,
                        coordsystem="cartesian",
                    )  # , periodicity='xy')

                    fieldline_x = np.zeros(len(fieldline))
                    fieldline_y = np.zeros(len(fieldline))
                    fieldline_z = np.zeros(len(fieldline))
                    fieldline_x[:] = fieldline[:, 1]
                    fieldline_y[:] = fieldline[:, 0]
                    fieldline_z[:] = fieldline[:, 2]

                    self.ax.plot(
                        fieldline_x,
                        fieldline_y,
                        fieldline_z,
                        color=(0.420, 0.502, 1.000),
                        linewidth=0.5,
                        zorder=4000,
                    )
