from __future__ import annotations

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, colors

from scipy.ndimage import maximum_filter, label, find_objects, minimum_filter

from msat.pyvis.fieldline3d import fieldline3d

from mhsflex.field3d import b3d
from mhsflex.pp import (
    btemp,
    bpressure,
    bdensity,
    dpressure,
    ddensity,
    fpressure,
    fdensity,
)


rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

cmap = colors.LinearSegmentedColormap.from_list(
    "cmap",
    (
        (0.000, (0.000, 0.000, 0.000)),
        (0.500, (0.659, 0.659, 0.659)),
        (1.000, (1.000, 1.000, 1.000)),
    ),
)
c1 = (1.000, 0.224, 0.376)
c2 = (0.420, 0.502, 1.000)
norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)

t_photosphere = 5600.0  # Photospheric temperature
t_corona = 2.0 * 10.0**6  # Coronal temperature

g_solar = 272.2  # kg/m^3
kB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
mbar = 1.67262 * 10**-27  # mean molecular weight (proton mass)
rho0 = 2.7 * 10**-4  # plasma density at z = 0 in kg/(m^3)
p0 = t_photosphere * kB * rho0 / mbar  # plasma pressure in kg/(s^2 m)
mu0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)


class Field3d:

    def __init__(
        self,
        path2file: str,
        a,
        b,
        alpha,
        z0,
        deltaz,
    ):

        self.path2file = path2file

        self.other_read(path2file)

        print(self.nx)
        print(self.ny)
        print(self.nz)
        print(self.px)
        print(self.py)
        print(self.pz)

        print(self.bz.shape)
        print(self.x.shape)
        print(self.y.shape)
        print(self.z.shape)

        self.nf = min(self.nx, self.ny)

        self.a = a
        self.b = b
        self.alpha = alpha
        self.z0 = z0
        self.deltaz = deltaz

        t0 = (t_photosphere + t_corona * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        h = kB * t0 / (mbar * g_solar) * 10**-6

        self.b0 = (
            self.bz.max()
        )  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        self.pB0 = (self.b0 * 10**-4) ** 2 / (
            2 * mu0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        self.beta0 = p0 / self.pB0  # Plasma Beta, ration plasma to magnetic pressure
        self.h_photo = h / t0 * t_photosphere

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

        self.field = np.zeros((2 * self.ny, 2 * self.nx, self.nz, 3))
        self.dfield = np.zeros((2 * self.ny, 2 * self.nx, self.nz, 3))

        self.x_big = (
            np.arange(2.0 * self.nx) * 2.0 * self.xmax / (2.0 * self.nx - 1) - self.xmax
        )
        self.y_big = (
            np.arange(2.0 * self.ny) * 2.0 * self.ymax / (2.0 * self.ny - 1) - self.ymax
        )

        self.sinks = self.bz.copy()
        self.sources = self.bz.copy()

        self.btemp = np.zeros_like(self.z)
        self.bpres = np.zeros_like(self.z)
        self.bden = np.zeros_like(self.z)
        self.dpres = np.zeros_like(
            self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
        )
        self.dden = np.zeros_like(
            self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
        )
        self.fpres = np.zeros_like(
            self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
        )
        self.fden = np.zeros_like(
            self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
        )

        self.detect_footpoints()

        b3d(self)

    @classmethod
    def from_fits(cls, name) -> Field3d:

        with astroopen("./obs/" + name + ".fits") as data:

            image = getdata("./obs/" + name + ".fits", ext=False)

            hdr = data[0].header

            dist = hdr["DSUN_OBS"]

            px_unit = hdr["CUNIT1"]
            py_unit = hdr["CUNIT2"]
            px_arcsec = hdr["CDELT1"]
            py_arcsec = hdr["CDELT2"]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap=cmap, norm=norm)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tick_params(direction="in", length=2, width=0.5)
        ax.invert_yaxis()
        plt.show()

        stx = int(input("First pixel x axis: "))
        lstx = int(input("Last pixel x axis: "))
        sty = int(input("First pixel y axis: "))
        lsty = int(input("Last pixel y axis: "))

        image = image[sty:lsty, stx:lstx]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap=cmap, norm=norm)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tick_params(direction="in", length=2, width=0.5)
        ax.invert_yaxis()
        plt.show()

        nx = image.shape[1]
        ny = image.shape[0]

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        pz = 90.0 * 10**-3

        nz = int(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        np.save("./data/" + name + "-param.npy", np.array((nx, ny, nz, px, py, pz)))
        np.save("./data/" + name + "-image.npy", image)
        np.save("./data/" + name + "-x.npy", x)
        np.save("./data/" + name + "-y.npy", y)
        np.save("./data/" + name + "-z.npy", z)

    def read(self):

        with open(self.path2file, "rb") as file:

            shape = np.fromfile(file, count=3, dtype=np.int32)

            self.nx, self.ny, self.nz = (int(n) for n in shape)

            print(self.nx, self.ny, self.nz)

            pixel = np.fromfile(file, count=3, dtype=np.float64)

            self.px, self.py, self.pz = (float(p) for p in pixel)

            print(self.px, self.py, self.pz)

            print(self.ny * self.nx)

            self.bz = np.fromfile(
                file,
                count=self.ny * self.nx,
                dtype=np.float64,
            ).reshape((self.ny, self.nx))

            self.x = np.fromfile(file, count=self.nx, dtype=np.float64)
            self.y = np.fromfile(file, count=self.ny, dtype=np.float64)
            self.z = np.fromfile(file, count=self.nz, dtype=np.float64)

    def other_read(self, name):

        param = np.load("./data/" + name + "-param.npy")
        self.nx, self.ny, self.nz, self.px, self.py, self.pz = (
            np.int32(param[0]),
            np.int32(param[1]),
            np.int32(param[2]),
            np.float64(param[3]),
            np.float64(param[4]),
            np.float64(param[5]),
        )

        self.bz = np.load("./data/" + name + "-image.npy")
        self.x = np.load("./data/" + name + "-x.npy")
        self.y = np.load("./data/" + name + "-y.npy")
        self.z = np.load("./data/" + name + "-z.npy")

    def plot(self, footpoints, view):

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.plot_magnetogram()

        if footpoints == 1:
            self.plot_fieldlines_footpoints(view)

        if footpoints == 0:
            self.plot_fieldlines_grid(view)

        if view == 0:
            self.ax.view_init(90, -90)  # type: ignore

            self.ax.set_xlabel("x", labelpad=10)
            self.ax.set_ylabel("y", labelpad=10)

            self.ax.set_xticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))
            self.ax.set_yticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))

            self.ax.set_zticklabels([])  # type: ignore
            self.ax.set_zlabel("")  # type: ignore

            [t.set_va("center") for t in self.ax.get_yticklabels()]  # type: ignore
            [t.set_ha("center") for t in self.ax.get_yticklabels()]  # type: ignore

            [t.set_va("center") for t in self.ax.get_xticklabels()]  # type: ignore
            [t.set_ha("center") for t in self.ax.get_xticklabels()]  # type: ignore

        if view == 1:
            self.ax.view_init(0, -90)  # type: ignore
            self.ax.set_xlabel("x", labelpad=5)
            self.ax.set_zlabel("z", labelpad=10)  # type: ignore

            self.ax.set_xticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))
            self.ax.set_zticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))  # type: ignore

            self.ax.set_yticklabels([])  # type: ignore
            self.ax.set_ylabel("")

            [t.set_va("center") for t in self.ax.get_xticklabels()]  # type: ignore
            [t.set_ha("center") for t in self.ax.get_xticklabels()]  # type: ignore

            [t.set_va("center") for t in self.ax.get_zticklabels()]  # type: ignore
            [t.set_ha("center") for t in self.ax.get_zticklabels()]  # type: ignore

        if view == 2:
            self.ax.view_init(30, 240, 0)  # type: ignore

            self.ax.set_xticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))
            self.ax.set_yticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))
            self.ax.set_zticks(np.arange(0, self.zmax + 1.0 * 10**-8, self.zmax / 5))  # type: ignore

            [t.set_va("bottom") for t in self.ax.get_yticklabels()]  # type: ignore
            [t.set_ha("right") for t in self.ax.get_yticklabels()]  # type: ignore

            [t.set_va("bottom") for t in self.ax.get_xticklabels()]  # type: ignore
            [t.set_ha("left") for t in self.ax.get_xticklabels()]  # type: ignore

            [t.set_va("top") for t in self.ax.get_zticklabels()]  # type: ignore
            [t.set_ha("center") for t in self.ax.get_zticklabels()]  # type: ignore

        plt.show()

    def plot_magnetogram(self):

        bphoto = self.field[:, :, 0, 2]

        x_grid, y_grid = np.meshgrid(self.x_big, self.y_big)
        self.ax.contourf(
            x_grid[self.ny : 2 * self.ny, self.nx : 2 * self.nx],
            y_grid[self.ny : 2 * self.ny, self.nx : 2 * self.nx],
            bphoto[self.ny : 2 * self.ny, self.nx : 2 * self.nx],
            1000,
            # norm=norm,
            cmap=cmap,
            offset=0.0,
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")  # type: ignore
        self.ax.grid(False)
        self.ax.set_zlim(self.zmin, self.zmax)  # type: ignore
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_box_aspect((self.ymax, self.xmax, self.zmax))  # type : ignore

        self.ax.xaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
        self.ax.xaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
        self.ax.yaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
        self.ax.yaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
        self.ax.zaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
        self.ax.zaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore

        self.ax.xaxis.pane.fill = False  # type : ignore
        self.ax.yaxis.pane.fill = False  # type : ignore
        self.ax.zaxis.pane.fill = False  # type : ignore

        [t.set_va("center") for t in self.ax.get_yticklabels()]  # type : ignore
        [t.set_ha("center") for t in self.ax.get_yticklabels()]  # type : ignore

        [t.set_va("top") for t in self.ax.get_xticklabels()]  # type : ignore
        [t.set_ha("center") for t in self.ax.get_xticklabels()]  # type : ignore

        [t.set_va("center") for t in self.ax.get_zticklabels()]  # type : ignore
        [t.set_ha("center") for t in self.ax.get_zticklabels()]  # type : ignore

        self.ax.view_init(90, -90)  # type: ignore

    def plot_fieldlines_footpoints(self, view):

        # x_0 = 1.0 * 10**-8
        # y_0 = 1.0 * 10**-8
        # dx = 0.1
        # dy = 0.1
        # nlinesmaxx = math.floor(self.xmax / dx)
        # nlinesmaxy = math.floor(self.ymax / dy)

        h1 = 1.0 / 100.0  # Initial step length for fieldline3D
        eps = 1.0e-8
        # Tolerance to which we require point on field line known for fieldline3D
        hmin = 0.0  # Minimum step length for fieldline3D
        hmax = 1.0  # Maximum step length for fieldline3D

        # Limit fieldline plot to original data size (rather than Seehafer size)
        boxedges = np.zeros((2, 3))

        # # Y boundaries must come first, X second due to switched order explained above
        boxedges[0, 0] = self.ymin
        boxedges[1, 0] = self.ymax
        boxedges[0, 1] = self.xmin
        boxedges[1, 1] = self.xmax
        boxedges[0, 2] = self.zmin
        boxedges[1, 2] = self.zmax

        nlinesmaxr = 2
        nlinesmaxphi = 5
        dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
        dphi = 2.0 * np.pi / nlinesmaxphi

        for ix in range(0, self.nx, int(self.nx / 30)):
            for iy in range(0, self.ny, int(self.nx / 30)):
                if self.sources[iy, ix] != 0 or self.sinks[iy, ix] != 0:

                    x_start = ix / (self.nx / self.xmax)
                    y_start = iy / (self.ny / self.ymax)

                    if self.bz[int(y_start), int(x_start)] < 0.0:
                        h1 = -h1

                    ystart = [y_start, x_start, 0.0]

                    fieldline = fieldline3d(
                        ystart,
                        self.field,
                        self.y_big,
                        self.x_big,
                        self.z,
                        h1,
                        hmin,
                        hmax,
                        eps,
                        oneway=False,
                        boxedge=boxedges,
                        gridcoord=False,
                        coordsystem="cartesian",
                    )  # , periodicity='xy')

                    if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                        fieldline[:, 2][0], 0.0
                    ):
                        # Need to give row direction first/ Y, then column direction/ X
                        self.ax.plot(
                            fieldline[:, 1],
                            fieldline[:, 0],
                            fieldline[:, 2],
                            color=c2,
                            linewidth=0.5,
                            zorder=4000,
                        )
                    else:
                        self.ax.plot(
                            fieldline[:, 1],
                            fieldline[:, 0],
                            fieldline[:, 2],
                            color=c2,
                            linewidth=0.5,
                            zorder=4000,
                        )

    def plot_fieldlines_grid(self, view):

        x_0 = 0.0
        y_0 = 0.0
        dx = self.xmax / 18.0
        dy = self.ymax / 18.0

        nlinesmaxx = math.floor(self.xmax / dx)
        nlinesmaxy = math.floor(self.ymax / dy)

        h1 = 1.0 / 100.0  # Initial step length for fieldline3D
        eps = 1.0e-8
        # Tolerance to which we require point on field line known for fieldline3D
        hmin = 0.0  # Minimum step length for fieldline3D
        hmax = 1.0  # Maximum step length for fieldline3D

        # Limit fieldline plot to original data size (rather than Seehafer size)
        boxedges = np.zeros((2, 3))

        # # Y boundaries must come first, X second due to switched order explained above
        boxedges[0, 0] = self.ymin
        boxedges[1, 0] = self.ymax
        boxedges[0, 1] = self.xmin
        boxedges[1, 1] = self.xmax
        boxedges[0, 2] = self.zmin
        boxedges[1, 2] = self.zmax

        for ilinesx in range(0, nlinesmaxx):
            for ilinesy in range(0, nlinesmaxy):
                x_start = x_0 + dx * ilinesx
                y_start = y_0 + dy * ilinesy

                if self.bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]
                # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
                fieldline = fieldline3d(
                    ystart,
                    self.field,
                    self.y_big,
                    self.x_big,
                    self.z,
                    h1,
                    hmin,
                    hmax,
                    eps,
                    oneway=False,
                    boxedge=boxedges,
                    gridcoord=False,
                    coordsystem="cartesian",
                )  # , periodicity='xy')

                if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                    fieldline[:, 2][0], 0.0
                ):
                    # Need to give row direction first/ Y, then column direction/ X
                    self.ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )
                else:
                    self.ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )

    # def find_center(self):

    #     neighborhood_size = 70
    #     threshold = 1.0

    #     data_max = maximum_filter(self.bz, neighborhood_size)  # mode ='reflect'
    #     maxima = self.bz == data_max
    #     data_min = minimum_filter(self.bz, neighborhood_size)
    #     minima = self.bz == data_min

    #     diff = (data_max - data_min) > threshold
    #     maxima[diff == 0] = 0
    #     minima[diff == 0] = 0

    #     labeled_sources, num_objects_sources = label(maxima)
    #     slices_sources = find_objects(labeled_sources)
    #     x_sources, y_sources = [], []

    #     labeled_sinks, num_objects_sinks = label(minima)
    #     slices_sinks = find_objects(labeled_sinks)
    #     x_sinks, y_sinks = [], []

    #     for dy, dx in slices_sources:
    #         x_center = (dx.start + dx.stop - 1) / 2
    #         x_sources.append(x_center / (self.nx / self.xmax))
    #         y_center = (dy.start + dy.stop - 1) / 2
    #         y_sources.append(y_center / (self.ny / self.ymax))

    #     for dy, dx in slices_sinks:
    #         x_center = (dx.start + dx.stop - 1) / 2
    #         x_sinks.append(x_center / (self.nx / self.xmax))
    #         y_center = (dy.start + dy.stop - 1) / 2
    #         y_sinks.append(y_center / (self.ny / self.ymax))

    #     self.sourcesx = x_sources
    #     self.sourcesy = y_sources

    #     self.sinksx = x_sinks
    #     self.sinksy = y_sinks

    # def plot_ss(self):

    #     x_plot = np.outer(self.y, np.ones(self.nx))
    #     y_plot = np.outer(self.x, np.ones(self.ny)).T

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     # ax.grid(color="white", linestyle="dotted", linewidth=0.5)
    #     ax.contourf(y_plot, x_plot, self.bz, 1000, cmap=cmap)
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     plt.tick_params(direction="in", length=2, width=0.5)
    #     ax.set_box_aspect(self.ymax / self.xmax)

    #     for i in range(0, len(self.sinksx)):

    #         xx = self.sinksx[i]
    #         yy = self.sinksy[i]
    #         ax.scatter(xx, yy, marker="x", color=c2)

    #     for i in range(0, len(self.sourcesx)):

    #         xx = self.sourcesx[i]
    #         yy = self.sourcesy[i]
    #         ax.scatter(xx, yy, marker="x", color=c1)

    #     sinks_label = mpatches.Patch(color=c2, label="Sinks")
    #     sources_label = mpatches.Patch(color=c1, label="Sources")

    #     plt.legend(handles=[sinks_label, sources_label], frameon=False)

    #     plt.show()

    def detect_footpoints(self):

        maxmask = self.sources < self.sources.max() * 0.4
        self.sources[maxmask != 0] = 0

        minmask = self.sinks < self.sinks.min() * 0.4
        self.sinks[minmask == 0] = 0

    def show_footpoints(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(
            np.outer(self.y, np.ones(self.ny)).T,
            np.outer(self.y, np.ones(self.nx)),
            self.bz,
            1000,
            cmap=cmap,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tick_params(direction="in", length=2, width=0.5)
        ax.set_box_aspect(self.ymax / self.xmax)

        for ix in range(0, self.nx, int(self.nx / 50)):
            for iy in range(0, self.ny, int(self.nx / 50)):
                if self.sources[iy, ix] != 0:
                    ax.scatter(
                        ix / (self.nx / self.xmax),
                        iy / (self.ny / self.ymax),
                        color=c2,
                        s=0.3,
                    )
                if self.sinks[iy, ix] != 0:
                    ax.scatter(
                        ix / (self.nx / self.xmax),
                        iy / (self.ny / self.ymax),
                        color=c2,
                        s=0.3,
                    )

        plt.show()

    def batm(self):

        self.btemp = btemp(self)
        self.bpres = bpressure(self)
        self.bden = bdensity(self)

    def vatm(self):

        self.dpres = dpressure(self)
        self.dden = ddensity(self)

    def fatm(self):

        self.btemp = btemp(self)
        self.bpres = bpressure(self)
        self.bden = bdensity(self)
        self.dpres = dpressure(self)
        self.dden = ddensity(self)
        self.fpres = fpressure(self)
        self.fden = fdensity(self)
