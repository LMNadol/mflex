import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, colors

from scipy.ndimage import maximum_filter, label, find_objects, minimum_filter

from msat.pyvis.fieldline3d import fieldline3d

from mhsflex.field3d import b3d
from mhsflex.pp import btemp, bpressure, bdensity


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

        self.read()

        self.nf = min(self.nx, self.ny)

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

        self.detect_footpoints()

        b3d(self)

    def read(self):

        with open(self.path2file, "rb") as file:
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
