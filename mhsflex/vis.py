from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, colors

from mhsflex.field3d import Field3dData

from msat.pyvis.fieldline3d import fieldline3d

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


def plot(
    data: Field3dData, footpoints_grid: bool, view: Literal["los", "side", "angular"]
):
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    fig = plt.figure()
    ax = fig.figure.add_subplot(111, projection="3d")
    plot_magnetogram(data, ax)

    if footpoints_grid:
        plot_fieldlines_grid(data, ax)
    else:
        sinks, sources = detect_footpoints(data)
        plot_fieldlines_footpoints(data, sinks, sources, ax)

    if view == "los":
        ax.view_init(90, -90)  # type: ignore

        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("y", labelpad=10)

        ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
        ax.set_yticks(np.arange(0, ymax + 1.0 * 10**-8, ymax / 5))

        ax.set_zticklabels([])  # type: ignore
        ax.set_zlabel("")  # type: ignore

        [t.set_va("center") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("center") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_xlabel("x", labelpad=5)
        ax.set_zlabel("z", labelpad=10)  # type: ignore

        ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
        ax.set_zticks(np.arange(0, zmax + 1.0 * 10**-8, zmax / 5))  # type: ignore

        ax.set_yticklabels([])  # type: ignore
        ax.set_ylabel("")

        [t.set_va("center") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("center") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore

    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore

        ax.set_xticks(np.arange(0, xmax + 1.0 * 10**-8, xmax / 5))
        ax.set_yticks(np.arange(0, ymax + 1.0 * 10**-8, ymax / 5))
        ax.set_zticks(np.arange(0, zmax + 1.0 * 10**-8, zmax / 5))  # type: ignore

        [t.set_va("bottom") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("right") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("bottom") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("left") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("top") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore

    plt.show()


def detect_footpoints(data: Field3dData) -> Tuple:

    sinks = data.bz.copy()
    sources = data.bz.copy()

    maxmask = sources < sources.max() * 0.4
    sources[maxmask != 0] = 0

    minmask = sinks < sinks.min() * 0.4
    sinks[minmask == 0] = 0

    return sinks, sources


def show_footpoints(data: Field3dData, sinks: np.ndarray, sources: np.ndarray) -> None:

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(
        np.outer(data.y, np.ones(data.ny)).T,
        np.outer(data.y, np.ones(data.nx)),
        data.bz,
        1000,
        cmap=cmap,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tick_params(direction="in", length=2, width=0.5)
    ax.set_box_aspect(ymax / xmax)

    for ix in range(0, data.nx, int(data.nx / 50)):
        for iy in range(0, data.ny, int(data.nx / 50)):
            if sources[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color=c2,
                    s=0.3,
                )
            if sinks[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color=c2,
                    s=0.3,
                )

        plt.show()


def plot_magnetogram(data: Field3dData, ax) -> None:

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x_grid, y_grid = np.meshgrid(x_big, y_big)
    ax.contourf(
        x_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx],
        y_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx],
        data.bz,
        1000,
        # norm=norm,
        cmap=cmap,
        offset=0.0,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.grid(False)
    ax.set_zlim(zmin, zmax)  # type: ignore
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect((ymax, xmax, zmax))  # type : ignore

    ax.xaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.2  # type : ignore
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0  # type : ignore

    ax.xaxis.pane.fill = False  # type : ignore
    ax.yaxis.pane.fill = False  # type : ignore
    ax.zaxis.pane.fill = False  # type : ignore

    [t.set_va("center") for t in ax.get_yticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_yticklabels()]  # type : ignore

    [t.set_va("top") for t in ax.get_xticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_xticklabels()]  # type : ignore

    [t.set_va("center") for t in ax.get_zticklabels()]  # type : ignore
    [t.set_ha("center") for t in ax.get_zticklabels()]  # type : ignore


def plot_fieldlines_footpoints(
    data: Field3dData, sinks: np.ndarray, sources: np.ndarray, ax
):
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ix in range(0, data.nx, int(data.nx / 30)):
        for iy in range(0, data.ny, int(data.ny / 30)):
            if sources[iy, ix] != 0 or sinks[iy, ix] != 0:

                x_start = ix / (data.nx / xmax)
                y_start = iy / (data.ny / ymax)

                if data.bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]

                fieldline = fieldline3d(
                    ystart,
                    data.field,
                    y_big,
                    x_big,
                    data.z,
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
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )
                else:
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color=c2,
                        linewidth=0.5,
                        zorder=4000,
                    )


def plot_fieldlines_grid(data: Field3dData, ax) -> None:

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x_0 = 0.0
    y_0 = 0.0
    dx = xmax / 18.0
    dy = ymax / 18.0

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if data.bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data.field,
                y_big,
                x_big,
                data.z,
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
                ax.plot(
                    fieldline[:, 1],
                    fieldline[:, 0],
                    fieldline[:, 2],
                    color=c2,
                    linewidth=0.5,
                    zorder=4000,
                )
            else:
                ax.plot(
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
