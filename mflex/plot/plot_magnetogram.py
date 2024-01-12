#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from mflex.plot.linetracer.fieldline3D import fieldline3d
from datetime import datetime
import math
from typing import List
from mflex.plot.linetracer.linecheck import fieldlinecheck


def plot_magnetogram_boundary(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]], nresol_x: int, nresol_y: int
) -> None:
    """
    Returns 2D plot of photospheric magnetic field at z=0.
    """

    x_arr = np.arange(nresol_x) * (nresol_x) / (nresol_x - 1)
    y_arr = np.arange(nresol_y) * (nresol_y) / (nresol_y - 1)
    x_plot = np.outer(y_arr, np.ones(nresol_x))
    y_plot = np.outer(x_arr, np.ones(nresol_y)).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(y_plot, x_plot, data_bz, 1000, cmap="bone")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def plot_magnetogram_boundary_3D(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
) -> None:
    """
    Returns 3D plot of photospheric magnetic field excluding field line extrapolation.
    """

    x_arr = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    y_arr = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        x_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        y_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        cmap="bone",
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim([zmin, zmax])
    ax.view_init(30, 245)
    ax.set_box_aspect((xmax, ymax, 1.0))
    plt.show()


def plot_fieldlines_grid(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    h1: float,
    hmin: float,
    hmax: float,
    eps: float,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    a: float,
    b: float,
    alpha: float,
    nf_max: float,
):
    """
    Returns 3D plot of photospheric magnetic field including field line extrapolation.
    """

    data_bz = data_b[:, :, 0, 2]
    x_arr = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    y_arr = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        x_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        y_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        cmap="bone",
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim([zmin, zmax])
    ax.view_init(90, -90)
    # ax.view_init(30, 240, 0)
    ax.set_box_aspect((xmax, ymax, 1))

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = 0.5
    dy = 0.5
    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = 0.0
    boxedges[1, 0] = ymax
    boxedges[0, 1] = 0.0
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data_b,
                y_arr,
                x_arr,
                z_arr,
                h1,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )  # , periodicity='xy')

            # Plot fieldlines
            fieldline_x = np.zeros(len(fieldline))
            fieldline_y = np.zeros(len(fieldline))
            fieldline_z = np.zeros(len(fieldline))
            fieldline_x[:] = fieldline[:, 0]
            fieldline_y[:] = fieldline[:, 1]
            fieldline_z[:] = fieldline[:, 2]

            # Need to give row direction first/ Y, then column direction/ X
            ax.plot(
                fieldline_y,
                fieldline_x,
                fieldline_z,
                color="yellow",
                linewidth=0.5,
                zorder=4000,
            )

    """current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/mflex/tests/solo_L2_phi-hrt-blos_20220307T000609_V01_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + str(nf_max)
        + "_"
        + dt_string
        + ".png"
    )
    ax.set_zticklabels([])
    plt.savefig(plotname, dpi=300)"""

    plt.show()


def plot_fieldlines_polar(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    h1: float,
    hmin: float,
    hmax: float,
    eps: float,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    a: float,
    b: float,
    alpha: float,
    nf_max: float,
    name: str,
):
    """
    Returns 3D plot of photospheric magnetic field including field line extrapolation.
    """

    data_bz = data_b[:, :, 0, 2]
    x_arr = np.arange(2 * nresol_x) * (xmax - xmin) / (2 * nresol_x - 1) + xmin
    y_arr = np.arange(2 * nresol_y) * (ymax - ymin) / (2 * nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        x_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        y_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        cmap="bone",
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_zlim([zmin, zmax])
    # ax.view_init(90, -90)
    ax.view_init(30, -115, 0)

    ax.set_box_aspect((1, 1, 1))

    nlinesmaxr = 2
    nlinesmaxphi = 10
    x_0 = 1.2 / np.pi + 1.0
    y_0 = 1.2 / np.pi + 1.0
    dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = 0.0
    boxedges[1, 0] = ymax
    boxedges[0, 1] = 0.0
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data_b,
                y_arr,
                x_arr,
                z_arr,
                h1,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )  # , periodicity='xy')

            # Plot fieldlines
            fieldlines = fieldlinecheck(fieldline, 0.0, xmax, 0.0, ymax)

            for line in fieldlines:
                fieldline_x = np.zeros(len(line))
                fieldline_y = np.zeros(len(line))
                fieldline_z = np.zeros(len(line))
                fieldline_x[:] = line[:, 1]
                fieldline_y[:] = line[:, 0]
                fieldline_z[:] = line[:, 2]

                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline_y,
                    fieldline_x,
                    fieldline_z,
                    color="blue",
                    linewidth=0.5,
                    zorder=4000,
                )

    """plotname = "/Users/lilli/Desktop/mflex/nw2019_paper/figure3" + name + ".png"
    plt.savefig(plotname, dpi=300)
    """
    ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)
    """
    plotname = "/Users/lilli/Desktop/mflex/nw2019_paper/figure4" + name + ".png"
    plt.savefig(plotname, dpi=300)"""

    """current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/mflex/tests/vonmises_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + str(nf_max)
        + "_"
        + dt_string
        + ".png"
    )
    ax.set_zticklabels([])
    plt.savefig(plotname, dpi=300)"""

    plt.show()
