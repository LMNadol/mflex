import matplotlib.pyplot as plt
import numpy as np
from mflex.plot.linetracer.fieldline3D import fieldline3d
from datetime import datetime
import math
from mflex.plot.linetracer.linecheck import fieldlinecheck
from matplotlib import colors


def plot_magnetogram_boundary(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    cmap: str = "bone",
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
    ax.contourf(y_plot, x_plot, data_bz, 1000, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plotname = "/Users/lilli/Desktop/mflex/seminar2024/SOAR_full"
    plt.savefig(plotname, dpi=300)
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
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore
    ax.view_init(30, 245)  # type: ignore
    ax.set_box_aspect((xmax, ymax, 1.0))  # type: ignore
    plt.show()


def plot_fieldlines_grid(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    L: np.float64,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    stepsize: float = 0.1,
    view: str = "top",
    cmap="gist_grey",
):
    """
    Returns 3D plot of photospheric magnetic field including field line extrapolation.
    """

    h1 = L / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = L  # Maximum step length for fieldline3D

    nresol_x = int(data_b.shape[1])
    nresol_y = int(data_b.shape[0])
    nresol_z = int(data_b.shape[2])

    data_bz = data_b[:, :, 0, 2]
    x_arr = np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
    y_arr = np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        x_grid[int(nresol_y / 2) : nresol_y, int(nresol_x / 2) : nresol_x],
        y_grid[int(nresol_y / 2) : nresol_y, int(nresol_x / 2) : nresol_x],
        data_bz[int(nresol_y / 2) : nresol_y, int(nresol_x / 2) : nresol_x],
        1000,
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
    ax.set_box_aspect((xmax, ymax, zmax))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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

            # MAYBE HAVE TO CHANGE ORDER OF X AND Y, SEE DALMATIAN PLOT ROUTINE FOR CORRECT ORDER
            ax.plot(
                fieldline_y,
                fieldline_x,
                fieldline_z,
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    return fig


def plot_fieldlines_issi_rmhd(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    L: np.float64,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    stepsize: float = 0.1,
    view: str = "top",
    cmap="gist_grey",
):
    """
    Returns 3D plot of photospheric magnetic field including field line extrapolation.
    """

    h1 = L / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = L  # Maximum step length for fieldline3D

    nresol_x = int(data_b.shape[1])
    nresol_y = int(data_b.shape[0])
    nresol_z = int(data_b.shape[2])

    data_bz = data_b[:, :, 0, 2]
    x_arr = np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
    y_arr = np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
    z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(
        x_grid,
        y_grid,
        data_bz,
        1000,
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
    ax.set_box_aspect((xmax, ymax, zmax))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    return fig


def plot_fieldlines_issi_analytical(
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
    stepsize: float = 0.1,
    view: str = "top",
    cmap="bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore
    ax.set_box_aspect((xmax, ymax, zmax))  # type: ignore

    return fig


def plot_fieldlines_soar(
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
    stepsize: float = 0.1,
    view: str = "top",
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
    ax.set_box_aspect((xmax, ymax, 4 * zmax))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    # plotname = (
    #     "/Users/lilli/Desktop/mflex/seminar2024/SOAR_BL2_"
    #     + str(a)
    #     + "_"
    #     + str(b)
    #     + "_"
    #     + str(alpha)
    #     + "_"
    #     + view
    #     + "_"
    #     + dt_string
    #     + ".png"
    # )
    # ax.set_zticklabels([])  # type: ignore
    # plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_soar_paper(
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
    stepsize: float = 0.1,
    view: str = "top",
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, zmax])  # type: ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore

    ax.set_box_aspect((xmax, ymax, zmax))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/Paper/SOAR1_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + view
        + "_"
        + dt_string
        + ".png"
    )
    plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_sdo_paper(
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
    stepsize: float = 0.1,
    view: str = "top",
    cmap: str = "bone",
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
    norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)  # type : ignore
    ax.contourf(
        x_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        y_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        norm=norm,
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim(zmin, zmax)  # type: ignore
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(False)  # type : ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
        ax.set_zlabel("")  # type: ignore
        ax.set_xlabel("x", labelpad=15)
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore
        ax.set_xlabel("x", labelpad=30)
        ax.set_ylabel("")
        ax.set_zticks(np.arange(0, zmax + 1, 5))  # type: ignore

    ax.set_box_aspect((xmax, ymax, 3 * zmax))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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

            if np.isclose(fieldline_z[-1], 0.0) and np.isclose(fieldline_z[0], 0.0):
                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline_y,
                    fieldline_x,
                    fieldline_z,
                    color=(0.420, 0.502, 1.000),
                    linewidth=0.5,
                    zorder=4000,
                )
            else:
                ax.plot(
                    fieldline_y,
                    fieldline_x,
                    fieldline_z,
                    color=(0.420, 0.502, 1.000),
                    linewidth=0.5,
                    zorder=4000,
                )

    current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    ax.xaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore

    ax.xaxis.pane.fill = False  # type : ignore
    ax.yaxis.pane.fill = False  # type : ignore
    ax.zaxis.pane.fill = False  # type : ignore

    [t.set_va("center") for t in ax.get_yticklabels()]
    [t.set_ha("center") for t in ax.get_yticklabels()]

    [t.set_va("top") for t in ax.get_xticklabels()]
    [t.set_ha("center") for t in ax.get_xticklabels()]

    [t.set_va("center") for t in ax.get_zticklabels()]
    [t.set_ha("center") for t in ax.get_zticklabels()]

    plotname = "/Users/lilli/Desktop/Colortests/color_test_" + view + ".png"
    plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_sdo_paper_zoom(
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
    z0: np.float64,
    a: float,
    b: float,
    alpha: float,
    stepsize: float = 0.1,
    view: str = "top",
    cmap: str = "bone",
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
    norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)  # type : ignore
    ax.contourf(
        x_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        y_grid[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        data_bz[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x],
        1000,
        norm=norm,
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, 2 * z0])  # type: ignore
    ax.grid(False)  # type : ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
        ax.set_zlabel("")  # type: ignore
        ax.set_xlabel("x", labelpad=15)
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore
        ax.set_xlabel("x", labelpad=75)
        ax.set_ylabel("")
        ax.set_zticks(np.arange(0, 2 * z0 + 1, 2))  # type: ignore

    ax.set_box_aspect((xmax, ymax, 10 * z0))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
    boxedges[1, 2] = 2 * z0

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

            if np.isclose(fieldline_z[-1], 0.0) and np.isclose(fieldline_z[0], 0.0):
                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline_y,
                    fieldline_x,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.25,
                    zorder=4000,
                )
            else:
                ax.plot(
                    fieldline_y,
                    fieldline_x,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.25,
                    zorder=4000,
                )

    current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/Paper/hmi_m_45s_2024_05_07_07_31_30_tai_magnetogram_mflex_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + view
        + "_"
        + dt_string
        + ".png"
    )
    plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_soar_paper_zoom(
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
    z0: np.float64,
    a: float,
    b: float,
    alpha: float,
    stepsize: float = 0.1,
    view: str = "top",
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    ax.set_zlim([zmin, 2 * z0])  # type: ignore

    if view == "top":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
    if view == "angular":
        ax.view_init(30, 240, 0)  # type: ignore
    if view == "side":
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore

    ax.set_box_aspect((xmax, ymax, 4 * z0))  # type: ignore

    x_0 = 1.0 * 10**-8
    y_0 = 1.0 * 10**-8
    dx = stepsize
    dy = stepsize
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
    boxedges[1, 2] = 2 * z0

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
                color="magenta",
                linewidth=0.25,
                zorder=4000,
            )

    current_time = datetime.now()
    dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    plotname = (
        "/Users/lilli/Desktop/Paper/SOAR1_"
        + str(a)
        + "_"
        + str(b)
        + "_"
        + str(alpha)
        + "_"
        + view
        + "_"
        + dt_string
        + ".png"
    )
    plt.savefig(plotname, dpi=300)

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
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)  # type: ignore
    # ax.view_init(30, -115, 0)

    ax.set_box_aspect((2, 2, 2))  # type: ignore

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 0.18
    y_0 = 1.2 / np.pi + 0.18
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
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="blue",
                    linewidth=0.5,
                    zorder=4000,
                )

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 1.05
    y_0 = 1.2 / np.pi + 1.05
    dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="blue",
                    linewidth=0.5,
                    zorder=4000,
                )

    """plotname = "/Users/lilli/Desktop/mflex/nw2019_paper/figure3" + name + ".png"
    plt.savefig(plotname, dpi=300)
    """
    ax.set_zlim([zmin, zmax])  # type: ignore
    # ax.view_init(0, -90)

    """
    plotname = "/Users/lilli/Desktop/mflex/nw2019_paper/figure4" + name + ".pnßg"
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


def plot_fieldlines_polar_large(
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
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    # ax.view_init(90, -90)
    ax.view_init(30, -115, 0)  # type: ignore

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = (1.2 / np.pi + 1.0) * 4.0 + 0.8
    y_0 = (1.2 / np.pi + 1.0) * 4.0 + 0.8
    dr = 1.0 / 2.0 / (nlinesmaxr + 1.0)
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
            # print("ystart", y_start, x_start, 0.0)
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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

            fieldline_x = np.zeros(len(fieldline))
            fieldline_y = np.zeros(len(fieldline))
            fieldline_z = np.zeros(len(fieldline))
            fieldline_x[:] = fieldline[:, 1]
            fieldline_y[:] = fieldline[:, 0]
            fieldline_z[:] = fieldline[:, 2]

            # Need to give row direction first/ Y, then column direction/ X

            ax.plot(
                fieldline_x,
                fieldline_y,
                fieldline_z,
                color="blue",
                linewidth=0.5,
                zorder=4000,
            )
            """
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
                )"""

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = (1.2 / np.pi + 1.0) * 4.0 + 0.8 - 2.6
    y_0 = (1.2 / np.pi + 1.0) * 4.0 + 0.8 - 2.6
    dr = 1.0 / 2.0 / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # print("ystart", y_start, x_start, 0.0)
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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

            fieldline_x = np.zeros(len(fieldline))
            fieldline_y = np.zeros(len(fieldline))
            fieldline_z = np.zeros(len(fieldline))
            fieldline_x[:] = fieldline[:, 1]
            fieldline_y[:] = fieldline[:, 0]
            fieldline_z[:] = fieldline[:, 2]

            # Need to give row direction first/ Y, then column direction/ X

            # MAYBE HAVE TO CHANGE ORDER OF X AND Y
            ax.plot(
                fieldline_x,
                fieldline_y,
                fieldline_z,
                color="blue",
                linewidth=0.5,
                zorder=4000,
            )
            """
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
                )"""

    """plotname = "/Users/lilli/Desktop/mflex/nw2019_paper/figure3" + name + ".png"
    plt.savefig(plotname, dpi=300)
    """
    ax.set_zlim([zmin, zmax])  # type: ignore
    ax.set_box_aspect((10, 10, 3))  # type: ignore
    ax.view_init(0, -90)  # type: ignore

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


def plot_fieldlines_polar_thesis(
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
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)  # type: ignore
    # ax.view_init(30, -115, 0)  # type: ignore

    ax.set_box_aspect((2, 2, 2))  # type: ignore

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 0.18
    y_0 = 1.2 / np.pi + 0.18
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
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 1.05
    y_0 = 1.2 / np.pi + 1.05
    dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

    ax.set_zlim([zmin, zmax])  # type: ignore
    ax.view_init(0, -90)  # type: ignore
    plotname = (
        "/Users/lilli/Desktop/Thesis_vonNeu/"
        + name
        + "_"
        + str(a)
        + "_"
        + str(alpha)
        + ".png"
    )
    plt.savefig(plotname, dpi=300)

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


def plot_fieldlines_polar_paper(
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
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)  # type: ignore

    ax.set_box_aspect((2, 2, 2))  # type: ignore

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 0.18
    y_0 = 1.2 / np.pi + 0.18
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
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 1.05
    y_0 = 1.2 / np.pi + 1.05
    dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

    ax.set_zlim([zmin, zmax])  # type: ignore
    # ax.view_init(0, -90)  # type: ignore
    ax.view_init(30, -115, 0)  # type: ignore
    ax.set_yticks([])
    ax.set_yticklabels([])
    plotname = (
        "/Users/lilli/Desktop/Paper/" + name + "_" + str(a) + "_" + str(alpha) + ".png"
    )
    plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_dalmatian_paper(
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
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)  # type: ignore

    ax.set_box_aspect((2, 2, 2))  # type: ignore

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    nlinesmaxr = 2
    nlinesmaxphi = 5
    # x_0 = -1.2 / np.pi + 1.0
    # y_0 = -1.2 / np.pi + 1.0
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

                if data_bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]
                # ax.scatter(y_start, x_start, 0.0, s=0.5)
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

                    ax.plot(
                        fieldline_x,
                        fieldline_y,
                        fieldline_z,
                        color=(0.420, 0.502, 1.000),
                        linewidth=0.5,
                        zorder=4000,
                    )

    # nlinesmaxr = 2
    # nlinesmaxphi = 5
    # x_0 = 1.2 / np.pi + 1.05
    # y_0 = 1.2 / np.pi + 1.05
    # dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    # dphi = 2.0 * np.pi / nlinesmaxphi

    # for ilinesr in range(0, nlinesmaxr):
    #     for ilinesphi in range(0, nlinesmaxphi):
    #         x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
    #         y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

    #         if data_bz[int(y_start), int(x_start)] < 0.0:
    #             h1 = -h1

    #         ystart = [y_start, x_start, 0.0]
    #         # ax.scatter(y_start, x_start, 0.0, s=0.5)
    #         # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
    #         fieldline = fieldline3d(
    #             ystart,
    #             data_b,
    #             y_arr,
    #             x_arr,
    #             z_arr,
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
    #         fieldlines = fieldlinecheck(fieldline, 0.0, xmax, 0.0, ymax)

    #         for line in fieldlines:
    #             fieldline_x = np.zeros(len(line))
    #             fieldline_y = np.zeros(len(line))
    #             fieldline_z = np.zeros(len(line))
    #             fieldline_x[:] = line[:, 1]
    #             fieldline_y[:] = line[:, 0]
    #             fieldline_z[:] = line[:, 2]

    #             # Need to give row direction first/ Y, then column direction/ X
    #             ax.plot(
    #                 fieldline_y,
    #                 fieldline_x,
    #                 fieldline_z,
    #                 color="magenta",
    #                 linewidth=0.5,
    #                 zorder=4000,
    #             )

    ax.xaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0  # type : ignore
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.2  # type : ignore

    ax.xaxis.pane.fill = False  # type : ignore
    ax.yaxis.pane.fill = False  # type : ignore
    ax.zaxis.pane.fill = False  # type : ignore

    [t.set_va("center") for t in ax.get_yticklabels()]
    [t.set_ha("center") for t in ax.get_yticklabels()]

    [t.set_va("top") for t in ax.get_xticklabels()]
    [t.set_ha("center") for t in ax.get_xticklabels()]

    [t.set_va("center") for t in ax.get_zticklabels()]
    [t.set_ha("center") for t in ax.get_zticklabels()]

    ax.set_zlim([zmin, zmax])  # type: ignore
    ax.view_init(90, -90)  # type: ignore
    # ax.view_init(30, -115, 0)  # type: ignore

    ax.grid(False)  # type : ignore
    ax.set_zticks([])
    ax.set_zticklabels([])

    plotname = (
        "/Users/lilli/Desktop/Paper/"
        + name
        # + "_2_"
        + str(a)
        + "_"
        + str(alpha)
        + ".png"
    )
    plt.savefig(plotname, dpi=300)

    plt.show()


def plot_fieldlines_polar_paper2(
    data_b: np.ndarray[np.float64, np.dtype[np.float64]],
    data_b2: np.ndarray[np.float64, np.dtype[np.float64]],
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
    cmap: str = "bone",
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
        cmap=cmap,
        offset=0.0,
    )
    # Have to have Xgrid first, Ygrid second, as Contourf expects x-axis/ columns first, then y-axis/rows
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    # ax.set_zlim([zmin, zmax])
    ax.view_init(0, -90)  # type: ignore

    ax.set_box_aspect((2, 2, 2))  # type: ignore

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 0.18
    y_0 = 1.2 / np.pi + 0.18
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
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

            fieldline2 = fieldline3d(
                ystart,
                data_b2,
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
            fieldlines2 = fieldlinecheck(fieldline2, 0.0, xmax, 0.0, ymax)

            for line in fieldlines2:
                fieldline_x2 = np.zeros(len(line))
                fieldline_y2 = np.zeros(len(line))
                fieldline_z2 = np.zeros(len(line))
                fieldline_x2[:] = line[:, 1]
                fieldline_y2[:] = line[:, 0]
                fieldline_z2[:] = line[:, 2]

                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline_x2,
                    fieldline_y2,
                    fieldline_z2,
                    color="greenyellow",
                    linewidth=0.5,
                    zorder=4000,
                )

    nlinesmaxr = 2
    nlinesmaxphi = 5
    x_0 = 1.2 / np.pi + 1.05
    y_0 = 1.2 / np.pi + 1.05
    dr = 1.0 / 2.0 * np.sqrt(1 / 10.0) / (nlinesmaxr + 1.0)
    dphi = 2.0 * np.pi / nlinesmaxphi

    for ilinesr in range(0, nlinesmaxr):
        for ilinesphi in range(0, nlinesmaxphi):
            x_start = x_0 + (ilinesr + 1.0) * dr * np.cos(ilinesphi * dphi)
            y_start = y_0 + (ilinesr + 1.0) * dr * np.sin(ilinesphi * dphi)

            if data_bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # ax.scatter(y_start, x_start, 0.0, s=0.5)
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
                    fieldline_x,
                    fieldline_y,
                    fieldline_z,
                    color="magenta",
                    linewidth=0.5,
                    zorder=4000,
                )

            fieldline2 = fieldline3d(
                ystart,
                data_b2,
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
            fieldlines2 = fieldlinecheck(fieldline2, 0.0, xmax, 0.0, ymax)

            for line in fieldlines2:
                fieldline_x2 = np.zeros(len(line))
                fieldline_y2 = np.zeros(len(line))
                fieldline_z2 = np.zeros(len(line))
                fieldline_x2[:] = line[:, 1]
                fieldline_y2[:] = line[:, 0]
                fieldline_z2[:] = line[:, 2]

                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline_x2,
                    fieldline_y2,
                    fieldline_z2,
                    color="greenyellow",
                    linewidth=0.5,
                    zorder=4000,
                )

    ax.set_zlim([zmin, zmax])  # type: ignore
    # ax.view_init(0, -90)  # type: ignore
    ax.view_init(30, -115, 0)  # type: ignore
    # ax.set_yticks([])
    # ax.set_yticklabels([])
    plotname = (
        "/Users/lilli/Desktop/Paper/" + name + "_" + str(a) + "_" + str(alpha) + ".png"
    )
    plt.savefig(plotname, dpi=300)

    plt.show()
