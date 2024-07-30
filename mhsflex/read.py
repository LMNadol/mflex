from astropy.io.fits import open as astroopen
from astropy.io.fits import getdata

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc, colors

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
norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)


def read_fits(name):

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
