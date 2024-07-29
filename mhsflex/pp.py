import numpy as np
from mhsflex.ff import f, dfdz, f_low, dfdz_low

t_photosphere = 5600.0  # Photospheric temperature
t_corona = 2.0 * 10.0**6  # Coronal temperature

g_solar = 272.2  # kg/m^3
kB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
mbar = 1.67262 * 10**-27  # mean molecular weight (proton mass)
rho0 = 2.7 * 10**-4  # plasma density at z = 0 in kg/(m^3)
p0 = t_photosphere * kB * rho0 / mbar  # plasma pressure in kg/(s^2 m)
mu0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)


def btemp(field):

    t0 = (t_photosphere + t_corona * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(field.z0 / field.deltaz))

    return t0 + t1 * np.tanh((field.z - field.z0) / field.deltaz)


def bpressure(field):

    t0 = (t_photosphere + t_corona * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(field.z0 / field.deltaz))
    h = kB * t0 / (mbar * g_solar) * 10**-6

    q1 = field.deltaz / (2.0 * h * (1.0 + t1 / t0))
    q2 = field.deltaz / (2.0 * h * (1.0 - t1 / t0))
    q3 = field.deltaz * (t1 / t0) / (h * (1.0 - (t1 / t0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (field.z - field.z0) / field.deltaz)
        / (1.0 + np.exp(-2.0 * (field.z - field.z0) / field.deltaz))
        / (1.0 + np.tanh(field.z0 / field.deltaz))
    )
    p2 = (1.0 - np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh((field.z - field.z0) / field.deltaz)
    )
    p3 = (1.0 + t1 / t0 * np.tanh((field.z - field.z0) / field.deltaz)) / (
        1.0 - t1 / t0 * np.tanh(field.z0 / field.deltaz)
    )

    return (p1**q1) * (p2**q2) * (p3**q3)


def bdensity(field):

    t0 = (t_photosphere + t_corona * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(field.z0 / field.deltaz))

    temp0 = t0 - t1 * np.tanh(field.z0 / field.deltaz)
    dummypres = bpressure(field)
    dummytemp = btemp(field)

    return dummypres / dummytemp * temp0


def dpressure(field):

    bz_matrix = field.field[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 2]
    z_matrix = np.zeros_like(bz_matrix)
    z_matrix[:, :, :] = field.z

    return -f(z_matrix, field.z0, field.deltaz, field.a, field.b) * bz_matrix**2.0 / 2.0


def ddensity(field):

    bz_matrix = field.field[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 2]
    z_matrix = np.zeros_like(bz_matrix)
    z_matrix[:, :, :] = field.z

    bdotbz_matrix = np.zeros_like(bz_matrix)

    bdotbz_matrix = (
        field.field[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 0]
        * field.dfield[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 0]
        + field.field[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 1]
        * field.dfield[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 1]
        + field.field[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 2]
        * field.dfield[field.ny : 2 * field.ny, field.nx : 2 * field.nx, :, 2]
    )

    return (
        dfdz(z_matrix, field.z0, field.deltaz, field.a, field.b) * bz_matrix**2 / 2.0
        + f(z_matrix, field.z0, field.deltaz, field.a, field.b) * bdotbz_matrix
    )


def fpressure(field):

    bp_matrix = np.zeros_like(field.dpres)
    bp_matrix[:, :, :] = field.bpres

    return field.b0**2.0 / mu0 * 10**-8 * (field.beta0 / 2.0 * bp_matrix + field.dpres)


def fdensity(field):

    t0 = (t_photosphere + t_corona * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    h = kB * t0 / (mbar * g_solar) * 10**-6

    bd_matrix = np.zeros_like(field.dden)
    bd_matrix[:, :, :] = field.bden

    return (
        field.b0**2.0
        / (mu0 * g_solar)
        * 10**-14
        * (field.beta0 / (2.0 * h) * t0 / t_photosphere * bd_matrix + field.dden)
    )
