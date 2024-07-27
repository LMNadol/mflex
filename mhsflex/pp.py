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

# b0 = 500.0  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
# pB0 = (b0 * 10**-4) ** 2 / (2 * mu0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
# beta0 = p0 / pB0  # Plasma Beta, ration plasma to magnetic pressure
# h_photo = h / t0 * t_photosphere


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
