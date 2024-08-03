from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from functools import cached_property

from mhsflex.b3d import b3d
from mhsflex.field2d import Field2dData

from mhsflex.switch import f, dfdz, f_low, dfdz_low

T_PHOTOSPHERE = 5600.0  # Photospheric temperature
T_CORONA = 2.0 * 10.0**6  # Coronal temperature

G_SOLAR = 272.2  # kg/m^3
KB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
MBAR = 1.67262 * 10**-27  # mean molecular weight (proton mass)
RHO0 = 2.7 * 10**-4  # plasma density at z = 0 in kg/(m^3)
P0 = T_PHOTOSPHERE * KB * RHO0 / MBAR  # plasma pressure in kg/(s^2 m)
MU0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)


@dataclass
class Field3dData:
    nx: np.int32
    ny: np.int32
    nz: np.int32
    nf: np.int32
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray
    field: np.ndarray
    dfield: np.ndarray

    a: float
    b: float
    alpha: float
    z0: np.float64
    deltaz: np.float64

    @cached_property
    def btemp(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))

        return T0 + T1 * np.tanh((self.z - self.z0) / self.deltaz)

    @cached_property
    def bpressure(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))
        H = KB * T0 / (MBAR * G_SOLAR) * 10**-6

        q1 = self.deltaz / (2.0 * H * (1.0 + T1 / T0))
        q2 = self.deltaz / (2.0 * H * (1.0 - T1 / T0))
        q3 = self.deltaz * (T1 / T0) / (H * (1.0 - (T1 / T0) ** 2))

        p1 = (
            2.0
            * np.exp(-2.0 * (self.z - self.z0) / self.deltaz)
            / (1.0 + np.exp(-2.0 * (self.z - self.z0) / self.deltaz))
            / (1.0 + np.tanh(self.z0 / self.deltaz))
        )
        p2 = (1.0 - np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh((self.z - self.z0) / self.deltaz)
        )
        p3 = (1.0 + T1 / T0 * np.tanh((self.z - self.z0) / self.deltaz)) / (
            1.0 - T1 / T0 * np.tanh(self.z0 / self.deltaz)
        )

        return (p1**q1) * (p2**q2) * (p3**q3)

    @cached_property
    def bdensity(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))

        temp0 = T0 - T1 * np.tanh(self.z0 / self.deltaz)
        dummypres = self.bpressure
        dummytemp = self.btemp

        return dummypres / dummytemp * temp0

    @cached_property
    def dpressure(self) -> np.ndarray:

        bz_matrix = self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        return -f(z_matrix, self.z0, self.deltaz, self.a, self.b) * bz_matrix**2.0 / 2.0

    @cached_property
    def ddensity(self) -> np.ndarray:

        bz_matrix = self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        bdotbz_matrix = np.zeros_like(bz_matrix)

        bdotbz_matrix = (
            self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
            * self.dfield[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
            + self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 1]
            * self.dfield[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 1]
            + self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
            * self.dfield[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
        )

        return (
            dfdz(z_matrix, self.z0, self.deltaz, self.a, self.b) * bz_matrix**2 / 2.0
            + f(z_matrix, self.z0, self.deltaz, self.a, self.b) * bdotbz_matrix
        )

    @cached_property
    def fpressure(self) -> np.ndarray:

        bp_matrix = np.zeros_like(self.dpressure)
        bp_matrix[:, :, :] = self.bpressure

        B0 = self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        PB0 = (B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

        return B0**2.0 / MU0 * 10**-8 * (BETA0 / 2.0 * bp_matrix + self.dpressure)

    @cached_property
    def fdensity(self) -> np.ndarray:

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        H = KB * T0 / (MBAR * G_SOLAR) * 10**-6
        B0 = self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        PB0 = (B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

        bd_matrix = np.zeros_like(self.ddensity)
        bd_matrix[:, :, :] = self.bdensity

        return (
            B0**2.0
            / (MU0 * G_SOLAR)
            * 10**-14
            * (BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + self.ddensity)
        )


def calculate_magfield(
    field2d: Field2dData,
    a: float,
    b: float,
    alpha: float,
    z0: np.float64,
    deltaz: np.float64,
    asymptotic=True,
    tanh=True,
) -> Field3dData:

    mf3d, dbz3d = b3d(field2d, a, b, alpha, z0, deltaz, asymptotic, tanh)

    data = Field3dData(
        nx=field2d.nx,
        ny=field2d.ny,
        nz=field2d.nz,
        nf=field2d.nf,
        x=field2d.x,
        y=field2d.y,
        z=field2d.z,
        bz=field2d.bz,
        field=mf3d,
        dfield=dbz3d,
        a=a,
        b=b,
        alpha=alpha,
        z0=z0,
        deltaz=deltaz,
    )

    return data
