from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import pickle

from functools import cached_property

from mhsflex.b3d import b3d

from mhsflex.field2d import Field2dData

from mhsflex.switch import f, dfdz, f_low, dfdz_low

T_PHOTOSPHERE = 5600.0  # Photospheric temperature
T_CORONA = 2.0 * 10.0**6  # Coronal temperature

G_SOLAR = 272.2  # m/s^2
KB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
MBAR = 1.67262 * 10**-27  # mean molecular weight (proton mass)
RHO0 = 2.7 * 10**-4  # plasma density at z = 0 in kg/(m^3)
P0 = T_PHOTOSPHERE * KB * RHO0 / MBAR  # plasma pressure in kg/(s^2 m)
MU0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)

L = 10**6  # Lengthscale Mm


@dataclass
class Field3dData:
    """
    Dataclass of type Field3dData with the following attributes:
    ------------------------------------------------------------------------------------------------------
    Taken from Field2dData object which Field3dData object is based on:
    nx, ny, nz  :   Dimensions of 3D magnetic field, usually nx and ny determined by magnetogram size,
                    while nz defined by user through height to which extrapolation is carried out.
    nf          :   Number of Fourier modes used in calculation of magnetic field vector, usually
                    nf = min(nx, ny) is taken. To do: split into nfx, nfy, sucht that all possible modes
                    in both directions can be used.
    px, py, pz  :   Pixel sizes in x-, y-, z-direction, in normal length scale (Mm).
    x, y, z     :   1D arrays of grid points on which magnetic field is given with shapes (nx,), (ny,)
                    and (nz,) respectively.
    bz          :   Bottom boundary magentogram of size (ny, nx,). Indexing of vectors done in this order,
                    such that, following intuition, x-direction corresponds to latitudinal extension and
                    y-direction to longitudinal extension of the magnetic field.
    ------------------------------------------------------------------------------------------------------
    New attributes:
    field       :   3D magnetic field vector of size (ny, nx, nz, 3,) which contains magnetic field data in
                    Gauss in the shapes By = field(:, :, :, 0), Bx = field(:, :, :, 1) and
                    Bz = field(:, :, :, 2).
    dfield      :   3D vector of size (ny, nx, nz, 3,) containing partial derivatives of Bz in Gauss in the
                    shape Bzdy = field(:, :, :, 0), Bzdx = field(:, :, :, 1) and Bzdz = field(:, :, :, 2).
    a           :   Amplitude parameter of function f(z).
    b           :   Switch parameter of function f(z).
    alpha       :   Poloidal/toroidal ratio parameter in equation (10) of Neukirch and Wiegelmann (2019).
    z0          :   Height around which transtion from non-force-free to force-free takes place.
    deltaz      :   Width of region over which transition from non-force-free to force-free takes place.
    tanh        :   Boolean paramter determining if Low or N+W/N+W-A height profile is used for calculation
                    of plasma pressure, plasma density, current density and Lorentz force.
    ------------------------------------------------------------------------------------------------------
    """

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

    tanh: bool

    def save(self, path):
        for name, attribute in self.__dict__.items():
            name = ".".join((name, "pkl"))
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @classmethod
    def load(cls, path):
        my_model = {}
        for name in cls.__annotations__:
            file_name = ".".join((name, "pkl"))
            with open("/".join((path, file_name)), "rb") as f:
                my_model[name] = pickle.load(f)
        return cls(**my_model)

    @cached_property
    def btemp(self) -> np.ndarray:
        """
        Calculate background temperature according to hyperbolic tangent height profile.
        """

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))

        return T0 + T1 * np.tanh((self.z - self.z0) / self.deltaz)

    @cached_property
    def bpressure(self) -> np.ndarray:
        """
        Calculate background pressure according to hyperbolic tangent height profile.
        """

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        H = KB * T0 / (MBAR * G_SOLAR) / L  # in m

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
        """
        Calculate background density according to hyperbolic tangent height profile.
        """

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin

        temp0 = T0 - T1 * np.tanh(self.z0 / self.deltaz)  # in Kelvin
        dummypres = self.bpressure  # normalised
        dummytemp = self.btemp / temp0  # normalised

        return dummypres / dummytemp

    @cached_property
    def dpressure(self) -> np.ndarray:
        """
        Calculate variation in pressure described by equation (30) in Neukirch and Wiegelmann (2019).
        """

        bz_matrix = self.field[
            self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2
        ]  # in Gauss
        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        B0 = self.field[:, :, 0, 2].max()  # in Gauss

        if self.tanh:
            return (
                -f(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                / 2.0
                * bz_matrix**2.0
                / B0**2.0
            )
        else:
            kappa = 1 / self.z0
            a = self.a * (1 - np.tanh(-self.z0 / self.deltaz))
            return -f_low(z_matrix, a, kappa) / 2.0 * bz_matrix**2.0 / B0**2.0

    @cached_property
    def ddensity(self) -> np.ndarray:
        """
        Calculate variation in pressure described by equation (31) in Neukirch and Wiegelmann (2019).
        """

        bz_matrix = self.field[
            self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2
        ]  # in Gauss
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
        )  # in Gauss**2

        B0 = self.field[:, :, 0, 2].max()  # in Gauss

        if self.tanh:
            return (
                dfdz(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                / 2.0
                * bz_matrix**2
                / B0**2
                + f(z_matrix, self.z0, self.deltaz, self.a, self.b)  # normalised
                * bdotbz_matrix  # normalised
                / B0**2
            )
        else:
            kappa = 1 / self.z0
            a = self.a * (1 - np.tanh(-self.z0 / self.deltaz))
            return (
                dfdz_low(z_matrix, a, kappa) / 2.0 * bz_matrix**2 / B0**2
                + f_low(z_matrix, a, kappa) * bdotbz_matrix / B0**2
            )

    @cached_property
    def fpressure(self) -> np.ndarray:
        """
        Calculate full pressure described by equation (14) in Neukirch and Wiegelmann (2019).
        """

        bp_matrix = np.zeros_like(self.dpressure)
        bp_matrix[:, :, :] = self.bpressure

        B0 = self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
        PB0 = (B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
        BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

        return BETA0 / 2.0 * bp_matrix + self.dpressure  # * (B0 * 10**-4) ** 2.0 / MU0

    @cached_property
    def fdensity(self) -> np.ndarray:
        """
        Calculate full density described by equation (15) in Neukirch and Wiegelmann (2019).
        """

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        H = KB * T0 / (MBAR * G_SOLAR) / L
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
            BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + self.ddensity
        )  #  *(B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L)

    @cached_property
    def lf3D(self) -> np.ndarray:
        """
        Calculate Lorentz force at all grid points.
        """

        return lf3d(self)

    @cached_property
    def j3D(self) -> np.ndarray:
        """
        Calculate current density at all grid points.
        """

        return j3d(self)


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
    """
    Create Field3dData object from Field2dData object and choosen paramters.
    """

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
        tanh=tanh,
    )

    return data


def btemp_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Calculate background temperature using linear interpolation between given temperatures temps
    at given heights heights.
    """

    temp = np.zeros_like(field3d.z)

    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    for iz, z in enumerate(field3d.z):

        h_index = 0

        for i in range(0, len(heights) - 1):
            if z >= heights[i] and z <= heights[i + 1]:
                h_index = i

        temp[iz] = temps[h_index] + (temps[h_index + 1] - temps[h_index]) / (
            heights[h_index + 1] - heights[h_index]
        ) * (z - heights[h_index])

    return temp


def bpressure_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Calculate background pressure resulting from linear interpolated temperature btemp_linear.
    """

    temp = np.zeros_like(field3d.z)

    for iheight, height in enumerate(heights):
        if height == field3d.z0:
            T0 = temps[iheight]

    H = KB * T0 / (MBAR * G_SOLAR) / L

    for iz, z in enumerate(field3d.z):

        h_index = 0

        for i in range(0, len(heights) - 1):
            if heights[i] <= z <= heights[i + 1]:
                h_index = i

        pro = 1.0
        for j in range(0, h_index):
            qj = (temps[j + 1] - temps[j]) / (heights[j + 1] - heights[j])
            expj = -T0 / (H * qj)
            tempj = (
                abs(temps[j] + qj * (heights[j + 1] - heights[j])) / temps[j]
            ) ** expj
            pro = pro * tempj

        q = (temps[h_index + 1] - temps[h_index]) / (
            heights[h_index + 1] - heights[h_index]
        )
        tempz = (abs(temps[h_index] + q * (z - heights[h_index])) / temps[h_index]) ** (
            -T0 / (H * q)
        )

        temp[iz] = pro * tempz

    return temp


def bdensity_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Calculate background density resulting from linear interpolated temperature btemp_linear.
    """

    temp0 = temps[0]
    dummypres = bpressure_linear(field3d, heights, temps)
    dummytemp = btemp_linear(field3d, heights, temps)

    return dummypres / dummytemp * temp0


def fpressure_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Calculate full pressure resulting from linear interpolated temperature btemp_linear.
    """

    bp_matrix = np.zeros_like(field3d.dpressure)
    bp_matrix[:, :, :] = bpressure_linear(field3d, heights, temps)

    B0 = field3d.field[
        :, :, 0, 2
    ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
    PB0 = (B0 * 10**-4) ** 2 / (2 * MU0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
    BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

    return (BETA0 / 2.0 * bp_matrix + field3d.dpressure) * (B0 * 10**-4) ** 2.0 / MU0


def fdensity_linear(
    field3d: Field3dData, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Calculate full density resulting from linear interpolated temperature btemp_linear.
    """

    for iheight, height in enumerate(heights):
        if height == field3d.z0:
            T0 = temps[iheight]

    H = KB * T0 / (MBAR * G_SOLAR) / L
    B0 = field3d.field[
        :, :, 0, 2
    ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
    PB0 = (B0 * 10**-4) ** 2 / (2 * MU0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
    BETA0 = P0 / PB0  # Plasma Beta, ration plasma to magnetic pressure

    bd_matrix = np.zeros_like(field3d.ddensity)
    bd_matrix[:, :, :] = bdensity_linear(field3d, heights, temps)

    return (
        (BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + field3d.ddensity)
        * (B0 * 10**-4) ** 2.0
        / (MU0 * G_SOLAR * L)
    )


def j3d(field3d: Field3dData) -> np.ndarray:
    """
    Return current density, calucated from magnetic field as j = (alpha B + curl(0,0,f(z)Bz))/ mu0.
    """

    j = np.zeros_like(field3d.field)

    j[:, :, :, 2] = field3d.alpha * field3d.field[:, :, :, 2] * 10**-4

    f_matrix = np.zeros_like(field3d.dfield[:, :, :, 0])
    f_matrix[:, :, :] = f(field3d.z, field3d.z0, field3d.deltaz, field3d.a, field3d.b)

    j[:, :, :, 0] = (
        field3d.alpha * field3d.field[:, :, :, 1] * 10**-4
        + f_matrix * field3d.dfield[:, :, :, 0] * 10**-4
    )

    j[:, :, :, 1] = (
        field3d.alpha * field3d.field[:, :, :, 0] * 10**-4
        + f_matrix * field3d.dfield[:, :, :, 1] * 10**-4
    )
    return j / MU0


def lf3d(field3d: Field3dData) -> np.ndarray:
    """
    Calculate Lorentz force.
    """

    j = j3d(field3d)

    lf = np.zeros_like(field3d.field)

    lf[:, :, :, 0] = (
        j[:, :, :, 1] * field3d.field[:, :, :, 2] * 10**-4
        - j[:, :, :, 2] * field3d.field[:, :, :, 1] * 10**-4
    )
    lf[:, :, :, 1] = (
        j[:, :, :, 2] * field3d.field[:, :, :, 0] * 10**-4
        - j[:, :, :, 0] * field3d.field[:, :, :, 2] * 10**-4
    )
    lf[:, :, :, 2] = (
        j[:, :, :, 0] * field3d.field[:, :, :, 1] * 10**-4
        - j[:, :, :, 1] * field3d.field[:, :, :, 0] * 10**-4
    )

    return lf
