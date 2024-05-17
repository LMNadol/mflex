import numpy as np
from dataclasses import dataclass


@dataclass
class Magfield:
    nx: int
    ny: int
    nz: int
    bx: np.ndarray  # type: ignore
    by: np.ndarray  # type: ignore
    bz: np.ndarray  # type: ignore
    bzdx: np.ndarray  # type: ignore
    bzdy: np.ndarray  # type: ignore
    bzdz: np.ndarray  # type: ignore
    x: np.ndarray  # type: ignore
    y: np.ndarray  # type: ignore
    z: np.ndarray  # type: ignore
