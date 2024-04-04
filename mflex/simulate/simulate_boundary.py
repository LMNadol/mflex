#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def dipole(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Dipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx = np.pi * (x - 1.0)
    yy = np.pi * (y - 1.0)
    mu_x1 = 1.2 / np.pi + 1.0
    mu_y1 = mu_x1
    mu_x2 = -mu_x1
    mu_y2 = -mu_y1
    kappa_x1 = 10.0
    kappa_y1 = kappa_x1
    kappa_x2 = kappa_y1
    kappa_y2 = kappa_x1

    return np.exp(kappa_x1 * np.cos(xx - mu_x1)) / (
        2.0 * np.pi * np.i0(kappa_x1)
    ) * np.exp(kappa_y1 * np.cos(yy - mu_y1)) / (
        2.0 * np.pi * np.i0(kappa_y1)
    ) - np.exp(
        kappa_x2 * np.cos(xx - mu_x2)
    ) / (
        2.0 * np.pi * np.i0(kappa_x2)
    ) * np.exp(
        kappa_y2 * np.cos(yy - mu_y2)
    ) / (
        2.0 * np.pi * np.i0(kappa_y2)
    )


def dipole2(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Dipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx = np.pi * (x / 3.0 - 5.0 / 3.0)
    yy = np.pi * (y / 3.0 - 5.0 / 3.0)
    mu_x1 = 1.2 / np.pi + 1.0
    mu_y1 = mu_x1
    mu_x2 = -mu_x1
    mu_y2 = -mu_y1
    kappa_x1 = 10.0
    kappa_y1 = kappa_x1
    kappa_x2 = kappa_y1
    kappa_y2 = kappa_x1

    return np.exp(kappa_x1 * np.cos(xx - mu_x1)) / (
        2.0 * np.pi * np.i0(kappa_x1)
    ) * np.exp(kappa_y1 * np.cos(yy - mu_y1)) / (
        2.0 * np.pi * np.i0(kappa_y1)
    ) - np.exp(
        kappa_x2 * np.cos(xx - mu_x2)
    ) / (
        2.0 * np.pi * np.i0(kappa_x2)
    ) * np.exp(
        kappa_y2 * np.cos(yy - mu_y2)
    ) / (
        2.0 * np.pi * np.i0(kappa_y2)
    )


def dipole_large(
    x: np.float64,
    y: np.float64,
    default_x: np.float64 = np.float64(5.0),
    default_y: np.float64 = np.float64(5.0),
) -> np.float64:
    """
    Returns value of Dipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    if (x <= 8.0) and (x >= 2.0):
        if (y <= 8.0) and (y >= 2.0):
            return dipole2(x, y)
        else:
            return dipole2(default_x, default_y)
    else:
        return dipole2(default_x, default_y)


def non_periodic(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Non-periodic-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx = np.pi * (x - 1.0)
    yy = np.pi * (y - 1.0)
    mu_x = 1.0
    mu_y = -mu_x
    kappa_x = 20.0
    kappa_y = kappa_x
    mu_x1 = mu_x
    mu_y1 = mu_y
    mu_x2 = -mu_x
    mu_y2 = -mu_y
    mu_x3 = mu_x2
    mu_y3 = mu_y1

    return (
        np.exp(kappa_x * np.cos(xx - mu_x1))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y1))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x2))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y2))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x3))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y3))
        / (2.0 * np.pi * np.i0(kappa_y))
    )


def dalmatian(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Multipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx = np.pi * (x - 1.0)
    yy = np.pi * (y - 1.0)
    mu_x = 1.0
    mu_y = -mu_x
    kappa_x = 10.0
    kappa_y = kappa_x
    mu_x1 = mu_x
    mu_y1 = mu_y
    mu_x2 = -1.2
    mu_y2 = -1.2
    mu_x3 = -2.4
    mu_y3 = 1.9
    mu_x4 = 2.1
    mu_y4 = -1.6
    mu_x5 = -1.5
    mu_y5 = 1.2
    mu_x6 = 2.5
    mu_y6 = 0.0
    mu_x7 = 0.0
    mu_y7 = -2.0
    mu_x8 = -1.0
    mu_y8 = -2.4

    return (
        +np.exp(kappa_x * np.cos(xx - mu_x1))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y1))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x2))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y2))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x3))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y3))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x4))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y4))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x5))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y5))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x6))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y6))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x7))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y7))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x8))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y8))
        / (2.0 * np.pi * np.i0(kappa_y))
    )
