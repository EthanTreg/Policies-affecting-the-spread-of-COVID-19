"""
Uses imported data to fit models to it

main()
    Main function to control the format of the input data and curve fitting
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def linear(x: np.ndarray, m: float, c: float):
    """
    Linear equation

    Parameters
    ----------
    x : ndarray
        Input data
    m : float
        Proportionality factor
    c : float
        y intercept

    Returns
    -------
    y : ndarray
        Output data
    """
    return m * x + c


def logarithmic(x: np.ndarray, A: float, m: float, c: float):
    """
    Logarithmic equation

    A * ln(m * x) + c

    Parameters
    ----------
    x : ndarray
        Input data
    A : float
        Maximum value
    m : float
        Scale
    c : float
        Offset

    Returns
    -------
    y : ndarray
        Output data
    """
    return A * np.log(m * x) + c


def sigmoid(x: np.ndarray, A: float, m: float, c: float):
    """
    Sigmoid function

    A / (1 + exp(-m * x + c))

    Parameters
    x : ndarray
        Input data
    A : float
        Maximum value
    m : float
        Gradient
    c : float
        Offset

    Returns
    -------
    y : ndarray
        Output data
    """
    return A / (1 + np.exp(- m * x + c))


def main():
    # Cummulative days per month
    month_days = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))

    # Retieve data files
    date, cases, cumsum = np.loadtxt("Project 1/Data/data_2022-Apr-01 (1).csv", unpack=True, usecols=(3, 4, 5), skiprows=1, delimiter=',', dtype=str)
    cumsum = cumsum.astype(float)[::-1] * 250e3 / 67e6
    cases = np.delete(cases, -1).astype(float)[::-1] * 250e3 / 70e6

    # Format dates into numbers from 0
    date = np.vstack(np.char.split(date[::-1], sep='-')).astype(int)
    date[:, :2] -= np.array([2021, 1])
    day = date[:, 0] * 365 + month_days[date[:, 1]] + date[:, 2]
    day += 400 - np.min(day)

    # Fit function to data
    popt, pcov = curve_fit(sigmoid, day, cumsum, p0=[180e3, 0.06, 8])
    model_y = sigmoid(np.arange(day[-1]), *popt)

    plt.figure(figsize=(21, 9), constrained_layout=True)
    plt.plot(day, cumsum)
    plt.plot(np.arange(day[-1]), model_y)
    # plt.show()


main()
