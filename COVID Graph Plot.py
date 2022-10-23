"""
Combines the results from several simulation runs and plots the combination with uncertainty regions

Methods
-------
main()
    Main function to define the input data and process the control of the plotting
"""


import pickle
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt


def avg_std(x: np.ndarray, axis: int=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the average values and standard deviation of a 2D array along a given dimension

    Parameters
    ----------
    x : ndarray
        Input 2D array
    axis : int
        Axis to average over

    Returns
    -------
    mean : ndarray
        Average over given dimension
    std : ndarray
        Standard deviation over given dimension
    """
    return np.mean(x, axis=axis), np.std(x, axis=axis) / np.sqrt(x.shape[0])


def shape_correction(x: list):
    """
    Corrects the shape for arrays with different shapes by copying values across each array given by their index

    Parameters
    ----------
    x : list
        Arrays of different shapes

    Returns
    -------
    x : list
        list of arrays with matching shapes
    """
    for i, j in np.ndindex(len(x), len(x)):
        if i != j:
            indices = np.argwhere(np.invert(np.isin(x[i][:, 0], x[j][:, 0])))[:, 0]
            x[j] = np.vstack((x[j], x[i][indices, :]))
            x[j] = x[j][np.argsort(x[j][:, 0])]

    return x


def subplot(ax: plt.Axes, x: np.ndarray, y_avg: np.ndarray, y_std: np.ndarray, title: str, label='', vlines: list=[], vcolor: list=[], vline_text: list=[], vtext_xy: list=[], major=20, minor=16):
    """
    Fills out each subplot with data and formats it

    Parameters
    ----------
    ax : axes object
        Axes to be plotted
    x : ndarray
        x data
    y_avg : ndarray
        Average y values
    y_std : ndarray
        Standard deviation of y values
    title : str
        Title of subplot
    label : str, default = ''
        Label for plot
    vlines : list, default = []
        x data vertical lines to plotted
    vcolor : list, default = []
        Colour of vertical lines
    vline_text : list, default = []
        Text paired with vertical lines
    vtext_xy : list, default = []
        Relative coordinates of text
    major : int, default = 20
        Size of primary text
    minor : int, default = 16
        Size of secondary text
    """
    ax.set_title(title, fontsize=major)
    ax.plot(x, y_avg, label=label)

    # Plot uncertainty region
    if type(y_std) == np.ndarray:
        ax.fill_between(x, y_avg - y_std, y_avg + y_std, alpha=0.5)

    ax.tick_params(axis='y', labelsize=minor)

    # Plot vertical lines and text
    for i in range(len(vlines)):
        ax.axvline(vlines[i], c=vcolor[i], alpha=0.3)

        if len(vline_text) != 0:
            ax.text(vtext_xy[i, 0], vtext_xy[i, 1], vline_text[i], transform=ax.transAxes, fontsize=minor)


def plot(new_infections: np.ndarray, recovered: np.ndarray, rt: np.ndarray, tests: np.ndarray, isolating: np.ndarray, vaccinated: np.ndarray):
    """
    Plots averaged data from simulation

    Parameters
    ----------
    new_infections : ndarray
        Number of infections per day
    recovered : ndarray
        Number of recoveries per day
    rt : ndarray
        Effective reproductive value per day
    tests : ndarray
        Number of tests and positive tests per day
    isolating : ndarray
        Number of isolating people per day
    vaccinated : ndarray
        Number of people with first dose and both doses per day
    """
    nrows = 3
    minor = 16
    vlines = []
    vcolor = []
    vline_text = []
    vtext_xy = []
    # Lockdowns
    vlines = [90, 150, 300, 330, 360, 420]
    vcolor = ['r', 'g', 'r', 'g', 'r', 'g']
    # Social distancing
    # vlines = [60]
    # vcolor = ['r']
    # vline_text = ['Social distancing starts']
    # vtext_xy = np.array(((0.11, 0.85),))

    if np.count_nonzero(tests[0, :, 1]) != 0:
        nrows += 2

    if np.count_nonzero(vaccinated[:, 1]) != 0:
        nrows += 1

    _, ax = plt.subplots(nrows=nrows, sharex=True, constrained_layout=True, figsize=(16, 3 * nrows))

    # Infections
    infections_avg_std = avg_std(new_infections)
    subplot(ax[0], np.arange(new_infections.shape[1]), infections_avg_std[0], infections_avg_std[1], 'a) New Infections', vlines=vlines, vcolor=vcolor, vline_text=vline_text, vtext_xy=vtext_xy)

    # Recoveries
    recovered_avg_std = avg_std(recovered)
    subplot(ax[1], np.arange(recovered.shape[1]), recovered_avg_std[0], recovered_avg_std[1], 'b) Recovered', vlines=vlines, vcolor=vcolor, vline_text=vline_text, vtext_xy=vtext_xy)

    # Effective reproductive value
    rt_avg_std = avg_std(rt)
    subplot(ax[2], rt_avg_std[0][:, 0], rt_avg_std[0][:, 1], rt_avg_std[1][:, 1], r'c) $R_t$', vlines=vlines, vcolor=vcolor, vline_text=vline_text, vtext_xy=vtext_xy)
    ax[2].text(0.75, 0.75, f'$R_t$ final value: {round(rt_avg_std[0][-1, 1], 2)}', transform=ax[2].transAxes, fontsize=minor)

    # Tests and number of isolating people
    if np.count_nonzero(tests[0, :, 1]) != 0:
        tests_avg_std = avg_std(tests)
        subplot(ax[3], tests_avg_std[0][:, 0], tests_avg_std[0][:, 1], tests_avg_std[1][:, 1], 'd) Testing', label='Tests taken')
        subplot(ax[3], tests_avg_std[0][:, 0], tests_avg_std[0][:, 2], tests_avg_std[1][:, 2], 'd) Testing', label='Positive tests')
        ax[3].legend(loc='upper left', fontsize=minor)

        isolating_avg_std = avg_std(isolating)
        subplot(ax[4], isolating_avg_std[0][:, 0], isolating_avg_std[0][:, 1], isolating_avg_std[1][:, 1], 'e) Isolating')

    # Vaccinations
    if np.count_nonzero(vaccinated[:, 1]) != 0:
        subplot(ax[-1], vaccinated[:, 0], vaccinated[:, 1] + vaccinated[:, 2], None, 'd) Vaccinated', label='First dose')
        subplot(ax[-1], vaccinated[:, 0], vaccinated[:, 2], None, 'd) Vaccinated', label='Second dose')
        ax[-1].legend(loc='upper left', fontsize=minor)

    plt.xlabel('Days', fontsize=minor)
    plt.xticks(fontsize=minor)
    plt.show()


def main():
    # Variables
    days = 600
    files = [22, 35, 36]

    # Constants
    results_location = 'Project 1/Results/Sim Results {}'
    new_infections = []
    recovered = []
    rt = []
    tests = []
    isolating = []

    # Load results from each file
    for i, num_file in enumerate(files):
        with open(results_location.format(num_file) + f'/Day {days - 1}', 'rb') as result_file:
            results = pickle.load(result_file)

        new_infections.append(results['new infections'])
        recovered.append(results['recovered'])
        rt.append(results['rt'])
        tests.append(results['tests'])
        isolating.append(results['isolating'])
        vaccinated = results['vaccinated']

    # Correct shape of Rt
    rt = shape_correction(rt)

    plot(np.array(new_infections), np.array(recovered), np.array(rt), np.array(tests), np.array(isolating), vaccinated)


main()
